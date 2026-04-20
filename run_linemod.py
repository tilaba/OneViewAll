# Copyright (c) 2026 [OneViewAll].
# Licensed under the MIT License.
#
# Note: This module is designed to interface with systems licensed under 
# the NVIDIA Source Code License and maintains non-commercial compatibility.

from scipy.spatial.transform import Rotation as R
from Utils import *
import json, os, sys, time
import numpy as np
import torch
import trimesh
from pathlib import Path
from scipy.spatial import cKDTree

from datareader import *
from estimater import *


# =========================================================
# Pose metrics
# =========================================================
class PoseMetrics:
    def __init__(self, dataset_root):
        self.meta_data = self._build_dataset_metadata(dataset_root)

    def _build_dataset_metadata(self, path):
        meta = {}
        models_dir = Path(path) / "models"

        with open(models_dir / "models_info.json") as f:
            info_json = json.load(f)

        for ply in models_dir.glob("obj_*.ply"):
            ob_id = int(ply.stem.split("_")[1])
            mesh = trimesh.load(ply)

            meta[ob_id] = {
                "diameter": info_json[str(ob_id)]["diameter"],
                "symmetry": info_json[str(ob_id)].get("symmetry", "none"),
                "mesh": mesh,
            }

        return meta

    def _apply_transform_to_points(self, pts, tf):
        if len(tf.shape) >= 3 and tf.shape[-3] != pts.shape[-2]:
            tf = tf[..., None, :, :]
        return (tf[..., :-1, :-1] @ pts[..., None] + tf[..., :-1, -1:])[..., 0]

    def compute_add_error(self, pred, gt, pts, diameter):
        pred_pts = self._apply_transform_to_points(pts, pred)
        gt_pts = self._apply_transform_to_points(pts, gt)
        return np.linalg.norm(pred_pts - gt_pts, axis=-1).mean() / diameter

    def compute_adds_error(self, pred, gt, pts, diameter):
        pred_pts = self._apply_transform_to_points(pts, pred)
        gt_pts = self._apply_transform_to_points(pts, gt)
        tree = cKDTree(pred_pts)
        d, _ = tree.query(gt_pts, k=1, workers=-1)
        return d.mean() / diameter

    def compute_add_success_0_1d(self, pred, gt, model_pts, ob_id):
        diameter = self.meta_data[ob_id]["diameter"] / 1000
        return self.compute_add_error(pred, gt, model_pts, diameter)

    def compute_pose_decomposition_error(self, pred, gt):
        R1, R2 = pred[:3, :3], gt[:3, :3]
        t1, t2 = pred[:3, 3], gt[:3, 3]

        rot = R1 @ R2.T
        rot_err = np.linalg.norm(R.from_matrix(rot).as_rotvec()) * 180 / np.pi
        trans_err = np.linalg.norm(t1 - t2)

        return rot_err, trans_err


# =========================================================
# Mask utilities
# =========================================================
def get_object_mask(reader, i_frame, ob_id, detect_type):
    mask = reader.get_mask(i_frame, ob_id)

    if detect_type == "box":
        vs, us = np.where(mask > 0)
        valid = np.zeros_like(mask, dtype=bool)
        valid[vs.min():vs.max(), us.min():us.max()] = 1
        return valid

    if detect_type == "mask":
        return mask > 0 if mask is not None else None

    if detect_type == "detected":
        m = cv2.imread(reader.color_files[i_frame].replace("rgb", "mask_cosypose"), -1)
        return m == ob_id

    raise RuntimeError("Unknown detect_type")


# =========================================================
# Inference worker
# =========================================================
def run_inference_worker(
    reader,
    i_frames,
    est: OneRefPose,
    ob_id=None,
    device="cuda:0",
    metrics=None,
    model_pts=None,
    error_dict=None,
):
    torch.cuda.set_device(device)
    est.to_device(device)

    result = NestDict()

    for i_frame in i_frames:

        color = reader.get_color(i_frame)
        depth = reader.get_depth(i_frame)
        ob_mask = get_object_mask(reader, i_frame, ob_id, detect_type)

        if ob_mask is None:
            continue

        # ---------------- GT ----------------
        gt_pose = reader.get_gt_pose(i_frame, ob_id)
        est.gt_pose = gt_pose

        # ---------------- Prediction ----------------
        pred_pose = est.register(
            K=reader.K,
            rgb=color,
            depth=depth,
            ob_mask=ob_mask,
            ob_id=ob_id,
        )

        # ---------------- Error ----------------
        add_err = metrics.compute_add_success_0_1d(
            pred_pose, gt_pose, model_pts, ob_id
        )

        rot_err, trans_err = metrics.compute_pose_decomposition_error(
            pred_pose, gt_pose
        )

        error_dict[ob_id].append(add_err)

        # ---------------- OUTPUT ----------------
        print(f"\n[Frame {i_frame} | Object {ob_id}]")
        print("GT Pose:\n", gt_pose)
        print("Pred Pose:\n", pred_pose)

        print(
            f"ADD(-S): {add_err:.6f} | "
            f"Rot: {rot_err:.2f} deg | "
            f"Trans: {trans_err * 1000:.2f} mm"
        )

        result[reader.get_video_id()][reader.id_strs[i_frame]][ob_id] = pred_pose

    return result, error_dict


# =========================================================
# AUC
# =========================================================
def compute_auc_sklearn(errs, max_val=0.1, step=0.001):
    from sklearn import metrics

    errs = np.sort(np.array(errs))
    X = np.arange(0, max_val + step, step)
    Y = np.zeros_like(X)

    for i, x in enumerate(X):
        Y[i] = (errs <= x).mean()
        if Y[i] >= 1:
            break

    return metrics.auc(X, Y) / max_val


# =========================================================
# Pipeline
# =========================================================
def run_pipeline():
    manager = multiprocessing.Manager()
    error_dict = manager.dict()

    reader_tmp = LinemodReader(f"{opt.linemod_dir}/test/000002")
    metrics = PoseMetrics(opt.linemod_dir)

    mesh_tmp = trimesh.primitives.Box(extents=np.ones(3)).to_mesh()

    est = OneRefPose(
        model_pts=mesh_tmp.vertices.copy(),
        model_normals=mesh_tmp.vertex_normals.copy(),
        mesh=mesh_tmp,
        scorer=None,
        refiner=None,
    )

    res = NestDict()

    # ---------------- objects ----------------
    # for ob_id in reader_tmp.ob_ids:
    for ob_id in [1]:
        error_dict[ob_id] = manager.list()

        # ALWAYS use GT mesh (reconstructed removed)
        mesh = reader_tmp.get_gt_mesh(ob_id)

        est._init_object_geometry(
            mesh.vertices.copy(),
            mesh.vertex_normals.copy(),
            mesh=mesh,
        )

        reader = LinemodReader(f"{opt.linemod_dir}/test/{ob_id:06d}")

        for i in range(len(reader.color_files)):
            out, error_dict = run_inference_worker(
                reader,
                [i],
                est,
                ob_id,
                "cuda:0",
                metrics,
                mesh.vertices.copy(),
                error_dict,
            )

            for v in out:
                for s in out[v]:
                    res[v][s][ob_id] = out[v][s][ob_id]

    # ---------------- final metrics ----------------
    all_errors = []
    for ob_id in error_dict:
        all_errors.extend(list(error_dict[ob_id]))

    all_errors = np.array(all_errors)

    print("\n========== FINAL METRICS ==========")
    print("Mean:", np.mean(all_errors))
    print("Median:", np.median(all_errors))
    print("ADD@0.1:", np.mean(all_errors <= 0.1))


# =========================================================
# Entry
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--linemod_dir", type=str)

    opt = parser.parse_args()
    set_seed(0)

    detect_type = "mask"

    start = time.time()
    run_pipeline()
    print("Total time:", time.time() - start)