# Copyright (c) 2026 [OneViewAll].
# Licensed under the MIT License.

from scipy.spatial.transform import Rotation as R
from Utils import *
import json, os, sys, time
import numpy as np
import torch
import trimesh
from pathlib import Path
from scipy.spatial import cKDTree
import multiprocessing

from datareader import *
from pose_calculation import *

CROP_RATIO = 1.2


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
            info = info_json[str(ob_id)]
            has_sym = ('symmetries_continuous' in info) or ('symmetries_discrete' in info)
            meta[ob_id] = {
                "diameter": info["diameter"],
                "mesh": mesh,
                "has_symmetry": has_sym,
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

    def compute_normalized_error(self, pred, gt, model_pts, ob_id):
        diameter_m = self.meta_data[ob_id]["diameter"] / 1000.0
        if self.meta_data[ob_id]["has_symmetry"]:
            return self.compute_adds_error(pred, gt, model_pts, diameter_m)
        return self.compute_add_error(pred, gt, model_pts, diameter_m)

    def compute_pose_decomposition_error(self, pred, gt):
        R1, R2 = pred[:3, :3], gt[:3, :3]
        rot = R1 @ R2.T
        rot_err = np.linalg.norm(R.from_matrix(rot).as_rotvec()) * 180 / np.pi
        trans_err = np.linalg.norm(pred[:3, 3] - gt[:3, 3])
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
def run_inference_worker(reader, i_frames, est, ob_id=None,
                         device="cuda:0", metrics=None, model_pts=None, error_dict=None,
                         detect_type="mask"):
    torch.cuda.set_device(device)
    est.to_device(device)
    result = NestDict()

    for i_frame in i_frames:
        color = reader.get_color(i_frame)
        depth = reader.get_depth(i_frame)
        ob_mask = get_object_mask(reader, i_frame, ob_id, detect_type)
        if ob_mask is None:
            print(f"[worker] frame {i_frame}: no mask, skip")
            continue
        gt_pose = reader.get_gt_pose(i_frame, ob_id)
        est.gt_pose = gt_pose

        pred_pose = est.register(K=reader.K, rgb=color, depth=depth,
                                 ob_mask=ob_mask, ob_id=ob_id)

        err = metrics.compute_normalized_error(pred_pose, gt_pose, model_pts, ob_id)
        rot_err, trans_err = metrics.compute_pose_decomposition_error(pred_pose, gt_pose)
        error_dict[ob_id].append(err)

        print(f"\n[Frame {i_frame} | Object {ob_id}]")
        print(f"ADD(-S): {err:.6f} | Rot: {rot_err:.2f} deg | Trans: {trans_err*1000:.2f} mm")
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
# CLI helpers
# =========================================================
def _parse_int_list(s):
    if s is None or str(s).strip() == "":
        return None
    return [int(v.strip()) for v in str(s).split(",") if v.strip() != ""]


def _resolve_skip_ref(opt):
    if opt.eval_skip_ref is None:
        return opt.ref_split == "test"
    return bool(opt.eval_skip_ref)


def _split_reader(linemod_dir, split, ob_id, cache=None):
    key = (split, ob_id)
    if cache is not None and key in cache:
        return cache[key]
    r = LinemodReader(f"{linemod_dir}/{split}/{ob_id:06d}")
    if cache is not None:
        cache[key] = r
    return r


def _diameter_m_from_metrics(metrics, ob_id):
    return metrics.meta_data[ob_id]["diameter"] / 1000.0


# =========================================================
# Frame selection
# =========================================================
def run_frame_selection(detect_type="mask"):
    ob_ids = _parse_int_list(opt.ob_ids) or [opt.ob_id]
    skip_ref = _resolve_skip_ref(opt)
    metrics = PoseMetrics(opt.linemod_dir)
    all_rows = []

    est = OneRefPose(scorer=None, refiner=None, ref_data_root=opt.ref_data_root)
    est.to_device("cuda:0")

    for ob_id in ob_ids:
        cache = {}
        reader_ref = _split_reader(opt.linemod_dir, opt.ref_split, ob_id, cache)
        reader_eval = _split_reader(opt.linemod_dir, "test", ob_id, cache)
        model_pts = reader_eval.get_gt_mesh(ob_id).vertices.copy()
        diam_m = _diameter_m_from_metrics(metrics, ob_id)

        n_ref = len(reader_ref.color_files)
        n_eval = len(reader_eval.color_files)

        explicit = _parse_int_list(opt.ref_frames)
        if explicit is not None:
            candidate_frames = explicit
        elif opt.ref_frame >= 0:
            candidate_frames = [opt.ref_frame]
        else:
            candidate_frames = list(range(n_ref))

        bad = [k for k in candidate_frames if k < 0 or k >= n_ref]
        if bad:
            raise SystemExit(f"[select] ob_id={ob_id:02d}: frames {bad} out of range [0,{n_ref-1}]")

        print(f"\n[select] ob_id={ob_id:02d} split={opt.ref_split} "
              f"n_candidates={len(candidate_frames)} n_eval={n_eval} skip_ref={skip_ref}")

        for k in candidate_frames:
            # [PATCH] 每个候选帧的 probe_dir（多帧必须分开），统一由 use_frame_as_ref 保存
            probe_dir = None
            if opt.save_ref_probe:
                probe_dir = os.path.join(opt.save_ref_probe, f"ob{ob_id:02d}_frame{k:04d}")
                os.makedirs(probe_dir, exist_ok=True)

            est.use_frame_as_ref(
                reader_ref, k, ob_id,
                is_symmetric=(opt.ref_sym_axis >= 0),
                symmetry_axis=opt.ref_sym_axis,
                crop_ratio=CROP_RATIO,
                device=opt.ref_device,
                save_to_disk=False,
                mesh_diameter=diam_m, crop_to_ref=True,
                probe_dir=probe_dir,     # [PATCH] 内部落盘 7 文件
            )
            if probe_dir is not None:
                print(f"[probe] saved 7 files + xyz_map.npy → {probe_dir}")

            est.refiner.save_ref_probe_dir = probe_dir
            est.scorer.save_ref_probe_dir = probe_dir

            errs, rots, transs = [], [], []
            n_skipped = 0
            for i in range(n_eval):
                if skip_ref and opt.ref_split == "test" and i == k:
                    n_skipped += 1
                    continue
                color = reader_eval.get_color(i)
                depth = reader_eval.get_depth(i)
                ob_mask = get_object_mask(reader_eval, i, ob_id, detect_type)
                if ob_mask is None:
                    continue
                gt_pose = reader_eval.get_gt_pose(i, ob_id)
                pred_pose = est.register(K=reader_eval.K, rgb=color, depth=depth,
                                         ob_mask=ob_mask, ob_id=ob_id)
                err = metrics.compute_normalized_error(pred_pose, gt_pose, model_pts, ob_id)
                rot_err, trans_err = metrics.compute_pose_decomposition_error(pred_pose, gt_pose)
                errs.append(err); rots.append(rot_err); transs.append(trans_err)

            errs = np.asarray(errs)
            row = {
                "ob_id": ob_id, "ref_split": opt.ref_split, "ref_frame": k,
                "n_eval": len(errs), "n_skipped": n_skipped,
                "mean": float(errs.mean()) if len(errs) else float("inf"),
                "median": float(np.median(errs)) if len(errs) else float("inf"),
                "sr@0.1": float((errs <= 0.1).mean()) if len(errs) else 0.0,
                "auc@0.1": compute_auc_sklearn(errs) if len(errs) else 0.0,
                "rot_mean": float(np.mean(rots)) if rots else float("inf"),
                "trans_mean_mm": float(np.mean(transs)) * 1000 if transs else float("inf"),
            }
            all_rows.append(row)
            print(f"[select] ob_id={ob_id:02d} ref_frame={k:04d} n_eval={row['n_eval']} "
                  f"mean={row['mean']:.6f} sr@0.1={row['sr@0.1']:.4f} "
                  f"auc@0.1={row['auc@0.1']:.4f}")

        obj_rows = sorted([r for r in all_rows if r["ob_id"] == ob_id],
                          key=lambda r: (r["mean"], -r["sr@0.1"]))
        best = obj_rows[0]
        print(f"[select] BEST ob_id={ob_id:02d} ref_frame={best['ref_frame']} "
              f"mean={best['mean']:.6f} sr@0.1={best['sr@0.1']:.4f}")

        if opt.save_ref:
            if opt.ref_data_root is None:
                raise ValueError("--save_ref requires --ref_data_root")
            est.use_frame_as_ref(
                reader_ref, best["ref_frame"], ob_id,
                is_symmetric=(opt.ref_sym_axis >= 0),
                symmetry_axis=opt.ref_sym_axis,
                crop_ratio=CROP_RATIO,
                device=opt.ref_device,
                save_to_disk=True,
                mesh_diameter=diam_m, crop_to_ref=True,
            )
            print(f"[select] best reference saved for ob_id={ob_id:02d}")

    import csv
    out_dir = opt.save_ref_probe or "."
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "ref_selection.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\n[select] results saved to {out_csv}")
    return all_rows


# =========================================================
# Single-reference pipeline
# =========================================================
def run_pipeline(detect_type="mask"):
    ref_frames = _parse_int_list(opt.ref_frames)
    if ref_frames is not None:
        if len(ref_frames) != 1:
            raise SystemExit(f"single-reference mode requires one --ref_frames element; got {ref_frames}")
        ref_frame = ref_frames[0]
    elif opt.ref_frame >= 0:
        ref_frame = opt.ref_frame
    else:
        raise SystemExit(
            "run_pipeline: --ref_frame is required in single-reference mode "
            "(or use --select_ref to sweep candidates)."
        )

    ob_ids = _parse_int_list(opt.ob_ids) or [opt.ob_id]
    if len(ob_ids) != 1:
        raise SystemExit(f"single-reference mode requires one object; got {ob_ids}")
    ob_id = ob_ids[0]

    skip_ref = _resolve_skip_ref(opt)

    manager = multiprocessing.Manager()
    error_dict = manager.dict()

    cache = {}
    reader_eval = _split_reader(opt.linemod_dir, "test", ob_id, cache)
    metrics = PoseMetrics(opt.linemod_dir)
    diam_m = _diameter_m_from_metrics(metrics, ob_id)
    model_pts = reader_eval.get_gt_mesh(ob_id).vertices.copy()

    est = OneRefPose(scorer=None, refiner=None, ref_data_root=opt.ref_data_root)
    est.to_device("cuda:0")

    reader_ref = _split_reader(opt.linemod_dir, opt.ref_split, ob_id, cache)
    n_ref = len(reader_ref.color_files)
    if ref_frame < 0 or ref_frame >= n_ref:
        raise SystemExit(f"[pipeline] ref_frame={ref_frame} out of range [0,{n_ref-1}]")
    print(f"[pipeline] ob_id={ob_id:02d} reference: split={opt.ref_split} "
          f"frame={ref_frame}  skip_ref={skip_ref}  diameter={diam_m:.4f}m  "
          f"crop_ratio={CROP_RATIO}")

    save_to_disk = bool(opt.save_ref)
    if opt.save_ref and not opt.ref_data_root:
        print("[pipeline] WARNING: --save_ref set but --ref_data_root is None; skipping disk save")
        save_to_disk = False

    # [PATCH] probe_dir 直接在 use_frame_as_ref 里落盘
    probe_dir = None
    if opt.save_ref_probe:
        probe_dir = opt.save_ref_probe
        os.makedirs(probe_dir, exist_ok=True)

    est.use_frame_as_ref(
        reader_ref, ref_frame, ob_id,
        is_symmetric=(opt.ref_sym_axis >= 0),
        symmetry_axis=opt.ref_sym_axis,
        crop_ratio=CROP_RATIO,
        device=opt.ref_device,
        save_to_disk=save_to_disk,
        mesh_diameter=diam_m, crop_to_ref=True,
        probe_dir=probe_dir,   # [PATCH] 内部落盘 7 文件
    )
    if save_to_disk:
        print(f"[pipeline] reference saved to {os.path.join(opt.ref_data_root, str(ob_id))}")
    if probe_dir is not None:
        print(f"[probe] saved 7 files + xyz_map.npy → {probe_dir}")
        print(f"[probe] reference-side npy tensors will be dumped once")
        est.refiner.save_ref_probe_dir = probe_dir
        est.scorer.save_ref_probe_dir = probe_dir

    error_dict[ob_id] = manager.list()
    res = NestDict()

    for i in range(len(reader_eval.color_files)):
        if skip_ref and opt.ref_split == "test" and i == ref_frame:
            print(f"[pipeline] skip reference frame {i} during evaluation")
            continue
        out, error_dict = run_inference_worker(
            reader_eval, [i], est, ob_id, "cuda:0", metrics, model_pts, error_dict,
            detect_type=detect_type)
        for v in out:
            for s in out[v]:
                res[v][s][ob_id] = out[v][s][ob_id]

    all_errors = np.array(list(error_dict[ob_id]))
    np.savetxt("all_errors.txt", all_errors, fmt="%.6f")

    print("\n========== OVERALL METRICS ==========")
    print(f"Reference:      {opt.ref_split}/frame {ref_frame} (skip_ref={skip_ref})")
    print(f"n_evaluated:    {len(all_errors)}")
    if len(all_errors):
        print(f"Mean:           {all_errors.mean():.6f}")
        print(f"Median:         {np.median(all_errors):.6f}")
        print(f"ADD(-S)@0.1d:   {np.mean(all_errors <= 0.1):.4f}")
        print(f"AUC@0.1:        {compute_auc_sklearn(all_errors):.4f}")


# =========================================================
# Entry
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--linemod_dir", type=str, required=True)
    parser.add_argument("--ob_id", type=int, default=1)
    parser.add_argument("--ob_ids", type=str, default=None)
    parser.add_argument("--ref_frame", type=int, default=-1)
    parser.add_argument("--ref_frames", type=str, default=None)
    parser.add_argument("--ref_split", type=str, default="test",
                        choices=["train", "test"])
    parser.add_argument("--eval_skip_ref", dest="eval_skip_ref",
                        action="store_true", default=None)
    parser.add_argument("--eval_include_ref", dest="eval_skip_ref",
                        action="store_false")
    parser.add_argument("--ref_data_root", type=str, default=None)
    parser.add_argument("--save_ref", action="store_true")
    parser.add_argument("--save_ref_probe", type=str, default=None)
    parser.add_argument("--select_ref", action="store_true")
    parser.add_argument("--ref_sym_axis", type=int, default=-1)
    parser.add_argument("--ref_device", type=str, default="cuda")
    parser.add_argument("--detect_type", type=str, default="mask",
                        choices=["mask", "box", "detected"])

    opt = parser.parse_args()
    set_seed(0)
    detect_type = opt.detect_type

    if opt.select_ref:
        run_frame_selection(detect_type=detect_type)
        sys.exit(0)

    start = time.time()
    run_pipeline(detect_type=detect_type)
    print("Total time:", time.time() - start)