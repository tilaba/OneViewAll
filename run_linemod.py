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

# Linemod 对称物体：eggbox=10, glue=11 ADD-S；其余使用 ADD
SYMMETRIC_OBJ_IDS = {10, 11}


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

    def use_adds(self, ob_id):
        """对称物体（eggbox、glue）用 ADD-S，其余用 ADD。"""
        return int(ob_id) in SYMMETRIC_OBJ_IDS

    def metric_name(self, ob_id):
        return "ADD-S" if self.use_adds(ob_id) else "ADD"

    def compute_normalized_error(self, pred, gt, model_pts, ob_id):
        diameter_m = self.meta_data[ob_id]["diameter"] / 1000.0
        if self.use_adds(ob_id):
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


def _mask_bbox_area(reader, i_frame, ob_id, min_visib=8):
    m = reader.get_mask(i_frame, ob_id)
    if m is None:
        return None
    m = np.asarray(m)
    if m.ndim > 2:
        m = m[..., 0]
    vs, us = np.where(m > 0)
    if len(vs) == 0:
        return None
    w = int(us.max() - us.min() + 1)
    h = int(vs.max() - vs.min() + 1)
    if w < min_visib or h < min_visib:
        return None
    return w, h, w * h


def select_ref_frame_by_mask_area(reader, ob_id, min_visib=8):
    n = len(reader.color_files)
    best_frame = None
    best_area = -1
    best_wh = None
    for i in range(n):
        info = _mask_bbox_area(reader, i, ob_id, min_visib=min_visib)
        if info is None:
            continue
        w, h, area = info
        if area > best_area:
            best_area = area
            best_frame = i
            best_wh = (w, h)
    return best_frame, best_area, best_wh


# =========================================================
# Evaluation sampling
# =========================================================
def _eval_indices(n, stride=1, max_frames=0, skip=None):
    """
    Build the list of evaluation frame indices.

      stride     : sample every `stride` frames (1 = all frames)
      max_frames : cap the number of evaluated frames after sampling
                   (0 = unlimited)
      skip       : optional iterable of indices to exclude (e.g. the
                   reference frame when evaluating on test split)

    Returns:
      idxs      : list[int] of frame indices to evaluate
      n_skipped : number of skipped indices that would otherwise have
                  been evaluated
    """
    stride = max(1, int(stride))
    max_frames = int(max_frames)

    idxs = list(range(0, n, stride))
    if max_frames > 0:
        idxs = idxs[:max_frames]

    if skip:
        skip_set = set(int(x) for x in skip)
        kept = [i for i in idxs if i not in skip_set]
        n_skipped = len(idxs) - len(kept)
        return kept, n_skipped

    return idxs, 0


# =========================================================
# Inference worker
# =========================================================
def run_inference_worker(reader, i_frames, est, ob_id=None,
                         device="cuda:0", metrics=None, model_pts=None, error_dict=None,
                         detect_type="mask"):
    torch.cuda.set_device(device)
    est.to_device(device)
    result = NestDict()

    metric = metrics.metric_name(ob_id)

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
        print(f"{metric}: {err:.6f} | Rot: {rot_err:.2f} deg | "
              f"Trans: {trans_err*1000:.2f} mm")
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
# Frame selection (sweep tool, optional)
# =========================================================
def run_frame_selection(detect_type="mask"):
    ob_ids = _parse_int_list(opt.ob_ids) or [opt.ob_id]
    skip_ref = _resolve_skip_ref(opt)
    metrics = PoseMetrics(opt.linemod_dir)
    all_rows = []

    eval_stride = max(1, int(getattr(opt, "eval_stride", 1)))
    max_eval_frames = int(getattr(opt, "max_eval_frames", 0))
    crop_ratio = float(opt.crop_ratio)

    est = OneRefPose(scorer=None, refiner=None, ref_data_root=opt.ref_data_root)
    est.to_device("cuda:0")

    for ob_id in ob_ids:
        cache = {}
        reader_ref = _split_reader(opt.linemod_dir, opt.ref_split, ob_id, cache)
        reader_eval = _split_reader(opt.linemod_dir, "test", ob_id, cache)
        model_pts = reader_eval.get_gt_mesh(ob_id).vertices.copy()
        diam_m = _diameter_m_from_metrics(metrics, ob_id)
        metric = metrics.metric_name(ob_id)

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
            raise SystemExit(
                f"[select] ob_id={ob_id:02d}: frames {bad} out of range [0,{n_ref-1}]")

        print(f"\n[select] ob_id={ob_id:02d} metric={metric} split={opt.ref_split} "
              f"n_candidates={len(candidate_frames)} n_eval={n_eval} "
              f"eval_stride={eval_stride} max_eval_frames={max_eval_frames} "
              f"crop_ratio={crop_ratio} skip_ref={skip_ref}")

        for k in candidate_frames:
            probe_dir = None
            if opt.save_ref_probe:
                probe_dir = os.path.join(opt.save_ref_probe, f"ob{ob_id:02d}_frame{k:04d}")
                os.makedirs(probe_dir, exist_ok=True)

            est.use_frame_as_ref(
                reader_ref, k, ob_id,
                is_symmetric=(opt.ref_sym_axis >= 0),
                symmetry_axis=opt.ref_sym_axis,
                crop_ratio=crop_ratio,
                device=opt.ref_device,
                save_to_disk=False,
                mesh_diameter=diam_m, crop_to_ref=True,
                probe_dir=probe_dir,
            )
            if probe_dir is not None:
                print(f"[probe] saved 7 files + xyz_map.npy → {probe_dir}")

            est.refiner.save_ref_probe_dir = probe_dir
            est.scorer.save_ref_probe_dir = probe_dir

            skip_set = {k} if (skip_ref and opt.ref_split == "test") else set()
            eval_idxs, n_skipped = _eval_indices(
                n_eval, eval_stride, max_eval_frames, skip=skip_set)

            errs, rots, transs = [], [], []
            for i in eval_idxs:
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
                "ob_id": ob_id, "metric": metric,
                "ref_split": opt.ref_split, "ref_frame": k,
                "n_eval": len(errs), "n_skipped": n_skipped,
                "mean": float(errs.mean()) if len(errs) else float("inf"),
                "median": float(np.median(errs)) if len(errs) else float("inf"),
                "acc@0.1": float((errs <= 0.1).mean()) if len(errs) else 0.0,
                "auc@0.1": compute_auc_sklearn(errs) if len(errs) else 0.0,
                "rot_mean": float(np.mean(rots)) if rots else float("inf"),
                "trans_mean_mm": float(np.mean(transs)) * 1000 if transs else float("inf"),
                "eval_stride": eval_stride,
                "max_eval_frames": max_eval_frames,
                "crop_ratio": crop_ratio,
            }
            all_rows.append(row)
            print(f"[select] ob_id={ob_id:02d} ref_frame={k:04d} n_eval={row['n_eval']} "
                  f"mean={row['mean']:.6f} {metric}@0.1={row['acc@0.1']:.4f} "
                  f"auc@0.1={row['auc@0.1']:.4f}")

        obj_rows = sorted([r for r in all_rows if r["ob_id"] == ob_id],
                          key=lambda r: (r["mean"], -r["acc@0.1"]))
        best = obj_rows[0]
        print(f"[select] BEST ob_id={ob_id:02d} ref_frame={best['ref_frame']} "
              f"mean={best['mean']:.6f} {metric}@0.1={best['acc@0.1']:.4f} "
              f"auc@0.1={best['auc@0.1']:.4f}")

        if opt.save_ref:
            if opt.ref_data_root is None:
                raise ValueError("--save_ref requires --ref_data_root")
            est.use_frame_as_ref(
                reader_ref, best["ref_frame"], ob_id,
                is_symmetric=(opt.ref_sym_axis >= 0),
                symmetry_axis=opt.ref_sym_axis,
                crop_ratio=crop_ratio,
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
# Single-reference pipeline (multi-object supported)
# =========================================================
def _print_summary(per_obj):
    """打印 per-object 指标表 + overall 汇总。"""
    if not per_obj:
        print("\n[summary] no results")
        return

    metric_names = set(r["metric"] for r in per_obj)
    header_metric = metric_names.pop() if len(metric_names) == 1 else "ADD/ADD-S"

    print(f"\nPer-object {header_metric} @ 0.1d metrics")
    print("=" * 50)

    all_errs = []
    for r in per_obj:
        errs = r["errs"]
        m = r["metric"]
        if len(errs) == 0:
            print(f"Object {r['ob_id']:02d}: no valid frames")
            continue
        mean_v = float(errs.mean())
        med_v = float(np.median(errs))
        acc = float(np.mean(errs <= 0.1))
        auc = compute_auc_sklearn(errs)
        print(f"Object {r['ob_id']:02d}: Mean={mean_v:.6f}  Median={med_v:.6f}  "
              f"{m}@0.1d={acc:.4f}  AUC@0.1={auc:.4f}")
        all_errs.append(errs)

    if not all_errs:
        return

    all_errs = np.concatenate(all_errs)
    print(f"\n========== OVERALL {header_metric} @ 0.1d ==========")
    print(f"Mean:           {all_errs.mean():.6f}")
    print(f"Median:         {np.median(all_errs):.6f}")
    print(f"{header_metric}@0.1d:     {np.mean(all_errs <= 0.1):.4f}")
    print(f"AUC@0.1:        {compute_auc_sklearn(all_errs):.4f}")


def run_pipeline(detect_type="mask"):
    ref_frames = _parse_int_list(opt.ref_frames)
    if ref_frames is not None and len(ref_frames) != 1:
        raise SystemExit(
            f"single-reference mode requires one --ref_frames element; got {ref_frames}")

    ob_ids = _parse_int_list(opt.ob_ids) or [opt.ob_id]

    skip_ref = _resolve_skip_ref(opt)
    crop_ratio = float(opt.crop_ratio)
    eval_stride = max(1, int(getattr(opt, "eval_stride", 1)))
    max_eval_frames = int(getattr(opt, "max_eval_frames", 0))

    metrics = PoseMetrics(opt.linemod_dir)

    est = OneRefPose(scorer=None, refiner=None, ref_data_root=opt.ref_data_root)
    est.to_device("cuda:0")

    cache = {}
    per_obj = []

    for ob_id in ob_ids:
        reader_eval = _split_reader(opt.linemod_dir, "test", ob_id, cache)
        reader_ref = _split_reader(opt.linemod_dir, opt.ref_split, ob_id, cache)
        diam_m = _diameter_m_from_metrics(metrics, ob_id)
        model_pts = reader_eval.get_gt_mesh(ob_id).vertices.copy()
        metric = metrics.metric_name(ob_id)
        n_ref = len(reader_ref.color_files)

        # ---- reference frame selection ----
        if ref_frames is not None:
            ref_frame = int(ref_frames[0])
            ref_source = "explicit(--ref_frames)"
        elif opt.ref_frame >= 0:
            ref_frame = int(opt.ref_frame)
            ref_source = "explicit(--ref_frame)"
        else:
            ref_frame, ref_area, ref_wh = select_ref_frame_by_mask_area(
                reader_ref, ob_id, min_visib=opt.ref_min_visib)
            if ref_frame is None:
                print(f"[pipeline] ob_id={ob_id:02d}: cannot auto-select reference frame, skip")
                continue
            ref_source = f"auto(mask_area={ref_area}, w={ref_wh[0]}, h={ref_wh[1]})"

        if ref_frame < 0 or ref_frame >= n_ref:
            print(f"[pipeline] ob_id={ob_id:02d}: ref_frame={ref_frame} out of range "
                  f"[0,{n_ref-1}], skip")
            continue

        print(f"\n[pipeline] ob_id={ob_id:02d} metric={metric} reference: "
              f"split={opt.ref_split} frame={ref_frame}  source={ref_source}  "
              f"skip_ref={skip_ref}  diameter={diam_m:.4f}m  crop_ratio={crop_ratio}  "
              f"eval_stride={eval_stride}  max_eval_frames={max_eval_frames}")

        # ---- set reference ----
        save_to_disk = bool(opt.save_ref)
        if opt.save_ref and not opt.ref_data_root:
            print("[pipeline] WARNING: --save_ref set but --ref_data_root is None; "
                  "skipping disk save")
            save_to_disk = False

        probe_dir = opt.save_ref_probe
        if probe_dir:
            probe_dir = os.path.join(opt.save_ref_probe, f"ob{ob_id:02d}")
            os.makedirs(probe_dir, exist_ok=True)

        est.use_frame_as_ref(
            reader_ref, ref_frame, ob_id,
            is_symmetric=(opt.ref_sym_axis >= 0),
            symmetry_axis=opt.ref_sym_axis,
            crop_ratio=crop_ratio,
            device=opt.ref_device,
            save_to_disk=save_to_disk,
            mesh_diameter=diam_m, crop_to_ref=True,
            probe_dir=probe_dir,
        )
        if save_to_disk:
            print(f"[pipeline] reference saved to "
                  f"{os.path.join(opt.ref_data_root, str(ob_id))}")
        if probe_dir is not None:
            print(f"[probe] saved 7 files + xyz_map.npy → {probe_dir}")
            est.refiner.save_ref_probe_dir = probe_dir
            est.scorer.save_ref_probe_dir = probe_dir

        # ---- evaluate ----
        skip_set = {ref_frame} if (skip_ref and opt.ref_split == "test") else set()
        eval_idxs, n_skipped = _eval_indices(
            len(reader_eval.color_files), eval_stride, max_eval_frames, skip=skip_set)
        if n_skipped:
            print(f"[pipeline] skip reference frame {ref_frame} during evaluation")
        print(f"[pipeline] eval frames: {len(eval_idxs)} "
              f"(stride={eval_stride}, max={max_eval_frames}, skipped={n_skipped})")

        errs = []
        for i in eval_idxs:
            color = reader_eval.get_color(i)
            depth = reader_eval.get_depth(i)
            ob_mask = get_object_mask(reader_eval, i, ob_id, detect_type)
            if ob_mask is None:
                continue
            gt_pose = reader_eval.get_gt_pose(i, ob_id)
            est.gt_pose = gt_pose
            pred_pose = est.register(K=reader_eval.K, rgb=color, depth=depth,
                                     ob_mask=ob_mask, ob_id=ob_id)
            err = metrics.compute_normalized_error(pred_pose, gt_pose, model_pts, ob_id)
            rot_err, trans_err = metrics.compute_pose_decomposition_error(pred_pose, gt_pose)
            errs.append(err)
            print(f"[Frame {i} | Object {ob_id}] {metric}: {err:.6f} | "
                  f"Rot: {rot_err:.2f} deg | Trans: {trans_err*1000:.2f} mm")

        errs = np.asarray(errs)
        np.savetxt(f"all_errors_ob{ob_id:02d}.txt", errs, fmt="%.6f")
        per_obj.append({"ob_id": ob_id, "metric": metric, "errs": errs})

    # ---- 全部跑完后统一汇总 ----
    _print_summary(per_obj)


# =========================================================
# Entry
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--linemod_dir", type=str, required=True)
    parser.add_argument("--ob_id", type=int, default=1)
    parser.add_argument("--ob_ids", type=str, default=None,
                        help="Comma-separated object ids, e.g. '1,2,4,5'. "
                             "Takes precedence over --ob_id.")
    parser.add_argument("--ref_frame", type=int, default=-1,
                        help="Explicit reference frame im_id. "
                             "-1 = auto-select the frame with the largest "
                             "mask bbox area.")
    parser.add_argument("--ref_frames", type=str, default=None,
                        help="Comma-separated candidate frames for the sweep "
                             "tool (--select_ref). In single-reference mode "
                             "exactly one element is required.")
    parser.add_argument("--ref_split", type=str, default="test",
                        choices=["train", "test"])
    parser.add_argument("--eval_skip_ref", dest="eval_skip_ref",
                        action="store_true", default=None)
    parser.add_argument("--eval_include_ref", dest="eval_skip_ref",
                        action="store_false")
    parser.add_argument("--eval_stride", type=int, default=1,
                        help="Evaluate every N-th frame (1 = all frames). "
                             "Use a value >1 to speed up sweep.")
    parser.add_argument("--max_eval_frames", type=int, default=0,
                        help="Cap the number of evaluated frames per object "
                             "after stride sampling (0 = unlimited).")
    parser.add_argument("--ref_data_root", type=str, default=None)
    parser.add_argument("--save_ref", action="store_true")
    parser.add_argument("--save_ref_probe", type=str, default=None)
    parser.add_argument("--select_ref", action="store_true",
                        help="Run the sweep tool over --ref_frames / all frames "
                             "instead of the single-reference pipeline.")
    parser.add_argument("--ref_sym_axis", type=int, default=-1)
    parser.add_argument("--ref_device", type=str, default="cuda")
    parser.add_argument("--ref_min_visib", type=int, default=8,
                        help="Auto-select reference frame only among frames whose "
                             "mask bbox width and height are >= this value.")
    parser.add_argument("--detect_type", type=str, default="mask",
                        choices=["mask", "box", "detected"])
    parser.add_argument("--crop_ratio", type=float, default=CROP_RATIO,
                        help="Crop window ratio used for reference/render crops "
                             "(default: %(default)s)")

    opt = parser.parse_args()
    set_seed(0)
    detect_type = opt.detect_type

    if opt.select_ref:
        run_frame_selection(detect_type=detect_type)
        sys.exit(0)

    start = time.time()
    run_pipeline(detect_type=detect_type)
    print("Total time:", time.time() - start)