# Copyright (c) 2026 [OneViewAll].
# Licensed under the MIT License.
#
# TLESS BOP evaluation using OneRefPose (from pose_calculation).
# This file contains NO FoundationPose code.
#
# Data layout expected:
#   <tless_dir>/
#     ├── models/{models_info.json, obj_XXXXXX.ply}   ← 使用 models（重建 mesh），
#     │                                                不用 models_cad：
#     │                                                models/ 的 models_info.json
#     │                                                里 diameter 字段有效，
#     │                                                models_cad/ 可能为空或
#     │                                                diameter=0，会让
#     │                                                capture_ref_data_from_frame
#     │                                                内部的归一化矩阵奇异。
#     └── test_primesense/000001/
#           ├── rgb/            (filename stem = frame_id, e.g. 000254.png)
#           ├── depth/
#           ├── mask_visib/     ({frame:06d}_{gt_idx:06d}.png)
#           ├── scene_gt.json
#           ├── scene_camera.json
#           └── scene_gt_info.json   (optional, for visib_fract filtering)

from Utils import *
import os, sys, json, time, argparse, glob
import logging
import faulthandler
import traceback
from pathlib import Path
import numpy as np
import torch
import trimesh
import yaml
import cv2
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm

from datareader import *
from pose_calculation import *       # OneRefPose

DEFAULT_YCB_CLASSES = {
    "can":     [2, 5, 7, 10],
    "box":     [3, 4, 8, 9, 61],
    "bottles": [6, 21],
    "blocks":  [36, 51, 52],
    "others":  [11, 19, 24, 25, 35, 37, 40],
}


# =========================================================
# [TEST-ONLY] 小工具
# =========================================================
def _sync_cuda():
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _pt(msg):
    print(msg, flush=True)


def _check_reader_inputs(reader, frame, ob_id, tag=""):
    """
    [TEST-ONLY] 在 use_frame_as_ref 前检查输入，帮助区分「数据坏」还是「代码坏」。
    只打印，不修改任何东西。
    """
    prefix = f"  [precheck{(' ' + tag) if tag else ''}]"

    # ---- 1) K 可逆性 ----
    try:
        K = reader.K
        K_np = K.detach().cpu().numpy() if torch.is_tensor(K) else np.asarray(K)
        det = float(np.linalg.det(K_np))
        K_finite = bool(np.isfinite(K_np).all())
        _pt(f"{prefix} K det={det:.6e}  finite={K_finite}  "
            f"K[2,2]={K_np[2,2] if K_np.shape==(3,3) else 'N/A'}")
        if (not K_finite) or abs(det) < 1e-12:
            _pt(f"{prefix} WARNING: K is singular/non-finite! "
                f"scene_camera.json['cam_K'] may be malformed.")
    except Exception as e:
        _pt(f"{prefix} K check failed: {type(e).__name__}: {e}")

    # ---- 2) depth 有效性 ----
    try:
        d = reader.get_depth(frame)
        if d is None:
            _pt(f"{prefix} WARNING: depth is None for frame={frame}")
        else:
            d = np.asarray(d)
            nz = int((d > 1e-6).sum())
            _pt(f"{prefix} depth shape={d.shape}  non_zero={nz}/{d.size}  "
                f"min={float(d.min()):.4f}  max={float(d.max()):.4f}")
            if nz == 0:
                _pt(f"{prefix} WARNING: depth is all zero for frame={frame} "
                    f"(scale / depth_scale may be wrong)")
    except Exception as e:
        _pt(f"{prefix} depth check failed: {type(e).__name__}: {e}")

    # ---- 3) mask 有效性 ----
    try:
        m = reader.get_mask(frame, ob_id, type="mask_visib")
        if m is None:
            _pt(f"{prefix} WARNING: mask is None for frame={frame} ob_id={ob_id}")
        else:
            m = np.asarray(m)
            vs, us = np.where(m > 0)
            _pt(f"{prefix} mask valid_px={len(vs)}  "
                f"bbox={(int(us.min()), int(vs.min()), int(us.max()), int(vs.max())) if len(vs) else None}")
            if len(vs) == 0:
                _pt(f"{prefix} WARNING: mask is empty")
    except Exception as e:
        _pt(f"{prefix} mask check failed: {type(e).__name__}: {e}")

    # ---- 4) GT pose 有效性 ----
    try:
        p = reader.get_gt_pose(frame, ob_id)
        if p is None:
            _pt(f"{prefix} WARNING: gt_pose is None")
        else:
            R = p[:3, :3]
            det = float(np.linalg.det(R))
            finite = bool(np.isfinite(p).all())
            _pt(f"{prefix} gt_pose finite={finite}  det(R)={det:.6f}  "
                f"t={np.round(p[:3, 3], 4)}")
            if (not finite) or abs(det) < 1e-8:
                _pt(f"{prefix} WARNING: gt_pose R is singular / non-finite")
    except Exception as e:
        _pt(f"{prefix} gt_pose check failed: {type(e).__name__}: {e}")


# =========================================================
# TLESS BOP reader
# =========================================================
class TlessBopReader:
    def __init__(self, video_dir, models_dir,
                 zfar=1.5, min_visib=8, min_visib_frac=0.0):
        self.video_dir = video_dir
        self.models_dir = models_dir
        self.zfar = zfar
        self._dataset_dir = str(video_dir)
        self.min_visib = int(min_visib)
        self.min_visib_frac = float(min_visib_frac)
        self._load()

    @staticmethod
    def _stem_to_idx(stem):
        try:
            return int(stem)
        except ValueError:
            try:
                return int(stem.split('_')[-1])
            except ValueError:
                return None

    def _build_frame_map(self, subdir):
        d = {}
        folder = os.path.join(self.video_dir, subdir)
        if not os.path.isdir(folder):
            return d
        for p in glob.glob(os.path.join(folder, '*')):
            stem = os.path.splitext(os.path.basename(p))[0]
            idx = self._stem_to_idx(stem)
            if idx is not None:
                d[idx] = p
        return d

    def _load(self):
        with open(os.path.join(self.video_dir, 'scene_gt.json')) as f:
            self._scene_gt = {int(k): v for k, v in json.load(f).items()}
        with open(os.path.join(self.video_dir, 'scene_camera.json')) as f:
            sc = json.load(f)
        self._scene_camera = {int(k): v for k, v in sc.items()}

        info_path = os.path.join(self.video_dir, 'scene_gt_info.json')
        self._scene_gt_info = {}
        if os.path.isfile(info_path):
            with open(info_path) as f:
                raw = json.load(f)
            self._scene_gt_info = {int(k): v for k, v in raw.items()}

        self.K = np.array(next(iter(sc.values()))['cam_K'],
                          dtype=np.float64).reshape(3, 3)

        self._rgb_map   = self._build_frame_map('rgb')
        self._depth_map = self._build_frame_map('depth')
        self._frames    = sorted(set(self._rgb_map) & set(self._depth_map))

        self.color_files = [self._rgb_map[i]   for i in self._frames]
        self.depth_files = [self._depth_map[i] for i in self._frames]
        self.id_strs = {i: f'{i:06d}' for i in self._frames}

        ob_set = set()
        for entries in self._scene_gt.values():
            for g in entries:
                ob_set.add(int(g['obj_id']))
        self.ob_ids = sorted(ob_set)

        self.symmetry_tfs = {}
        m_info_path = os.path.join(self.models_dir, 'models_info.json')
        if os.path.isfile(m_info_path):
            with open(m_info_path) as f:
                info = json.load(f)
            for oid in self.ob_ids:
                tfs = [np.eye(4, dtype=np.float64)]
                for sf in (info.get(str(oid), {}).get('symmetries_discrete') or []):
                    tfs.append(np.array(sf, dtype=np.float64).reshape(4, 4))
                self.symmetry_tfs[oid] = tfs

        self._img_hw = None
        if self._frames:
            _img = cv2.imread(self._rgb_map[self._frames[0]])
            if _img is not None:
                self._img_hw = _img.shape[:2]
                fx = self.K[0, 0]
                W = _img.shape[1]
                if fx / max(W, 1) > 5 or fx / max(W, 1) < 0.1:
                    print(f'[WARN] K[0,0]={fx:.2f} vs img_w={W} '
                          f'(ratio={fx/max(W,1):.3f}) — scene_camera.json may not '
                          f'match actual rgb size ({_img.shape[0]}x{W})!', flush=True)

    def get_video_id(self):
        return os.path.basename(self.video_dir.rstrip('/'))

    def get_color(self, i):
        p = self._rgb_map.get(i)
        return cv2.imread(p) if p is not None else None

    def get_depth(self, i):
        p = self._depth_map.get(i)
        if p is None:
            return None
        d = cv2.imread(p, -1)
        if d is None:
            return None
        scale = float(self._scene_camera.get(i, {}).get('depth_scale', 1.0))
        return d.astype(np.float32) / scale

    def get_instance_ids_in_image(self, i):
        return [int(g['obj_id']) for g in self._scene_gt.get(i, [])]

    def list_obj_ids(self):
        return list(self.ob_ids)

    def _gt_idx(self, i, oid):
        for j, g in enumerate(self._scene_gt.get(i, [])):
            if int(g['obj_id']) == oid:
                return j
        return None

    def get_gt_pose(self, i, oid):
        j = self._gt_idx(i, oid)
        if j is None:
            return None
        g = self._scene_gt[i][j]
        p = np.eye(4, dtype=np.float64)
        p[:3, :3] = np.array(g['cam_R_m2c']).reshape(3, 3)
        p[:3, 3] = np.array(g['cam_t_m2c'])
        return p

    def get_mask(self, i, oid, type='mask_visib', **kwargs):
        j = self._gt_idx(i, oid)
        if j is None:
            return None
        path = os.path.join(self.video_dir, type, f'{i:06d}_{j:06d}.png')
        return cv2.imread(path, -1) if os.path.isfile(path) else None

    def get_gt_mesh(self, oid):
        return trimesh.load(os.path.join(self.models_dir, f'obj_{oid:06d}.ply'))

    def _visib_fract(self, i, oid):
        j = self._gt_idx(i, oid)
        if j is None:
            return 1.0
        info = self._scene_gt_info.get(i)
        if not info or j >= len(info):
            return 1.0
        return float(info[j].get('visib_fract', 1.0))

    def _mask_bbox(self, i, oid):
        m = self.get_mask(i, oid, type='mask_visib')
        if m is None:
            return None
        vs, us = np.where(m > 0)
        if len(vs) == 0:
            return None
        u0, u1 = int(us.min()), int(us.max())
        v0, v1 = int(vs.min()), int(vs.max())
        return u0, v0, u1, v1

    def frames_with_bbox(self, oid):
        candidates = sorted(k for k, e in self._scene_gt.items()
                            if any(int(g['obj_id']) == oid for g in e))
        out = []
        for k in candidates:
            if k not in self._rgb_map or k not in self._depth_map:
                continue
            if self._visib_fract(k, oid) < self.min_visib_frac:
                continue
            bb = self._mask_bbox(k, oid)
            if bb is None:
                continue
            u0, v0, u1, v1 = bb
            w = u1 - u0 + 1
            h = v1 - v0 + 1
            if w < self.min_visib or h < self.min_visib:
                continue
            out.append((k, w * h, w, h))
        return out

    def frames_for_obj(self, oid):
        return [k for k, _, _, _ in self.frames_with_bbox(oid)]


# =========================================================
# Pose metrics
# =========================================================
class PoseMetrics:
    def __init__(self, dataset_root, metric="bop"):
        assert metric in ("bop", "ycbv"), f"Unknown metric: {metric}"
        self.metric = metric
        self.meta_data = self._build_dataset_metadata(dataset_root)

    @staticmethod
    def _has_symmetry(info):
        for k in ("symmetries", "symmetries_continuous", "symmetries_discrete"):
            if info.get(k):
                return True
        return False

    def _build_dataset_metadata(self, path):
        meta = {}
        # ---- [方案A] 使用 models/ 而非 models_cad/ ----
        models_dir = Path(path) / "models"
        with open(models_dir / "models_info.json") as f:
            info_json = json.load(f)
        for ply in models_dir.glob("obj_*.ply"):
            ob_id = int(ply.stem.split("_")[1])
            info = info_json.get(str(ob_id), {})
            diam = info.get("diameter", None)
            if diam is None or float(diam) <= 1e-6:
                print(f"[PoseMetrics] WARN: ob_id={ob_id:02d} "
                      f"diameter={diam} (models_info.json['{ob_id}'] 可能残缺；"
                      f"mesh_diameter=0 会让 capture_ref_data_from_frame 内部的"
                      f"归一化矩阵奇异)", flush=True)
            meta[ob_id] = {
                "diameter_mm": float(diam) if diam is not None else 200.0,
                "mesh": trimesh.load(ply),
                "has_symmetry": self._has_symmetry(info),
            }
        return meta

    def _apply(self, pts, tf):
        if len(tf.shape) >= 3 and tf.shape[-3] != pts.shape[-2]:
            tf = tf[..., None, :, :]
        return (tf[..., :-1, :-1] @ pts[..., None] + tf[..., :-1, -1:])[..., 0]

    def add_m(self, pred, gt, pts):
        p = self._apply(pts, pred)
        g = self._apply(pts, gt)
        return float(np.linalg.norm(p - g, axis=-1).mean())

    def adds_m(self, pred, gt, pts):
        p = self._apply(pts, pred)
        g = self._apply(pts, gt)
        d, _ = cKDTree(p).query(g, k=1, workers=-1)
        return float(d.mean())

    def compute_all_errors(self, pred, gt, model_pts, ob_id):
        add_m  = self.add_m(pred, gt, model_pts)
        adds_m = self.adds_m(pred, gt, model_pts)

        diam_m = self.meta_data[ob_id]["diameter_mm"] / 1000.0
        add_n  = add_m  / diam_m
        adds_n = adds_m / diam_m

        has_sym = self.meta_data[ob_id]["has_symmetry"]
        selected_m = adds_m if has_sym else add_m
        selected_n = adds_n if has_sym else add_n
        selected_name = "ADD-S" if has_sym else "ADD"

        return {
            "add_m": add_m, "adds_m": adds_m, "selected_m": selected_m,
            "add_n": add_n, "adds_n": adds_n, "selected_n": selected_n,
            "selected_name": selected_name,
            "has_symmetry": has_sym,
        }

    def compute_pose_decomposition_error(self, pred, gt):
        rot = pred[:3, :3] @ gt[:3, :3].T
        rot_err = np.linalg.norm(Rot.from_matrix(rot).as_rotvec()) * 180 / np.pi
        trans_err = np.linalg.norm(pred[:3, 3] - gt[:3, 3])
        return rot_err, trans_err


def compute_auc(errs, max_val=0.1):
    errs = np.asarray(errs)
    if len(errs) == 0:
        return 0.0
    clipped = np.clip(errs, 0, max_val)
    return float(np.mean(1.0 - (clipped / max_val)))


def compute_acc(errs, threshold=0.1):
    errs = np.asarray(errs)
    if len(errs) == 0:
        return 0.0
    return float(np.mean(errs < threshold) * 100.0)


# =========================================================
# Mask helper
# =========================================================
def get_object_mask(reader, i_frame, ob_id, detect_type):
    if detect_type == "box":
        mask = reader.get_mask(i_frame, ob_id, type="mask_visib")
        if mask is None:
            return None
        H, W = mask.shape[:2]
        vs, us = np.where(mask > 0)
        if len(vs) == 0:
            return None
        valid = np.zeros((H, W), dtype=bool)
        valid[vs.min():vs.max() + 1, us.min():us.max() + 1] = 1
        return valid
    if detect_type == "mask":
        m = reader.get_mask(i_frame, ob_id, type="mask_visib")
        return (m > 0) if m is not None else None
    raise RuntimeError(f"Unknown detect_type: {detect_type}")


# =========================================================
# Reader helpers
# =========================================================
def _reader_ob_ids(reader):
    try:
        return list(reader.ob_ids)
    except (AttributeError, TypeError):
        return []


def _discover_video_dirs(tless_dir):
    return sorted(glob.glob(f"{tless_dir}/test_primesense/*"))


def _report_inventory(tless_dir, verbose=True, log_path=None):
    """
    扫描 <tless_dir>/test_primesense/* 下的 scene_gt.json，
    打印两个方向的映射，并返回 (video_to_obs, ob_to_videos)。
    """
    video_dirs = _discover_video_dirs(tless_dir)
    video_to_obs = {}
    ob_to_videos = {}

    for v_idx, vd in enumerate(video_dirs):
        vid = os.path.basename(vd.rstrip('/'))
        gt_path = os.path.join(vd, 'scene_gt.json')
        obs = []
        if os.path.isfile(gt_path):
            try:
                with open(gt_path) as f:
                    raw = json.load(f)
                ob_set = set()
                for entries in raw.values():
                    for g in entries:
                        ob_set.add(int(g['obj_id']))
                obs = sorted(ob_set)
            except Exception as e:
                print(f"[inventory] WARN: failed to read {gt_path}: "
                      f"{type(e).__name__}: {e}", flush=True)
        video_to_obs[v_idx] = (vid, obs)
        for oid in obs:
            ob_to_videos.setdefault(oid, []).append(v_idx)

    lines = []

    def _out(s):
        lines.append(s)
        if verbose:
            print(s, flush=True)

    _out("\n========== [Inventory] video_dir -> ob_ids ==========")
    for v_idx in sorted(video_to_obs):
        vid, obs = video_to_obs[v_idx]
        if obs:
            _out(f"  v{v_idx:02d} ({vid}): {len(obs)} obj(s) -> {obs}")
        else:
            _out(f"  v{v_idx:02d} ({vid}): (empty / no scene_gt)")

    _out("\n========== [Inventory] ob_id -> video_dirs ==========")
    for oid in sorted(ob_to_videos):
        vs = ob_to_videos[oid]
        _out(f"  ob_id={oid:02d}: appears in {len(vs)} video(s) -> {vs}")

    if log_path:
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'w') as f:
                f.write("\n".join(lines) + "\n")
            if verbose:
                print(f"[inventory] saved -> {log_path}", flush=True)
        except Exception as e:
            print(f"[inventory] WARN: failed to write {log_path}: "
                  f"{type(e).__name__}: {e}", flush=True)

    return video_to_obs, ob_to_videos


def _get_reader(tless_dir, v_idx, cache=None):
    if cache is not None and v_idx in cache:
        return cache[v_idx]
    video_dirs = _discover_video_dirs(tless_dir)
    # ---- [方案A] 使用 models/ 而非 models_cad/ ----
    models_dir = os.path.join(tless_dir, 'models')
    r = TlessBopReader(video_dirs[v_idx], models_dir,
                       min_visib=opt.min_visib,
                       min_visib_frac=opt.min_visib_frac)
    if cache is not None:
        cache[v_idx] = r
    return r


# =========================================================
# Reference candidates
# =========================================================
def _find_ref_candidates(tless_dir, ob_id,
                         ref_video_idx=-1, ref_frame=-1,
                         cache=None, verbose=True, max_candidates=20):
    video_dirs = _discover_video_dirs(tless_dir)
    n = len(video_dirs)
    if n == 0:
        return []

    if verbose:
        print(f"  [scan] Scanning {n} videos for ob_id={ob_id} ...", flush=True)

    pools = []
    for v_idx in range(n):
        r = _get_reader(tless_dir, v_idx, cache)
        if ob_id not in r.ob_ids:
            if verbose:
                print(f"  [scan] v{v_idx:02d} ({r.get_video_id()}): "
                      f"ob_id={ob_id:02d} NOT present", flush=True)
            continue
        infos = r.frames_with_bbox(ob_id)
        if not infos:
            if verbose:
                print(f"  [scan] v{v_idx:02d} ({r.get_video_id()}): "
                      f"ob_id={ob_id:02d} present but no usable frames", flush=True)
            continue
        if verbose:
            print(f"  [scan] v{v_idx:02d} ({r.get_video_id()}): "
                  f"{len(infos)} usable frames, "
                  f"bbox range w=[{min(i[2] for i in infos)}..{max(i[2] for i in infos)}], "
                  f"h=[{min(i[3] for i in infos)}..{max(i[3] for i in infos)}]",
                  flush=True)
        for f, area, w, h in infos:
            if v_idx == ref_video_idx and f == ref_frame:
                prio = 0
            elif v_idx == ref_video_idx:
                prio = 1
            else:
                prio = 2
            pools.append((prio, -area, v_idx, f, w, h, area))

    if not pools:
        return []

    pools.sort(key=lambda t: (t[0], t[1]))
    candidates = [(p[2], p[3], p[4], p[5], p[6]) for p in pools[:max_candidates]]

    if verbose:
        top = candidates[:5]
        print(f"  [scan] {len(candidates)} candidates, top: "
              f"{[(v, f, w, h) for v, f, w, h, _ in top]}", flush=True)

    return candidates


# =========================================================
# Single-frame inference
# =========================================================
def run_inference_worker(reader, i_frame, est, ob_id, device, metrics, model_pts,
                         detect_type="mask",
                         current_eval_idx=None, total_eval_frames=None):
    color = reader.get_color(i_frame)
    depth = reader.get_depth(i_frame)
    if color is None or depth is None:
        return None, None, None, None, None

    ob_mask = get_object_mask(reader, i_frame, ob_id, detect_type)
    if ob_mask is None or ob_mask.sum() < 4:
        return None, None, None, None, None

    gt_pose = reader.get_gt_pose(i_frame, ob_id)
    if gt_pose is None:
        return None, None, None, None, None

    est.gt_pose = gt_pose
    pred_pose = est.register(K=reader.K, rgb=color, depth=depth,
                             ob_mask=ob_mask, ob_id=ob_id, iteration=5)

    err_dict = metrics.compute_all_errors(pred_pose, gt_pose, model_pts, ob_id)
    rot_err, trans_err = metrics.compute_pose_decomposition_error(pred_pose, gt_pose)

    add_n  = err_dict["add_n"]
    adds_n = err_dict["adds_n"]
    selected_n    = err_dict["selected_n"]
    selected_name = err_dict["selected_name"]

    idx_str = f"{current_eval_idx}/{total_eval_frames}" \
              if current_eval_idx is not None else f"{i_frame}"

    print(f"\n[Frame {idx_str} | Video {reader._dataset_dir.split('/')[-1]} | "
          f"RawFrame {i_frame} | Object {ob_id}]", flush=True)
    print("GT Pose:\n", np.round(gt_pose, 8), flush=True)
    print("Pred Pose:\n", np.round(pred_pose, 8), flush=True)
    print(f"ADD: {add_n:.6f} | ADD-S: {adds_n:.6f} | "
          f"{selected_name}: {selected_n:.6f} | "
          f"Rot: {rot_err:.2f} deg | Trans: {trans_err * 1000:.2f} mm", flush=True)

    return pred_pose, gt_pose, err_dict, rot_err, trans_err


def _load_ycb_classes(path):
    if path is None:
        return DEFAULT_YCB_CLASSES
    with open(path) as f:
        raw = yaml.safe_load(f)
    return {str(k): [int(x) for x in v] for k, v in raw.items()}


# =========================================================
# Pipeline
# =========================================================
def run_pipeline(detect_type="mask"):
    ob_ids = ([int(v) for v in opt.ob_ids.split(",") if v.strip()]
              if opt.ob_ids else [opt.ob_id])
    multi_ob = len(ob_ids) > 1
    skip_ref = bool(opt.eval_skip_ref)
    metric = opt.metric
    stride = max(1, opt.eval_stride)

    _pt("[pipeline] CUDA preflight ...")
    try:
        if not torch.cuda.is_available():
            _pt("[pipeline] WARNING: torch.cuda.is_available() == False "
                "(OneRefPose 里硬编码 device='cuda' 会失败)")
        else:
            _pt(f"[pipeline] CUDA device 0: {torch.cuda.get_device_name(0)}")
            _t0 = time.time()
            _ = torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
            _pt(f"[pipeline] CUDA preflight OK ({time.time()-_t0:.2f}s)")
    except Exception as e:
        _pt(f"[pipeline] CUDA preflight FAILED: {type(e).__name__}: {e}")
        raise

    # ---- [方案A] PoseMetrics 内部现在读 <tless_dir>/models ----
    metrics = PoseMetrics(opt.tless_dir, metric=metric)
    ycb_classes = _load_ycb_classes(opt.ycb_classes) if metric == "ycbv" else None

    video_dirs = _discover_video_dirs(opt.tless_dir)
    if not video_dirs:
        raise SystemExit(f"[pipeline] No videos under {opt.tless_dir}/test_primesense/")

    print(f"[pipeline] Evaluating {len(video_dirs)} videos, {len(ob_ids)} objects, "
          f"metric={metric}, crop_ratio={opt.crop_ratio}, stride={stride}, "
          f"min_visib={opt.min_visib}, min_visib_frac={opt.min_visib_frac}, "
          f"ref_device={opt.ref_device}, "
          f"ref_crop_to_ref={opt.ref_crop_to_ref}, "
          f"models_dir=<tless_dir>/models", flush=True)

    reader_cache = {}
    reader0 = _get_reader(opt.tless_dir, 0, reader_cache)
    print(f"[pipeline] video 0 ob_ids: {_reader_ob_ids(reader0)}", flush=True)

    inventory_log = (os.path.join(opt.debug_dir, "inventory.log")
                     if opt.debug_dir else None)
    video_to_obs, ob_to_videos = _report_inventory(
        opt.tless_dir, verbose=True, log_path=inventory_log)

    print("\n========== [Inventory] target ob_ids -> video_dirs ==========",
          flush=True)
    for ob_id in ob_ids:
        vs = ob_to_videos.get(ob_id, [])
        if vs:
            print(f"  target ob_id={ob_id:02d}: appears in "
                  f"{len(vs)} video(s) -> {vs}", flush=True)
        else:
            print(f"  target ob_id={ob_id:02d}: NOT FOUND in any video "
                  f"(check --tless_dir / ob_ids)", flush=True)

    all_errors = {}
    all_res = NestDict()

    for i_ob, ob_id in enumerate(ob_ids):

        print(f"\n========== [obj {i_ob + 1}/{len(ob_ids)}] ob_id={ob_id:02d} ==========",
              flush=True)

        # ---- 打印 diameter 便于排查 ----
        try:
            di_mm = metrics.meta_data[ob_id]["diameter_mm"]
            _pt(f"  [meta] ob_id={ob_id:02d} diameter_mm={di_mm}  "
                f"has_symmetry={metrics.meta_data[ob_id]['has_symmetry']}")
            if di_mm is None or float(di_mm) <= 1e-9:
                _pt(f"  [meta] WARNING: diameter_mm is zero/negative! "
                    f"models/models_info.json['{ob_id}']['diameter'] may be "
                    f"missing/wrong; capture_ref_data_from_frame 可能因此奇异。")
                # 用 mesh 实际尺寸兜底，避免 mesh_diameter=0
                try:
                    verts = np.asarray(metrics.meta_data[ob_id]["mesh"].vertices)
                    di_mm = float(np.linalg.norm(verts.max(0) - verts.min(0))) * 1000.0
                    _pt(f"  [meta] fallback diameter_mm={di_mm:.3f} (from mesh bbox)")
                except Exception as e:
                    _pt(f"  [meta] fallback also failed: {type(e).__name__}: {e}")
        except Exception as e:
            di_mm = 200.0
            _pt(f"  [meta] WARNING: cannot read diameter for ob_id={ob_id}: "
                f"{type(e).__name__}: {e}")

        candidates = _find_ref_candidates(
            opt.tless_dir, ob_id,
            ref_video_idx=opt.ref_video_idx,
            ref_frame=opt.ref_frame,
            cache=reader_cache, verbose=True,
        )
        if not candidates:
            print(f"[pipeline] WARNING: no usable reference frame for ob_id={ob_id}",
                  flush=True)
            continue

        est = None
        ref_v_idx = ref_k = ref_w = ref_h = None

        for ci, (v_idx, frame, w, h, area) in enumerate(candidates):
            _pt(f"  [ref try {ci+1}/{len(candidates)}] "
                f"v{v_idx:02d} f{frame:04d} (bbox {w}x{h}, area={area})")

            reader_ref = _get_reader(opt.tless_dir, v_idx, reader_cache)

            _check_reader_inputs(reader_ref, frame, ob_id,
                                 tag=f"v{v_idx:02d}f{frame:04d}")

            probe_dir = None
            if opt.save_ref_probe:
                probe_dir = (os.path.join(opt.save_ref_probe, f"ob{ob_id:02d}")
                             if multi_ob else opt.save_ref_probe)
                os.makedirs(probe_dir, exist_ok=True)

            diam_m = di_mm / 1000.0
            if not (diam_m > 1e-9):
                _pt(f"  [ref] ERROR: mesh_diameter={diam_m} <= 0 "
                    f"(di_mm={di_mm}); 跳过该候选以免 capture_ref_data_from_frame "
                    f"内部 linalg.inv 奇异。")
                continue

            save_to_disk = bool(opt.save_ref) and bool(opt.ref_data_root)

            _pt(f"  [ref] use args: crop_ratio={opt.crop_ratio} "
                f"crop_to_ref={opt.ref_crop_to_ref} "
                f"ref_device={opt.ref_device} mesh_diameter={diam_m}")

            # ---------- (A) 构造 OneRefPose ----------
            _t0 = time.time()
            _pt(f"  [ref] constructing OneRefPose(ob_id={ob_id}, "
                f"ref_data_root={opt.ref_data_root}) ...")
            try:
                cand_est = OneRefPose(scorer=None, refiner=None,
                                      ref_data_root=opt.ref_data_root, ob_id=ob_id)
            except Exception:
                _pt(f"  [ref] OneRefPose.__init__ FAILED")
                traceback.print_exc()
                if opt.stop_on_ref_fail:
                    raise
                continue
            _sync_cuda()
            _pt(f"  [ref] OneRefPose ready ({time.time()-_t0:.2f}s)")

            # ---------- (B) to_device ----------
            _t0 = time.time()
            _pt(f"  [ref] to_device('cuda:0') ...")
            try:
                cand_est.to_device("cuda:0")
            except Exception:
                _pt(f"  [ref] to_device FAILED")
                traceback.print_exc()
                del cand_est
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                if opt.stop_on_ref_fail:
                    raise
                continue
            _sync_cuda()
            _pt(f"  [ref] to_device done ({time.time()-_t0:.2f}s)")

            # ---------- (C) use_frame_as_ref ----------
            _t0 = time.time()
            _pt(f"  [ref] use_frame_as_ref(frame={frame}, "
                f"crop_ratio={opt.crop_ratio}, crop_to_ref={opt.ref_crop_to_ref}) ...")
            try:
                cand_est.use_frame_as_ref(
                    reader_ref, frame, ob_id,
                    is_symmetric=False,
                    symmetry_axis=opt.ref_sym_axis,
                    crop_ratio=opt.crop_ratio,
                    device=opt.ref_device,
                    save_to_disk=save_to_disk,
                    mesh_diameter=diam_m,
                    crop_to_ref=opt.ref_crop_to_ref,
                    probe_dir=probe_dir,
                )
            except Exception:
                _pt(f"  [ref] use_frame_as_ref FAILED "
                    f"v{v_idx:02d} f{frame:04d}  (完整 traceback 如下) ")
                traceback.print_exc()
                _pt(f"  [ref] 提示：错误发生在 capture_ref_data_from_frame 内部，"
                    f"常见原因是某个 4x4/3x3 矩阵奇异。可尝试 "
                    f"`--no_ref_crop_to_ref`、`--ref_device cpu`、"
                    f"`--crop_ratio 1.0` 或 `--crop_ratio 1.05`。")
                del cand_est
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                if opt.stop_on_ref_fail:
                    raise
                continue
            _sync_cuda()
            _pt(f"  [ref] use_frame_as_ref done ({time.time()-_t0:.2f}s)")

            est = cand_est
            ref_v_idx, ref_k, ref_w, ref_h = v_idx, frame, w, h
            _pt(f"  [ref] OK accepted v{ref_v_idx:02d} f{ref_k:04d}")
            break

        if est is None:
            print(f"[pipeline] ERROR: all {len(candidates)} candidate reference frames "
                  f"failed for ob_id={ob_id}; skip.", flush=True)
            continue

        print("\n" + "=" * 50, flush=True)
        print(f"Selected Reference Frame: Video [{ref_v_idx:02d}] "
              f"Frame [{ref_k:04d}] bbox {ref_w}x{ref_h}", flush=True)
        print("=" * 50 + "\n", flush=True)

        reader_ref = _get_reader(opt.tless_dir, ref_v_idx, reader_cache)

        save_vis_dir = opt.debug_dir
        if not save_vis_dir:
            if opt.save_ref_probe:
                save_vis_dir = (os.path.join(opt.save_ref_probe, f"ob{ob_id:02d}")
                                if multi_ob else opt.save_ref_probe)
            elif opt.ref_data_root:
                save_vis_dir = os.path.join(opt.ref_data_root, str(ob_id))
            else:
                save_vis_dir = "debug_ref_vis"

        os.makedirs(save_vis_dir, exist_ok=True)
        abs_save_vis_dir = os.path.abspath(save_vis_dir)

        if getattr(est, 'ref_data', None) is not None \
                and getattr(est.ref_data, 'rgb', None) is not None:
            crop_rgb  = est.ref_data.rgb.detach().cpu().permute(1, 2, 0)\
                .numpy().astype(np.uint8)
            crop_mask = (est.ref_data.xyz_map[2] > 1e-5).cpu().numpy()\
                .astype(np.uint8) * 255

            cv2.imwrite(os.path.join(
                abs_save_vis_dir,
                f"ref_ob{ob_id:02d}_v{ref_v_idx:02d}_f{ref_k:04d}_crop_rgb.png"),
                cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))

            cv2.imwrite(os.path.join(
                abs_save_vis_dir,
                f"ref_ob{ob_id:02d}_v{ref_v_idx:02d}_f{ref_k:04d}_crop_mask.png"),
                crop_mask)

            overlay = crop_rgb.copy()
            overlay[crop_mask > 0] = [0, 255, 0]
            blended = cv2.addWeighted(overlay, 0.3, crop_rgb, 0.7, 0)
            cv2.imwrite(os.path.join(
                abs_save_vis_dir,
                f"ref_ob{ob_id:02d}_v{ref_v_idx:02d}_f{ref_k:04d}_crop_overlay.png"),
                blended)

        if opt.save_ref and opt.ref_data_root:
            print(f"[pipeline] Ref saved -> "
                  f"{os.path.join(opt.ref_data_root, str(ob_id))}", flush=True)
        if opt.save_ref_probe:
            if getattr(est, 'refiner', None) is not None:
                est.refiner.save_ref_probe_dir = opt.save_ref_probe
            if getattr(est, 'scorer', None) is not None:
                est.scorer.save_ref_probe_dir = opt.save_ref_probe

        model_pts = reader0.get_gt_mesh(ob_id).vertices.copy()

        if metric == "bop":
            all_errors[ob_id] = {"add_m": [], "adds_m": [], "selected_m": []}
        else:
            all_errors[ob_id] = {"add": [], "adds": []}

        target_seq_count = 0
        total_eval_frames = 0
        for v_idx in range(len(video_dirs)):
            r = _get_reader(opt.tless_dir, v_idx, reader_cache)
            for k in r.frames_for_obj(ob_id):
                if skip_ref and v_idx == ref_v_idx and k == ref_k:
                    continue
                if target_seq_count % stride == 0:
                    total_eval_frames += 1
                target_seq_count += 1

        if opt.max_eval_frames > 0:
            total_eval_frames = min(total_eval_frames, opt.max_eval_frames)
            print(f"  [eval] Target ob_id={ob_id:02d} limited to "
                  f"{total_eval_frames} frames...", flush=True)
        else:
            print(f"  [eval] Target ob_id={ob_id:02d} appears in "
                  f"{total_eval_frames} eval frames (stride={stride}). Starting...",
                  flush=True)

        pbar = tqdm(total=total_eval_frames, desc=f"ob_id={ob_id:02d}",
                    dynamic_ncols=True)
        evaluated_count = 0
        n_skipped = 0
        target_seq_count = 0

        for v_idx in range(len(video_dirs)):
            if opt.max_eval_frames > 0 and evaluated_count >= opt.max_eval_frames:
                break

            r = _get_reader(opt.tless_dir, v_idx, reader_cache)

            for k in r.frames_for_obj(ob_id):
                if opt.max_eval_frames > 0 and evaluated_count >= opt.max_eval_frames:
                    break

                if skip_ref and v_idx == ref_v_idx and k == ref_k:
                    n_skipped += 1
                    continue

                if target_seq_count % stride != 0:
                    target_seq_count += 1
                    continue

                target_seq_count += 1
                pbar.set_postfix({"v": f"{v_idx:02d}", "f": f"{k:04d}"})

                try:
                    pred_pose, gt_pose, err_dict, rot_err, trans_err = \
                        run_inference_worker(
                            r, k, est, ob_id, "cuda:0", metrics, model_pts,
                            detect_type,
                            current_eval_idx=evaluated_count + 1,
                            total_eval_frames=total_eval_frames,
                        )
                except Exception:
                    print(f"\n[register error] v{v_idx:02d} f{k:04d}", flush=True)
                    traceback.print_exc()
                    evaluated_count += 1
                    pbar.update(1)
                    continue

                if err_dict is None:
                    evaluated_count += 1
                    pbar.update(1)
                    continue

                if metric == "bop":
                    all_errors[ob_id]["add_m"].append(err_dict["add_m"])
                    all_errors[ob_id]["adds_m"].append(err_dict["adds_m"])
                    all_errors[ob_id]["selected_m"].append(err_dict["selected_m"])
                else:
                    all_errors[ob_id]["add"].append(err_dict["add_m"])
                    all_errors[ob_id]["adds"].append(err_dict["adds_m"])

                all_res[r.get_video_id()][r.id_strs[k]][ob_id] = pred_pose
                evaluated_count += 1
                pbar.update(1)

        pbar.close()

        if metric == "bop":
            errs = np.array(all_errors[ob_id]["selected_m"])
        else:
            errs = np.array(all_errors[ob_id]["adds"])

        if len(errs) == 0:
            print(f"[pipeline] WARNING: no valid eval frames for ob_id={ob_id}",
                  flush=True)
            continue

        if opt.debug_dir:
            np.savetxt(os.path.join(opt.debug_dir, f"errors_ob{ob_id:02d}.txt"),
                       errs, fmt="%.6f")

        del est
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    if metric == "bop":
        _finalize_bop(all_errors, metrics)
    else:
        _finalize_ycbv(all_errors, ycb_classes)

    if opt.debug_dir:
        os.makedirs(opt.debug_dir, exist_ok=True)
        with open(os.path.join(opt.debug_dir, "tless_res.yml"), "w") as f:
            yaml.safe_dump(make_yaml_dumpable(all_res), f)
        print(f"\n[pipeline] Final results saved to {opt.debug_dir}", flush=True)


def _finalize_bop(all_errors, metrics):
    print("\n========== BOP evaluation (threshold = 10 cm) ==========", flush=True)
    if not all_errors:
        print("(no results)", flush=True)
        return

    per_obj = []
    for ob_id, v in sorted(all_errors.items()):
        if not v["selected_m"]:
            continue
        add_m  = np.asarray(v["add_m"])
        adds_m = np.asarray(v["adds_m"])
        sel_m  = np.asarray(v["selected_m"])
        has_sym = metrics.meta_data[ob_id]["has_symmetry"]

        a_add  = compute_auc(add_m,  max_val=0.1) * 100.0
        a_adds = compute_auc(adds_m, max_val=0.1) * 100.0
        a_sel  = compute_auc(sel_m,  max_val=0.1) * 100.0

        acc_add  = compute_acc(add_m,  threshold=0.1)
        acc_adds = compute_acc(adds_m, threshold=0.1)
        acc_sel  = compute_acc(sel_m,  threshold=0.1)

        per_obj.append({
            "ob_id": ob_id, "n": len(sel_m), "has_sym": has_sym,
            "add_auc": a_add, "adds_auc": a_adds, "sel_auc": a_sel,
            "add_acc": acc_add, "adds_acc": acc_adds, "sel_acc": acc_sel,
        })

        print(f"\n========== [Object {ob_id:02d}] ==========", flush=True)
        print(f" n_evaluated:        {len(sel_m)}", flush=True)
        print(f" sym (has_symmetry): {'Y' if has_sym else '-'}", flush=True)
        print(f"   ADD AUC:          {a_add:.2f}    ADD@10cm:     {acc_add:.2f}%", flush=True)
        print(f"   ADD-S AUC:        {a_adds:.2f}    ADD-S@10cm:   {acc_adds:.2f}%", flush=True)
        print(f"   ADD(-S) AUC:      {a_sel:.2f}    ADD(-S)@10cm: {acc_sel:.2f}%  "
              f"(Selected: {'ADD-S' if has_sym else 'ADD'})", flush=True)

    if not per_obj:
        return

    def _mean(key):
        return float(np.mean([p[key] for p in per_obj]))

    print(f"\n========== [OVERALL MEAN] ==========", flush=True)
    print(f" Objects:            {len(per_obj)} (per-object equal weight)", flush=True)
    print(f"   ADD AUC:          {_mean('add_auc'):.2f}    ADD@10cm:     {_mean('add_acc'):.2f}%", flush=True)
    print(f"   ADD-S AUC:        {_mean('adds_auc'):.2f}    ADD-S@10cm:   {_mean('adds_acc'):.2f}%", flush=True)
    print(f"   ADD(-S) AUC:      {_mean('sel_auc'):.2f}    ADD(-S)@10cm: {_mean('sel_acc'):.2f}%", flush=True)


def _finalize_ycbv(all_errors, ycb_classes):
    print("\n========== YCB-Video (5-class mean, threshold=10 cm) ==========",
          flush=True)
    if not all_errors:
        print("(no results)", flush=True)
        return

    per_obj = {}
    for ob_id, d in sorted(all_errors.items()):
        if not d["adds"]:
            continue
        adds = np.asarray(d["adds"])
        add  = np.asarray(d["add"])
        per_obj[ob_id] = {
            "n": len(adds),
            "adds_auc": compute_auc(adds, max_val=0.1) * 100.0,
            "add_auc":  compute_auc(add,  max_val=0.1) * 100.0,
            "adds_acc": compute_acc(adds, threshold=0.1),
            "add_acc":  compute_acc(add,  threshold=0.1),
        }

    if not per_obj:
        return

    assigned = set()
    print(f"\n{'class':>10} {'n_obj':>6} {'n_eval':>8} "
          f"{'ADD-S AUC':>12} {'ADD AUC':>10} {'ADD-S@10cm':>12} {'ADD@10cm':>10}",
          flush=True)
    cls_adds, cls_add, cls_adds_acc, cls_add_acc = [], [], [], []

    for cls_name in ["can", "box", "bottles", "blocks", "others"]:
        ids = [i for i in ycb_classes.get(cls_name, []) if i in per_obj]
        if not ids:
            continue
        assigned.update(ids)
        a     = float(np.mean([per_obj[i]["adds_auc"]  for i in ids]))
        b     = float(np.mean([per_obj[i]["add_auc"]   for i in ids]))
        a_acc = float(np.mean([per_obj[i]["adds_acc"]  for i in ids]))
        b_acc = float(np.mean([per_obj[i]["add_acc"]   for i in ids]))
        n     = int(sum(per_obj[i]["n"] for i in ids))
        cls_adds.append(a); cls_add.append(b)
        cls_adds_acc.append(a_acc); cls_add_acc.append(b_acc)
        print(f"{cls_name:>10} {len(ids):>6} {n:>8} "
              f"{a:>12.2f} {b:>10.2f} {a_acc:>11.2f}% {b_acc:>9.2f}%", flush=True)

    unassigned = sorted(set(per_obj.keys()) - assigned)
    if unassigned:
        print(f"\n[WARN] Objects not in any YCB class: {unassigned}", flush=True)

    if cls_adds:
        print("-" * 88, flush=True)
        print(f"{'MEAN':>10} {'':>6} {'':>8} "
              f"{np.mean(cls_adds):>12.2f} {np.mean(cls_add):>10.2f} "
              f"{np.mean(cls_adds_acc):>11.2f}% {np.mean(cls_add_acc):>9.2f}%",
              flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tless_dir", type=str, required=True,
                        help="e.g. /root/data/tless_data/tless")
    parser.add_argument("--ob_id",  type=int, default=1)
    parser.add_argument("--ob_ids", type=str, default=None,
                        help='comma-separated, e.g. "1,2,5"')
    parser.add_argument("--ref_video_idx", type=int, default=-1)
    parser.add_argument("--ref_frame",     type=int, default=-1)

    parser.add_argument("--eval_skip_ref", action="store_true", default=True)
    parser.add_argument("--eval_include_ref", dest="eval_skip_ref",
                        action="store_false")
    parser.add_argument("--eval_stride", type=int, default=1)

    parser.add_argument("--ref_data_root",  type=str, default=None)
    parser.add_argument("--save_ref",       action="store_true")
    parser.add_argument("--save_ref_probe", type=str, default=None)
    parser.add_argument("--ref_sym_axis",   type=int, default=-1)
    parser.add_argument("--ref_device",     type=str, default="cuda",
                        help="传给 use_frame_as_ref(device=...)。"
                             "默认 cuda；排错时可试 cpu。")

    parser.add_argument("--ref_crop_to_ref", action="store_true", default=True,
                        help="传给 use_frame_as_ref(crop_to_ref=...)。")
    parser.add_argument("--no_ref_crop_to_ref", dest="ref_crop_to_ref",
                        action="store_false")

    parser.add_argument("--stop_on_ref_fail", action="store_true", default=False,
                        help="第一个候选参考帧失败就抛出异常退出。")

    parser.add_argument("--metric", type=str, default="bop",
                        choices=["bop", "ycbv"])
    parser.add_argument("--ycb_classes", type=str, default=None)

    parser.add_argument("--detect_type", type=str, default="mask",
                        choices=["mask", "box"])
    parser.add_argument("--debug_dir", type=str, default="")
    parser.add_argument("--crop_ratio", type=float, default=1.1)
    parser.add_argument("--max_eval_frames", type=int, default=0)

    parser.add_argument("--min_visib", type=int, default=8)
    parser.add_argument("--min_visib_frac", type=float, default=0.0)

    parser.add_argument("--dump_after", type=float, default=0.0)
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    opt = parser.parse_args()
    set_seed(0)

    logging.basicConfig(
        level=getattr(logging, opt.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    faulthandler.enable()
    if opt.dump_after > 0:
        faulthandler.dump_traceback_later(int(opt.dump_after), repeat=True)
        _pt(f"[test] faulthandler will dump stacks every "
            f"{opt.dump_after:.0f}s (PID={os.getpid()})")

    start = time.time()
    run_pipeline(detect_type=opt.detect_type)
    print(f"\n[pipeline] Total wall time: {time.time() - start:.2f}s", flush=True)