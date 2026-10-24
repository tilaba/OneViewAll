# test_real275.py
# -*- coding: utf-8 -*-
"""
REAL275 / NOCS 位姿评估（OneRefPose）。
AR 四条硬约束：
  #1 数据错误立即 raise（模型/直径/GT/mask_id 缺失）。
  #2 pred=None 计零召回（inf，不删样本）。
  #3 指标异常 → NaN，不算 0 分；完整 AR 需三项齐全。
  #4 全目标集 + 完整子集两套 AR 都报，显式追踪每个请求对象状态。

注意：本脚本产出的是"REAL275 自定义协议下的 BOP 风格 AR"，
不是官方 REAL275 / BOP 评测结果。启用前请核实：
  - 模型尺度、GT 平移单位、原点/坐标轴一致性；
  - 逐模型对称性（当前 class 1/2/4 → Y 轴连续旋转为自定义假设）。
"""

from Utils import *
import os, sys, json, time, argparse, glob
import logging, faulthandler, traceback, tempfile, pickle, re
import numpy as np
import torch
import trimesh
import yaml as _pyyaml
import cv2
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from tqdm import tqdm

from datareader import *
from pose_calculation import OneRefPose

_INTERNAL_OB_KEY_MODE = "class_id"
_GLOBAL_MODEL_ID_TO_NAME = None
_GLOBAL_MODEL_DIAMETER_MM_CACHE = {}


def _sync_cuda():
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _pt(msg):
    print(msg, flush=True)


def _sanitize_key(k):
    return re.sub(r"[^A-Za-z0-9_\-\.]", "_", str(k))


def orthogonalize_rotation(R_in):
    U, _, Vt = np.linalg.svd(R_in)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    return R


def read_gt_class_ids(data, n_instances, source=""):
    for key in ("gt_class_ids", "gt_classes", "classes"):
        value = data.get(key)
        if value is None:
            continue
        values = np.asarray(value).reshape(-1)
        if len(values) != n_instances:
            raise ValueError(f"{source}: {key} len={len(values)} vs {n_instances}")
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError(f"{source}: {key} non-numeric")
        if not np.isfinite(values).all():
            raise ValueError(f"{source}: {key} NaN/Inf")
        ids = values.astype(np.int64)
        if not np.array_equal(values, ids):
            raise ValueError(f"{source}: {key} non-integer")
        return ids, key
    raise KeyError(f"{source}: missing gt_class_ids/gt_classes/classes")


def parse_meta_models(meta_path):
    """_meta.txt -> {inst_id: {class_id, model_name, symmetry_id}}；支持 3/4 列。"""
    if not os.path.isfile(meta_path):
        return {}
    out = {}
    with open(meta_path, "r") as f:
        for line in f:
            w = line.strip().split()
            if len(w) < 3:
                continue
            try:
                inst_id, cls_id = int(w[0]), int(w[1])
            except ValueError:
                continue
            if len(w) == 3:
                name, sym = w[2], 0
            else:
                try:
                    sym = int(w[2])
                except ValueError:
                    sym = 0
                name = w[3]
            if name.endswith(".npz"):
                name = name[:-4]
            out[inst_id] = {"class_id": cls_id, "model_name": name, "symmetry_id": sym}
    return out


def pose_m_to_mm(pose_m):
    p = np.asarray(pose_m, dtype=np.float64).copy()
    p[:3, 3] *= 1000.0
    return p


def compute_model_diameter_mm(model_pts_m, model_id=None, verbose=False):
    """真实 BOP 直径（mm）。失败 raise，禁止 bbox 回退。"""
    if model_id is not None and model_id in _GLOBAL_MODEL_DIAMETER_MM_CACHE:
        return _GLOBAL_MODEL_DIAMETER_MM_CACHE[model_id]
    pts = np.asarray(model_pts_m, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 4:
        raise ValueError(f"model_points < 4 (got {len(pts)})")
    if not np.isfinite(pts).all():
        raise ValueError("model_points 含 NaN/Inf")
    try:
        from scipy.spatial import ConvexHull
        hull = pts[ConvexHull(pts).vertices]
    except Exception:
        hull = pts
    max_sq = 0.0
    for s in range(0, len(hull), 1024):
        d2 = cdist(hull[s:s + 1024], hull, metric="sqeuclidean")
        max_sq = max(max_sq, float(d2.max()))
    diam_mm = float(np.sqrt(max_sq)) * 1000.0
    if not (np.isfinite(diam_mm) and diam_mm > 0):
        raise ValueError(f"真实直径无效: {diam_mm}")
    if verbose:
        _pt(f"[diam] model_id={model_id} 直径={diam_mm:.2f} mm (verts={len(pts)}, hull={len(hull)})")
    if model_id is not None:
        _GLOBAL_MODEL_DIAMETER_MM_CACHE[model_id] = diam_mm
    return diam_mm


def _default_symmetry_transforms(class_id, symmetry_id=0, max_step_rad=0.05):
    """[自定义假设] class 1/2/4 视为绕 Y 轴连续旋转对称。symmetry_id 未使用。"""
    eye = {"R": np.eye(3, dtype=np.float32), "t": np.zeros((3, 1), dtype=np.float32)}
    if class_id not in (1, 2, 4):
        return [eye]
    axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    n = max(2, int(np.ceil(2 * np.pi / max_step_rad)))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
                 dtype=np.float32)
    syms = [eye]
    for k in range(1, n):
        th = 2.0 * np.pi * k / n
        R = np.eye(3, dtype=np.float32) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
        syms.append({"R": R.astype(np.float32), "t": np.zeros((3, 1), dtype=np.float32)})
    return syms


class ARAccumulator:
    """先定 n_targets，逐目标追踪 done/fail_reason。"""

    def __init__(self, n_targets, vsd_taus, vsd_thetas, mssd_thetas, mspd_thetas):
        if n_targets <= 0:
            raise ValueError("评估目标集为空，AR 未定义")
        self.n_targets = int(n_targets)
        self.vsd_thetas = np.asarray(vsd_thetas, dtype=np.float64)
        self.mssd_thetas = np.asarray(mssd_thetas, dtype=np.float64)
        self.mspd_thetas = np.asarray(mspd_thetas, dtype=np.float64)
        self.errors = {
            "vsd": np.full((self.n_targets, len(vsd_taus)), np.inf, dtype=np.float64),
            "mssd": np.full(self.n_targets, np.inf, dtype=np.float64),
            "mspd": np.full(self.n_targets, np.inf, dtype=np.float64),
        }
        self.done = {k: np.zeros(self.n_targets, dtype=bool) for k in self.errors}
        self.fail_reasons = {k: [] for k in self.errors}

    def mark_prediction_missing(self, index):
        for name in self.errors:
            self.done[name][index] = True

    def mark_metric_unavailable(self, name, reason):
        if name not in self.fail_reasons:
            raise KeyError(f"unknown metric {name}")
        self.fail_reasons[name].append(str(reason))

    def record(self, name, index, compute):
        if name not in self.errors:
            raise KeyError(f"unknown metric {name}")
        self.done[name][index] = True
        try:
            value = np.asarray(compute(), dtype=np.float64)
            expected = (len(self.errors["vsd"][0]),) if name == "vsd" else ()
            if value.shape != expected:
                raise ValueError(f"形状 {value.shape}, 预期 {expected}")
            if not np.isfinite(value).all():
                raise ValueError(f"NaN/Inf: {value}")
            if np.any(value < 0):
                raise ValueError(f"负值: {value}")
            self.errors[name][index] = value
        except Exception as exc:
            self.fail_reasons[name].append(f"target={index}: {type(exc).__name__}: {exc}")

    def summarize(self):
        thresholds = {"vsd": self.vsd_thetas, "mssd": self.mssd_thetas, "mspd": self.mspd_thetas}
        result = {"n_targets": self.n_targets}
        for name, errors in self.errors.items():
            reasons = list(self.fail_reasons[name])
            missing = int(np.count_nonzero(~self.done[name]))
            if missing:
                reasons.append(f"{missing} 个目标未完成处理")
            complete = (len(reasons) == 0)
            result[f"{name}_complete"] = complete
            result[f"{name}_fail_reason"] = reasons[0] if reasons else None
            if complete:
                if name == "vsd":
                    ok = errors[:, :, None] < thresholds[name][None, None, :]
                else:
                    ok = errors[:, None] < thresholds[name][None, :]
                result[f"ar_{name}"] = float(ok.mean())
            else:
                result[f"ar_{name}"] = float("nan")
        parts = [result["ar_vsd"], result["ar_mssd"], result["ar_mspd"]]
        result["ar"] = (float(np.mean(parts)) if np.all(np.isfinite(parts)) else float("nan"))
        geo = [result["ar_mssd"], result["ar_mspd"]]
        result["mean_mssd_mspd"] = (float(np.mean(geo)) if np.all(np.isfinite(geo)) else float("nan"))
        return result


class Real275BopReader:
    MASK_SUFFIXES = ["_mask.png", "_mask_visib.png", "_label.png"]
    NOCS_DEFAULT_K = np.array([[591.0125, 0.0, 322.525],
                               [0.0, 590.16775, 244.11084],
                               [0.0, 0.0, 1.0]], dtype=np.float64)

    def __init__(self, video_dir, gts_root, scene_id, obj_models_dir=None, cam_K=None,
                 orthogonalize_R=True, depth_scale=1000.0, min_visib=8,
                 ob_key_mode="class_id", model_id_to_name=None):
        self.video_dir = video_dir
        self.gts_root = gts_root
        self.scene_id = int(scene_id)
        self.obj_models_dir = obj_models_dir
        self._dataset_dir = str(video_dir)
        self.K = np.array(cam_K, dtype=np.float64) if cam_K is not None else self.NOCS_DEFAULT_K.copy()
        self.orthogonalize_R = bool(orthogonalize_R)
        self.depth_scale = float(depth_scale)
        self.min_visib = int(min_visib)
        if ob_key_mode not in ("class_id", "gt_idx", "model_name", "model_id"):
            raise ValueError(f"ob_key_mode: {ob_key_mode}")
        self.ob_key_mode = ob_key_mode
        self.model_id_to_name = dict(model_id_to_name) if model_id_to_name else {}
        self.model_name_to_id = {v: k for k, v in self.model_id_to_name.items()}
        self._rgb_map, self._depth_map, self._mask_map = {}, {}, {}
        self._scene_gt, self._mesh_cache = {}, {}
        self.ob_ids = []
        self._load()

    def _key_of(self, item):
        if self.ob_key_mode == "class_id":
            return item["class_id"]
        if self.ob_key_mode == "gt_idx":
            return item["gt_idx"]
        if self.ob_key_mode == "model_name":
            return item["model_name"]
        mn = item.get("model_name")
        return self.model_name_to_id.get(mn) if mn else None

    def _load(self):
        for p in glob.glob(os.path.join(self.video_dir, "*")):
            fn = os.path.basename(p)
            stem = fn.split("_")[0]
            if not stem.isdigit():
                continue
            f = int(stem)
            if fn.endswith("_color.png"):
                self._rgb_map[f] = p
            elif fn.endswith("_depth.png"):
                self._depth_map[f] = p
            else:
                for suf in self.MASK_SUFFIXES:
                    if fn.endswith(suf):
                        self._mask_map[f] = p
                        break
        self._frames = sorted(set(self._rgb_map) & set(self._depth_map))
        self.id_strs = {i: f"{i:04d}" for i in self._frames}

        pattern = os.path.join(self.gts_root, f"results_real_test_scene_{self.scene_id}_*.pkl")
        gt_files = sorted(glob.glob(pattern))
        if not gt_files:
            raise FileNotFoundError(f"GT not found: {pattern}")

        reported_meta = reported_pkl = False
        ambiguous = missing_model = missing_class = missing_mask = 0
        n_scale_warn = 0

        for fp in gt_files:
            m = re.search(r"_scene_(\d+)_(\d+)\.pkl$", os.path.basename(fp))
            if m is None:
                continue
            frame_id = int(m.group(2))
            with open(fp, "rb") as fh:
                data = pickle.load(fh)
            gt_RTs = np.asarray(data["gt_RTs"], dtype=np.float64)
            if gt_RTs.ndim != 3 or gt_RTs.shape[1:] != (4, 4) or not np.isfinite(gt_RTs).all():
                raise ValueError(f"{fp}: gt_RTs invalid")
            n_inst = len(gt_RTs)

            # 诊断：R 块奇异值（是否含非均匀/均匀尺度）
            for gi in range(n_inst):
                sv = np.linalg.svd(gt_RTs[gi, :3, :3], compute_uv=False)
                if not np.allclose(sv, 1.0, atol=1e-2):
                    if n_scale_warn < 5:
                        _pt(f"[WARN] {fp} inst={gi}: R 奇异值={sv.round(4)} 含非单位尺度")
                    n_scale_warn += 1

            inst_to_info = parse_meta_models(os.path.join(self.video_dir, f"{frame_id:04d}_meta.txt"))
            if inst_to_info and not reported_meta:
                _pt(f"[reader] {self.get_video_id()}: meta.txt")
                reported_meta = True

            pkl_class_ids = None
            if not inst_to_info:
                try:
                    pkl_class_ids, field = read_gt_class_ids(data, n_inst, source=fp)
                    if not reported_pkl:
                        _pt(f"[reader] {self.get_video_id()}: pkl '{field}'")
                        reported_pkl = True
                except (KeyError, ValueError):
                    pkl_class_ids = None

            pkl_mask_ids = data.get("gt_mask_ids", None)
            if pkl_mask_ids is not None:
                pkl_mask_ids = np.asarray(pkl_mask_ids).reshape(-1).astype(np.int64)
                if len(pkl_mask_ids) != n_inst:
                    raise ValueError(f"{fp}: gt_mask_ids 长度不符")

            items = []
            for gi in range(n_inst):
                pose = gt_RTs[gi].copy()
                if self.orthogonalize_R:
                    pose[:3, :3] = orthogonalize_rotation(pose[:3, :3])

                # 原则 #1：先确定 mask_id；优先用 pkl 显式映射
                if pkl_mask_ids is not None:
                    mask_id = int(pkl_mask_ids[gi])
                elif inst_to_info:
                    mask_id = gi + 1  # 未显式提供，按 1-based 惯例（有风险）
                else:
                    mask_id = None
                    missing_mask += 1

                # 原则 #1：用 mask_id 查 meta；禁止 gi+1 假设
                if inst_to_info:
                    if mask_id is None:
                        raise RuntimeError(f"{fp}: GT[{gi}] 无 mask_id 但有 meta")
                    info = inst_to_info.get(mask_id)
                    if info is None:
                        raise RuntimeError(
                            f"{fp}: GT[{gi}] mask_id={mask_id} 在 meta 中不存在 "
                            f"(meta inst_ids={sorted(inst_to_info.keys())})")
                    cid = info.get("class_id")
                    mn = info.get("model_name")
                    sym_id = info.get("symmetry_id", 0)
                else:
                    cid = int(pkl_class_ids[gi]) if pkl_class_ids is not None else None
                    mn = None
                    sym_id = 0

                if cid is None:
                    missing_class += 1
                if mn is None:
                    missing_model += 1

                items.append({"gt_idx": gi, "class_id": cid, "mask_id": mask_id,
                              "pose": pose, "model_name": mn, "symmetry_id": sym_id})

            if self.ob_key_mode == "class_id":
                n_missing = sum(1 for it in items if it["class_id"] is None)
                if n_missing == n_inst:
                    raise RuntimeError(f"{fp}: no class_id")
                valid = [it["class_id"] for it in items if it["class_id"] is not None]
                if len(valid) != len(set(valid)):
                    ambiguous += 1
            if self.ob_key_mode in ("model_name", "model_id"):
                n_missing = sum(1 for it in items if it["model_name"] is None)
                if n_missing == n_inst:
                    raise RuntimeError(f"{fp}: no model_name")
                valid = [it["model_name"] for it in items if it["model_name"] is not None]
                if len(valid) != len(set(valid)):
                    ambiguous += 1

            self._scene_gt[frame_id] = items

        keys = set()
        for items in self._scene_gt.values():
            for item in items:
                k = self._key_of(item)
                if k is not None:
                    keys.add(str(k) if self.ob_key_mode == "model_name" else int(k))
        self.ob_ids = sorted(keys)

        _pt(f"[reader] {self.get_video_id()}: mode={self.ob_key_mode}, frames={len(self._scene_gt)}, keys={self.ob_ids}")
        for tag, val in [("ambiguous", ambiguous), ("missing_model", missing_model),
                         ("missing_class", missing_class), ("missing_mask", missing_mask),
                         ("scale_warn", n_scale_warn)]:
            if val:
                _pt(f"[WARN] {self.get_video_id()}: {val} {tag}")

    def _match_items(self, frame, ob_key):
        if self.ob_key_mode == "model_id":
            mn = self.model_id_to_name.get(ob_key)
            if mn is None:
                return []
            return [it for it in self._scene_gt.get(frame, []) if it.get("model_name") == mn]
        return [it for it in self._scene_gt.get(frame, []) if self._key_of(it) == ob_key]

    def _get_single_item(self, frame, ob_key):
        items = self._match_items(frame, ob_key)
        return items[0] if len(items) == 1 else None

    def get_video_id(self):
        return os.path.basename(self.video_dir.rstrip('/'))

    def get_color(self, i):
        p = self._rgb_map.get(i)
        if p is None:
            return None
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None

    def get_depth(self, i):
        p = self._depth_map.get(i)
        if p is None:
            return None
        d = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if d is None:
            return None
        if d.ndim == 3:
            d = d[:, :, 1].astype(np.uint16) * 256 + d[:, :, 2].astype(np.uint16)
        elif d.ndim != 2:
            raise ValueError(f"depth shape {d.shape}")
        d = d.astype(np.float32)
        d[d == 32001] = 0.0
        d[~np.isfinite(d)] = 0.0
        return d / self.depth_scale

    def get_instance_ids_in_image(self, i):
        return [k for k in self.ob_ids if self._get_single_item(i, k) is not None]

    def list_obj_ids(self):
        return list(self.ob_ids)

    def get_gt_pose(self, i, ob_key):
        item = self._get_single_item(i, ob_key)
        return item["pose"].copy() if item is not None else None

    def get_mask(self, i, ob_key, type="mask_visib", **kwargs):
        item = self._get_single_item(i, ob_key)
        if item is None:
            return None
        if item.get("mask_id") is None:
            raise RuntimeError(f"frame={i} ob_key={ob_key}: mask_id 未映射")
        path = self._mask_map.get(i)
        if path is None:
            return None
        labels = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if labels is None:
            return None
        if labels.ndim == 3:
            labels = labels[:, :, -1]
        if labels.ndim != 2:
            raise ValueError(f"{path}: mask not 2D")
        mask = labels == item["mask_id"]
        return mask.astype(np.uint8) * 255 if np.any(mask) else None

    def _find_obj_models_dir(self):
        base = self.obj_models_dir if self.obj_models_dir else \
            os.path.join(os.path.dirname(os.path.dirname(self.gts_root)), "obj_models")
        for sub in ("real_test", "real_val", "train", "val", "real_train"):
            cand = os.path.join(base, sub)
            if os.path.isdir(cand):
                return cand
        raise FileNotFoundError(f"obj_models not found: {base}")

    def _load_mesh_obj(self, model_name):
        if model_name in self._mesh_cache:
            return self._mesh_cache[model_name]
        obj_dir = self._find_obj_models_dir()
        path = next((os.path.join(obj_dir, c) for c in
                     (f"{model_name}.obj", f"{model_name}_norm.obj", f"{model_name}.ply")
                     if os.path.isfile(os.path.join(obj_dir, c))), None)
        if path is None:
            raise FileNotFoundError(f"Mesh not found: {model_name}")
        mesh = trimesh.load(path, force="mesh")
        if mesh.vertices.shape[0] < 4:
            raise RuntimeError(f"Mesh too few verts: {path}")
        self._mesh_cache[model_name] = mesh
        _pt(f"[mesh] loaded {model_name} ({mesh.vertices.shape[0]} verts)")
        return mesh

    def get_gt_mesh(self, ob_key, frame=None):
        if self.ob_key_mode == "model_id":
            mn = self.model_id_to_name.get(ob_key)
            if mn is None:
                raise RuntimeError(f"model_id={ob_key} unmapped")
            return self._load_mesh_obj(mn)
        if self.ob_key_mode == "model_name":
            return self._load_mesh_obj(ob_key)
        model_names = set()
        frames = [frame] if frame is not None else list(self._scene_gt.keys())
        for f in frames:
            for it in self._match_items(f, ob_key):
                if it.get("model_name"):
                    model_names.add(it["model_name"])
        if not model_names:
            raise RuntimeError(f"ob_key={ob_key}: no model_name")
        if len(model_names) > 1:
            raise RuntimeError(f"ob_key={ob_key}: multiple model_name {model_names}")
        return self._load_mesh_obj(model_names.pop())

    def get_model_path(self, ob_key, frame=None):
        if self.ob_key_mode == "model_id":
            mn = self.model_id_to_name.get(ob_key)
        elif self.ob_key_mode == "model_name":
            mn = ob_key
        else:
            model_names = set()
            frames = [frame] if frame is not None else list(self._scene_gt.keys())
            for f in frames:
                for it in self._match_items(f, ob_key):
                    if it.get("model_name"):
                        model_names.add(it["model_name"])
            if len(model_names) != 1:
                return None
            mn = next(iter(model_names))
        if mn is None:
            return None
        obj_dir = self._find_obj_models_dir()
        for cand in (f"{mn}.obj", f"{mn}_norm.obj", f"{mn}.ply"):
            p = os.path.join(obj_dir, cand)
            if os.path.isfile(p):
                return p
        return None

    def _mask_bbox(self, i, ob_key):
        m = self.get_mask(i, ob_key)
        if m is None:
            return None
        vs, us = np.where(m > 0)
        if len(vs) == 0:
            return None
        return int(us.min()), int(vs.min()), int(us.max()), int(vs.max())

    def frames_with_bbox(self, ob_key):
        out = []
        for frame in sorted(self._scene_gt):
            if frame not in self._rgb_map or frame not in self._depth_map:
                continue
            if self._get_single_item(frame, ob_key) is None:
                continue
            bbox = self._mask_bbox(frame, ob_key)
            if bbox is None:
                continue
            u0, v0, u1, v1 = bbox
            w, h = u1 - u0 + 1, v1 - v0 + 1
            if w < self.min_visib or h < self.min_visib:
                continue
            out.append((frame, w * h, w, h))
        return out

    def frames_for_obj(self, ob_key):
        return [k for k, _, _, _ in self.frames_with_bbox(ob_key)]


class PoseMetrics:
    def __init__(self, adds_sample=2048):
        self.adds_sample = int(adds_sample)
        self._rng = np.random.default_rng(0)

    def _subsample(self, pts):
        n = len(pts)
        return pts if n <= self.adds_sample else pts[self._rng.choice(n, self.adds_sample, replace=False)]

    def compute_add_norm(self, pred, gt, pts, diameter_m):
        if pts is None or len(pts) == 0 or not (diameter_m > 0):
            return float("nan")
        pts = self._subsample(np.asarray(pts, dtype=np.float64))
        pp = pts @ pred[:3, :3].T + pred[:3, 3]
        gp = pts @ gt[:3, :3].T + gt[:3, 3]
        return float(np.linalg.norm(pp - gp, axis=1).mean() / diameter_m)

    def compute_adds_norm(self, pred, gt, pts, diameter_m):
        if pts is None or len(pts) == 0 or not (diameter_m > 0):
            return float("nan")
        pts = self._subsample(np.asarray(pts, dtype=np.float64))
        pp = pts @ pred[:3, :3].T + pred[:3, 3]
        gp = pts @ gt[:3, :3].T + gt[:3, 3]
        d, _ = cKDTree(pp).query(gp, k=1, workers=-1)
        return float(d.mean() / diameter_m)

    def compute_pose_decomposition_error(self, pred, gt):
        rot = pred[:3, :3] @ gt[:3, :3].T
        rot_err = np.linalg.norm(Rot.from_matrix(rot).as_rotvec()) * 180 / np.pi
        trans_err = np.linalg.norm(pred[:3, 3] - gt[:3, 3])
        return float(rot_err), float(trans_err)

    def compute_all_errors(self, pred, gt, model_pts, ob_key, diameter_m):
        rot_err, trans_err = self.compute_pose_decomposition_error(pred, gt)
        return {"trans_mm": trans_err * 1000.0, "rot_deg": rot_err,
                "add_norm": self.compute_add_norm(pred, gt, model_pts, diameter_m),
                "adds_norm": self.compute_adds_norm(pred, gt, model_pts, diameter_m)}


def compute_acc(errs, threshold=10.0):
    errs = np.asarray(errs, dtype=np.float64)
    return float(np.mean(errs < threshold) * 100.0) if len(errs) else 0.0


def compute_auc_norm(errs, max_val=0.1):
    errs = np.asarray(errs, dtype=np.float64).reshape(-1)
    if len(errs) == 0:
        return 0.0
    if max_val <= 0:
        raise ValueError("max_val <= 0")
    valid = np.isfinite(errs) & (errs >= 0)
    contrib = np.zeros(len(errs), dtype=np.float64)
    contrib[valid] = np.clip(1.0 - errs[valid] / max_val, 0.0, 1.0)
    return float(contrib.mean())


def _try_import_bop_lib():
    try:
        from bop_toolkit_lib import pose_error
        try:
            from bop_toolkit_lib import renderer_vispy
            return pose_error, renderer_vispy
        except Exception:
            return pose_error, None
    except Exception:
        return None, None


def _export_obj_as_ply_mm(obj_path, obj_id):
    if obj_path is None or not os.path.isfile(obj_path):
        return None
    tmp = os.path.join(tempfile.gettempdir(), "real275_ar_ply_mm_cache")
    os.makedirs(tmp, exist_ok=True)
    base = os.path.splitext(os.path.basename(obj_path))[0]
    src_stat = os.stat(obj_path)
    # 缓存键包含 mtime + size，避免源文件更新后复用旧 mesh
    out = os.path.join(tmp, f"{base}_obj{obj_id}_m{int(src_stat.st_mtime)}_s{src_stat.st_size}_mm.ply")
    if os.path.isfile(out):
        return out
    try:
        mesh_mm = trimesh.load(obj_path, force="mesh").copy()
        mesh_mm.apply_scale(1000.0)
        mesh_mm.export(out)
        return out
    except Exception as e:
        _pt(f"[AR] obj->ply_mm failed: {obj_path} ({e})")
        return None


def compute_ar_for_model_v2(model_path, model_pts_m, model_id, class_id, symmetry_id,
                            ar_samples, real275_dir, reader_cache, obj_id_for_renderer,
                            width=640, height=480, use_symmetry=True, max_frames=200,
                            verbose=True):
    """ar_samples 覆盖完整计划集（pred 可为 None）。返回 dict，*_reason=None 表示该项可用。"""
    diag_n = len(ar_samples) if ar_samples else 0

    pose_error, renderer_vispy = _try_import_bop_lib()
    if pose_error is None:
        return {"ok": False, "reason": "bop_toolkit_lib.pose_error 不可用", "n_targets": diag_n}
    if not ar_samples:
        return {"ok": False, "reason": "ar_samples 为空", "n_targets": 0}
    if model_path is None or not os.path.isfile(model_path):
        return {"ok": False, "reason": f"模型路径无效: {model_path!r}", "n_targets": diag_n}

    if max_frames > 0 and len(ar_samples) > max_frames:
        idx = np.linspace(0, len(ar_samples) - 1, max_frames).astype(int)
        samples = [ar_samples[i] for i in idx]
    else:
        samples = list(ar_samples)
    n = len(samples)

    model_pts_mm = (np.asarray(model_pts_m, dtype=np.float64) * 1000.0).astype(np.float32)
    try:
        diameter_mm = compute_model_diameter_mm(model_pts_m, model_id=model_id)
    except Exception as e:
        return {"ok": False, "reason": f"真实直径计算失败: {e}", "n_targets": n}

    syms = _default_symmetry_transforms(class_id, symmetry_id=symmetry_id, max_step_rad=0.05) \
        if use_symmetry else [{"R": np.eye(3, dtype=np.float32), "t": np.zeros((3, 1), dtype=np.float32)}]
    for s in syms:
        s["t"] = np.asarray(s["t"], dtype=np.float32) * 1000.0

    rnd = None
    vispy_fail_reason = None
    if renderer_vispy is not None:
        ply_mm = _export_obj_as_ply_mm(model_path, obj_id_for_renderer)
        if ply_mm is None:
            vispy_fail_reason = "obj->ply_mm 转换失败"
        else:
            try:
                rnd = renderer_vispy.RendererVispy(width, height, mode="depth")
                rnd.add_object(int(obj_id_for_renderer), ply_mm)
                _pt(f"[AR] RendererVispy OK ({width}x{height})")
            except Exception as e:
                vispy_fail_reason = f"RendererVispy init failed: {e}"
                rnd = None
    else:
        vispy_fail_reason = "bop_toolkit_lib.renderer_vispy 不可用"

    vsd_taus = np.arange(0.05, 0.51, 0.05)
    vsd_thetas = np.arange(0.05, 0.51, 0.05)
    mssd_thetas = np.arange(0.05, 0.51, 0.05)
    mspd_thetas = np.arange(5.0, 51.0, 5.0) * (width / 640.0)
    delta_mm = 15.0

    acc = ARAccumulator(n_targets=n, vsd_taus=vsd_taus, vsd_thetas=vsd_thetas,
                        mssd_thetas=mssd_thetas, mspd_thetas=mspd_thetas)
    if rnd is None:
        acc.mark_metric_unavailable("vsd", vispy_fail_reason or "renderer=None")

    n_pred_fail = 0

    def _compute_vsd(sample, pred_mm, gt_mm):
        if rnd is None:
            raise RuntimeError("renderer unavailable")
        reader = _get_reader(real275_dir, sample["v_idx"], reader_cache)
        K = np.asarray(reader.K, dtype=np.float64)
        depth_m = reader.get_depth(sample["frame_id"])
        if depth_m is None:
            raise RuntimeError(f"深度缺失 v{sample['v_idx']} f{sample['frame_id']}")
        depth_mm = np.asarray(depth_m, dtype=np.float64) * 1000.0
        if depth_mm.shape != (height, width):
            raise ValueError(f"深度尺寸 {depth_mm.shape} != {(height, width)}")
        e = pose_error.vsd(pred_mm[:3, :3], pred_mm[:3, 3].reshape(3, 1),
                           gt_mm[:3, :3], gt_mm[:3, 3].reshape(3, 1),
                           depth_test=depth_mm, K=K, delta=delta_mm,
                           taus=vsd_taus.tolist(), normalized_by_diameter=True,
                           diameter=diameter_mm, renderer=rnd,
                           obj_id=int(obj_id_for_renderer), cost_type="step")
        return np.asarray(e, dtype=np.float64).reshape(-1)

    def _compute_mssd(pred_mm, gt_mm):
        e = pose_error.mssd(pred_mm[:3, :3], pred_mm[:3, 3].reshape(3, 1),
                            gt_mm[:3, :3], gt_mm[:3, 3].reshape(3, 1), model_pts_mm, syms)
        return float(e) / diameter_mm

    def _compute_mspd(sample, pred_mm, gt_mm):
        K = np.asarray(_get_reader(real275_dir, sample["v_idx"], reader_cache).K, dtype=np.float64)
        return float(pose_error.mspd(pred_mm[:3, :3], pred_mm[:3, 3].reshape(3, 1),
                                     gt_mm[:3, :3], gt_mm[:3, 3].reshape(3, 1),
                                     K, model_pts_mm, syms))

    for i, s in enumerate(samples):
        gt, pred = s.get("gt"), s.get("pred")
        if gt is None:
            raise RuntimeError(f"目标缺少 GT: v_idx={s.get('v_idx')}, frame_id={s.get('frame_id')}")
        if pred is None:
            n_pred_fail += 1
            acc.mark_prediction_missing(i)
            continue
        gt_mm = pose_m_to_mm(gt)
        pred_mm = pose_m_to_mm(pred)
        acc.record("mssd", i, lambda p=pred_mm, g=gt_mm: _compute_mssd(p, g))
        acc.record("mspd", i, lambda sm=s, p=pred_mm, g=gt_mm: _compute_mspd(sm, p, g))
        acc.record("vsd", i, lambda sm=s, p=pred_mm, g=gt_mm: _compute_vsd(sm, p, g))

    result = acc.summarize()
    result.update({"ok": True, "reason": None, "n_pred_fail": n_pred_fail,
                   "diameter_mm": diameter_mm, "vispy_fail_reason": vispy_fail_reason})

    _pt(f"[AR] n={n} pred_fail={n_pred_fail} diameter={diameter_mm:.2f} mm")
    for name in ("vsd", "mssd", "mspd"):
        _pt(f"[AR]   {name.upper():5s} complete={result[f'{name}_complete']} "
            f"ar={result[f'ar_{name}']:.4f} reason={result[f'{name}_fail_reason']}")
    _pt(f"[AR]   AR = {result['ar']:.4f} (2项均值={result['mean_mssd_mspd']:.4f})")
    return result


def get_object_mask(reader, i_frame, ob_key, detect_type):
    if detect_type == "box":
        mask = reader.get_mask(i_frame, ob_key)
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
        m = reader.get_mask(i_frame, ob_key)
        return (m > 0) if m is not None else None
    raise RuntimeError(f"Unknown detect_type: {detect_type}")


def _discover_video_dirs(real275_dir):
    return sorted(glob.glob(os.path.join(real275_dir, "real_test", "scene_*")))


def _collect_model_names_in_reader(reader):
    models = set()
    for items in reader._scene_gt.values():
        for it in items:
            if it.get("model_name"):
                models.add(it["model_name"])
    return sorted(models)


def _get_reader(real275_dir, v_idx, cache=None):
    if cache is not None and v_idx in cache:
        return cache[v_idx]
    video_dirs = _discover_video_dirs(real275_dir)
    sid = int(os.path.basename(video_dirs[v_idx]).split("_")[1])
    r = Real275BopReader(
        video_dirs[v_idx], os.path.join(real275_dir, "gts", "real_test"), sid,
        obj_models_dir=opt.obj_models_dir, cam_K=opt.cam_K,
        min_visib=opt.min_visib, ob_key_mode=_INTERNAL_OB_KEY_MODE,
        model_id_to_name=_GLOBAL_MODEL_ID_TO_NAME)
    if cache is not None:
        cache[v_idx] = r
    return r


def _report_inventory(real275_dir, log_path=None, cache=None):
    video_dirs = _discover_video_dirs(real275_dir)
    scene_to_models, model_to_scenes = {}, {}
    lines = []

    def emit(text):
        lines.append(text)
        print(text, flush=True)

    emit("\n========== [Inventory] ==========")
    for v_idx in range(len(video_dirs)):
        reader = _get_reader(real275_dir, v_idx, cache)
        vid = reader.get_video_id()
        models = _collect_model_names_in_reader(reader)
        scene_to_models[v_idx] = (vid, models)
        for mn in models:
            model_to_scenes.setdefault(mn, []).append(v_idx)
        emit(f"\nv{v_idx:02d} ({vid}) models={models}")
        for mn in models:
            rf = ri = amb = 0
            for frame in reader._scene_gt:
                nm = sum(1 for it in reader._scene_gt[frame] if it.get("model_name") == mn)
                if nm:
                    rf += 1
                    ri += nm
                    amb += int(nm > 1)
            try:
                usable = len(reader.frames_with_bbox(mn))
            except Exception:
                usable = "N/A"
            emit(f"  {mn}: GT帧={rf}, 实例={ri}, 歧义={amb}, 可用帧={usable}")

    emit("\n========== scene_id -> models ==========")
    for v in sorted(scene_to_models):
        emit(f"  scene_id={v} ({scene_to_models[v][0]}) {len(scene_to_models[v][1])} models")

    emit("\n========== model -> scene_ids ==========")
    for mn in sorted(model_to_scenes):
        vids = sorted(model_to_scenes[mn])
        emit(f"  {mn}: {vids} ({[scene_to_models[v][0] for v in vids]})")

    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def _build_model_id_registry(reader_cache, video_dirs):
    raw = {}
    for v_idx in range(len(video_dirs)):
        r = reader_cache.get(v_idx)
        if r is None:
            continue
        for items in r._scene_gt.values():
            for it in items:
                mn = it.get("model_name")
                if not mn:
                    continue
                info = raw.setdefault(mn, {"class_id": it.get("class_id"), "scene_ids": set(), "usable_frames": 0})
                info["scene_ids"].add(v_idx)
                if it.get("class_id") is not None:
                    info["class_id"] = it["class_id"]
        if r.ob_key_mode == "model_name":
            for mn in r.ob_ids:
                if mn in raw:
                    try:
                        raw[mn]["usable_frames"] += len(r.frames_with_bbox(mn))
                    except Exception:
                        pass

    sorted_names = sorted(raw.keys(), key=lambda mn: (raw[mn]["class_id"] if raw[mn]["class_id"] is not None else 999, mn))
    model_id_to_name, model_name_to_id, model_info = {}, {}, {}
    for new_id, mn in enumerate(sorted_names, start=1):
        model_id_to_name[new_id] = mn
        model_name_to_id[mn] = new_id
        scene_ids = sorted(raw[mn]["scene_ids"])
        model_info[new_id] = {
            "model_id": new_id, "model_name": mn, "class_id": raw[mn]["class_id"],
            "scene_ids": scene_ids,
            "scene_names": [reader_cache[v].get_video_id() for v in scene_ids],
            "usable_frames": raw[mn]["usable_frames"]}
    return model_id_to_name, model_name_to_id, model_info


def _print_model_id_table(model_info):
    print("\n========== [Inventory] model_id 表 ==========")
    print(f"{'model_id':>8} | {'class_id':>8} | {'model_name':<34} | {'scene_ids':<16} | scene_names | 可用帧数")
    print("-" * 130)
    for mid in sorted(model_info):
        i = model_info[mid]
        print(f"{mid:>8} | {str(i['class_id']):>8} | {i['model_name']:<34} | "
              f"{str(i['scene_ids']):<16} | {str(i['scene_names']):<28} | {i['usable_frames']}")
    print("[TIP] --model_id N / --model_ids 1,3,5 / --model_ids all")


def _collect_inventory_dict(real275_dir, cache=None):
    global _INTERNAL_OB_KEY_MODE, _GLOBAL_MODEL_ID_TO_NAME
    prev_mode, prev_map = _INTERNAL_OB_KEY_MODE, _GLOBAL_MODEL_ID_TO_NAME
    _INTERNAL_OB_KEY_MODE, _GLOBAL_MODEL_ID_TO_NAME = "model_name", None
    video_dirs = _discover_video_dirs(real275_dir)
    scenes = []
    try:
        for v_idx in range(len(video_dirs)):
            r = _get_reader(real275_dir, v_idx, cache)
            vid = r.get_video_id()
            models_in_scene = _collect_model_names_in_reader(r)
            models_list = []
            for mn in models_in_scene:
                cid = None
                for items in r._scene_gt.values():
                    for it in items:
                        if it.get("model_name") == mn:
                            cid = it.get("class_id")
                            break
                    if cid is not None:
                        break
                rf = ri = amb = 0
                for frame in r._scene_gt:
                    nm = sum(1 for it in r._scene_gt[frame] if it.get("model_name") == mn)
                    if nm:
                        rf += 1
                        ri += nm
                        amb += int(nm > 1)
                try:
                    usable = len(r.frames_with_bbox(mn))
                except Exception:
                    usable = 0
                models_list.append({"model_name": mn, "class_id": cid, "raw_frames": int(rf),
                                    "raw_instances": int(ri), "ambiguous": int(amb),
                                    "usable_frames": int(usable)})
            scenes.append({"v_idx": int(v_idx), "video_id": vid, "models": models_list})
    finally:
        _INTERNAL_OB_KEY_MODE, _GLOBAL_MODEL_ID_TO_NAME = prev_mode, prev_map
    return {"scenes": scenes}


def _export_registry_json(real275_dir, out_path, cache=None):
    print(f"[export] {real275_dir} ...", flush=True)
    video_dirs = _discover_video_dirs(real275_dir)
    if not video_dirs:
        raise SystemExit("[export] No videos")
    if cache is None:
        cache = {}
    inventory = _collect_inventory_dict(real275_dir, cache=cache)
    _, _, model_info = _build_model_id_registry(cache, video_dirs)
    data = {
        "version": 1, "real275_dir": os.path.abspath(real275_dir),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "min_visib": int(opt.min_visib), "obj_key_mode": "model_name",
        "scenes": inventory["scenes"],
        "models_registry": [model_info[mid] for mid in sorted(model_info)]}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[export] saved -> {out_path}", flush=True)
    return data


def _load_registry_json(path, expected_min_visib=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[registry] read fail: {e}", flush=True)
        return None
    if data.get("version") != 1:
        return None
    if expected_min_visib is not None and data.get("min_visib") != expected_min_visib:
        print("[registry] min_visib mismatch", flush=True)
        return None
    model_id_to_name, model_name_to_id, model_info = {}, {}, {}
    for m in data["models_registry"]:
        mid = int(m["model_id"])
        model_id_to_name[mid] = m["model_name"]
        model_name_to_id[m["model_name"]] = mid
        model_info[mid] = m
    return {"scenes": data["scenes"]}, model_id_to_name, model_name_to_id, model_info


def _print_inventory_from_json(inventory, log_path=None):
    lines = []

    def emit(text):
        lines.append(text)
        print(text, flush=True)

    emit("\n========== [Inventory] (from JSON) ==========")
    for scene in inventory["scenes"]:
        v_idx, vid = scene["v_idx"], scene["video_id"]
        emit(f"\nv{v_idx:02d} ({vid}) models={[m['model_name'] for m in scene['models']]}")
        for m in scene["models"]:
            emit(f"  {m['model_name']}: raw={m.get('raw_frames', 0)}, usable={m.get('usable_frames', 0)}")

    emit("\n========== scene_id -> models ==========")
    for scene in inventory["scenes"]:
        emit(f"  scene_id={scene['v_idx']} ({scene['video_id']}) {len(scene['models'])} models")

    emit("\n========== model -> scene_ids ==========")
    model_to_scenes = {}
    for scene in inventory["scenes"]:
        for m in scene["models"]:
            model_to_scenes.setdefault(m["model_name"], []).append(scene["v_idx"])
    scene_id_to_name = {s["v_idx"]: s["video_id"] for s in inventory["scenes"]}
    for mn in sorted(model_to_scenes):
        vids = sorted(model_to_scenes[mn])
        emit(f"  {mn}: {vids} ({[scene_id_to_name[v] for v in vids]})")
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def _find_ref_candidates(real275_dir, ob_key, ref_scene, ref_frame=-1, ref_stride=1,
                         cache=None, verbose=True, max_candidates=20):
    r = _get_reader(real275_dir, ref_scene, cache)
    if ob_key not in r.ob_ids:
        if verbose:
            print(f"  [scan] v{ref_scene:02d}: ob_key absent", flush=True)
        return []
    infos = r.frames_with_bbox(ob_key)
    if not infos:
        return []
    if ref_frame >= 0:
        infos = [t for t in infos if t[0] == ref_frame]
        if not infos:
            return []
    elif ref_stride > 1:
        infos = sorted(infos, key=lambda x: x[0])[::ref_stride]
    if verbose:
        print(f"  [scan] v{ref_scene:02d} ({r.get_video_id()}): {len(infos)} candidates", flush=True)
    pools = [(0 if f == ref_frame else (1 if f == infos[0][0] else 2), -area, ref_scene, f, w, h, area)
             for f, area, w, h in infos]
    pools.sort(key=lambda t: (t[0], t[1]))
    return [(p[2], p[3], p[4], p[5], p[6]) for p in pools[:max_candidates]]


def run_pose_inference(reader, i_frame, est, ob_key, detect_type="mask",
                       pred_translation_scale=1.0):
    """只做推理。返回 (pred_np, gt_pose) 或 (None, None)。
    原则 #2：任何失败都返回 None，交由调用方决定计零召回。
    """
    color = reader.get_color(i_frame)
    depth = reader.get_depth(i_frame)
    if color is None or depth is None:
        return None, None
    ob_mask = get_object_mask(reader, i_frame, ob_key, detect_type)
    if ob_mask is None or ob_mask.sum() < 4:
        return None, None
    gt_pose = reader.get_gt_pose(i_frame, ob_key)
    if gt_pose is None:
        return None, None

    pred_pose = est.register(K=reader.K, rgb=color, depth=depth,
                             ob_mask=ob_mask, ob_id=ob_key, iteration=5)
    if pred_pose is None:
        raise RuntimeError("register returned None")
    pred_np = (pred_pose.detach().cpu().numpy() if torch.is_tensor(pred_pose)
               else np.asarray(pred_pose))
    pred_np = np.asarray(pred_np, dtype=np.float64).copy()
    if pred_np.shape != (4, 4):
        raise ValueError(f"pred shape {pred_np.shape}")
    if not np.isfinite(pred_np).all():
        raise ValueError("pred NaN/Inf")
    pred_np[:3, 3] *= pred_translation_scale
    return pred_np, gt_pose


def _count_eval_frames(eval_v_idx_list, ob_key, ref_v_idx, ref_k, skip_ref,
                       eval_stride, max_eval_global, max_eval_per_scene,
                       reader_cache, real275_dir, verbose=True):
    total = 0
    target_seq_count = 0
    per_scene = {}
    for v_idx in eval_v_idx_list:
        if max_eval_global > 0 and total >= max_eval_global:
            break
        r = _get_reader(real275_dir, v_idx, reader_cache)
        scene_cnt = 0
        for k in r.frames_for_obj(ob_key):
            if max_eval_global > 0 and total >= max_eval_global:
                break
            if max_eval_per_scene > 0 and scene_cnt >= max_eval_per_scene:
                break
            if skip_ref and v_idx == ref_v_idx and k == ref_k:
                continue
            if target_seq_count % eval_stride != 0:
                target_seq_count += 1
                continue
            target_seq_count += 1
            total += 1
            scene_cnt += 1
        per_scene[int(v_idx)] = int(scene_cnt)
    return int(total), per_scene


def _aggregate_objects_ar(per_obj, requested_keys=None, statuses=None):
    """原则 #4：全目标集 vs 完整子集 AR 分离报告。
    requested_keys: 用户请求的对象列表
    statuses: {ob_key: "evaluated"|"no_ref"|"ref_failed"|"no_frames"|"no_reachable_scene"}
    """
    if requested_keys is None:
        requested_keys = [p["ob_key"] for p in per_obj]
    if statuses is None:
        statuses = {k: "evaluated" for k in requested_keys}

    n_requested = len(requested_keys)
    n_reachable = len(per_obj)
    n_ref_failed = sum(1 for k in requested_keys
                       if statuses.get(k) in ("ref_failed", "no_frames"))
    n_no_ref = sum(1 for k in requested_keys
                   if statuses.get(k) in ("no_ref", "no_reachable_scene"))

    keys = ("ar_vsd", "ar_mssd", "ar_mspd")
    complete = [row for row in per_obj if all(np.isfinite(row[k]) for k in keys)]
    n_complete = len(complete)

    out = {
        "n_objects_requested": n_requested,
        "n_objects_reachable": n_reachable,
        "n_objects_complete": n_complete,
        "n_objects_ref_failed": n_ref_failed,
        "n_objects_no_ref": n_no_ref,
        "object_coverage": (n_complete / n_requested) if n_requested else 0.0,
        "full_set_ar": float("nan"),
        "complete_subset_ar": float("nan"),
        "unreachable_reasons": [],
    }
    for k in requested_keys:
        st = statuses.get(k, "unknown")
        if st != "evaluated":
            out["unreachable_reasons"].append(f"{k}: {st}")

    if complete:
        for key in keys:
            out[f"complete_subset_{key}"] = float(np.mean([row[key] for row in complete]))
        out["complete_subset_ar"] = float(
            np.mean([out[f"complete_subset_{k}"] for k in keys]))
        # 全目标集 AR：只有全部请求对象都进入评估且分项完整才可报告
        if n_complete == n_requested and n_requested > 0:
            out["full_set_ar"] = out["complete_subset_ar"]
    return out


def run_pipeline(detect_type="mask"):
    global _INTERNAL_OB_KEY_MODE, _GLOBAL_MODEL_ID_TO_NAME

    skip_ref = bool(opt.eval_skip_ref)
    eval_stride = max(1, opt.eval_stride)
    ref_stride = max(1, opt.ref_stride)
    max_eval_global = int(opt.max_eval_frames)
    max_eval_per_scene = int(opt.max_eval_frames_per_scene)

    _pt("[pipeline] CUDA preflight ...")
    if torch.cuda.is_available():
        _pt(f"[pipeline] CUDA: {torch.cuda.get_device_name(0)}")
        _ = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        _pt("[pipeline] CUDA OK")
    else:
        _pt("[pipeline] WARNING: cuda unavailable")

    if opt.compute_ar:
        _pt("[pipeline] AR deps check ...")
        pe_ok = rv_ok = False
        try:
            from bop_toolkit_lib import pose_error  # noqa
            pe_ok = True
            _pt("  [AR-deps] pose_error: OK")
        except Exception as e:
            _pt(f"  [AR-deps] pose_error: FAILED ({e})")
        try:
            from bop_toolkit_lib import renderer_vispy  # noqa
            rv_ok = True
            _pt("  [AR-deps] renderer_vispy: OK")
        except Exception as e:
            _pt(f"  [AR-deps] renderer_vispy: FAILED ({e}) → VSD 不可用")
        if opt.compute_ar and opt.ar_use_symmetry:
            _pt("  [AR-WARN] 对称性为自定义假设（class 1/2/4 → Y 轴连续旋转）；"
                "symmetry_id 未使用。请核实模型定义。")

    metrics = PoseMetrics()
    video_dirs = _discover_video_dirs(opt.real275_dir)
    if not video_dirs:
        raise SystemExit(f"[pipeline] No videos under {opt.real275_dir}/real_test/")

    registry_json_path = opt.registry_json or os.path.join(
        os.path.abspath(opt.real275_dir), "obj_models", "model_registry.json")
    inventory_log = os.path.join(opt.debug_dir, "inventory.log") if opt.debug_dir else None

    reader_cache_tmp = {}
    inventory = None
    cached = None
    if not opt.rebuild_registry and os.path.isfile(registry_json_path):
        cached = _load_registry_json(registry_json_path, expected_min_visib=opt.min_visib)

    if cached is None:
        print(f"[pipeline] registry miss, scanning: {registry_json_path}", flush=True)
        _INTERNAL_OB_KEY_MODE = "model_name"
        _GLOBAL_MODEL_ID_TO_NAME = None
        _report_inventory(opt.real275_dir, log_path=inventory_log, cache=reader_cache_tmp)
        model_id_to_name, model_name_to_id, model_info = _build_model_id_registry(reader_cache_tmp, video_dirs)
    else:
        inventory, model_id_to_name, model_name_to_id, model_info = cached
        print(f"[pipeline] registry loaded: {registry_json_path}", flush=True)
        _print_inventory_from_json(inventory, log_path=inventory_log)

    _print_model_id_table(model_info)

    if opt.ob_key_mode == "model_id":
        _INTERNAL_OB_KEY_MODE, _GLOBAL_MODEL_ID_TO_NAME, reader_cache = "model_id", model_id_to_name, {}
    elif opt.ob_key_mode == "model_name":
        _INTERNAL_OB_KEY_MODE, _GLOBAL_MODEL_ID_TO_NAME, reader_cache = "model_name", None, reader_cache_tmp
    else:
        _INTERNAL_OB_KEY_MODE, _GLOBAL_MODEL_ID_TO_NAME, reader_cache = opt.ob_key_mode, None, {}

    print(f"[pipeline] {len(video_dirs)} videos, mode={opt.ob_key_mode}, internal={_INTERNAL_OB_KEY_MODE}, "
          f"crop={opt.crop_ratio}, stride={eval_stride}, min_visib={opt.min_visib}, "
          f"pred_scale={opt.pred_translation_scale}, max_eval_global={max_eval_global}, "
          f"max_eval_per_scene={max_eval_per_scene}, compute_ar={opt.compute_ar}, "
          f"ar_max_frames={opt.ar_max_frames}", flush=True)
    if opt.compute_ar and opt.ar_max_frames > 0:
        _pt(f"  [AR-WARN] AR 每对象最多抽 {opt.ar_max_frames} 帧；"
            "正式全计划集用 --ar_max_frames 0")

    def _model_ids_from_opt():
        if opt.model_ids:
            return sorted(model_info.keys()) if opt.model_ids.strip().lower() == "all" else \
                [int(x.strip()) for x in opt.model_ids.split(",") if x.strip()]
        if opt.model_id is not None and opt.model_id >= 0:
            return [int(opt.model_id)]
        if opt.ob_ids:
            return [int(x) for x in opt.ob_ids]
        if opt.ob_id is not None and opt.ob_id >= 0:
            return [int(opt.ob_id)]
        return []

    if opt.ob_key_mode == "model_id":
        if opt.ob_names:
            ob_keys = [s.strip() for s in opt.ob_names.split(",") if s.strip()]
        else:
            ids = _model_ids_from_opt() or sorted(model_info.keys())
            if not ids:
                raise SystemExit("[pipeline] no valid model_id")
            ob_keys = []
            for i in ids:
                if int(i) in model_id_to_name:
                    ob_keys.append(int(i))
                else:
                    print(f"[pipeline] WARNING: model_id={i} absent", flush=True)
        if not ob_keys:
            raise SystemExit("[pipeline] no valid model_id")
    elif opt.ob_key_mode == "model_name":
        if opt.ob_names:
            ob_keys = [s.strip() for s in opt.ob_names.split(",") if s.strip()]
        elif opt.model_ids or (opt.model_id is not None and opt.model_id >= 0):
            ids = _model_ids_from_opt()
            ob_keys = [model_id_to_name[int(i)] for i in ids if int(i) in model_id_to_name]
        else:
            ob_keys = sorted(model_name_to_id.keys())
    else:
        if opt.ob_ids:
            ob_keys = [int(v) for v in opt.ob_ids]
        elif opt.ob_id is not None and opt.ob_id >= 0:
            ob_keys = [int(opt.ob_id)]
        else:
            raise SystemExit("[pipeline] class_id/gt_idx 模式需 --ob_id 或 --ob_ids")

    if not ob_keys:
        raise SystemExit("[pipeline] no ob_keys")
    multi_ob = len(ob_keys) > 1
    print(f"[pipeline] {len(ob_keys)} key(s): {ob_keys}", flush=True)

    all_errors = {}
    all_res = NestDict()
    requested_ob_keys = list(ob_keys)  # 原则 #4：原始请求集
    object_status = {}  # {ob_key: 状态}，原则 #4：显式追踪

    for i_ob, ob_key in enumerate(ob_keys):
        object_status[ob_key] = "pending"
        if opt.ob_key_mode == "model_id":
            model_id, model_name = int(ob_key), model_id_to_name[int(ob_key)]
            key_label = f"model_id={model_id} | {model_name}"
        elif opt.ob_key_mode == "model_name":
            model_name, model_id = ob_key, model_name_to_id.get(ob_key)
            key_label = f"model_id={model_id} | {model_name}" if model_id else model_name
        else:
            model_name = model_id = None
            key_label = str(ob_key)

        print(f"\n========== [key {i_ob + 1}/{len(ob_keys)}] {key_label} ==========", flush=True)

        candidate_scene_ids = None
        if inventory is not None and model_id is not None and model_id in model_info:
            candidate_scene_ids = list(model_info[model_id].get("scene_ids", []))

        if candidate_scene_ids is not None:
            eval_v_idx_list = [v for v in candidate_scene_ids if 0 <= v < len(video_dirs)
                               and len(_get_reader(opt.real275_dir, v, reader_cache).frames_for_obj(ob_key)) > 0]
        else:
            eval_v_idx_list = [v for v in range(len(video_dirs))
                               if len(_get_reader(opt.real275_dir, v, reader_cache).frames_for_obj(ob_key)) > 0]

        if not eval_v_idx_list:
            print(f"[pipeline] WARNING: key={ob_key} 不在任何 scene，skip", flush=True)
            object_status[ob_key] = "no_reachable_scene"
            continue

        eval_scene_names = [reader_cache[v].get_video_id() for v in eval_v_idx_list]
        print(f"[pipeline] key={ob_key} 评估 scene: "
              f"[{', '.join(f'{s}({n})' for s, n in zip(eval_v_idx_list, eval_scene_names))}]", flush=True)

        if opt.ref_video_idx >= 0 and opt.ref_video_idx in eval_v_idx_list:
            ref_scene = opt.ref_video_idx
        else:
            ref_scene = min(eval_v_idx_list)
        print(f"[pipeline] ref_scene={ref_scene} ({reader_cache[ref_scene].get_video_id()})", flush=True)

        candidates = _find_ref_candidates(opt.real275_dir, ob_key, ref_scene,
                                          ref_frame=opt.ref_frame, ref_stride=ref_stride,
                                          cache=reader_cache, verbose=True)
        if not candidates:
            print(f"[pipeline] WARNING: no usable ref for {ob_key}, skip", flush=True)
            object_status[ob_key] = "no_ref"
            continue

        est = None
        ref_v_idx = ref_k = ref_w = ref_h = None
        for ci, (v_idx, frame, w, h, area) in enumerate(candidates):
            _pt(f"  [ref try {ci+1}/{len(candidates)}] v{v_idx:02d} f{frame:04d} (bbox {w}x{h})")
            reader_ref = _get_reader(opt.real275_dir, v_idx, reader_cache)
            probe_dir = None
            if opt.save_ref_probe:
                probe_dir = (os.path.join(opt.save_ref_probe, f"key_{_sanitize_key(ob_key)}")
                             if multi_ob else opt.save_ref_probe)
                os.makedirs(probe_dir, exist_ok=True)
            try:
                mesh_diameter_m = float(np.linalg.norm(reader_ref.get_gt_mesh(ob_key, frame=frame).extents))
            except Exception as e:
                _pt(f"  [ref] WARN: mesh load failed ({type(e).__name__}: {e})")
                mesh_diameter_m = None
            save_to_disk = bool(opt.save_ref) and bool(opt.ref_data_root)
            try:
                cand_est = OneRefPose(scorer=None, refiner=None, ref_data_root=opt.ref_data_root, ob_id=ob_key)
                cand_est.to_device("cuda:0")
                cand_est.use_frame_as_ref(reader_ref, frame, ob_key, is_symmetric=False,
                                          symmetry_axis=opt.ref_sym_axis, crop_ratio=opt.crop_ratio,
                                          device=opt.ref_device, save_to_disk=save_to_disk,
                                          mesh_diameter=mesh_diameter_m,
                                          crop_to_ref=opt.ref_crop_to_ref, probe_dir=probe_dir)
            except Exception:
                _pt(f"  [ref] use_frame_as_ref FAILED v{v_idx:02d} f{frame:04d}")
                traceback.print_exc()
                if 'cand_est' in locals():
                    del cand_est
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                if opt.stop_on_ref_fail:
                    raise
                continue
            _sync_cuda()
            est = cand_est
            ref_v_idx, ref_k, ref_w, ref_h = v_idx, frame, w, h
            _pt(f"  [ref] OK v{ref_v_idx:02d} f{ref_k:04d}")
            break

        if est is None:
            print(f"[pipeline] ERROR: all ref failed for {ob_key}, skip", flush=True)
            object_status[ob_key] = "ref_failed"
            continue

        print(f"\n{'=' * 50}\nSelected Ref: Video [{ref_v_idx:02d}] Frame [{ref_k:04d}] bbox {ref_w}x{ref_h}\n{'=' * 50}\n", flush=True)

        reader_ref = _get_reader(opt.real275_dir, ref_v_idx, reader_cache)
        sk = _sanitize_key(ob_key)

        save_vis_dir = "runs/real275"
        if opt.debug_dir:
            save_vis_dir = os.path.join(opt.debug_dir, "ref_vis")
        elif opt.save_ref_probe:
            save_vis_dir = opt.save_ref_probe
        elif opt.ref_data_root:
            save_vis_dir = os.path.join(opt.ref_data_root, str(ob_key))
        if multi_ob:
            save_vis_dir = os.path.join(save_vis_dir, f"key_{sk}")
        os.makedirs(save_vis_dir, exist_ok=True)
        abs_save_vis_dir = os.path.abspath(save_vis_dir)

        if getattr(est, 'ref_data', None) is not None and getattr(est.ref_data, 'rgb', None) is not None:
            crop_rgb = est.ref_data.rgb.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            crop_mask = (est.ref_data.xyz_map[2] > 1e-5).cpu().numpy().astype(np.uint8) * 255
            cv2.imwrite(os.path.join(abs_save_vis_dir, f"ref_ob_{sk}_v{ref_v_idx:02d}_f{ref_k:04d}_crop_rgb.png"),
                        cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(abs_save_vis_dir, f"ref_ob_{sk}_v{ref_v_idx:02d}_f{ref_k:04d}_crop_mask.png"), crop_mask)
            overlay = crop_rgb.copy()
            overlay[crop_mask > 0] = [0, 255, 0]
            cv2.imwrite(os.path.join(abs_save_vis_dir, f"ref_ob_{sk}_v{ref_v_idx:02d}_f{ref_k:04d}_crop_overlay.png"),
                        cv2.cvtColor(cv2.addWeighted(overlay, 0.3, crop_rgb, 0.7, 0), cv2.COLOR_RGB2BGR))

        full_rgb, full_mask = reader_ref.get_color(ref_k), reader_ref.get_mask(ref_k, ob_key)
        if full_rgb is not None and full_mask is not None:
            full_bgr = cv2.cvtColor(full_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(abs_save_vis_dir, f"ref_ob_{sk}_v{ref_v_idx:02d}_f{ref_k:04d}_full_rgb.png"), full_bgr)
            cv2.imwrite(os.path.join(abs_save_vis_dir, f"ref_ob_{sk}_v{ref_v_idx:02d}_f{ref_k:04d}_full_mask.png"), full_mask)
            full_overlay = full_bgr.copy()
            full_overlay[full_mask > 0] = [0, 255, 0]
            blended = cv2.addWeighted(full_overlay, 0.4, full_bgr, 0.6, 0)
            vs, us = np.where(full_mask > 0)
            if len(vs):
                cv2.rectangle(blended, (int(us.min()), int(vs.min())), (int(us.max()), int(vs.max())), (0, 0, 255), 2)
            cv2.imwrite(os.path.join(abs_save_vis_dir, f"ref_ob_{sk}_v{ref_v_idx:02d}_f{ref_k:04d}_full_overlay.png"), blended)

        try:
            model_pts = reader_ref.get_gt_mesh(ob_key, frame=ref_k).vertices.copy()
        except Exception as e:
            raise RuntimeError(f"[原则#1] key={ob_key}: 加载真实 mesh 失败: {e}") from e
        model_pts = np.asarray(model_pts, dtype=np.float64)
        if model_pts.ndim != 2 or model_pts.shape[1] != 3 or len(model_pts) < 4 or not np.isfinite(model_pts).all():
            raise ValueError(f"[原则#1] key={ob_key}: 模型顶点无效 shape={model_pts.shape}")

        try:
            diameter_mm_real = compute_model_diameter_mm(model_pts, model_id=model_id, verbose=True)
        except Exception as e:
            raise RuntimeError(f"[原则#1] key={ob_key}: 真实直径计算失败: {e}") from e
        diameter_m = diameter_mm_real / 1000.0

        _model_path, _class_id, _symmetry_id = None, None, 0
        try:
            _model_path = reader_ref.get_model_path(ob_key, frame=ref_k)
        except Exception as e:
            _pt(f"  [AR-WARN] get_model_path 异常: {e}")
        try:
            if model_id is not None and model_id in model_info:
                _class_id = model_info[model_id].get("class_id")
            item_ref = reader_ref._get_single_item(ref_k, ob_key)
            if item_ref is not None:
                _symmetry_id = int(item_ref.get("symmetry_id", 0) or 0)
                if _class_id is None:
                    _class_id = item_ref.get("class_id")
        except Exception:
            pass

        all_errors[ob_key] = {
            "trans": [], "rot": [], "n_fail": 0, "add_norm": [], "adds_norm": [],
            "diameter_m": float(diameter_m), "diameter_mm": float(diameter_mm_real),
            "model_pts": model_pts.copy(), "model_path": _model_path,
            "class_id": _class_id, "symmetry_id": _symmetry_id,
            "cam_K": np.asarray(reader_ref.K, dtype=np.float64).copy(),
            "scenes": list(eval_v_idx_list), "scene_names": list(eval_scene_names),
            "model_id": model_id, "model_name": model_name,
            "per_scene": {v: {"trans": [], "rot": [], "n_fail": 0, "add_norm": [], "adds_norm": []}
                          for v in eval_v_idx_list},
            "ar_samples": []}

        total_eval_frames, per_scene_counts = _count_eval_frames(
            eval_v_idx_list, ob_key, ref_v_idx, ref_k, skip_ref, eval_stride,
            max_eval_global, max_eval_per_scene, reader_cache, opt.real275_dir)
        print(f"  [eval] total_eval_frames={total_eval_frames} per_scene={per_scene_counts}", flush=True)

        if total_eval_frames <= 0:
            print("  [eval] 无帧可评，skip", flush=True)
            object_status[ob_key] = "no_frames"
            del est
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            continue

        pbar = tqdm(total=total_eval_frames, desc=f"key={sk}", dynamic_ncols=True)
        evaluated_count = 0
        target_seq_count = 0

        for v_idx in eval_v_idx_list:
            if max_eval_global > 0 and evaluated_count >= max_eval_global:
                break
            r = _get_reader(opt.real275_dir, v_idx, reader_cache)
            scene_evaluated = 0
            for k in r.frames_for_obj(ob_key):
                if max_eval_global > 0 and evaluated_count >= max_eval_global:
                    break
                if max_eval_per_scene > 0 and scene_evaluated >= max_eval_per_scene:
                    break
                if skip_ref and v_idx == ref_v_idx and k == ref_k:
                    continue
                if target_seq_count % eval_stride != 0:
                    target_seq_count += 1
                    continue
                target_seq_count += 1
                pbar.set_postfix({"v": f"{v_idx:02d}", "f": f"{k:04d}", "n": f"{evaluated_count + 1}/{total_eval_frames}"})

                # 原则 #2：pred 可 None；原则 #4：每帧都进 ar_samples
                sample_record = {"v_idx": int(v_idx), "frame_id": int(k), "pred": None, "gt": None}

                pred_pose = gt_pose = None
                err_dict = None
                try:
                    pred_pose, gt_pose = run_pose_inference(
                        r, k, est, ob_key, detect_type,
                        pred_translation_scale=opt.pred_translation_scale)
                except Exception:
                    print(f"\n[register error] v{v_idx:02d} f{k:04d}", flush=True)
                    traceback.print_exc()
                    pred_pose = gt_pose = None

                # 独立记录 AR sample（与指标计算解耦）
                if opt.compute_ar:
                    if pred_pose is not None and gt_pose is not None:
                        sample_record["pred"] = np.asarray(pred_pose, dtype=np.float64).copy()
                        sample_record["gt"] = np.asarray(gt_pose, dtype=np.float64).copy()
                    all_errors[ob_key]["ar_samples"].append(sample_record)

                # 独立计算非 AR 指标（失败不污染 AR）
                if pred_pose is not None and gt_pose is not None:
                    try:
                        err_dict = metrics.compute_all_errors(pred_pose, gt_pose, model_pts, ob_key, diameter_m)
                    except Exception:
                        print(f"\n[metric error] v{v_idx:02d} f{k:04d} (ADD/ADD-S 计算失败，不影响 AR)", flush=True)
                        traceback.print_exc()
                        err_dict = None

                if err_dict is None:
                    for key in ("trans", "rot", "add_norm", "adds_norm"):
                        all_errors[ob_key][key].append(float("inf"))
                        all_errors[ob_key]["per_scene"][v_idx][key].append(float("inf"))
                    all_errors[ob_key]["n_fail"] += 1
                    evaluated_count += 1
                    scene_evaluated += 1
                    pbar.update(1)
                    continue

                trans_mm, rot_err = err_dict["trans_mm"], err_dict["rot_deg"]
                all_errors[ob_key]["trans"].append(trans_mm)
                all_errors[ob_key]["rot"].append(rot_err)
                all_errors[ob_key]["add_norm"].append(err_dict["add_norm"])
                all_errors[ob_key]["adds_norm"].append(err_dict["adds_norm"])
                all_errors[ob_key]["per_scene"][v_idx]["trans"].append(trans_mm)
                all_errors[ob_key]["per_scene"][v_idx]["rot"].append(rot_err)
                all_errors[ob_key]["per_scene"][v_idx]["add_norm"].append(err_dict["add_norm"])
                all_errors[ob_key]["per_scene"][v_idx]["adds_norm"].append(err_dict["adds_norm"])
                all_res[r.get_video_id()][r.id_strs[k]][str(ob_key)] = pred_pose
                evaluated_count += 1
                scene_evaluated += 1
                pbar.update(1)

        pbar.close()

        if opt.compute_ar:
            n_ar = len(all_errors[ob_key]["ar_samples"])
            n_ok = sum(1 for s in all_errors[ob_key]["ar_samples"] if s.get("pred") is not None)
            _pt(f"  [AR-info] key={ob_key}: ar_samples 总数={n_ar} pred_ok={n_ok}")

        if opt.debug_dir:
            os.makedirs(opt.debug_dir, exist_ok=True)
            for name, arr in [("trans_mm", all_errors[ob_key]["trans"]),
                              ("rot_deg", all_errors[ob_key]["rot"]),
                              ("add_norm", all_errors[ob_key]["add_norm"]),
                              ("adds_norm", all_errors[ob_key]["adds_norm"])]:
                fmt = "%.2f" if name in ("trans_mm", "rot_deg") else "%.6f"
                np.savetxt(os.path.join(opt.debug_dir, f"errors_ob_{sk}_{name}.txt"), np.array(arr), fmt=fmt)

        object_status[ob_key] = "evaluated"
        del est
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    _finalize(all_errors, reader_cache, requested_ob_keys=requested_ob_keys, object_status=object_status)

    if opt.debug_dir:
        os.makedirs(opt.debug_dir, exist_ok=True)
        with open(os.path.join(opt.debug_dir, "real275_res.yml"), "w") as f:
            _pyyaml.safe_dump(make_yaml_dumpable(all_res), f)
        with open(os.path.join(opt.debug_dir, "model_scene_coverage.txt"), "w", encoding="utf-8") as f:
            f.write("# model_id | model_name | scene_id(scene_name) | n_eval_frames | status\n")
            for k in requested_ob_keys:
                v = all_errors.get(k)
                st = object_status.get(k, "unknown")
                if v is None:
                    f.write(f"model_id=N/A | ob_key={k}: status={st}\n")
                    continue
                pairs = ", ".join(f"{s}({n})" for s, n in zip(v.get("scenes", []), v.get("scene_names", [])))
                f.write(f"model_id={v.get('model_id')} | {v.get('model_name')}: "
                        f"[{pairs}] n_frames={len(v['trans'])} status={st}\n")
        print(f"\n[pipeline] Final results saved to {opt.debug_dir}", flush=True)


def _finalize(all_errors, reader_cache=None, requested_ob_keys=None, object_status=None):
    print("\n========== REAL275 位姿评估 ==========", flush=True)
    if reader_cache is None:
        reader_cache = {}
    if requested_ob_keys is None:
        requested_ob_keys = list(all_errors.keys())
    if object_status is None:
        object_status = {k: "evaluated" for k in requested_ob_keys}

    if not all_errors and not requested_ob_keys:
        return

    per_obj = []
    for ob_key, v in sorted(all_errors.items(), key=lambda kv: str(kv[0])):
        if not v["trans"]:
            continue
        trans = np.asarray(v["trans"], dtype=np.float64)
        rot = np.asarray(v["rot"], dtype=np.float64)
        add_n = np.asarray(v.get("add_norm", []), dtype=np.float64)
        adds_n = np.asarray(v.get("adds_norm", []), dtype=np.float64)
        diam, diam_mm = float(v.get("diameter_m", float("nan"))), float(v.get("diameter_mm", float("nan")))
        scenes, scene_names = v.get("scenes", []), v.get("scene_names", [])
        mid, mname, per_scene = v.get("model_id"), v.get("model_name"), v.get("per_scene", {})

        finite = np.isfinite(trans) & np.isfinite(rot)
        n_total, n_ok = len(trans), int(finite.sum())
        n_fail = int((~finite).sum())
        mean_trans = float(trans[finite].mean()) if n_ok else float("nan")
        mean_rot = float(rot[finite].mean()) if n_ok else float("nan")
        f_add, f_adds = np.isfinite(add_n), np.isfinite(adds_n)
        mean_add = float(add_n[f_add].mean()) if f_add.any() else float("nan")
        mean_adds = float(adds_n[f_adds].mean()) if f_adds.any() else float("nan")

        def _sr(arr, thr):
            return float((arr <= thr).mean() * 100.0) if len(arr) else 0.0

        ar_res = None
        if opt.compute_ar:
            ar_samples = v.get("ar_samples", [])
            if ar_samples:
                try:
                    ar_res = compute_ar_for_model_v2(
                        model_path=v.get("model_path"), model_pts_m=v.get("model_pts"),
                        model_id=mid, class_id=v.get("class_id"), symmetry_id=v.get("symmetry_id", 0),
                        ar_samples=ar_samples, real275_dir=opt.real275_dir, reader_cache=reader_cache,
                        obj_id_for_renderer=int(mid) if mid is not None else 1,
                        width=int(opt.ar_width), height=int(opt.ar_height),
                        use_symmetry=bool(opt.ar_use_symmetry), max_frames=int(opt.ar_max_frames),
                        verbose=True)
                except Exception as e:
                    _pt(f"[AR] compute_ar_for_model_v2 异常: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    ar_res = {"ok": False, "reason": f"{type(e).__name__}: {e}", "n_targets": len(ar_samples)}
            else:
                ar_res = {"ok": False, "reason": "ar_samples 为空", "n_targets": 0}

        def _ag(k, default=float("nan")):
            if ar_res is None:
                return default
            val = ar_res.get(k)
            try:
                return float(val) if val is not None else default
            except (TypeError, ValueError):
                return default

        per_obj.append({
            "ob_key": ob_key, "model_id": mid, "model_name": mname,
            "n_total": n_total, "n_ok": n_ok, "n_fail": n_fail,
            "scenes": scenes, "scene_names": scene_names, "diameter_m": diam, "diameter_mm": diam_mm,
            "mean_trans": mean_trans, "mean_rot": mean_rot,
            "mean_add_norm": mean_add, "mean_adds_norm": mean_adds,
            "sr_add_005": _sr(add_n, 0.05), "sr_add_01": _sr(add_n, 0.1), "sr_add_02": _sr(add_n, 0.2),
            "sr_adds_005": _sr(adds_n, 0.05), "sr_adds_01": _sr(adds_n, 0.1), "sr_adds_02": _sr(adds_n, 0.2),
            "auc_add_01": compute_auc_norm(add_n, 0.1), "auc_adds_01": compute_auc_norm(adds_n, 0.1),
            "acc_10": compute_acc(trans, 10.0), "acc_20": compute_acc(trans, 20.0),
            "joint_5d_5cm": float(np.mean((rot < 5.0) & (trans < 50.0)) * 100.0) if n_total else 0.0,
            "joint_10d_5cm": float(np.mean((rot < 10.0) & (trans < 50.0)) * 100.0) if n_total else 0.0,
            "ar_ok": bool(ar_res.get("ok")) if ar_res else False,
            "ar_reason": (ar_res.get("reason") if ar_res and not ar_res.get("ok") else None),
            "ar_vsd": _ag("ar_vsd"), "ar_mssd": _ag("ar_mssd"), "ar_mspd": _ag("ar_mspd"),
            "ar": _ag("ar"), "mean_mssd_mspd": _ag("mean_mssd_mspd"),
            "ar_n_targets": int(ar_res.get("n_targets", 0)) if ar_res else 0,
            "ar_n_pred_fail": int(ar_res.get("n_pred_fail", 0)) if ar_res else 0,
            "vsd_complete": bool(ar_res.get("vsd_complete", False)) if ar_res else False,
            "vsd_fail_reason": ar_res.get("vsd_fail_reason") if ar_res else None,
            "mssd_complete": bool(ar_res.get("mssd_complete", False)) if ar_res else False,
            "mssd_fail_reason": ar_res.get("mssd_fail_reason") if ar_res else None,
            "mspd_complete": bool(ar_res.get("mspd_complete", False)) if ar_res else False,
            "mspd_fail_reason": ar_res.get("mspd_fail_reason") if ar_res else None})

        key_label = f"model_id={mid} | {mname}" if mid is not None and mname is not None else str(ob_key)
        scene_pairs = ", ".join(f"{s}({n})" for s, n in zip(scenes, scene_names))
        print(f"\n========== [key {key_label}] ==========")
        print(f" 涉及 scene: [{scene_pairs}]  diameter={diam:.4f} m ({diam_mm:.2f} mm)")
        print(f" 计划评估: {n_total}  成功: {n_ok}  失败: {n_fail}")
        if n_ok > 0:
            print(f"  成功样本均值: Trans {mean_trans:.2f} mm | Rot {mean_rot:.2f} deg")
        print(f"  ADD Mean: {mean_add:.6f}  ADD-S Mean: {mean_adds:.6f}")
        if opt.compute_ar:
            if ar_res is None:
                print("  AR: ✗ compute_ar_for_model_v2 返回 None")
            elif not ar_res.get("ok"):
                print(f"  AR: ✗ {ar_res.get('reason', '未知')}")
            else:
                print(f"  --- BOP AR (n_targets={ar_res.get('n_targets', 0)}, pred_fail={ar_res.get('n_pred_fail', 0)}) ---")
                for name, label in (("vsd", "VSD"), ("mssd", "MSSD"), ("mspd", "MSPD")):
                    val = ar_res.get(f"ar_{name}", float("nan"))
                    if np.isfinite(val):
                        print(f"  AR_{label:4s} = {val:.4f}  "
                              f"({'complete' if ar_res.get(f'{name}_complete') else 'partial'})")
                    else:
                        print(f"  AR_{label:4s} = N/A  ({ar_res.get(f'{name}_fail_reason') or '未知'})")
                ar_val = ar_res.get("ar", float("nan"))
                print(f"  AR = {ar_val:.4f} (BOP 3 项均值)" if np.isfinite(ar_val)
                      else "  AR = N/A（三个分项未全部完整可用）")
                two = ar_res.get("mean_mssd_mspd", float("nan"))
                if np.isfinite(two):
                    print(f"  (MSSD+MSPD 两项均值 = {two:.4f}，**不是 BOP AR**)")

        if per_scene:
            print("\n  [per-scene breakdown]")
            for v_idx in sorted(per_scene):
                ps = per_scene[v_idx]
                if not ps.get("trans"):
                    continue
                st = np.asarray(ps["trans"], dtype=np.float64)
                sr = np.asarray(ps["rot"], dtype=np.float64)
                sa = np.asarray(ps["add_norm"], dtype=np.float64)
                ss = np.asarray(ps["adds_norm"], dtype=np.float64)
                f = np.isfinite(st) & np.isfinite(sr)
                s_name = scene_names[scenes.index(v_idx)] if v_idx in scenes else str(v_idx)
                print(f"    scene_id={v_idx} ({s_name}): n={len(st)} ok={int(f.sum())} fail={int((~f).sum())} | "
                      f"Trans={st[f].mean() if f.any() else float('nan'):.2f}mm "
                      f"Rot={sr[f].mean() if f.any() else float('nan'):.2f}deg ")

    # 原则 #4：即使 per_obj 为空，也报告请求集 / 不可达原因
    if not per_obj:
        print("\n  [结果] 无任何对象进入评估")
        for k in requested_ob_keys:
            print(f"    - {k}: {object_status.get(k, 'unknown')}")
    else:
        print("\n========== [MODEL COVERAGE] ==========")
        for p in per_obj:
            scene_pairs = ", ".join(f"{s}({n})" for s, n in zip(p["scenes"], p["scene_names"]))
            mid_str = f"model_id={p['model_id']}" if p.get("model_id") is not None else "model_id=N/A"
            print(f"  {mid_str} | {p.get('model_name')}: scene_count={len(p['scenes'])}, "
                  f"scenes=[{scene_pairs}], n_frames={p['n_total']}")

        def _mean(key):
            vals = [p[key] for p in per_obj if np.isfinite(p[key])]
            return float(np.mean(vals)) if vals else float("nan")

        print("\n========== [OVERALL MEAN] (per-object equal weight) ==========")
        print(f" Objects(evaluated): {len(per_obj)}")
        for label, key, fmt in [("Mean Trans", "mean_trans", "%.2f mm"), ("Mean Rot", "mean_rot", "%.2f deg"),
                                ("Mean ADD", "mean_add_norm", "%.6f"), ("Mean ADD-S", "mean_adds_norm", "%.6f")]:
            print(f"   {label}: " + fmt % _mean(key))

    if opt.compute_ar:
        print("\n========== [OVERALL AR] ==========")
        agg = _aggregate_objects_ar(per_obj, requested_keys=requested_ob_keys, statuses=object_status)
        print(f" 请求对象: {agg['n_objects_requested']}  "
              f"可达: {agg['n_objects_reachable']}  "
              f"完整: {agg['n_objects_complete']}  "
              f"参考失败: {agg['n_objects_ref_failed']}  "
              f"无候选参考/场景: {agg['n_objects_no_ref']}")
        print(f" 完整子集覆盖率（完整/请求）: {agg['object_coverage'] * 100:.1f}%")
        if agg["unreachable_reasons"]:
            print(" [不可达对象]")
            for r in agg["unreachable_reasons"]:
                print(f"    - {r}")

        if "complete_subset_ar" in agg and np.isfinite(agg["complete_subset_ar"]):
            n_c = agg["n_objects_complete"]
            print(f"\n  --- 完整指标子集 AR (n={n_c}) ---")
            for k in ("ar_vsd", "ar_mssd", "ar_mspd", "ar"):
                print(f"  {k.upper()}: {agg.get('complete_subset_' + k, float('nan')):.4f}")
        if np.isfinite(agg["full_set_ar"]):
            print(f"\n  --- 全目标集 AR (n={agg['n_objects_requested']} 全部完整) ---\n"
                  f"  AR: {agg['full_set_ar']:.4f}")
        else:
            print(f"\n  [全目标集 AR] N/A（请求 {agg['n_objects_requested']}，"
                  f"完整 {agg['n_objects_complete']}，含不可达/不完整对象）")
    else:
        print("\n   AR: ⊘ 未启用")


def inspect_gt(real275_dir):
    root = os.path.join(real275_dir, "gts", "real_test")
    files = sorted(glob.glob(os.path.join(root, "*.pkl")))
    print(f"[inspect] {len(files)} pkl under {root}")
    if not files:
        return
    with open(files[0], "rb") as f:
        d = pickle.load(f)
    print(f"\n===== {files[0]} =====")
    for k, v in d.items():
        try:
            arr = np.asarray(v)
            print(f"{k:20s} type={type(v).__name__:15s} shape={arr.shape} dtype={arr.dtype}")
            print(f"    {'values' if arr.size <= 20 else 'first 5'}: {arr if arr.size <= 20 else arr.flatten()[:5]}")
        except Exception as e:
            print(f"{k:20s} type={type(v).__name__} ({e})")

    m = re.search(r"_scene_(\d+)_(\d+)\.pkl$", os.path.basename(files[0]))
    if m is None:
        return
    scene_id_str, frame_id = m.group(1), int(m.group(2))
    meta_path = os.path.join(real275_dir, "real_test", f"scene_{scene_id_str}", f"{frame_id:04d}_meta.txt")
    print(f"\n[inspect] meta.txt: {meta_path}")
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as f:
            for i, line in enumerate(f):
                print(f"  line {i}: {line.rstrip()}")
                if i > 10:
                    print("  ...")
                    break
        inst_to_info = parse_meta_models(meta_path)
        print("  parsed inst_id -> (class_id, model_name, symmetry_id):")
        for inst_id in sorted(inst_to_info):
            info = inst_to_info[inst_id]
            print(f"    inst={inst_id}: {info['class_id']}, {info['model_name']}, {info['symmetry_id']}")
    else:
        print("  (missing)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real275_dir", type=str, required=True)
    parser.add_argument("--model_id", type=int, default=None)
    parser.add_argument("--model_ids", type=str, default=None)
    parser.add_argument("--ob_id", type=int, default=None)
    parser.add_argument("--ob_ids", type=str, default=None)
    parser.add_argument("--ob_names", type=str, default=None)
    parser.add_argument("--ref_video_idx", type=int, default=-1)
    parser.add_argument("--ref_frame", type=int, default=-1)
    parser.add_argument("--ref_stride", type=int, default=1)
    parser.add_argument("--cam_K", type=str, default=None)
    parser.add_argument("--obj_models_dir", type=str, default=None)
    parser.add_argument("--eval_skip_ref", action="store_true", default=True)
    parser.add_argument("--eval_include_ref", dest="eval_skip_ref", action="store_false")
    parser.add_argument("--eval_stride", type=int, default=1)
    parser.add_argument("--max_eval_frames", type=int, default=0)
    parser.add_argument("--max_eval_frames_per_scene", type=int, default=0)
    parser.add_argument("--ref_data_root", type=str, default=None)
    parser.add_argument("--save_ref", action="store_true")
    parser.add_argument("--save_ref_probe", type=str, default=None)
    parser.add_argument("--ref_sym_axis", type=int, default=-1)
    parser.add_argument("--ref_device", type=str, default="cuda")
    parser.add_argument("--ref_crop_to_ref", action="store_true", default=True)
    parser.add_argument("--no_ref_crop_to_ref", dest="ref_crop_to_ref", action="store_false")
    parser.add_argument("--stop_on_ref_fail", action="store_true", default=False)
    parser.add_argument("--detect_type", type=str, default="mask", choices=["mask", "box"])
    parser.add_argument("--debug_dir", type=str, default="")
    parser.add_argument("--crop_ratio", type=float, default=1.0)
    parser.add_argument("--min_visib", type=int, default=8)
    parser.add_argument("--pred_translation_scale", type=float, default=1.0)
    parser.add_argument("--ob_key_mode", type=str, default="model_id",
                        choices=["class_id", "gt_idx", "model_name", "model_id"])
    parser.add_argument("--inspect_gt", action="store_true", default=False)
    parser.add_argument("--export_registry_json", type=str, default=None)
    parser.add_argument("--registry_json", type=str, default=None)
    parser.add_argument("--rebuild_registry", action="store_true", default=False)
    parser.add_argument("--compute_ar", dest="compute_ar", action="store_true", default=True)
    parser.add_argument("--no_compute_ar", dest="compute_ar", action="store_false")
    parser.add_argument("--ar_width", type=int, default=640)
    parser.add_argument("--ar_height", type=int, default=480)
    parser.add_argument("--ar_max_frames", type=int, default=200,
                        help="AR 每对象最多抽 N 帧（0=全部，正式评估建议 0）")
    parser.add_argument("--ar_use_symmetry", action="store_true", default=True)
    parser.add_argument("--no_ar_use_symmetry", dest="ar_use_symmetry", action="store_false")

    opt = parser.parse_args()
    if opt.ob_ids:
        opt.ob_ids = [int(x) for x in opt.ob_ids.split(",")]
    if opt.cam_K:
        opt.cam_K = np.array([float(x) for x in opt.cam_K.replace(",", " ").split()]).reshape(3, 3)
    else:
        opt.cam_K = None

    set_seed(0)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        stream=sys.stdout, force=True)
    faulthandler.enable()

    if opt.inspect_gt:
        inspect_gt(opt.real275_dir)
        raise SystemExit(0)
    if opt.export_registry_json:
        _export_registry_json(opt.real275_dir, opt.export_registry_json, cache={})
        raise SystemExit(0)

    start = time.time()
    run_pipeline(detect_type=opt.detect_type)
    print(f"\n[pipeline] Total wall time: {time.time() - start:.2f}s", flush=True)