# test_real275.py
# -*- coding: utf-8 -*-
"""
REAL275 / NOCS 位姿评估，使用 OneRefPose。
"""

from Utils import *
import os, sys, json, time, argparse, glob
import logging
import faulthandler
import traceback
import numpy as np
import torch
import trimesh
import yaml as _pyyaml
import cv2
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import cKDTree
from tqdm import tqdm
import pickle
import re

from datareader import *
from pose_calculation import OneRefPose


_INTERNAL_OB_KEY_MODE = "class_id"
_GLOBAL_MODEL_ID_TO_NAME = None


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
    supported_keys = ("gt_class_ids", "gt_classes", "classes")
    for key in supported_keys:
        value = data.get(key)
        if value is None:
            continue
        values = np.asarray(value).reshape(-1)
        if len(values) != n_instances:
            raise ValueError(f"{source}: {key} 长度={len(values)} 但 gt_RTs={n_instances}")
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError(f"{source}: {key} 非数值")
        if not np.isfinite(values).all():
            raise ValueError(f"{source}: {key} 含 NaN/Inf")
        ids = values.astype(np.int64)
        if not np.array_equal(values, ids):
            raise ValueError(f"{source}: {key} 含非整数类别 ID")
        return ids, key
    raise KeyError(f"{source}: 缺少类别字段 {supported_keys}")


def parse_meta_models(meta_path):
    if not os.path.isfile(meta_path):
        return {}
    out = {}
    with open(meta_path, "r") as f:
        for line in f:
            words = line.strip().split()
            if len(words) < 3:
                continue
            try:
                inst_id = int(words[0])
                cls_id = int(words[1])
            except ValueError:
                continue
            if len(words) == 3:
                name = words[2]
            elif len(words) >= 4:
                name = words[3]
            else:
                continue
            if name.endswith(".npz"):
                name = name[:-4]
            out[inst_id] = {"class_id": cls_id, "model_name": name}
    return out


class Real275BopReader:
    MASK_SUFFIXES = ["_mask.png", "_mask_visib.png", "_label.png"]
    NOCS_DEFAULT_K = np.array([
        [591.0125,   0.0,     322.525 ],
        [  0.0,    590.16775, 244.11084],
        [  0.0,      0.0,       1.0    ],
    ], dtype=np.float64)

    def __init__(self, video_dir, gts_root, scene_id,
                 obj_models_dir=None, cam_K=None,
                 orthogonalize_R=True, depth_scale=1000.0,
                 min_visib=8, ob_key_mode="class_id",
                 model_id_to_name=None):
        self.video_dir = video_dir
        self.gts_root = gts_root
        self.scene_id = int(scene_id)
        self.obj_models_dir = obj_models_dir
        self._dataset_dir = str(video_dir)
        self.K = (np.array(cam_K, dtype=np.float64)
                  if cam_K is not None else self.NOCS_DEFAULT_K.copy())
        self.orthogonalize_R = bool(orthogonalize_R)
        self.depth_scale = float(depth_scale)
        self.min_visib = int(min_visib)
        if ob_key_mode not in ("class_id", "gt_idx", "model_name", "model_id"):
            raise ValueError(f"ob_key_mode 不支持: {ob_key_mode}")
        self.ob_key_mode = ob_key_mode
        self.model_id_to_name = dict(model_id_to_name) if model_id_to_name else {}
        self.model_name_to_id = {v: k for k, v in self.model_id_to_name.items()}
        self._rgb_map = {}
        self._depth_map = {}
        self._mask_map = {}
        self._scene_gt = {}
        self._mesh_cache = {}
        self.ob_ids = []
        self._load()

    def _key_of(self, item):
        if self.ob_key_mode == "class_id":
            return item["class_id"]
        if self.ob_key_mode == "gt_idx":
            return item["gt_idx"]
        if self.ob_key_mode == "model_name":
            return item["model_name"]
        if self.ob_key_mode == "model_id":
            mn = item.get("model_name")
            return self.model_name_to_id.get(mn) if mn else None
        raise RuntimeError(f"unknown ob_key_mode={self.ob_key_mode}")

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

        pattern = os.path.join(
            self.gts_root, f"results_real_test_scene_{self.scene_id}_*.pkl")
        gt_files = sorted(glob.glob(pattern))
        if not gt_files:
            raise FileNotFoundError(f"没有找到 GT 文件: {pattern}")

        reported_meta = False
        reported_pkl = False
        ambiguous_count = 0
        missing_model_count = 0
        missing_class_count = 0

        for fp in gt_files:
            match = re.search(r"_scene_(\d+)_(\d+)\.pkl$", os.path.basename(fp))
            if match is None:
                continue
            frame_id = int(match.group(2))
            with open(fp, "rb") as fh:
                data = pickle.load(fh)
            gt_RTs = np.asarray(data["gt_RTs"], dtype=np.float64)
            if gt_RTs.ndim != 3 or gt_RTs.shape[1:] != (4, 4):
                raise ValueError(f"{fp}: gt_RTs 应为 (N,4,4)")
            if not np.isfinite(gt_RTs).all():
                raise ValueError(f"{fp}: gt_RTs 含 NaN/Inf")
            n_instances = len(gt_RTs)

            meta_path = os.path.join(self.video_dir, f"{frame_id:04d}_meta.txt")
            inst_to_info = parse_meta_models(meta_path)
            if inst_to_info and not reported_meta:
                _pt(f"[reader] {self.get_video_id()}: 使用 meta.txt")
                reported_meta = True

            pkl_class_ids = None
            if not inst_to_info:
                try:
                    pkl_class_ids, field = read_gt_class_ids(
                        data, n_instances, source=fp)
                    if not reported_pkl:
                        _pt(f"[reader] {self.get_video_id()}: 使用 pkl '{field}'")
                        reported_pkl = True
                except (KeyError, ValueError):
                    pkl_class_ids = None

            items = []
            for gt_idx in range(n_instances):
                pose = gt_RTs[gt_idx].copy()
                if self.orthogonalize_R:
                    pose[:3, :3] = orthogonalize_rotation(pose[:3, :3])
                info = inst_to_info.get(gt_idx + 1, {})
                class_id = info.get("class_id")
                model_name = info.get("model_name")
                if class_id is None and pkl_class_ids is not None:
                    class_id = int(pkl_class_ids[gt_idx])
                if class_id is None:
                    missing_class_count += 1
                if model_name is None:
                    missing_model_count += 1
                items.append({
                    "gt_idx": gt_idx,
                    "class_id": class_id,
                    "mask_id": gt_idx + 1,
                    "pose": pose,
                    "model_name": model_name,
                })

            if self.ob_key_mode == "class_id":
                n_missing = sum(1 for it in items if it["class_id"] is None)
                if n_missing == n_instances:
                    raise RuntimeError(
                        f"{fp}: 无法确定 class_id（meta 缺失且 pkl 无类别字段）")
                valid_ids = [it["class_id"] for it in items if it["class_id"] is not None]
                if len(valid_ids) != len(set(valid_ids)):
                    ambiguous_count += 1

            if self.ob_key_mode in ("model_name", "model_id"):
                n_missing = sum(1 for it in items if it["model_name"] is None)
                if n_missing == n_instances:
                    raise RuntimeError(
                        f"{fp}: 无法确定 model_name（meta 缺失）")
                valid_names = [it["model_name"] for it in items
                               if it["model_name"] is not None]
                if len(valid_names) != len(set(valid_names)):
                    ambiguous_count += 1

            self._scene_gt[frame_id] = items

        keys = set()
        for items in self._scene_gt.values():
            for item in items:
                k = self._key_of(item)
                if k is None:
                    continue
                keys.add(str(k) if self.ob_key_mode == "model_name" else int(k))
        self.ob_ids = sorted(keys)

        _pt(f"[reader] {self.get_video_id()}: mode={self.ob_key_mode}, "
            f"GT frames={len(self._scene_gt)}, ob_keys={self.ob_ids}")
        if ambiguous_count:
            _pt(f"[WARN] {self.get_video_id()}: {ambiguous_count} 帧同 key 多实例")
        if missing_model_count:
            _pt(f"[WARN] {self.get_video_id()}: {missing_model_count} 项无 model_name")
        if missing_class_count:
            _pt(f"[WARN] {self.get_video_id()}: {missing_class_count} 项无 class_id")

    def _match_items(self, frame, ob_key):
        if self.ob_key_mode == "model_id":
            mn = self.model_id_to_name.get(ob_key)
            if mn is None:
                return []
            return [it for it in self._scene_gt.get(frame, [])
                    if it.get("model_name") == mn]
        return [it for it in self._scene_gt.get(frame, [])
                if self._key_of(it) == ob_key]

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
            raise ValueError(f"不支持的深度形状: {d.shape}")
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
        path = self._mask_map.get(i)
        if path is None:
            return None
        labels = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if labels is None:
            return None
        if labels.ndim == 3:
            labels = labels[:, :, -1]
        if labels.ndim != 2:
            raise ValueError(f"{path}: mask 应为二维")
        mask = labels == item["mask_id"]
        return mask.astype(np.uint8) * 255 if np.any(mask) else None

    def _find_obj_models_dir(self):
        if self.obj_models_dir:
            base = self.obj_models_dir
        else:
            r27 = os.path.dirname(os.path.dirname(self.gts_root))
            base = os.path.join(r27, "obj_models")
        for sub in ("real_test", "real_val", "train", "val", "real_train"):
            cand = os.path.join(base, sub)
            if os.path.isdir(cand):
                return cand
        raise FileNotFoundError(f"obj_models 目录不存在: {base}")

    def _load_mesh_obj(self, model_name):
        if model_name in self._mesh_cache:
            return self._mesh_cache[model_name]
        obj_dir = self._find_obj_models_dir()
        candidates = [
            os.path.join(obj_dir, f"{model_name}.obj"),
            os.path.join(obj_dir, f"{model_name}_norm.obj"),
            os.path.join(obj_dir, f"{model_name}.ply"),
        ]
        path = next((c for c in candidates if os.path.isfile(c)), None)
        if path is None:
            raise FileNotFoundError(f"Mesh not found: {model_name}")
        mesh = trimesh.load(path, force="mesh")
        if mesh.vertices.shape[0] == 0:
            raise RuntimeError(f"Mesh empty: {path}")
        self._mesh_cache[model_name] = mesh
        _pt(f"[mesh] loaded {model_name} ({mesh.vertices.shape[0]} verts)")
        return mesh

    def get_gt_mesh(self, ob_key, frame=None):
        if self.ob_key_mode == "model_id":
            mn = self.model_id_to_name.get(ob_key)
            if mn is None:
                raise RuntimeError(f"model_id={ob_key} 无映射")
            return self._load_mesh_obj(mn)

        if self.ob_key_mode == "model_name":
            return self._load_mesh_obj(ob_key)

        model_names = set()
        frames = [frame] if frame is not None else list(self._scene_gt.keys())
        for f in frames:
            for it in self._match_items(f, ob_key):
                mn = it.get("model_name")
                if mn:
                    model_names.add(mn)
        if not model_names:
            raise RuntimeError(f"ob_key={ob_key}: 找不到 model_name")
        if len(model_names) > 1:
            raise RuntimeError(f"ob_key={ob_key}: 多个 model_name {model_names}")
        return self._load_mesh_obj(model_names.pop())

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
            w = u1 - u0 + 1
            h = v1 - v0 + 1
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
        if n <= self.adds_sample:
            return pts
        return pts[self._rng.choice(n, self.adds_sample, replace=False)]

    def compute_add_norm(self, pred, gt, pts, diameter_m):
        if pts is None or len(pts) == 0 or not (diameter_m > 0):
            return float("nan")
        pts = self._subsample(np.asarray(pts, dtype=np.float64))
        pred_pts = pts @ pred[:3, :3].T + pred[:3, 3]
        gt_pts = pts @ gt[:3, :3].T + gt[:3, 3]
        return float(np.linalg.norm(pred_pts - gt_pts, axis=1).mean() / diameter_m)

    def compute_adds_norm(self, pred, gt, pts, diameter_m):
        if pts is None or len(pts) == 0 or not (diameter_m > 0):
            return float("nan")
        pts = self._subsample(np.asarray(pts, dtype=np.float64))
        pred_pts = pts @ pred[:3, :3].T + pred[:3, 3]
        gt_pts = pts @ gt[:3, :3].T + gt[:3, 3]
        tree = cKDTree(pred_pts)
        d, _ = tree.query(gt_pts, k=1, workers=-1)
        return float(d.mean() / diameter_m)

    def compute_pose_decomposition_error(self, pred, gt):
        rot = pred[:3, :3] @ gt[:3, :3].T
        rot_err = np.linalg.norm(Rot.from_matrix(rot).as_rotvec()) * 180 / np.pi
        trans_err = np.linalg.norm(pred[:3, 3] - gt[:3, 3])
        return float(rot_err), float(trans_err)

    def compute_all_errors(self, pred, gt, model_pts, ob_key, diameter_m):
        rot_err, trans_err = self.compute_pose_decomposition_error(pred, gt)
        add_norm = self.compute_add_norm(pred, gt, model_pts, diameter_m)
        adds_norm = self.compute_adds_norm(pred, gt, model_pts, diameter_m)
        return {"trans_mm": trans_err * 1000.0, "rot_deg": rot_err,
                "add_norm": add_norm, "adds_norm": adds_norm}


def compute_acc(errs, threshold=10.0):
    errs = np.asarray(errs, dtype=np.float64)
    return float(np.mean(errs < threshold) * 100.0) if len(errs) else 0.0


def compute_auc_norm(errs, max_val=0.1, step=0.001):
    errs = np.asarray(errs, dtype=np.float64)
    errs = errs[np.isfinite(errs)]
    if len(errs) == 0:
        return 0.0
    X = np.arange(0, max_val + step, step)
    Y = np.array([float((errs <= x).mean()) for x in X])
    area = np.trapezoid(Y, X) if hasattr(np, "trapezoid") else np.trapz(Y, X)
    return float(area / max_val)


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
            mn = it.get("model_name")
            if mn:
                models.add(mn)
    return sorted(models)


def _get_reader(real275_dir, v_idx, cache=None):
    if cache is not None and v_idx in cache:
        return cache[v_idx]
    video_dirs = _discover_video_dirs(real275_dir)
    gts_root = os.path.join(real275_dir, "gts", "real_test")
    sid = int(os.path.basename(video_dirs[v_idx]).split("_")[1])
    r = Real275BopReader(
        video_dirs[v_idx], gts_root, sid,
        obj_models_dir=opt.obj_models_dir,
        cam_K=opt.cam_K,
        min_visib=opt.min_visib,
        ob_key_mode=_INTERNAL_OB_KEY_MODE,
        model_id_to_name=_GLOBAL_MODEL_ID_TO_NAME,
    )
    if cache is not None:
        cache[v_idx] = r
    return r


def _report_inventory(real275_dir, log_path=None, cache=None):
    video_dirs = _discover_video_dirs(real275_dir)
    scene_to_models = {}
    model_to_scenes = {}
    lines = []

    def emit(text):
        lines.append(text)
        print(text, flush=True)

    emit("\n========== [Inventory] 数据清单 ==========")
    for v_idx in range(len(video_dirs)):
        reader = _get_reader(real275_dir, v_idx, cache)
        vid = reader.get_video_id()
        models = _collect_model_names_in_reader(reader)
        scene_to_models[v_idx] = (vid, models)
        for mn in models:
            model_to_scenes.setdefault(mn, []).append(v_idx)
        emit(f"\nv{v_idx:02d} ({vid}) models={models}")
        for mn in models:
            raw_frames = raw_instances = ambiguous = 0
            for frame in reader._scene_gt:
                n_match = sum(1 for it in reader._scene_gt[frame]
                              if it.get("model_name") == mn)
                if n_match == 0:
                    continue
                raw_frames += 1
                raw_instances += n_match
                ambiguous += int(n_match > 1)
            try:
                usable = len(reader.frames_with_bbox(mn))
            except Exception:
                usable = "N/A"
            emit(f"  model={mn}: GT帧数={raw_frames}, GT实例数={raw_instances}, "
                 f"多实例歧义帧数={ambiguous}, 可用单实例帧数={usable}")

    emit("\n========== [Inventory] scene_id -> model_names ==========")
    for v_idx in sorted(scene_to_models):
        vid, models = scene_to_models[v_idx]
        emit(f"  scene_id={v_idx} ({vid}) 含 {len(models)} 个 model:")
        for mn in models:
            emit(f"    - {mn}")

    emit("\n========== [Inventory] model_name -> scene_ids ==========")
    for mn in sorted(model_to_scenes):
        vids = sorted(model_to_scenes[mn])
        names = [scene_to_models[v][0] for v in vids]
        emit(f"  {mn}: scene_ids={vids}  ({names})")

    if log_path:
        parent = os.path.dirname(os.path.abspath(log_path))
        os.makedirs(parent, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def _build_model_id_registry(reader_cache, video_dirs):
    raw = {}
    for v_idx in range(len(video_dirs)):
        r = reader_cache.get(v_idx)
        if r is None:
            continue
        for frame_items in r._scene_gt.values():
            for it in frame_items:
                mn = it.get("model_name")
                if not mn:
                    continue
                cid = it.get("class_id")
                info = raw.setdefault(mn, {
                    "class_id": cid, "scene_ids": set(), "usable_frames": 0})
                info["scene_ids"].add(v_idx)
                if cid is not None:
                    info["class_id"] = cid
        if r.ob_key_mode == "model_name":
            for mn in r.ob_ids:
                if mn not in raw:
                    continue
                try:
                    raw[mn]["usable_frames"] += len(r.frames_with_bbox(mn))
                except Exception:
                    pass

    def _sort_key(mn):
        cid = raw[mn]["class_id"]
        return (cid if cid is not None else 999, mn)

    sorted_names = sorted(raw.keys(), key=_sort_key)
    model_id_to_name = {}
    model_name_to_id = {}
    model_info = {}
    for new_id, mn in enumerate(sorted_names, start=1):
        model_id_to_name[new_id] = mn
        model_name_to_id[mn] = new_id
        r0 = raw[mn]
        scene_ids = sorted(r0["scene_ids"])
        scene_names = [reader_cache[v].get_video_id() for v in scene_ids]
        model_info[new_id] = {
            "model_id": new_id, "model_name": mn, "class_id": r0["class_id"],
            "scene_ids": scene_ids, "scene_names": scene_names,
            "usable_frames": r0["usable_frames"],
        }
    return model_id_to_name, model_name_to_id, model_info


def _print_model_id_table(model_info):
    print("\n========== [Inventory] model_id 表 ==========")
    header = (f"{'model_id':>8} | {'class_id':>8} | {'model_name':<34} | "
              f"{'scene_ids':<16} | scene_names                   | 可用帧数")
    print(header)
    print("-" * len(header))
    for mid in sorted(model_info):
        info = model_info[mid]
        print(f"{mid:>8} | {str(info['class_id']):>8} | "
              f"{info['model_name']:<34} | {str(info['scene_ids']):<16} | "
              f"{str(info['scene_names']):<28} | {info['usable_frames']}")
    print(f"\n[TIP] `--model_id N` 单个；`--model_ids 1,3,5` 多个；`--model_ids all` 全部。")


def _collect_inventory_dict(real275_dir, cache=None):
    global _INTERNAL_OB_KEY_MODE, _GLOBAL_MODEL_ID_TO_NAME
    prev_mode = _INTERNAL_OB_KEY_MODE
    prev_map = _GLOBAL_MODEL_ID_TO_NAME
    _INTERNAL_OB_KEY_MODE = "model_name"
    _GLOBAL_MODEL_ID_TO_NAME = None

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
                raw_frames = raw_instances = ambiguous = 0
                for frame in r._scene_gt:
                    n_match = sum(1 for it in r._scene_gt[frame]
                                  if it.get("model_name") == mn)
                    if n_match == 0:
                        continue
                    raw_frames += 1
                    raw_instances += n_match
                    ambiguous += int(n_match > 1)
                try:
                    usable = len(r.frames_with_bbox(mn))
                except Exception:
                    usable = 0
                models_list.append({
                    "model_name": mn, "class_id": cid,
                    "raw_frames": int(raw_frames), "raw_instances": int(raw_instances),
                    "ambiguous": int(ambiguous), "usable_frames": int(usable),
                })
            scenes.append({"v_idx": int(v_idx), "video_id": vid, "models": models_list})
    finally:
        _INTERNAL_OB_KEY_MODE = prev_mode
        _GLOBAL_MODEL_ID_TO_NAME = prev_map
    return {"scenes": scenes}


def _export_registry_json(real275_dir, out_path, cache=None):
    print(f"[export] 扫描 {real275_dir} ...", flush=True)
    video_dirs = _discover_video_dirs(real275_dir)
    if not video_dirs:
        raise SystemExit(f"[export] No videos under {real275_dir}/real_test/")
    if cache is None:
        cache = {}
    inventory = _collect_inventory_dict(real275_dir, cache=cache)
    _, _, model_info = _build_model_id_registry(cache, video_dirs)
    data = {
        "version": 1,
        "real275_dir": os.path.abspath(real275_dir),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "min_visib": int(opt.min_visib),
        "obj_key_mode": "model_name",
        "scenes": inventory["scenes"],
        "models_registry": [model_info[mid] for mid in sorted(model_info)],
    }
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[export] saved -> {out_path} ({len(data['scenes'])} scenes, "
          f"{len(data['models_registry'])} models)", flush=True)
    return data


def _load_registry_json(path, expected_min_visib=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[registry] 读取失败: {e}", flush=True)
        return None
    if data.get("version") != 1:
        return None
    if (expected_min_visib is not None
            and data.get("min_visib") != expected_min_visib):
        print(f"[registry] min_visib 不匹配，将重建", flush=True)
        return None
    inventory = {"scenes": data["scenes"]}
    model_id_to_name = {}
    model_name_to_id = {}
    model_info = {}
    for m in data["models_registry"]:
        mid = int(m["model_id"])
        model_id_to_name[mid] = m["model_name"]
        model_name_to_id[m["model_name"]] = mid
        model_info[mid] = m
    return inventory, model_id_to_name, model_name_to_id, model_info


def _print_inventory_from_json(inventory, log_path=None):
    lines = []

    def emit(text):
        lines.append(text)
        print(text, flush=True)

    emit("\n========== [Inventory] 数据清单（来自 JSON） ==========")
    for scene in inventory["scenes"]:
        v_idx, vid = scene["v_idx"], scene["video_id"]
        model_names = [m["model_name"] for m in scene["models"]]
        emit(f"\nv{v_idx:02d} ({vid}) models={model_names}")
        for m in scene["models"]:
            emit(f"  model={m['model_name']}: GT帧数={m.get('raw_frames', 0)}, "
                 f"GT实例数={m.get('raw_instances', 0)}, "
                 f"多实例歧义帧数={m.get('ambiguous', 0)}, "
                 f"可用单实例帧数={m.get('usable_frames', 0)}")
    emit("\n========== [Inventory] scene_id -> model_names ==========")
    for scene in inventory["scenes"]:
        model_names = [m["model_name"] for m in scene["models"]]
        emit(f"  scene_id={scene['v_idx']} ({scene['video_id']}) 含 "
             f"{len(model_names)} 个 model:")
        for mn in model_names:
            emit(f"    - {mn}")
    emit("\n========== [Inventory] model_name -> scene_ids ==========")
    model_to_scenes = {}
    for scene in inventory["scenes"]:
        for m in scene["models"]:
            model_to_scenes.setdefault(m["model_name"], []).append(scene["v_idx"])
    scene_id_to_name = {s["v_idx"]: s["video_id"] for s in inventory["scenes"]}
    for mn in sorted(model_to_scenes):
        vids = sorted(model_to_scenes[mn])
        emit(f"  {mn}: scene_ids={vids}  ({[scene_id_to_name[v] for v in vids]})")
    if log_path:
        parent = os.path.dirname(os.path.abspath(log_path))
        os.makedirs(parent, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def _find_ref_candidates(real275_dir, ob_key, ref_scene,
                         ref_frame=-1, ref_stride=1,
                         cache=None, verbose=True, max_candidates=20):
    r = _get_reader(real275_dir, ref_scene, cache)
    if ob_key not in r.ob_ids:
        if verbose:
            print(f"  [scan] v{ref_scene:02d}: ob_key 不在 scene 中", flush=True)
        return []
    infos = r.frames_with_bbox(ob_key)
    if not infos:
        if verbose:
            print(f"  [scan] v{ref_scene:02d}: 无可用帧", flush=True)
        return []

    if ref_frame >= 0:
        infos = [t for t in infos if t[0] == ref_frame]
        if not infos:
            if verbose:
                print(f"  [scan] v{ref_scene:02d}: ref_frame={ref_frame} 不在可用帧里",
                      flush=True)
            return []
    elif ref_stride > 1:
        infos = sorted(infos, key=lambda x: x[0])[::ref_stride]

    if verbose:
        print(f"  [scan] v{ref_scene:02d} ({r.get_video_id()}): "
              f"{len(infos)} candidates", flush=True)

    pools = []
    for f, area, w, h in infos:
        prio = 0 if f == ref_frame else (1 if f == infos[0][0] else 2)
        pools.append((prio, -area, ref_scene, f, w, h, area))
    pools.sort(key=lambda t: (t[0], t[1]))
    candidates = [(p[2], p[3], p[4], p[5], p[6]) for p in pools[:max_candidates]]
    if verbose:
        top = candidates[:5]
        print(f"  [scan] {len(candidates)} candidates, top: "
              f"{[(v, f, w, h) for v, f, w, h, _ in top]}", flush=True)
    return candidates


def run_inference_worker(reader, i_frame, est, ob_key, device, metrics,
                         model_pts, diameter_m, detect_type="mask",
                         pred_translation_scale=1.0,
                         current_eval_idx=None, total_eval_frames=None):
    color = reader.get_color(i_frame)
    depth = reader.get_depth(i_frame)
    if color is None or depth is None:
        return None, None, None, None, None
    ob_mask = get_object_mask(reader, i_frame, ob_key, detect_type)
    if ob_mask is None or ob_mask.sum() < 4:
        return None, None, None, None, None
    gt_pose = reader.get_gt_pose(i_frame, ob_key)
    if gt_pose is None:
        return None, None, None, None, None

    pred_pose = est.register(K=reader.K, rgb=color, depth=depth,
                             ob_mask=ob_mask, ob_id=ob_key, iteration=5)
    if pred_pose is None:
        raise RuntimeError("register 返回 None")
    pred_np = (pred_pose.detach().cpu().numpy() if torch.is_tensor(pred_pose)
               else np.asarray(pred_pose))
    pred_np = np.asarray(pred_np, dtype=np.float64).copy()
    if pred_np.shape != (4, 4):
        raise ValueError(f"预测位姿应为 (4,4)，实际 {pred_np.shape}")
    if not np.isfinite(pred_np).all():
        raise ValueError("预测位姿含 NaN/Inf")
    pred_np[:3, 3] *= pred_translation_scale
    pred_pose = pred_np

    err_dict = metrics.compute_all_errors(
        pred_np, gt_pose, model_pts, ob_key, diameter_m)
    rot_err = err_dict["rot_deg"]
    trans_mm = err_dict["trans_mm"]
    add_norm = err_dict["add_norm"]
    adds_norm = err_dict["adds_norm"]

    idx_str = (f"{current_eval_idx}/{total_eval_frames}"
               if current_eval_idx is not None and total_eval_frames is not None
               else (f"{current_eval_idx}" if current_eval_idx is not None
                     else f"{i_frame}"))
    add_ok = bool(np.isfinite(add_norm) and add_norm <= 0.1)
    adds_ok = bool(np.isfinite(adds_norm) and adds_norm <= 0.1)
    add_str = f"{add_norm:.6f}" if np.isfinite(add_norm) else "nan"
    adds_str = f"{adds_norm:.6f}" if np.isfinite(adds_norm) else "nan"

    print(f"\n[Frame {idx_str} | Video {reader._dataset_dir.split('/')[-1]} | "
          f"RawFrame {i_frame} | Key {ob_key}]", flush=True)
    print(f"Rot: {rot_err:.2f} deg | Trans: {trans_mm:.2f} mm | "
          f"ADD@0.1d: {'PASS' if add_ok else 'FAIL'} ({add_str}) | "
          f"ADD-S@0.1d: {'PASS' if adds_ok else 'FAIL'} ({adds_str})", flush=True)

    return pred_pose, gt_pose, err_dict, rot_err, trans_mm


def _count_eval_frames(eval_v_idx_list, ob_key, ref_v_idx, ref_k,
                       skip_ref, eval_stride, max_eval_global,
                       max_eval_per_scene, reader_cache, real275_dir):
    """预统计 total_eval_frames，与实际评估循环的过滤条件保持一致。"""
    total = 0
    target_seq_count = 0
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
    return total


def run_pipeline(detect_type="mask"):
    global _INTERNAL_OB_KEY_MODE, _GLOBAL_MODEL_ID_TO_NAME

    skip_ref = bool(opt.eval_skip_ref)
    eval_stride = max(1, opt.eval_stride)
    ref_stride = max(1, opt.ref_stride)
    max_eval_global = int(opt.max_eval_frames)
    max_eval_per_scene = int(opt.max_eval_frames_per_scene)

    _pt("[pipeline] CUDA preflight ...")
    try:
        if not torch.cuda.is_available():
            _pt("[pipeline] WARNING: torch.cuda.is_available() == False")
        else:
            _pt(f"[pipeline] CUDA device 0: {torch.cuda.get_device_name(0)}")
            _ = torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
            _pt("[pipeline] CUDA preflight OK")
    except Exception as e:
        _pt(f"[pipeline] CUDA preflight FAILED: {type(e).__name__}: {e}")
        raise

    metrics = PoseMetrics()
    video_dirs = _discover_video_dirs(opt.real275_dir)
    if not video_dirs:
        raise SystemExit(f"[pipeline] No videos under {opt.real275_dir}/real_test/")

    registry_json_path = opt.registry_json or os.path.join(
        os.path.abspath(opt.real275_dir), "obj_models", "model_registry.json")
    inventory_log = os.path.join(opt.debug_dir, "inventory.log") if opt.debug_dir else None

    reader_cache_tmp = {}
    cached = None
    if (not opt.rebuild_registry) and os.path.isfile(registry_json_path):
        cached = _load_registry_json(registry_json_path,
                                     expected_min_visib=opt.min_visib)

    inventory = None
    if cached is None:
        print(f"[pipeline] registry JSON 未命中，扫描数据集: {registry_json_path}",
              flush=True)
        _INTERNAL_OB_KEY_MODE = "model_name"
        _GLOBAL_MODEL_ID_TO_NAME = None
        _report_inventory(opt.real275_dir, log_path=inventory_log,
                          cache=reader_cache_tmp)
        model_id_to_name, model_name_to_id, model_info = \
            _build_model_id_registry(reader_cache_tmp, video_dirs)
    else:
        inventory, model_id_to_name, model_name_to_id, model_info = cached
        print(f"[pipeline] 从 registry JSON 加载: {registry_json_path}", flush=True)
        _print_inventory_from_json(inventory, log_path=inventory_log)

    _print_model_id_table(model_info)

    if opt.ob_key_mode == "model_id":
        _INTERNAL_OB_KEY_MODE = "model_id"
        _GLOBAL_MODEL_ID_TO_NAME = model_id_to_name
        reader_cache = {}
    elif opt.ob_key_mode == "model_name":
        _INTERNAL_OB_KEY_MODE = "model_name"
        _GLOBAL_MODEL_ID_TO_NAME = None
        reader_cache = reader_cache_tmp
    else:
        _INTERNAL_OB_KEY_MODE = opt.ob_key_mode
        _GLOBAL_MODEL_ID_TO_NAME = None
        reader_cache = {}

    print(f"[pipeline] Evaluating {len(video_dirs)} videos, "
          f"user_mode={opt.ob_key_mode}, internal_mode={_INTERNAL_OB_KEY_MODE}, "
          f"crop_ratio={opt.crop_ratio}, eval_stride={eval_stride}, "
          f"ref_stride={ref_stride}, min_visib={opt.min_visib}, "
          f"pred_scale={opt.pred_translation_scale}, "
          f"max_eval_global={max_eval_global}, "
          f"max_eval_per_scene={max_eval_per_scene}, "
          f"obj_models_dir={opt.obj_models_dir or '<auto>'}", flush=True)

    ob_keys = []

    def _model_ids_from_opt():
        if opt.model_ids:
            if opt.model_ids.strip().lower() == "all":
                return sorted(model_info.keys())
            return [int(x.strip()) for x in opt.model_ids.split(",") if x.strip()]
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
            ids = _model_ids_from_opt()
            if not ids:
                ids = sorted(model_info.keys())
                _pt(f"[pipeline] 未指定 model_id，默认评估全部 {len(ids)} 个")
            for i in ids:
                if int(i) in model_id_to_name:
                    ob_keys.append(int(i))
                else:
                    print(f"[pipeline] WARNING: model_id={i} 不存在，跳过", flush=True)
        if not ob_keys:
            raise SystemExit("[pipeline] no valid model_id")
    elif opt.ob_key_mode == "model_name":
        if opt.ob_names:
            ob_keys = [s.strip() for s in opt.ob_names.split(",") if s.strip()]
        elif opt.model_ids or (opt.model_id is not None and opt.model_id >= 0):
            ids = _model_ids_from_opt()
            for i in ids:
                if int(i) in model_id_to_name:
                    ob_keys.append(model_id_to_name[int(i)])
        else:
            ob_keys = sorted(model_name_to_id.keys())
            _pt(f"[pipeline] model_name 模式默认评估全部 {len(ob_keys)} 个")
    else:
        if opt.ob_ids:
            ob_keys = [int(v) for v in opt.ob_ids]
        elif opt.ob_id is not None and opt.ob_id >= 0:
            ob_keys = [int(opt.ob_id)]
        else:
            raise SystemExit("[pipeline] class_id/gt_idx 模式需要 --ob_id 或 --ob_ids")

    if not ob_keys:
        raise SystemExit("[pipeline] no ob_keys")

    multi_ob = len(ob_keys) > 1
    print(f"[pipeline] {len(ob_keys)} key(s): {ob_keys}", flush=True)

    if opt.ob_key_mode in ("model_id", "model_name"):
        _pt("\n[INFO] 以 model_name 为语义单位：参考帧/mesh/直径与 model 一一对应")

    all_errors = {}
    all_res = NestDict()

    for i_ob, ob_key in enumerate(ob_keys):
        if opt.ob_key_mode == "model_id":
            model_id = int(ob_key)
            model_name = model_id_to_name[model_id]
            key_label = f"model_id={model_id} | {model_name}"
        elif opt.ob_key_mode == "model_name":
            model_name = ob_key
            model_id = model_name_to_id.get(model_name)
            key_label = f"model_id={model_id} | {model_name}" if model_id else model_name
        else:
            model_name = None
            model_id = None
            key_label = str(ob_key)

        print(f"\n========== [key {i_ob + 1}/{len(ob_keys)}] {key_label} ==========",
              flush=True)

        candidate_scene_ids = None
        if inventory is not None and model_id is not None and model_id in model_info:
            candidate_scene_ids = list(model_info[model_id].get("scene_ids", []))

        eval_v_idx_list = []
        if candidate_scene_ids is not None:
            for v_idx in candidate_scene_ids:
                if v_idx < 0 or v_idx >= len(video_dirs):
                    continue
                r = _get_reader(opt.real275_dir, v_idx, reader_cache)
                if len(r.frames_for_obj(ob_key)) > 0:
                    eval_v_idx_list.append(v_idx)
        else:
            for v_idx in range(len(video_dirs)):
                r = _get_reader(opt.real275_dir, v_idx, reader_cache)
                if len(r.frames_for_obj(ob_key)) > 0:
                    eval_v_idx_list.append(v_idx)

        if not eval_v_idx_list:
            print(f"[pipeline] WARNING: key={ob_key} 不存在于任何 scene，skip",
                  flush=True)
            continue

        eval_scene_names = [reader_cache[v].get_video_id() for v in eval_v_idx_list]
        scene_pairs = ", ".join(f"{s}({n})"
                                for s, n in zip(eval_v_idx_list, eval_scene_names))
        print(f"[pipeline] key={ob_key} 评估覆盖 scene: [{scene_pairs}]", flush=True)

        # ---- 决定参考 scene ----
        if opt.ref_video_idx >= 0:
            if opt.ref_video_idx in eval_v_idx_list:
                ref_scene = opt.ref_video_idx
                print(f"[pipeline] ref_scene={ref_scene} "
                      f"({reader_cache[ref_scene].get_video_id()}) [用户指定]",
                      flush=True)
            else:
                ref_scene = min(eval_v_idx_list)
                print(f"[pipeline] WARN: --ref_video_idx={opt.ref_video_idx} "
                      f"不含 key={ob_key}，回退到 min(scene_ids)={ref_scene} "
                      f"({reader_cache[ref_scene].get_video_id()})", flush=True)
        else:
            ref_scene = min(eval_v_idx_list)
            print(f"[pipeline] ref_scene={ref_scene} "
                  f"({reader_cache[ref_scene].get_video_id()}) [auto=min scene_id]",
                  flush=True)

        ref_frame = opt.ref_frame
        if ref_frame >= 0:
            print(f"[pipeline] ref_frame={ref_frame} (精确匹配)", flush=True)
        else:
            print(f"[pipeline] ref_frame=auto, ref_stride={ref_stride}", flush=True)

        candidates = _find_ref_candidates(
            opt.real275_dir, ob_key, ref_scene,
            ref_frame=ref_frame, ref_stride=ref_stride,
            cache=reader_cache, verbose=True,
        )
        if not candidates:
            print(f"[pipeline] WARNING: no usable ref for key={ob_key}, skip",
                  flush=True)
            continue

        est = None
        ref_v_idx = ref_k = ref_w = ref_h = None

        for ci, (v_idx, frame, w, h, area) in enumerate(candidates):
            _pt(f"  [ref try {ci+1}/{len(candidates)}] "
                f"v{v_idx:02d} f{frame:04d} (bbox {w}x{h}, area={area})")
            reader_ref = _get_reader(opt.real275_dir, v_idx, reader_cache)
            probe_dir = None
            if opt.save_ref_probe:
                probe_dir = (os.path.join(opt.save_ref_probe,
                                          f"key_{_sanitize_key(ob_key)}")
                             if multi_ob else opt.save_ref_probe)
                os.makedirs(probe_dir, exist_ok=True)

            try:
                ref_mesh = reader_ref.get_gt_mesh(ob_key, frame=frame)
                mesh_diameter_m = float(np.linalg.norm(ref_mesh.extents))
            except Exception as e:
                _pt(f"  [ref] WARN: mesh 加载失败 ({type(e).__name__}: {e})")
                mesh_diameter_m = None

            save_to_disk = bool(opt.save_ref) and bool(opt.ref_data_root)
            try:
                cand_est = OneRefPose(scorer=None, refiner=None,
                                      ref_data_root=opt.ref_data_root,
                                      ob_id=ob_key)
                cand_est.to_device("cuda:0")
                cand_est.use_frame_as_ref(
                    reader_ref, frame, ob_key,
                    is_symmetric=False,
                    symmetry_axis=opt.ref_sym_axis,
                    crop_ratio=opt.crop_ratio,
                    device=opt.ref_device,
                    save_to_disk=save_to_disk,
                    mesh_diameter=mesh_diameter_m,
                    crop_to_ref=opt.ref_crop_to_ref,
                    probe_dir=probe_dir,
                )
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
            print(f"[pipeline] ERROR: all ref candidates failed for key={ob_key}, skip",
                  flush=True)
            continue

        print("\n" + "=" * 50, flush=True)
        print(f"Selected Ref: Video [{ref_v_idx:02d}] Frame [{ref_k:04d}] "
              f"bbox {ref_w}x{ref_h}", flush=True)
        print("=" * 50 + "\n", flush=True)

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

        if (getattr(est, 'ref_data', None) is not None
                and getattr(est.ref_data, 'rgb', None) is not None):
            crop_rgb = (est.ref_data.rgb.detach().cpu().permute(1, 2, 0)
                        .numpy().astype(np.uint8))
            crop_mask = ((est.ref_data.xyz_map[2] > 1e-5).cpu().numpy()
                         .astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(
                abs_save_vis_dir,
                f"ref_ob_{sk}_v{ref_v_idx:02d}_f{ref_k:04d}_crop_rgb.png"),
                cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(
                abs_save_vis_dir,
                f"ref_ob_{sk}_v{ref_v_idx:02d}_f{ref_k:04d}_crop_mask.png"),
                crop_mask)
            overlay = crop_rgb.copy()
            overlay[crop_mask > 0] = [0, 255, 0]
            blended = cv2.addWeighted(overlay, 0.3, crop_rgb, 0.7, 0)
            cv2.imwrite(os.path.join(
                abs_save_vis_dir,
                f"ref_ob_{sk}_v{ref_v_idx:02d}_f{ref_k:04d}_crop_overlay.png"),
                cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

        full_rgb = reader_ref.get_color(ref_k)
        full_mask = reader_ref.get_mask(ref_k, ob_key)
        if full_rgb is not None and full_mask is not None:
            full_bgr = cv2.cvtColor(full_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(
                abs_save_vis_dir,
                f"ref_ob_{sk}_v{ref_v_idx:02d}_f{ref_k:04d}_full_rgb.png"), full_bgr)
            cv2.imwrite(os.path.join(
                abs_save_vis_dir,
                f"ref_ob_{sk}_v{ref_v_idx:02d}_f{ref_k:04d}_full_mask.png"), full_mask)
            full_overlay = full_bgr.copy()
            full_overlay[full_mask > 0] = [0, 255, 0]
            blended_full = cv2.addWeighted(full_overlay, 0.4, full_bgr, 0.6, 0)
            vs, us = np.where(full_mask > 0)
            if len(vs) > 0:
                u0, v0, u1, v1 = (int(us.min()), int(vs.min()),
                                  int(us.max()), int(vs.max()))
                cv2.rectangle(blended_full, (u0, v0), (u1, v1), (0, 0, 255), 2)
            cv2.imwrite(os.path.join(
                abs_save_vis_dir,
                f"ref_ob_{sk}_v{ref_v_idx:02d}_f{ref_k:04d}_full_overlay.png"),
                blended_full)

        try:
            model_pts = reader_ref.get_gt_mesh(ob_key, frame=ref_k).vertices.copy()
            diameter_m = float(np.linalg.norm(model_pts.max(0) - model_pts.min(0)))
            _pt(f"  [mesh] {model_pts.shape[0]} verts, diameter={diameter_m:.4f} m")
        except Exception as e:
            _pt(f"  [mesh] WARN: {e}，使用占位")
            model_pts = trimesh.creation.box(extents=(0.2, 0.2, 0.2)).vertices.copy()
            diameter_m = float(np.linalg.norm(model_pts.max(0) - model_pts.min(0)))

        all_errors[ob_key] = {
            "trans": [], "rot": [], "n_fail": 0,
            "add_norm": [], "adds_norm": [],
            "diameter": float(diameter_m),
            "scenes": list(eval_v_idx_list),
            "scene_names": list(eval_scene_names),
            "model_id": model_id, "model_name": model_name,
            "per_scene": {v_idx: {"trans": [], "rot": [], "n_fail": 0,
                                   "add_norm": [], "adds_norm": []}
                          for v_idx in eval_v_idx_list},
        }

        # ---- 预统计 total_eval_frames（跨所有含该 model 的 scene）----
        total_eval_frames = _count_eval_frames(
            eval_v_idx_list, ob_key, ref_v_idx, ref_k,
            skip_ref, eval_stride, max_eval_global,
            max_eval_per_scene, reader_cache, opt.real275_dir)
        print(f"  [eval] key={ob_key} total_eval_frames={total_eval_frames} "
              f"(scenes={len(eval_v_idx_list)}, eval_stride={eval_stride}, "
              f"max_eval_global={max_eval_global}, "
              f"max_eval_per_scene={max_eval_per_scene})", flush=True)

        if total_eval_frames <= 0:
            print(f"  [eval] 无可评估帧，skip key={ob_key}", flush=True)
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
                pbar.set_postfix({"v": f"{v_idx:02d}", "f": f"{k:04d}",
                                  "n": f"{evaluated_count + 1}/{total_eval_frames}"})

                try:
                    pred_pose, gt_pose, err_dict, rot_err, trans_mm = \
                        run_inference_worker(
                            r, k, est, ob_key, "cuda:0", metrics,
                            model_pts, diameter_m, detect_type,
                            pred_translation_scale=opt.pred_translation_scale,
                            current_eval_idx=evaluated_count + 1,
                            total_eval_frames=total_eval_frames,
                        )
                except Exception:
                    print(f"\n[register error] v{v_idx:02d} f{k:04d}", flush=True)
                    traceback.print_exc()
                    all_errors[ob_key]["trans"].append(float("inf"))
                    all_errors[ob_key]["rot"].append(float("inf"))
                    all_errors[ob_key]["add_norm"].append(float("inf"))
                    all_errors[ob_key]["adds_norm"].append(float("inf"))
                    all_errors[ob_key]["per_scene"][v_idx]["trans"].append(float("inf"))
                    all_errors[ob_key]["per_scene"][v_idx]["rot"].append(float("inf"))
                    all_errors[ob_key]["per_scene"][v_idx]["add_norm"].append(float("inf"))
                    all_errors[ob_key]["per_scene"][v_idx]["adds_norm"].append(float("inf"))
                    all_errors[ob_key]["n_fail"] += 1
                    evaluated_count += 1
                    scene_evaluated += 1
                    pbar.update(1)
                    continue

                if err_dict is None:
                    all_errors[ob_key]["trans"].append(float("inf"))
                    all_errors[ob_key]["rot"].append(float("inf"))
                    all_errors[ob_key]["add_norm"].append(float("inf"))
                    all_errors[ob_key]["adds_norm"].append(float("inf"))
                    all_errors[ob_key]["per_scene"][v_idx]["trans"].append(float("inf"))
                    all_errors[ob_key]["per_scene"][v_idx]["rot"].append(float("inf"))
                    all_errors[ob_key]["per_scene"][v_idx]["add_norm"].append(float("inf"))
                    all_errors[ob_key]["per_scene"][v_idx]["adds_norm"].append(float("inf"))
                    all_errors[ob_key]["n_fail"] += 1
                    evaluated_count += 1
                    scene_evaluated += 1
                    pbar.update(1)
                    continue

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

        if opt.debug_dir:
            os.makedirs(opt.debug_dir, exist_ok=True)
            for name, arr in [("trans_mm", all_errors[ob_key]["trans"]),
                              ("rot_deg", all_errors[ob_key]["rot"]),
                              ("add_norm", all_errors[ob_key]["add_norm"]),
                              ("adds_norm", all_errors[ob_key]["adds_norm"])]:
                fmt = "%.2f" if name in ("trans_mm", "rot_deg") else "%.6f"
                np.savetxt(os.path.join(opt.debug_dir,
                                        f"errors_ob_{sk}_{name}.txt"),
                           np.array(arr), fmt=fmt)

        del est
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    _finalize(all_errors)

    if opt.debug_dir:
        os.makedirs(opt.debug_dir, exist_ok=True)
        with open(os.path.join(opt.debug_dir, "real275_res.yml"), "w") as f:
            _pyyaml.safe_dump(make_yaml_dumpable(all_res), f)
        with open(os.path.join(opt.debug_dir, "model_scene_coverage.txt"),
                  "w", encoding="utf-8") as f:
            f.write("# model_id | model_name | scene_id(scene_name) | n_eval_frames\n")
            for ob_key, v in all_errors.items():
                scenes = v.get("scenes", [])
                names = v.get("scene_names", [])
                pairs = ", ".join(f"{s}({n})" for s, n in zip(scenes, names))
                f.write(f"model_id={v.get('model_id')} | "
                        f"{v.get('model_name')}: [{pairs}]  "
                        f"n_frames={len(v['trans'])}\n")
        print(f"\n[pipeline] Final results saved to {opt.debug_dir}", flush=True)


def _finalize(all_errors):
    print("\n========== REAL275 位姿评估 ==========", flush=True)
    if not all_errors:
        return

    per_obj = []
    for ob_key, v in sorted(all_errors.items(), key=lambda kv: str(kv[0])):
        if not v["trans"]:
            continue
        trans = np.asarray(v["trans"], dtype=np.float64)
        rot = np.asarray(v["rot"], dtype=np.float64)
        add_n = np.asarray(v.get("add_norm", []), dtype=np.float64)
        adds_n = np.asarray(v.get("adds_norm", []), dtype=np.float64)
        diam = float(v.get("diameter", float("nan")))
        scenes = v.get("scenes", [])
        scene_names = v.get("scene_names", [])
        mid = v.get("model_id")
        mname = v.get("model_name")
        per_scene = v.get("per_scene", {})

        finite_tr = np.isfinite(trans) & np.isfinite(rot)
        n_total = len(trans)
        n_ok = int(finite_tr.sum())
        n_fail = int((~finite_tr).sum())
        mean_trans = float(trans[finite_tr].mean()) if n_ok > 0 else float("nan")
        mean_rot = float(rot[finite_tr].mean()) if n_ok > 0 else float("nan")

        f_add = np.isfinite(add_n)
        f_adds = np.isfinite(adds_n)
        mean_add = float(add_n[f_add].mean()) if f_add.any() else float("nan")
        mean_adds = float(adds_n[f_adds].mean()) if f_adds.any() else float("nan")

        sr_add_01 = float((add_n <= 0.1).mean() * 100.0) if len(add_n) else 0.0
        sr_adds_01 = float((adds_n <= 0.1).mean() * 100.0) if len(adds_n) else 0.0
        sr_add_02 = float((add_n <= 0.2).mean() * 100.0) if len(add_n) else 0.0
        sr_adds_02 = float((adds_n <= 0.2).mean() * 100.0) if len(adds_n) else 0.0
        sr_add_005 = float((add_n <= 0.05).mean() * 100.0) if len(add_n) else 0.0
        sr_adds_005 = float((adds_n <= 0.05).mean() * 100.0) if len(adds_n) else 0.0
        auc_add_01 = compute_auc_norm(add_n, max_val=0.1)
        auc_adds_01 = compute_auc_norm(adds_n, max_val=0.1)
        acc_10 = compute_acc(trans, 10.0)
        acc_20 = compute_acc(trans, 20.0)
        joint_5d_5cm = (float(np.mean((rot < 5.0) & (trans < 50.0)) * 100.0)
                        if n_total > 0 else 0.0)
        joint_10d_5cm = (float(np.mean((rot < 10.0) & (trans < 50.0)) * 100.0)
                         if n_total > 0 else 0.0)

        per_obj.append({
            "ob_key": ob_key, "model_id": mid, "model_name": mname,
            "n_total": n_total, "n_ok": n_ok, "n_fail": n_fail,
            "scenes": scenes, "scene_names": scene_names,
            "mean_trans": mean_trans, "mean_rot": mean_rot,
            "mean_add_norm": mean_add, "mean_adds_norm": mean_adds,
            "sr_add_005": sr_add_005, "sr_add_01": sr_add_01, "sr_add_02": sr_add_02,
            "sr_adds_005": sr_adds_005, "sr_adds_01": sr_adds_01, "sr_adds_02": sr_adds_02,
            "auc_add_01": auc_add_01, "auc_adds_01": auc_adds_01,
            "acc_10": acc_10, "acc_20": acc_20,
            "joint_5d_5cm": joint_5d_5cm, "joint_10d_5cm": joint_10d_5cm,
        })

        scene_pairs = ", ".join(f"{s}({n})" for s, n in zip(scenes, scene_names))
        key_label = (f"model_id={mid} | {mname}"
                     if mid is not None and mname is not None else str(ob_key))
        print(f"\n========== [key {key_label}] ==========", flush=True)
        print(f" 涉及 scene: [{scene_pairs}]  mesh_diameter={diam:.4f} m", flush=True)
        print(f" 计划评估: {n_total}  成功: {n_ok}  失败: {n_fail}", flush=True)
        if n_ok > 0:
            print(f"  成功样本均值: Trans {mean_trans:.2f} mm | Rot {mean_rot:.2f} deg",
                  flush=True)
        print(f"  ADD  Mean: {mean_add:.6f}", flush=True)
        print(f"  ADD-S Mean: {mean_adds:.6f}", flush=True)
        print(f"  ADD@0.05/0.1/0.2d: {sr_add_005:.2f}% / {sr_add_01:.2f}% / "
              f"{sr_add_02:.2f}%", flush=True)
        print(f"  ADD-S@0.05/0.1/0.2d: {sr_adds_005:.2f}% / {sr_adds_01:.2f}% / "
              f"{sr_adds_02:.2f}%", flush=True)
        print(f"  AUC@0.1d: ADD={auc_add_01:.4f}  ADD-S={auc_adds_01:.4f}", flush=True)
        print(f"  Recall@10mm: {acc_10:.2f}%  @20mm: {acc_20:.2f}%", flush=True)
        print(f"  Joint (5d&50mm): {joint_5d_5cm:.2f}%  "
              f"(10d&50mm): {joint_10d_5cm:.2f}%", flush=True)

        if per_scene:
            print(f"\n  [per-scene breakdown]", flush=True)
            for v_idx in sorted(per_scene):
                ps = per_scene[v_idx]
                if not ps.get("trans"):
                    continue
                s_trans = np.asarray(ps["trans"], dtype=np.float64)
                s_rot = np.asarray(ps["rot"], dtype=np.float64)
                s_add = np.asarray(ps["add_norm"], dtype=np.float64)
                s_adds = np.asarray(ps["adds_norm"], dtype=np.float64)
                s_finite = np.isfinite(s_trans) & np.isfinite(s_rot)
                s_n_total = len(s_trans)
                s_n_ok = int(s_finite.sum())
                s_n_fail = int((~s_finite).sum())
                s_mt = float(s_trans[s_finite].mean()) if s_n_ok else float("nan")
                s_mr = float(s_rot[s_finite].mean()) if s_n_ok else float("nan")
                s_sr01 = float((s_add <= 0.1).mean() * 100.0) if len(s_add) else 0.0
                s_srs01 = float((s_adds <= 0.1).mean() * 100.0) if len(s_adds) else 0.0
                s_name = scene_names[scenes.index(v_idx)] if v_idx in scenes else str(v_idx)
                print(f"    scene_id={v_idx} ({s_name}): "
                      f"n={s_n_total} ok={s_n_ok} fail={s_n_fail} | "
                      f"Trans={s_mt:.2f}mm Rot={s_mr:.2f}deg | "
                      f"ADD@0.1d={s_sr01:.2f}% ADD-S@0.1d={s_srs01:.2f}%", flush=True)

    if not per_obj:
        return

    print(f"\n========== [MODEL COVERAGE] ==========", flush=True)
    for p in per_obj:
        scene_pairs = ", ".join(f"{s}({n})" for s, n in zip(p["scenes"], p["scene_names"]))
        mid = p.get("model_id")
        mname = p.get("model_name")
        mid_str = f"model_id={mid}" if mid is not None else "model_id=N/A"
        print(f"  {mid_str} | {mname}: scene_count={len(p['scenes'])}, "
              f"scenes=[{scene_pairs}], n_frames={p['n_total']}", flush=True)

    def _mean(key):
        vals = [p[key] for p in per_obj if np.isfinite(p[key])]
        return float(np.mean(vals)) if vals else float("nan")

    print(f"\n========== [OVERALL MEAN] (per-object equal weight) ==========", flush=True)
    print(f" Objects: {len(per_obj)}", flush=True)
    print(f"   Mean Trans: {_mean('mean_trans'):.2f} mm", flush=True)
    print(f"   Mean Rot: {_mean('mean_rot'):.2f} deg", flush=True)
    print(f"   Mean ADD: {_mean('mean_add_norm'):.6f}", flush=True)
    print(f"   Mean ADD-S: {_mean('mean_adds_norm'):.6f}", flush=True)
    print(f"   ADD@0.1d: {_mean('sr_add_01'):.2f}%  ADD-S@0.1d: {_mean('sr_adds_01'):.2f}%",
          flush=True)
    print(f"   AUC@0.1d: ADD={_mean('auc_add_01'):.4f}  ADD-S={_mean('auc_adds_01'):.4f}",
          flush=True)
    print(f"   Recall@10mm: {_mean('acc_10'):.2f}%  @20mm: {_mean('acc_20'):.2f}%",
          flush=True)
    print(f"   Joint (5d&50mm): {_mean('joint_5d_5cm'):.2f}%  "
          f"(10d&50mm): {_mean('joint_10d_5cm'):.2f}%", flush=True)


def inspect_gt(real275_dir):
    root = os.path.join(real275_dir, "gts", "real_test")
    files = sorted(glob.glob(os.path.join(root, "*.pkl")))
    print(f"[inspect] 找到 {len(files)} 个 pkl 于 {root}")
    if not files:
        return
    with open(files[0], "rb") as f:
        d = pickle.load(f)
    print(f"\n===== {files[0]} =====")
    for k, v in d.items():
        try:
            arr = np.asarray(v)
            print(f"{k:20s} type={type(v).__name__:15s} "
                  f"shape={arr.shape} dtype={arr.dtype}")
            if arr.size <= 20:
                print(f"    values: {arr}")
            else:
                print(f"    first 5: {arr.flatten()[:5]}")
        except Exception as e:
            print(f"{k:20s} type={type(v).__name__}  (cannot array-ify: {e})")

    base_name = os.path.basename(files[0])
    m = re.search(r"_scene_(\d+)_(\d+)\.pkl$", base_name)
    if m is None:
        return
    scene_id_str, frame_id = m.group(1), int(m.group(2))
    scene_dir = os.path.join(real275_dir, "real_test", f"scene_{scene_id_str}")
    meta_path = os.path.join(scene_dir, f"{frame_id:04d}_meta.txt")
    print(f"\n[inspect] meta.txt: {meta_path}")
    inst_to_info = {}
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as f:
            for i, line in enumerate(f):
                print(f"  line {i}: {line.rstrip()}")
                if i > 10:
                    print("  ...")
                    break
        inst_to_info = parse_meta_models(meta_path)
        print(f"  parsed inst_id -> (class_id, model_name):")
        for inst_id in sorted(inst_to_info):
            info = inst_to_info[inst_id]
            print(f"    inst={inst_id}: class_id={info['class_id']}, "
                  f"model_name={info['model_name']}")
    else:
        print(f"  (不存在！)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real275_dir", type=str, required=True)
    parser.add_argument("--model_id", type=int, default=None)
    parser.add_argument("--model_ids", type=str, default=None)
    parser.add_argument("--ob_id", type=int, default=None)
    parser.add_argument("--ob_ids", type=str, default=None)
    parser.add_argument("--ob_names", type=str, default=None)
    parser.add_argument("--ref_video_idx", type=int, default=-1,
                        help="参考 scene；未指定则用含该 model 的 min(scene_id)")
    parser.add_argument("--ref_frame", type=int, default=-1,
                        help="参考帧编号；在 ref_scene 内精确匹配")
    parser.add_argument("--ref_stride", type=int, default=1)
    parser.add_argument("--cam_K", type=str, default=None)
    parser.add_argument("--obj_models_dir", type=str, default=None)
    parser.add_argument("--eval_skip_ref", action="store_true", default=True)
    parser.add_argument("--eval_include_ref", dest="eval_skip_ref",
                        action="store_false")
    parser.add_argument("--eval_stride", type=int, default=1)
    parser.add_argument("--max_eval_frames", type=int, default=0)
    parser.add_argument("--max_eval_frames_per_scene", type=int, default=0)
    parser.add_argument("--ref_data_root", type=str, default=None)
    parser.add_argument("--save_ref", action="store_true")
    parser.add_argument("--save_ref_probe", type=str, default=None)
    parser.add_argument("--ref_sym_axis", type=int, default=-1)
    parser.add_argument("--ref_device", type=str, default="cuda")
    parser.add_argument("--ref_crop_to_ref", action="store_true", default=True)
    parser.add_argument("--no_ref_crop_to_ref", dest="ref_crop_to_ref",
                        action="store_false")
    parser.add_argument("--stop_on_ref_fail", action="store_true", default=False)
    parser.add_argument("--detect_type", type=str, default="mask",
                        choices=["mask", "box"])
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

    opt = parser.parse_args()
    if opt.ob_ids:
        opt.ob_ids = [int(x) for x in opt.ob_ids.split(",")]
    if opt.cam_K:
        vals = [float(x) for x in opt.cam_K.replace(",", " ").split()]
        opt.cam_K = np.array(vals).reshape(3, 3)
    else:
        opt.cam_K = None

    set_seed(0)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
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