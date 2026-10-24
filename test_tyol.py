# test_tyol.py
# -*- coding: utf-8 -*-
"""
TYOL BOP AR evaluation using OneRefPose (from pose_calculation)。
不含 FoundationPose 代码。参考 test_real275.py 的 AR 实现。

主指标：BOP AR = mean(VSD_AR, MSSD_AR, MSPD_AR)，始终计算。
辅助参考：ADD / ADD-S 的 AUC 与 @10cm（--no_compute_adds 可关闭）

AR 硬约束（与 test_real275 一致）：
  #1 数据错误立即 raise（模型/直径/GT/mask 缺失）
  #2 pred=None 计零召回（inf，不删样本）
  #3 指标异常 → NaN，不算 0 分；完整 AR 需三项齐全
  #4 全目标集 + 完整子集两套 AR 都报
"""

from Utils import *
import os, sys, json, time, argparse, glob
import logging, faulthandler, traceback, tempfile
from pathlib import Path
import numpy as np
import torch
import trimesh
import yaml
import cv2
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm

from pose_calculation import OneRefPose


# =========================================================
# BOP 库
# =========================================================
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


def _sync_cuda():
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _pt(msg):
    print(msg, flush=True)


def _pose_m_to_mm(pose_m):
    p = np.asarray(pose_m, dtype=np.float64).copy()
    p[:3, 3] *= 1000.0
    return p


# =========================================================
# TYOL reader（米制接口）
# =========================================================
class TyolSceneReader:
    def __init__(self, scene_dir, models_dir, min_visib=8, min_visib_frac=0.0):
        self.video_dir = scene_dir
        self.models_dir = models_dir
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

        self._rgb_map = self._build_frame_map('rgb')
        self._depth_map = self._build_frame_map('depth')
        self._frames = sorted(set(self._rgb_map) & set(self._depth_map))
        self.id_strs = {i: f'{i:06d}' for i in self._frames}

        ob_set = set()
        for entries in self._scene_gt.values():
            for g in entries:
                ob_set.add(int(g['obj_id']))
        self.ob_ids = sorted(ob_set)

        self._multi_instance = {}
        for img_id, entries in self._scene_gt.items():
            counts = {}
            for gt_idx, g in enumerate(entries):
                oid = int(g['obj_id'])
                counts.setdefault(oid, []).append(gt_idx)
            for oid, idxs in counts.items():
                if len(idxs) > 1:
                    self._multi_instance[(img_id, oid)] = idxs

    def get_video_id(self):
        return os.path.basename(self.video_dir.rstrip('/'))

    def get_color(self, i):
        p = self._rgb_map.get(i)
        if p is None:
            return None
        bgr = cv2.imread(p)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None

    def get_depth(self, i):
        p = self._depth_map.get(i)
        if p is None:
            return None
        d = cv2.imread(p, -1)
        if d is None:
            return None
        scale = float(self._scene_camera.get(i, {}).get('depth_scale', 1.0))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid depth_scale={scale} @ {self.video_dir} f={i}")
        return d.astype(np.float32) * scale / 1000.0

    def get_K(self, i):
        if i in self._scene_camera and 'cam_K' in self._scene_camera[i]:
            return np.array(self._scene_camera[i]['cam_K'],
                            dtype=np.float64).reshape(3, 3)
        return self.K

    def _gt_idx(self, i, oid):
        matches = [j for j, g in enumerate(self._scene_gt.get(i, []))
                   if int(g['obj_id']) == oid]
        if not matches:
            return None
        return matches[0]

    def get_gt_pose(self, i, oid):
        j = self._gt_idx(i, oid)
        if j is None:
            return None
        g = self._scene_gt[i][j]
        p = np.eye(4, dtype=np.float64)
        p[:3, :3] = np.array(g['cam_R_m2c']).reshape(3, 3)
        p[:3, 3] = np.array(g['cam_t_m2c']) / 1000.0
        return p

    def get_mask(self, i, oid, type='mask_visib', **kwargs):
        j = self._gt_idx(i, oid)
        if j is None:
            return None
        path = os.path.join(self.video_dir, type, f'{i:06d}_{j:06d}.png')
        return cv2.imread(path, -1) if os.path.isfile(path) else None

    def get_gt_mesh(self, oid):
        mesh = trimesh.load(os.path.join(self.models_dir, f'obj_{oid:06d}.ply'))
        mesh.apply_scale(0.001)
        return mesh

    def _visib_fract(self, i, oid):
        j = self._gt_idx(i, oid)
        if j is None:
            return 1.0
        info = self._scene_gt_info.get(i)
        if not info or j >= len(info):
            return 1.0
        return float(info[j].get('visib_fract', 1.0))

    def _mask_bbox(self, i, oid):
        m = self.get_mask(i, oid)
        if m is None:
            return None
        vs, us = np.where(m > 0)
        if len(vs) == 0:
            return None
        return int(us.min()), int(vs.min()), int(us.max()), int(vs.max())

    def frames_with_bbox(self, oid):
        out = []
        for k in sorted(self._scene_gt):
            if k not in self._rgb_map or k not in self._depth_map:
                continue
            if self._visib_fract(k, oid) < self.min_visib_frac:
                continue
            bb = self._mask_bbox(k, oid)
            if bb is None:
                continue
            u0, v0, u1, v1 = bb
            w, h = u1 - u0 + 1, v1 - v0 + 1
            if w < self.min_visib or h < self.min_visib:
                continue
            out.append((k, w * h, w, h))
        return out

    def frames_for_obj(self, oid):
        return [k for k, _, _, _ in self.frames_with_bbox(oid)]


# =========================================================
# ARAccumulator
# =========================================================
class ARAccumulator:
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
        thresholds = {"vsd": self.vsd_thetas, "mssd": self.mssd_thetas,
                      "mspd": self.mspd_thetas}
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
        result["ar"] = float(np.mean(parts)) if np.all(np.isfinite(parts)) else float("nan")
        return result


# =========================================================
# BOP AR
# =========================================================
def compute_ar_for_tyol(reader, obj_id, models_dir, models_info, ar_samples,
                        use_symmetry=True, max_frames=0, vsd_delta_mm=15.0,
                        max_sym_disc_step=0.01):
    diag_n = len(ar_samples) if ar_samples else 0

    pose_error, renderer_vispy = _try_import_bop_lib()
    if pose_error is None:
        return {"ok": False, "reason": "bop_toolkit_lib.pose_error 不可用",
                "n_targets": diag_n}
    if not ar_samples:
        return {"ok": False, "reason": "ar_samples 为空", "n_targets": 0}

    if max_frames > 0 and len(ar_samples) > max_frames:
        idx = np.linspace(0, len(ar_samples) - 1, max_frames).astype(int)
        samples = [ar_samples[i] for i in idx]
    else:
        samples = list(ar_samples)
    n = len(samples)

    obj_path = os.path.join(models_dir, f'obj_{int(obj_id):06d}.ply')
    if not os.path.isfile(obj_path):
        return {"ok": False, "reason": f"模型路径无效: {obj_path}", "n_targets": n}

    try:
        mesh_mm = trimesh.load(obj_path, force="mesh", process=False)
    except Exception as e:
        return {"ok": False, "reason": f"模型加载失败: {e}", "n_targets": n}

    model_pts_mm = np.asarray(mesh_mm.vertices, dtype=np.float64)
    if model_pts_mm.ndim != 2 or model_pts_mm.shape[1] != 3 or len(model_pts_mm) < 4:
        return {"ok": False, "reason": "模型顶点无效", "n_targets": n}

    info = models_info.get(str(obj_id), {})
    diam_mm = info.get("diameter")
    if diam_mm is None or not np.isfinite(float(diam_mm)) or float(diam_mm) <= 0:
        return {"ok": False, "reason": f"diameter 缺失/非法 ({diam_mm})", "n_targets": n}
    diam_mm = float(diam_mm)

    if use_symmetry:
        from bop_toolkit_lib import misc
        try:
            syms = misc.get_symmetry_transformations(
                info, max_sym_disc_step=max_sym_disc_step)
        except Exception as e:
            return {"ok": False, "reason": f"symmetry 解析失败: {e}", "n_targets": n}
        if not syms:
            syms = [{"R": np.eye(3, dtype=np.float32),
                     "t": np.zeros((3, 1), dtype=np.float32)}]
    else:
        syms = [{"R": np.eye(3, dtype=np.float32),
                 "t": np.zeros((3, 1), dtype=np.float32)}]

    h0, w0 = None, None
    for s in samples:
        d_m = reader.get_depth(s["im_id"])
        if d_m is not None:
            h0, w0 = d_m.shape
            break
    if h0 is None or w0 is None:
        return {"ok": False, "reason": "所有样本 depth 均读取失败", "n_targets": n}

    rnd = None
    vispy_fail_reason = None
    if renderer_vispy is not None:
        try:
            rnd = renderer_vispy.RendererVispy(w0, h0, mode="depth")
            rnd.add_object(int(obj_id), obj_path)
            _pt(f"[AR] RendererVispy OK ({w0}x{h0})")
        except Exception as e:
            vispy_fail_reason = f"RendererVispy init failed: {e}"
            rnd = None
    else:
        vispy_fail_reason = "bop_toolkit_lib.renderer_vispy 不可用"

    vsd_taus = np.arange(0.05, 0.51, 0.05)
    vsd_thetas = np.arange(0.05, 0.51, 0.05)
    mssd_thetas = np.arange(0.05, 0.51, 0.05)
    mspd_thetas = np.arange(5.0, 51.0, 5.0) * (w0 / 640.0)

    acc = ARAccumulator(n_targets=n, vsd_taus=vsd_taus,
                        vsd_thetas=vsd_thetas,
                        mssd_thetas=mssd_thetas,
                        mspd_thetas=mspd_thetas)
    if rnd is None:
        acc.mark_metric_unavailable("vsd", vispy_fail_reason or "renderer=None")

    n_pred_fail = 0

    for i, s in enumerate(samples):
        gt_m = s.get("gt_m")
        pred_m = s.get("pred_m")
        if gt_m is None:
            raise RuntimeError(f"样本缺少 GT: im_id={s.get('im_id')}")
        if pred_m is None:
            n_pred_fail += 1
            acc.mark_prediction_missing(i)
            continue

        gt_mm = _pose_m_to_mm(gt_m)
        pred_mm = _pose_m_to_mm(pred_m)
        re, te = pred_mm[:3, :3], pred_mm[:3, 3:4]
        rg, tg = gt_mm[:3, :3], gt_mm[:3, 3:4]

        def _compute_mssd(re=re, te=te, rg=rg, tg=tg):
            e = pose_error.mssd(re, te, rg, tg, model_pts_mm, syms)
            return float(e) / diam_mm
        acc.record("mssd", i, _compute_mssd)

        def _compute_mspd(sm=s, re=re, te=te, rg=rg, tg=tg):
            K = reader.get_K(sm["im_id"])
            return float(pose_error.mspd(re, te, rg, tg, K,
                                         model_pts_mm, syms))
        acc.record("mspd", i, _compute_mspd)

        def _compute_vsd(sm=s, re=re, te=te, rg=rg, tg=tg):
            if rnd is None:
                raise RuntimeError("renderer unavailable")
            d_m = reader.get_depth(sm["im_id"])
            if d_m is None:
                raise RuntimeError(f"depth 缺失 im_id={sm['im_id']}")
            d_mm = np.asarray(d_m, dtype=np.float64) * 1000.0
            if d_mm.shape != (h0, w0):
                raise ValueError(f"depth shape {d_mm.shape} != ({h0},{w0})")
            K = reader.get_K(sm["im_id"])
            e = pose_error.vsd(re, te, rg, tg,
                               depth_test=d_mm, K=K, delta=vsd_delta_mm,
                               taus=vsd_taus.tolist(),
                               normalized_by_diameter=True,
                               diameter=diam_mm, renderer=rnd,
                               obj_id=int(obj_id), cost_type="step")
            return np.asarray(e, dtype=np.float64).reshape(-1)
        acc.record("vsd", i, _compute_vsd)

    result = acc.summarize()
    result.update({"ok": True, "reason": None, "n_pred_fail": n_pred_fail,
                   "diameter_mm": diam_mm, "vispy_fail_reason": vispy_fail_reason})

    _pt(f"[AR] obj={obj_id} n={n} pred_fail={n_pred_fail} "
        f"diam={diam_mm:.2f}mm")
    for name in ("vsd", "mssd", "mspd"):
        _pt(f"[AR]   {name.upper():5s} complete={result[f'{name}_complete']} "
            f"ar={result[f'ar_{name}']:.4f} reason={result[f'{name}_fail_reason']}")
    _pt(f"[AR]   AR = {result['ar']:.4f}")
    return result


# =========================================================
# PoseMetrics（辅助 ADD/ADD-S）
# =========================================================
class PoseMetrics:
    def __init__(self, models_dir):
        self.meta_data = self._load_meta(models_dir)

    def _load_meta(self, models_dir):
        with open(os.path.join(models_dir, "models_info.json")) as f:
            info_json = json.load(f)
        meta = {}
        for ply in glob.glob(os.path.join(models_dir, "obj_*.ply")):
            ob_id = int(os.path.basename(ply).split("_")[1].split(".")[0])
            info = info_json.get(str(ob_id), {})
            diam = info.get("diameter", None)
            meta[ob_id] = {
                "diameter_mm": float(diam) if diam is not None else 200.0,
                "has_symmetry": bool(info.get("symmetries_discrete")
                                     or info.get("symmetries_continuous")),
            }
        return meta

    def compute_all_errors(self, pred_m, gt_m, pts_m, ob_id):
        p = (pred_m[:3, :3] @ pts_m.T).T + pred_m[:3, 3]
        g = (gt_m[:3, :3] @ pts_m.T).T + gt_m[:3, 3]
        add_m = float(np.linalg.norm(p - g, axis=1).mean())
        d, _ = cKDTree(p).query(g, k=1, workers=-1)
        adds_m = float(d.mean())

        diam_m = self.meta_data[ob_id]["diameter_mm"] / 1000.0
        return {
            "add_m": add_m, "adds_m": adds_m,
            "add_n": add_m / diam_m, "adds_n": adds_m / diam_m,
        }


def compute_auc(errs, max_val=0.1):
    errs = np.asarray(errs)
    if len(errs) == 0:
        return 0.0
    return float(np.mean(1.0 - np.clip(errs, 0, max_val) / max_val))


def compute_acc(errs, threshold=0.1):
    errs = np.asarray(errs)
    if len(errs) == 0:
        return 0.0
    return float(np.mean(errs < threshold) * 100.0)


# =========================================================
# Mask
# =========================================================
def get_object_mask(reader, i_frame, ob_id, detect_type):
    if detect_type == "box":
        mask = reader.get_mask(i_frame, ob_id)
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
        m = reader.get_mask(i_frame, ob_id)
        return (m > 0) if m is not None else None
    raise RuntimeError(f"Unknown detect_type: {detect_type}")


# =========================================================
# 单帧推理
# =========================================================
def infer_one_frame(reader, i_frame, est, ob_id, detect_type):
    color = reader.get_color(i_frame)
    depth = reader.get_depth(i_frame)
    if color is None or depth is None:
        return None, None
    ob_mask = get_object_mask(reader, i_frame, ob_id, detect_type)
    if ob_mask is None or ob_mask.sum() < 4:
        return None, None
    gt_pose = reader.get_gt_pose(i_frame, ob_id)
    if gt_pose is None:
        return None, None
    pred = est.register(K=reader.K, rgb=color, depth=depth,
                        ob_mask=ob_mask, ob_id=ob_id, iteration=5)
    if pred is None:
        return None, gt_pose
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    pred = np.asarray(pred, dtype=np.float64)
    if pred.shape != (4, 4) or not np.isfinite(pred).all():
        return None, gt_pose
    return pred, gt_pose


# =========================================================
# 参考帧选择
# =========================================================
def select_ref_frame(reader, obj_id, target_im_ids, opt):
    if not target_im_ids:
        raise RuntimeError(f"scene={reader.get_video_id()} obj={obj_id} 无可评估帧")

    if opt.ref_image >= 0:
        if opt.ref_image not in target_im_ids:
            raise RuntimeError(
                f"--ref_image={opt.ref_image} 不在 targets 里；"
                f"可选 im_id: {target_im_ids[:10]}...")
        ref_frame = int(opt.ref_image)
    else:
        bbox_by_im = {}
        for f, area, w, h in reader.frames_with_bbox(obj_id):
            bbox_by_im[f] = area
        candidates = [(im, bbox_by_im.get(im, 0)) for im in target_im_ids]
        candidates.sort(key=lambda x: -x[1])
        if not candidates:
            raise RuntimeError("无候选参考帧")
        ref_frame = int(candidates[0][0])

    eval_im_ids = [im for im in target_im_ids if im != ref_frame] \
        if opt.eval_skip_ref else list(target_im_ids)
    if not eval_im_ids:
        raise RuntimeError(
            f"scene={reader.get_video_id()} obj={obj_id}: 参考帧是唯一评估帧，"
            f"无法评估（可加 --eval_include_ref）")
    return ref_frame, eval_im_ids


# =========================================================
# 单 scene 评估（AR 始终计算）
# =========================================================
def evaluate_scene(scene_dir, scene_id, models_dir, models_info, metrics,
                   opt, targets_per_scene):
    reader = TyolSceneReader(scene_dir, models_dir,
                             min_visib=opt.min_visib,
                             min_visib_frac=opt.min_visib_frac)
    if not reader.ob_ids:
        _pt(f"[scene {scene_id}] 无任何 obj_id，skip")
        return None

    if targets_per_scene:
        target_objs = set(targets_per_scene.keys())
        ob_ids = sorted(target_objs & set(reader.ob_ids))
        if not ob_ids:
            _pt(f"[scene {scene_id}] targets 中 obj 与 scene_gt 不匹配，skip")
            return None
    else:
        ob_ids = list(reader.ob_ids)

    results = {}
    for obj_id in ob_ids:
        if targets_per_scene:
            im_ids = sorted(targets_per_scene.get(obj_id, set()))
        else:
            im_ids = reader.frames_for_obj(obj_id)
        if not im_ids:
            _pt(f"[scene {scene_id}] obj={obj_id} 无评估帧，skip")
            continue

        try:
            ref_frame, eval_im_ids = select_ref_frame(reader, obj_id, im_ids, opt)
        except RuntimeError as e:
            _pt(f"[scene {scene_id}] obj={obj_id} 参考帧选择失败: {e}")
            results[obj_id] = {"status": "ref_failed", "reason": str(e),
                               "scene_id": scene_id, "obj_id": obj_id,
                               "ref_frame": None, "ar_result": None,
                               "n_pred_fail": 0, "n_total": 0}
            continue

        # ====== [新增] 推理帧抽样（仅影响推理帧数，不影响 AR 内部抽样）======
        _stride = max(1, int(getattr(opt, "eval_stride", 1)))
        if _stride > 1:
            eval_im_ids = eval_im_ids[::_stride]
        _max_ev = int(getattr(opt, "max_eval_frames", 0))
        if _max_ev > 0 and len(eval_im_ids) > _max_ev:
            eval_im_ids = eval_im_ids[:_max_ev]
        if not eval_im_ids:
            _pt(f"[scene {scene_id}] obj={obj_id} 抽样后无评估帧，skip")
            results[obj_id] = {"status": "no_frames",
                               "reason": "stride 抽样后为空",
                               "scene_id": scene_id, "obj_id": obj_id,
                               "ref_frame": ref_frame, "ar_result": None,
                               "n_pred_fail": 0, "n_total": 0}
            continue
        # =================================================================

        _pt(f"[scene {scene_id}] obj={obj_id} ref=im{ref_frame} "
            f"eval={len(eval_im_ids)}/{len(im_ids)}")

        est = None
        try:
            est = OneRefPose(scorer=None, refiner=None,
                             ref_data_root=opt.ref_data_root, ob_id=obj_id)
            est.to_device("cuda:0")
            reader.K = np.array(
                reader._scene_camera[ref_frame]['cam_K'],
                dtype=np.float64).reshape(3, 3)

            mesh_m = reader.get_gt_mesh(obj_id)
            diam_m = float(np.linalg.norm(np.asarray(mesh_m.extents)))
            if not (diam_m > 1e-9):
                raise RuntimeError(f"mesh_diameter={diam_m} <= 0")

            save_to_disk = bool(opt.save_ref) and bool(opt.ref_data_root)
            probe_dir = None
            if opt.save_ref_probe:
                probe_dir = os.path.join(opt.save_ref_probe,
                                         f"scene{scene_id:06d}_ob{obj_id:02d}")
                os.makedirs(probe_dir, exist_ok=True)

            est.use_frame_as_ref(
                reader, ref_frame, obj_id,
                is_symmetric=False,
                symmetry_axis=opt.ref_sym_axis,
                crop_ratio=opt.crop_ratio,
                device=opt.ref_device,
                save_to_disk=save_to_disk,
                mesh_diameter=diam_m,
                crop_to_ref=opt.ref_crop_to_ref,
                probe_dir=probe_dir,
            )
            _sync_cuda()
        except Exception:
            _pt(f"[scene {scene_id}] obj={obj_id} 参考初始化失败:")
            traceback.print_exc()
            if est is not None:
                try:
                    del est
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            if opt.stop_on_ref_fail:
                raise
            results[obj_id] = {"status": "ref_failed",
                               "reason": "use_frame_as_ref failed",
                               "scene_id": scene_id, "obj_id": obj_id,
                               "ref_frame": ref_frame, "ar_result": None,
                               "n_pred_fail": 0, "n_total": 0}
            continue

        model_pts_m = mesh_m.vertices.copy()
        n_pred_fail = 0
        ar_samples = []
        add_n_list, adds_n_list = [], []

        pbar = tqdm(total=len(eval_im_ids),
                    desc=f"s{scene_id:06d}/ob{obj_id}", dynamic_ncols=True)
        for im_id in eval_im_ids:
            pred_m, gt_m = infer_one_frame(reader, im_id, est, obj_id,
                                            opt.detect_type)
            if pred_m is None:
                n_pred_fail += 1
                ar_samples.append({"im_id": int(im_id), "pred_m": None,
                                   "gt_m": None if gt_m is None else gt_m.copy()})
                add_n_list.append(float("inf"))
                adds_n_list.append(float("inf"))
                pbar.update(1)
                continue

            ar_samples.append({"im_id": int(im_id),
                               "pred_m": pred_m.copy(),
                               "gt_m": gt_m.copy()})

            if not opt.no_compute_adds:
                try:
                    err = metrics.compute_all_errors(pred_m, gt_m,
                                                     model_pts_m, obj_id)
                    add_n_list.append(err["add_n"])
                    adds_n_list.append(err["adds_n"])
                except Exception:
                    add_n_list.append(float("inf"))
                    adds_n_list.append(float("inf"))
            pbar.update(1)
        pbar.close()

        # ---- AR 始终计算 ----
        ar_result = None
        try:
            ar_result = compute_ar_for_tyol(
                reader, obj_id, models_dir, models_info, ar_samples,
                use_symmetry=opt.ar_use_symmetry,
                max_frames=opt.ar_max_frames,
                vsd_delta_mm=opt.vsd_delta_mm,
                max_sym_disc_step=opt.symmetry_step,
            )
        except Exception as e:
            _pt(f"[AR] compute_ar_for_tyol 异常: {type(e).__name__}: {e}")
            traceback.print_exc()
            ar_result = {"ok": False,
                         "reason": f"{type(e).__name__}: {e}",
                         "n_targets": len(ar_samples)}

        results[obj_id] = {
            "status": "evaluated" if (ar_result and ar_result.get("ok"))
                       else "metric_failed",
            "reason": (ar_result.get("reason")
                        if ar_result and not ar_result.get("ok") else None),
            "scene_id": scene_id, "obj_id": obj_id,
            "ref_frame": ref_frame,
            "n_total": len(ar_samples),
            "n_pred_fail": n_pred_fail,
            "ar_result": ar_result,
            "add_n_list": add_n_list,
            "adds_n_list": adds_n_list,
        }

        del est
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return results


# =========================================================
# 汇总
# =========================================================
def _aggregate_objects_ar(per_obj_results, requested_keys, statuses):
    n_requested = len(requested_keys)
    n_reachable = sum(1 for k in requested_keys
                       if statuses.get(k) == "evaluated")
    n_ref_failed = sum(1 for k in requested_keys
                        if statuses.get(k) == "ref_failed")
    n_metric_failed = sum(1 for k in requested_keys
                          if statuses.get(k) == "metric_failed")

    keys = ("ar_vsd", "ar_mssd", "ar_mspd")
    complete = []
    for k in requested_keys:
        if statuses.get(k) != "evaluated":
            continue
        r = per_obj_results.get(k, {}).get("ar_result")
        if r is None:
            continue
        if all(np.isfinite(r.get(x, float("nan"))) for x in keys):
            complete.append((k, r))

    out = {
        "n_objects_requested": n_requested,
        "n_objects_reachable": n_reachable,
        "n_objects_complete": len(complete),
        "n_objects_ref_failed": n_ref_failed,
        "n_objects_metric_failed": n_metric_failed,
        "object_coverage": (len(complete) / n_requested) if n_requested else 0.0,
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
            vals = [r[key] for _, r in complete]
            out[f"complete_subset_{key}"] = float(np.mean(vals))
        out["complete_subset_ar"] = float(np.mean(
            [out[f"complete_subset_{k}"] for k in keys]))
        if len(complete) == n_requested and n_requested > 0:
            out["full_set_ar"] = out["complete_subset_ar"]
    return out


def _finalize(all_results, requested_keys, statuses):
    print("\n========== TYOL BOP AR evaluation ==========", flush=True)
    if not all_results:
        print("(no results)", flush=True)
        return

    per_obj_results = {}
    for key in sorted(all_results):
        r = all_results[key]
        per_obj_results[key] = r
        sid, oid = r["scene_id"], r["obj_id"]
        print(f"\n========== [scene {sid:06d} / obj {oid:02d}] ==========",
              flush=True)
        print(f"  status: {r['status']}  n_total={r.get('n_total', 0)}  "
              f"pred_fail={r.get('n_pred_fail', 0)}  ref_im={r.get('ref_frame')}",
              flush=True)
        ar = r.get("ar_result")
        if ar is None:
            print(f"  AR: 未计算", flush=True)
        elif not ar.get("ok"):
            print(f"  AR: ✗ {ar.get('reason', '未知')}", flush=True)
        else:
            print(f"  n_targets={ar.get('n_targets', 0)}  "
                  f"pred_fail={ar.get('n_pred_fail', 0)}  "
                  f"diam={ar.get('diameter_mm', 0):.2f}mm", flush=True)
            for name, label in (("vsd", "VSD"), ("mssd", "MSSD"),
                                 ("mspd", "MSPD")):
                val = ar.get(f"ar_{name}", float("nan"))
                if np.isfinite(val):
                    print(f"  AR_{label:4s} = {val:.4f}  "
                          f"({'complete' if ar.get(f'{name}_complete') else 'partial'})",
                          flush=True)
                else:
                    print(f"  AR_{label:4s} = N/A  "
                          f"({ar.get(f'{name}_fail_reason') or '未知'})",
                          flush=True)
            ar_val = ar.get("ar", float("nan"))
            print(f"  AR = {ar_val:.4f} (BOP 3 项均值)"
                  if np.isfinite(ar_val)
                  else "  AR = N/A（三个分项未全部完整可用）", flush=True)

        add_n = np.asarray(r.get("add_n_list", []))
        adds_n = np.asarray(r.get("adds_n_list", []))
        if len(add_n):
            print(f"  [aux] ADD AUC={compute_auc(add_n, 0.1)*100:.2f} "
                  f"ADD@0.1={compute_acc(add_n, 0.1):.2f}%  "
                  f"ADD-S AUC={compute_auc(adds_n, 0.1)*100:.2f} "
                  f"ADD-S@0.1={compute_acc(adds_n, 0.1):.2f}%", flush=True)

    print("\n========== [OVERALL AR] ==========", flush=True)
    agg = _aggregate_objects_ar(per_obj_results, requested_keys, statuses)
    print(f" 请求对象: {agg['n_objects_requested']}  "
          f"可达: {agg['n_objects_reachable']}  "
          f"完整: {agg['n_objects_complete']}  "
          f"参考失败: {agg['n_objects_ref_failed']}  "
          f"指标失败: {agg['n_objects_metric_failed']}", flush=True)
    print(f" 完整子集覆盖率: {agg['object_coverage'] * 100:.1f}%", flush=True)
    if agg["unreachable_reasons"]:
        print(" [不可达对象]", flush=True)
        for r in agg["unreachable_reasons"]:
            print(f"    - {r}", flush=True)

    if np.isfinite(agg["complete_subset_ar"]):
        n_c = agg["n_objects_complete"]
        print(f"\n  --- 完整指标子集 AR (n={n_c}) ---", flush=True)
        for k in ("ar_vsd", "ar_mssd", "ar_mspd"):
            val = agg.get(f"complete_subset_{k}", float("nan"))
            print(f"  {k.upper()}: {val:.4f}", flush=True)
        print(f"  AR: {agg['complete_subset_ar']:.4f}", flush=True)

    if np.isfinite(agg["full_set_ar"]):
        print(f"\n  --- 全目标集 AR (n={agg['n_objects_requested']}) ---\n"
              f"  AR: {agg['full_set_ar']:.4f}", flush=True)
    else:
        print(f"\n  [全目标集 AR] N/A（请求 {agg['n_objects_requested']}，"
              f"完整 {agg['n_objects_complete']}，含不可达/不完整对象）",
              flush=True)


# =========================================================
# targets 解析
# =========================================================
def _parse_targets(path):
    rows = json.load(open(path))
    out = {}
    for row in rows:
        sid = int(row["scene_id"])
        im = int(row["im_id"])
        oid = int(row["obj_id"])
        out.setdefault(sid, {}).setdefault(oid, set()).add(im)
    return out


# =========================================================
# Pipeline
# =========================================================
def run_pipeline():
    tyol_dir = opt.tyol_dir
    models_dir = opt.models_dir or os.path.join(tyol_dir, "models")

    _pt("[pipeline] CUDA preflight ...")
    if torch.cuda.is_available():
        _pt(f"[pipeline] CUDA: {torch.cuda.get_device_name(0)}")
        _ = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        _pt("[pipeline] CUDA OK")
    else:
        _pt("[pipeline] WARNING: cuda unavailable")

    _pt("[pipeline] AR deps check ...")
    try:
        from bop_toolkit_lib import pose_error  # noqa
        _pt("  [AR-deps] pose_error: OK")
    except Exception as e:
        _pt(f"  [AR-deps] pose_error: FAILED ({e}) → AR 将全 NaN")
    try:
        from bop_toolkit_lib import renderer_vispy  # noqa
        _pt("  [AR-deps] renderer_vispy: OK")
    except Exception as e:
        _pt(f"  [AR-deps] renderer_vispy: FAILED ({e}) → VSD 不可用")

    info_path = os.path.join(models_dir, "models_info.json")
    if not os.path.isfile(info_path):
        raise SystemExit(f"models_info.json 不存在: {info_path}")
    with open(info_path) as f:
        models_info = json.load(f)
    _pt(f"[pipeline] models_info loaded ({len(models_info)} objs)")

    metrics = PoseMetrics(models_dir)

    scene_dirs = sorted(glob.glob(os.path.join(tyol_dir, "test", "*")))
    scene_dirs = [d for d in scene_dirs if os.path.isdir(d)]
    if not scene_dirs:
        raise SystemExit(f"[pipeline] 没有 test/* 场景: {tyol_dir}/test/")
    _pt(f"[pipeline] 共 {len(scene_dirs)} 个 scene, models_dir={models_dir}")

    targets_path = None
    for cand in (
        opt.targets_json,
        os.path.join(tyol_dir, "test_targets_bop19.json"),
        os.path.join(tyol_dir, "tyol", "test_targets_bop19.json"),
    ):
        if cand and os.path.isfile(cand):
            targets_path = cand
            break
    targets = _parse_targets(targets_path) if targets_path else None
    _pt(f"[pipeline] targets: {targets_path or '未使用（评估所有可见帧）'}")

    if opt.obj_ids:
        selected = set(int(x.strip()) for x in opt.obj_ids.split(",") if x.strip())
    else:
        selected = None

    all_results = {}
    requested_keys = []
    statuses = {}

    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        try:
            scene_id = int(scene_name)
        except ValueError:
            _pt(f"[pipeline] skip 非法 scene 目录: {scene_dir}")
            continue

        if targets is not None and scene_id not in targets:
            continue

        if selected is not None:
            if targets is not None:
                scene_targets = {o: ims for o, ims in targets.get(scene_id, {}).items()
                                 if o in selected}
                if not scene_targets:
                    continue
            else:
                scene_targets = None
        else:
            scene_targets = targets.get(scene_id) if targets is not None else None

        try:
            tmp_reader = TyolSceneReader(
                scene_dir, models_dir,
                min_visib=opt.min_visib,
                min_visib_frac=opt.min_visib_frac)
            scene_objs = sorted(tmp_reader.ob_ids)
            del tmp_reader
        except Exception:
            _pt(f"[pipeline] reader 加载失败: {scene_dir}")
            traceback.print_exc()
            continue

        if scene_targets:
            scene_objs = [o for o in scene_objs if o in scene_targets]
        if selected is not None:
            scene_objs = [o for o in scene_objs if o in selected]
        if not scene_objs:
            continue

        for oid in scene_objs:
            key = f"{scene_id:06d}_{oid:02d}"
            requested_keys.append(key)
            statuses[key] = "pending"

        _pt(f"\n========== [scene {scene_id:06d}] ==========")
        rec = evaluate_scene(scene_dir, scene_id, models_dir, models_info,
                             metrics, opt, scene_targets)
        if not rec:
            for oid in scene_objs:
                key = f"{scene_id:06d}_{oid:02d}"
                statuses[key] = "no_frames"
            continue

        for oid, r in rec.items():
            key = f"{scene_id:06d}_{oid:02d}"
            all_results[key] = r
            statuses[key] = r["status"]

    _finalize(all_results, requested_keys, statuses)

    if opt.debug_dir:
        os.makedirs(opt.debug_dir, exist_ok=True)
        summary = {}
        for key, r in all_results.items():
            ar = r.get("ar_result") or {}
            summary[key] = {
                "scene_id": r["scene_id"], "obj_id": r["obj_id"],
                "status": r["status"], "ref_frame": r.get("ref_frame"),
                "n_total": r.get("n_total", 0),
                "n_pred_fail": r.get("n_pred_fail", 0),
                "ar": ar.get("ar"),
                "ar_vsd": ar.get("ar_vsd"),
                "ar_mssd": ar.get("ar_mssd"),
                "ar_mspd": ar.get("ar_mspd"),
            }
        with open(os.path.join(opt.debug_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        _pt(f"[pipeline] summary saved -> {opt.debug_dir}/summary.json")


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="test_tyol.py")
    parser.add_argument("--tyol_dir", type=str, required=True,
                        help="例如 /root/data/tyol（须包含 test/ 和 models/）")
    parser.add_argument("--models_dir", type=str, default=None,
                        help="默认 <tyol_dir>/models；可指定 models_eval")
    parser.add_argument("--obj_ids", type=str, default=None,
                        help="逗号分隔，例如 1,3,5；不传则全部")
    parser.add_argument("--targets_json", type=str, default=None,
                        help="可选；默认自动找 <tyol_dir>/test_targets_bop19.json "
                             "或 <tyol_dir>/tyol/test_targets_bop19.json")

    parser.add_argument("--ref_image", type=int, default=-1,
                        help="指定参考帧号 im_id；-1 表示在 scene 内按 mask 面积自动挑")
    parser.add_argument("--eval_skip_ref", action="store_true", default=True)
    parser.add_argument("--eval_include_ref", dest="eval_skip_ref",
                        action="store_false")

    parser.add_argument("--ref_data_root", type=str, default=None)
    parser.add_argument("--save_ref", action="store_true")
    parser.add_argument("--save_ref_probe", type=str, default=None)
    parser.add_argument("--ref_sym_axis", type=int, default=-1)
    parser.add_argument("--ref_device", type=str, default="cuda")
    parser.add_argument("--ref_crop_to_ref", action="store_true", default=True)
    parser.add_argument("--no_ref_crop_to_ref", dest="ref_crop_to_ref",
                        action="store_false")
    parser.add_argument("--stop_on_ref_fail", action="store_true", default=False)

    # AR 相关（始终计算，无开关）
    parser.add_argument("--ar_use_symmetry", action="store_true", default=True)
    parser.add_argument("--no_ar_use_symmetry", dest="ar_use_symmetry",
                        action="store_false")
    parser.add_argument("--ar_max_frames", type=int, default=0,
                        help="AR 每对象最多抽 N 帧（0=全部，正式评估建议 0）")
    parser.add_argument("--vsd_delta_mm", type=float, default=15.0,
                        help="VSD visibility tolerance（毫米）")
    parser.add_argument("--symmetry_step", type=float, default=0.01,
                        help="BOP Toolkit max_sym_disc_step")

    # ====== [新增] 推理帧抽样（sweep 加速用）======
    parser.add_argument("--eval_stride", type=int, default=1,
                        help="评估帧抽样步长（1=全帧；sweep 时建议 10~20）")
    parser.add_argument("--max_eval_frames", type=int, default=0,
                        help="每 obj 最多评估帧数（0=不限；sweep 时建议 30~50）")
    # ==============================================

    # 辅助 ADD/ADD-S（可选）
    parser.add_argument("--no_compute_adds", action="store_true", default=False,
                        help="关闭 ADD/ADD-S 辅助指标（仅算 BOP AR，速度更快）")

    parser.add_argument("--detect_type", type=str, default="mask",
                        choices=["mask", "box"])
    parser.add_argument("--crop_ratio", type=float, default=1.1)
    parser.add_argument("--debug_dir", type=str, default="")
    parser.add_argument("--min_visib", type=int, default=8)
    parser.add_argument("--min_visib_frac", type=float, default=0.0)
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    opt = parser.parse_args()
    set_seed(0)

    logging.basicConfig(
        level=getattr(logging, opt.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout, force=True,
    )
    faulthandler.enable()

    start = time.time()
    run_pipeline()
    print(f"\n[pipeline] Total wall time: {time.time() - start:.2f}s", flush=True)