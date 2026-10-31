# Copyright (c) 2026 [OneViewAll].
# Licensed under the MIT License.

from Utils import *
import logging
import os
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image as _Img

CROP_RATIO_DEFAULT = 1.2

from training.pose_score import PoseScore, RefData
from training.pose_refinement import PoseRefinement
from training.geometric_scorer import GeometricScorer


# =========================================================
# [PATCH] 内联保存函数（不再依赖 pose_score.save_ref_data / RefData.save）
# =========================================================
def _save_ref_data_impl(ref_data, save_dir) -> str:
    """
    Save a RefData-like object to `save_dir` in the SAME layout as
    `reference_database/linemod_real/<ob_id>/`:
        describe.txt  mask.png  meta.json  pose.txt  ref_data.pt  rgb.png  xyz.png
    Plus `xyz_map.npy` for load_ref_data_from_dir fallback.

    只依赖 ref_data 的 4 个属性：xyz_map / rgb / pose / meta（+ 可选 depth / ob_id），
    不依赖任何类方法，所以任何版本的 RefData 都能保存。
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    xyz = torch.as_tensor(ref_data.xyz_map).detach().cpu().float()
    rgb = torch.as_tensor(ref_data.rgb).detach().cpu().float()
    pose = torch.as_tensor(ref_data.pose).detach().cpu().float().reshape(4, 4)
    meta = dict(getattr(ref_data, "meta", None) or {})
    depth = getattr(ref_data, "depth", None)
    ob_id = getattr(ref_data, "ob_id", None)

    # ---- rgb.png ----
    rgb_hwc = rgb.permute(1, 2, 0) if (rgb.ndim == 3 and rgb.shape[0] == 3) else rgb
    if rgb_hwc.numel() > 0 and float(rgb_hwc.max()) <= 1.01:
        rgb_hwc = rgb_hwc * 255.0
    _Img.fromarray(rgb_hwc.clamp(0, 255).to(torch.uint8).numpy()).save(save_dir / "rgb.png")

    # ---- xyz_map.npy ----
    xyz_hwc = xyz.permute(1, 2, 0) if (xyz.ndim == 3 and xyz.shape[0] == 3) else xyz
    np.save(str(save_dir / "xyz_map.npy"), xyz_hwc.contiguous().numpy())

    # ---- xyz.png ----
    z = xyz_hwc[..., 2]
    valid = torch.isfinite(z) & (z > 1e-5)
    if valid.any():
        zmin, zmax = float(z[valid].min()), float(z[valid].max())
        if zmax - zmin < 1e-6:
            zmax = zmin + 1e-3
        zn = torch.zeros_like(z)
        zn[valid] = (z[valid] - zmin) / (zmax - zmin)
        r_c = torch.clamp(1.5 - (zn * 4 - 3).abs(), 0, 1)
        g_c = torch.clamp(1.5 - (zn * 4 - 2).abs(), 0, 1)
        b_c = torch.clamp(1.5 - (zn * 4 - 1).abs(), 0, 1)
        vis = torch.stack([r_c, g_c, b_c], dim=-1)
        vis[~valid] = 0
        _Img.fromarray((vis * 255).to(torch.uint8).numpy()).save(save_dir / "xyz.png")
    else:
        _Img.fromarray(np.zeros(tuple(xyz_hwc.shape[:2]), dtype=np.uint8)).save(save_dir / "xyz.png")

    # ---- mask.png ----
    _Img.fromarray((valid.numpy().astype(np.uint8) * 255), mode="L").save(save_dir / "mask.png")

    # ---- pose.txt ----
    np.savetxt(str(save_dir / "pose.txt"), pose.numpy(), fmt="%.8f")

    # ---- meta.json ----
    with open(save_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ---- describe.txt ----
    with open(save_dir / "describe.txt", "w") as f:
        f.write(f"ob_id: {ob_id}\n")
        f.write(f"crop_ratio: {meta.get('crop_ratio', CROP_RATIO_DEFAULT)}\n") # 读取真实的
        f.write(f"xyz_map.shape: {tuple(xyz.shape)}\n")
        f.write(f"rgb.shape: {tuple(rgb.shape)}\n")
        f.write(f"pose: ob_in_cam (4x4)\n")
        f.write(f"meta:\n{json.dumps(meta, indent=2)}\n")

    # ---- ref_data.pt ----
    torch.save({
        "xyz_map": xyz,
        "rgb": rgb,
        "pose": pose,
        "depth": None if depth is None else torch.as_tensor(depth).detach().cpu(),
        "meta": meta,
    }, save_dir / "ref_data.pt")

    print(f"[OneRefPose.save] wrote 7 files + xyz_map.npy → {save_dir}")
    return str(save_dir)


class OneRefPose:
    def __init__(self, scorer: PoseScore = None, refiner: PoseRefinement = None,
                 ref_data_root: str = None, ob_id: int = None):
        self.ref_data_root = ref_data_root
        self.ref_data: RefData = None
        self.diameter = 0.1
        self.gt_pose = None
        self.ignore_normal_flip = True
        self.pose_last = None

        if self.ref_data_root is None:
            logging.warning(
                "OneRefPose: ref_data_root is None. No reference data will be "
                "loaded from disk. Call `use_frame_as_ref(...)` before register()."
            )

        if ob_id is not None and self.ref_data_root is not None:
            self._init_object_geometry(ob_id=ob_id)

        self.scorer = scorer if scorer is not None else PoseScore(ref_data_root=ref_data_root)
        self.refiner = refiner if refiner is not None else PoseRefinement(ref_data_root=ref_data_root)
        self._inject_ref_data()
        self._build_rotation_hypotheses(min_n_views=40, inplane_step=180)

    # ======================================================
    # Reference data plumbing
    # ======================================================
    def _inject_ref_data(self):
        for m in (getattr(self, "scorer", None), getattr(self, "refiner", None)):
            if m is not None and hasattr(m, "set_ref_data"):
                m.set_ref_data(self.ref_data)

    def _load_ref_data(self, save_dir: Path, ob_id: int) -> RefData:
        from training.pose_score import load_ref_data_from_dir
        return load_ref_data_from_dir(save_dir, ob_id=ob_id)

    # ======================================================
    # Object initialization
    # ======================================================
    def _init_object_geometry(self, ob_id=1, save_dir=None):
        self.diameter = 0.1
        if save_dir is None:
            save_dir = os.path.join(self.ref_data_root, str(ob_id))
        save_dir = Path(save_dir)
        try:
            self.ref_data = self._load_ref_data(save_dir, ob_id)
        except Exception as e:
            print(f"[OneRefPose] Failed to load reference for ob_id {ob_id}: {e}")
            self.ref_data = None
            self._inject_ref_data()
            return
        r = self.ref_data
        print(f"[RefData] ob_id={ob_id}  xyz={tuple(r.xyz_map.shape)}  "
              f"rgb={tuple(r.rgb.shape)}  pose={tuple(r.pose.shape)}  meta={r.meta}")
        self._update_diameter_from_ref()
        self._inject_ref_data()

    def _update_diameter_from_ref(self):
        if self.ref_data is None:
            self.diameter = 0.1
            return
        xyz = self.ref_data.xyz_map
        if xyz.ndim == 3 and xyz.shape[0] == 3:
            xyz = xyz.permute(1, 2, 0)
        valid_mask = xyz[..., 2] > 0.001
        pts = xyz[valid_mask].reshape(-1, 3)
        if pts.numel() == 0:
            self.diameter = 0.1
            return
        mean = pts.mean(dim=0, keepdim=True)
        std = torch.clamp(pts.std(dim=0, keepdim=True), min=1e-6)
        z_scores = torch.abs((pts - mean) / std)
        pts_filtered = pts[(z_scores < 3.0).all(dim=1)]
        if pts_filtered.numel() < 3:
            pts_filtered = pts
        lower = pts_filtered.quantile(0.05, dim=0)
        upper = pts_filtered.quantile(0.95, dim=0)
        self.diameter = (upper - lower).norm().item()

    def _estimate_diameter_from_frame(self, reader, i_frame, ob_id):
        K = reader.K
        K_np = K.detach().cpu().numpy() if torch.is_tensor(K) else np.asarray(K)
        depth = reader.get_depth(i_frame)
        depth_np = depth.detach().cpu().numpy() if torch.is_tensor(depth) else np.asarray(depth)
        mask_np = np.asarray(reader.get_mask(i_frame, ob_id))
        xyz = depth2xyzmap(depth_np, K_np)
        xyz = np.asarray(xyz)
        if xyz.ndim == 3 and xyz.shape[0] == 3:
            xyz = xyz.transpose(1, 2, 0)
        pts = xyz[mask_np > 0]
        pts = pts[pts[..., 2] > 0.001]
        if pts.size == 0:
            return 0.1
        return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))

    def to_device(self, device="cuda:0"):
        for k in self.__dict__:
            v = self.__dict__[k]
            if torch.is_tensor(v) or isinstance(v, nn.Module):
                self.__dict__[k] = v.to(device)
        if self.ref_data is not None:
            for attr in ("xyz_map", "rgb", "pose"):
                t = getattr(self.ref_data, attr, None)
                if torch.is_tensor(t):
                    setattr(self.ref_data, attr, t.to(device))
        if self.refiner is not None and hasattr(self.refiner, "model"):
            self.refiner.model.to(device)
        if self.scorer is not None and hasattr(self.scorer, "model"):
            self.scorer.model.to(device)

    # ======================================================
    # [PATCH] Set a real frame as reference —— 保存逻辑统一在这里
    # ======================================================
    def use_frame_as_ref(self, reader, i_frame, ob_id, save_dir=None,
                         is_symmetric=False, symmetry_axis=-1,
                         crop_ratio=CROP_RATIO_DEFAULT,
                         device="cuda", save_to_disk=False,
                         mesh_diameter=None, crop_to_ref=True,
                         render_size=(160, 160),
                         probe_dir=None):
        """
        构建 RefData 并（可选）保存。

        保存目标：
          - save_to_disk=True  → save_dir（若 None 则 ref_data_root/<ob_id>）
                                 （长期参考，供推理加载）
          - probe_dir != None  → 直接 dump 到 probe_dir（调试用，独立触发）

        两者可以同时生效（内容一致，只是路径不同）。
        """
        from training.pose_score import capture_ref_data_from_frame

        if mesh_diameter is None and crop_to_ref:
            mesh_diameter = self._estimate_diameter_from_frame(reader, i_frame, ob_id)

        ref = capture_ref_data_from_frame(
            reader, i_frame, ob_id, K=reader.K,
            is_symmetric=is_symmetric, symmetry_axis=symmetry_axis,
            crop_ratio=crop_ratio, # <--- 修改点：使用真实的 crop_ratio 透传
            device=device,
            render_size=render_size, mesh_diameter=mesh_diameter,
            crop_to_ref=crop_to_ref,
        )
        self.ref_data = ref
        self._update_diameter_from_ref()
        self._inject_ref_data()
        logging.info(
            f"[OneRefPose] ref set: ob_id={ob_id} frame={i_frame} "
            f"crop_ratio={crop_ratio} crop_to_ref={crop_to_ref} " # <--- 修改点：打印真实的 crop_ratio
            f"diameter={self.diameter:.4f}"
        )

        # ---- 保存到 ref_data_root（长期参考）----
        if save_to_disk:
            target = save_dir
            if target is None:
                if self.ref_data_root is None:
                    raise ValueError(
                        "save_to_disk=True requires ref_data_root or save_dir")
                target = Path(self.ref_data_root) / str(ob_id)
            _save_ref_data_impl(ref, target)
            logging.info(f"[OneRefPose] reference saved to {target}")

        # ---- 保存到 probe_dir（调试 dump，独立于 save_to_disk）----
        if probe_dir is not None:
            _save_ref_data_impl(ref, probe_dir)
            logging.info(f"[OneRefPose] probe saved to {probe_dir}")

        return ref

    # ======================================================
    # Rotation hypothesis generation
    # ======================================================
    def _build_rotation_hypotheses(self, min_n_views=40, inplane_step=60):
        # cam_in_obs = sample_views_icosphere(n_views=min_n_views)
        cam_in_obs = sample_views_fibonacci(n_views=min_n_views)
        cam_in_obs = cam_in_obs[cam_in_obs[:, 2, 3] >= 0.35]
        rot_grid = []
        for i in range(len(cam_in_obs)):
            for angle in np.deg2rad(np.arange(0, 360, inplane_step)):
                R = euler_matrix(0, 0, angle)
                rot_grid.append(np.linalg.inv(cam_in_obs[i] @ R))
        self.rot_grid = torch.as_tensor(np.asarray(rot_grid), device="cuda",
                                        dtype=torch.float)
        logging.info(f"self.rot_grid: {self.rot_grid.shape}")

    # ======================================================
    # Pose sampling
    # ======================================================
    def _sample_pose_hypotheses(self, K, rgb, depth, mask, scene_pts=None):
        poses = self.rot_grid.clone()
        center = self._estimate_translation(depth, mask, K)
        poses[:, :3, 3] = torch.tensor(center, device="cuda").reshape(1, 3)
        return poses

    def _estimate_translation(self, depth, mask, K):
        depth = depth.detach().cpu().numpy() if torch.is_tensor(depth) else np.asarray(depth)
        K = K.detach().cpu().numpy() if torch.is_tensor(K) else np.asarray(K)
        mask = np.asarray(mask)
        vs, us = np.where(mask > 0)
        if len(us) == 0:
            return np.zeros(3)
        uc = (us.min() + us.max()) / 2.0
        vc = (vs.min() + vs.max()) / 2.0
        valid = (mask.astype(bool)) & (depth >= 0.001)
        if not valid.any():
            return np.zeros(3)
        zc = np.median(depth[valid])
        return (np.linalg.inv(K) @ np.array([uc, vc, 1]).reshape(3, 1) * zc).reshape(3)

    # ======================================================
    # Main inference API
    # ======================================================
    def register(self, K, rgb, depth, ob_mask, ob_id=None, iteration=3):
        depth = erode_depth(depth, radius=2, device="cuda")
        depth = bilateral_filter_depth(depth, radius=2, device="cuda")

        depth_np = depth.detach().cpu().numpy() if torch.is_tensor(depth) else np.asarray(depth)
        K_np = K.detach().cpu().numpy() if torch.is_tensor(K) else np.asarray(K)
        ob_mask_np = np.asarray(ob_mask)

        valid = (depth_np >= 0.001) & (ob_mask_np > 0)
        if valid.sum() < 4:
            print(f"[register] WARNING: only {int(valid.sum())} valid pixels, "
                  f"skip refinement/scoring")
            pose = np.eye(4)
            pose[:3, 3] = self._estimate_translation(depth_np, ob_mask_np, K_np)
            return pose

        poses = self._sample_pose_hypotheses(K_np, rgb, depth_np, ob_mask_np)
        logging.info(f"[register] sampled {len(poses)} pose hypotheses")
        
        center = self._estimate_translation(depth_np, ob_mask_np, K_np)
        poses[:, :3, 3] = torch.tensor(center, device="cuda")

        xyz_map = depth2xyzmap(depth_np, K_np)

        poses, _ = self.refiner.predict(
            rgb=rgb, depth=depth, K=K,
            ob_in_cams=poses,
            xyz_map=xyz_map,
            mesh_diameter=self.diameter,
            iteration=iteration, ob_id=ob_id)

        scores, _ = self.scorer.predict(
            rgb=rgb, depth=depth, K=K,
            ob_in_cams=poses, ob_mask=ob_mask,
            xyz_map=xyz_map,
            mesh_diameter=self.diameter, ob_id=ob_id)

        ids = torch.as_tensor(scores).argsort(descending=True)
        poses = torch.as_tensor(poses)[ids]
        self.pose_last = poses[0]
        return poses[0].cpu().numpy()

    def compute_add_err_to_gt_pose(self, poses):
        return -torch.ones(len(poses), device="cuda")