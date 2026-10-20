# Copyright (c) 2026 [OneViewAll].
# Licensed under the MIT License.
#
# Note: This module is designed to interface with systems licensed under 

from Utils import *
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from training.pose_score import PoseScore, RefData
from training.pose_refinement import PoseRefinement
from training.geometric_scorer import GeometricScorer


class OneRefPose:
    def __init__(
        self,
        scorer: PoseScore = None,
        refiner: PoseRefinement = None,
        ref_data_root: str = None,
        ob_id: int = None,
    ):
        self.ref_data_root = ref_data_root
        self.ref_data: RefData = None
        self.diameter = 0.1

        self.gt_pose = None
        self.ignore_normal_flip = True
        self.pose_last = None

        if self.ref_data_root is None:
            raise ValueError(
                "OneRefPose requires `ref_data_root` to locate reference data. "
                "Please pass `ref_data_root=...` (e.g. 'reference_database/linemod_real')."
            )

        if ob_id is not None:
            self._init_object_geometry(ob_id=ob_id)

        self.scorer = (
            scorer if scorer is not None
            else PoseScore(ref_data_root=ref_data_root)
        )
        self.refiner = (
            refiner if refiner is not None
            else PoseRefinement(ref_data_root=ref_data_root)
        )
        self._inject_ref_data()
        self._build_rotation_hypotheses(min_n_views=50, inplane_step=90)

    # ======================================================
    # Reference data plumbing
    # ======================================================
    def _inject_ref_data(self):
        scorer = getattr(self, "scorer", None)
        refiner = getattr(self, "refiner", None)
        if scorer is not None and hasattr(scorer, "set_ref_data"):
            scorer.set_ref_data(self.ref_data)
        if refiner is not None and hasattr(refiner, "set_ref_data"):
            refiner.set_ref_data(self.ref_data)

    # ======================================================
    # Loading helpers (self-contained, no RefData.from_file)
    # ======================================================
    @staticmethod
    def _load_pt_meta(pt_path: Path):
        """Return (meta, has_tensors, pt_data)."""
        if not pt_path.is_file():
            return {}, False, {}
        data = torch.load(str(pt_path), map_location="cpu")
        meta = data.get("meta", {}) or {}
        has_tensors = any(k in data for k in ("xyz_map", "rgb", "depth", "pose"))
        return meta, has_tensors, data

    @staticmethod
    def _read_rgb(path: Path) -> torch.Tensor:
        """Return (3, H, W) float32, 0-255, from PNG."""
        from PIL import Image
        arr = np.asarray(Image.open(str(path)).convert("RGB"), dtype=np.uint8)  # HWC
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()

    @staticmethod
    def _read_depth(path: Path) -> torch.Tensor:
        """Return (H, W) float32."""
        arr = np.load(str(path))
        t = torch.from_numpy(arr).float()
        if t.ndim == 3 and t.shape[0] == 1:
            t = t[0]
        return t

    @staticmethod
    def _read_xyz(path: Path) -> torch.Tensor:
        """Return (3, H, W) float32, from HWC npy."""
        arr = np.load(str(path))
        t = torch.from_numpy(arr).float()
        if t.ndim == 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1).contiguous()
        return t

    @staticmethod
    def _read_pose(path: Path) -> torch.Tensor:
        arr = np.loadtxt(str(path))
        return torch.tensor(arr, dtype=torch.float32).reshape(4, 4)

    def _load_ref_data(self, save_dir: Path, ob_id: int) -> RefData:
        """
        Load RefData from either:
          - New layout: ref_data.pt (meta only) + rgb.png / depth.npy /
            xyz_map.npy / pose.txt
          - Legacy layout: ref_data.pt with embedded tensors
        """
        pt_path = save_dir / "ref_data.pt"
        meta, has_tensors, pt_data = self._load_pt_meta(pt_path)

        if has_tensors:
            # Legacy path
            rgb = pt_data.get("rgb")
            depth = pt_data.get("depth")
            xyz_map = pt_data.get("xyz_map")
            pose = pt_data.get("pose", torch.eye(4))

            if not torch.is_tensor(rgb):   rgb = torch.as_tensor(rgb)
            if not torch.is_tensor(depth): depth = torch.as_tensor(depth)
            if not torch.is_tensor(xyz_map): xyz_map = torch.as_tensor(xyz_map)
            if not torch.is_tensor(pose):  pose = torch.as_tensor(pose)

            rgb = rgb.float()
            if rgb.ndim == 3 and rgb.shape[0] != 3 and rgb.shape[-1] == 3:
                rgb = rgb.permute(2, 0, 1).contiguous()

            depth = depth.float()
            if depth.ndim == 3 and depth.shape[0] == 1:
                depth = depth[0]

            xyz_map = xyz_map.float()
            if xyz_map.ndim == 3 and xyz_map.shape[0] != 3 and xyz_map.shape[-1] == 3:
                xyz_map = xyz_map.permute(2, 0, 1).contiguous()

            pose = pose.float().reshape(4, 4)
            return RefData(xyz_map=xyz_map, rgb=rgb, pose=pose,
                           depth=depth, meta=meta, ob_id=ob_id)

        # New layout: independent files
        rgb_png = save_dir / "rgb.png"
        depth_npy = save_dir / "depth.npy"
        xyz_npy = save_dir / "xyz_map.npy"
        pose_txt = save_dir / "pose.txt"
        meta_json = save_dir / "meta.json"

        missing = [str(p.name) for p, ok in (
            (rgb_png, rgb_png.is_file()),
            (depth_npy, depth_npy.is_file()),
            (xyz_npy, xyz_npy.is_file()),
        ) if not ok]
        if missing:
            raise FileNotFoundError(
                f"{save_dir}: missing {missing} (and ref_data.pt has no tensors). "
                f"Run tools/convert_ref_data.py first, or keep tensors inside ref_data.pt."
            )

        if not meta and meta_json.is_file():
            import json
            with open(meta_json) as f:
                meta = json.load(f)

        rgb = self._read_rgb(rgb_png)
        depth = self._read_depth(depth_npy)
        xyz_map = self._read_xyz(xyz_npy)
        pose = self._read_pose(pose_txt) if pose_txt.is_file() else torch.eye(4)

        return RefData(xyz_map=xyz_map, rgb=rgb, pose=pose,
                       depth=depth, meta=meta, ob_id=ob_id)

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
            print(f"[OneRefPose] Failed to load reference data for ob_id {ob_id}: {e}")
            self.ref_data = None
            self._inject_ref_data()
            return

        # ------------------------------------------------------------------
        # Print a summary of the loaded reference data.
        #   xyz_map: reference 3D point map, obtained by back-projecting the
        #       reference depth map with the camera intrinsics K
        #       (X = K^{-1} * [u, v, 1]^T * depth). It is NOT read directly
        #       from disk; it is derived from the reference depth.
        #       A foreground (object) mask is applied during construction so
        #       that ONLY the object's points are kept and the background is
        #       filtered out (background pixels are zeroed). This way the
        #       subsequent score / refinement stages and the diameter estimate
        #       operate on the object alone.
        #       - new layout: computed from save_dir/depth.npy + object mask + ref K
        #       - legacy layout: ref_data.pt["xyz_map"] (already precomputed and masked)
        #     `_load_ref_data` normalizes it to (3, H, W), so the shape printed
        #     here is (3, H, W).
        #   rgb:    reference RGB, from rgb.png or ref_data.pt["rgb"],
        #           normalized to (3, H, W). Provides color only.
        #   depth:  reference depth, from depth.npy or ref_data.pt["depth"],
        #           normalized to (H, W). Used to derive xyz_map.
        #   pose:   reference pose, from pose.txt or ref_data.pt["pose"], (4, 4).
        #   meta:   from ref_data.pt["meta"] (or meta.json), contains crop_ratio etc.
        # ------------------------------------------------------------------
        r = self.ref_data
        print(f"[RefData] ob_id={ob_id}  "
              f"xyz={tuple(r.xyz_map.shape)}  "
              f"rgb={tuple(r.rgb.shape)}  "
              f"depth={tuple(r.depth.shape)}  "
              f"pose={tuple(r.pose.shape)}  "
              f"meta={r.meta}")

        # Estimate diameter (using an HWC view)
        xyz = r.xyz_map
        if xyz.ndim == 3 and xyz.shape[0] == 3:
            xyz = xyz.permute(1, 2, 0)
        valid_mask = xyz[..., 2] > 0.001
        pts = xyz[valid_mask].reshape(-1, 3)
        if pts.numel() == 0:
            self.diameter = 0.1
            self._inject_ref_data()
            return

        mean = pts.mean(dim=0, keepdim=True)
        std = pts.std(dim=0, keepdim=True)
        std = torch.clamp(std, min=1e-6)
        z_scores = torch.abs((pts - mean) / std)
        filter_mask = (z_scores < 3.0).all(dim=1)
        pts_filtered = pts[filter_mask]
        if pts_filtered.numel() < 3:
            pts_filtered = pts

        lower = pts_filtered.quantile(0.05, dim=0)
        upper = pts_filtered.quantile(0.95, dim=0)
        self.diameter = (upper - lower).norm().item()

        self._inject_ref_data()

    def to_device(self, device="cuda:0"):
        for k in self.__dict__:
            v = self.__dict__[k]
            if torch.is_tensor(v) or isinstance(v, nn.Module):
                self.__dict__[k] = v.to(device)

        if self.ref_data is not None:
            # Move tensors manually, without relying on RefData.to()
            if torch.is_tensor(self.ref_data.xyz_map):
                self.ref_data.xyz_map = self.ref_data.xyz_map.to(device)
            if torch.is_tensor(self.ref_data.rgb):
                self.ref_data.rgb = self.ref_data.rgb.to(device)
            if torch.is_tensor(self.ref_data.pose):
                self.ref_data.pose = self.ref_data.pose.to(device)
            if torch.is_tensor(self.ref_data.depth):
                self.ref_data.depth = self.ref_data.depth.to(device)

        if self.refiner is not None:
            self.refiner.model.to(device)
        if self.scorer is not None:
            self.scorer.model.to(device)

    # ======================================================
    # Rotation hypothesis generation
    # ======================================================
    def _build_rotation_hypotheses(self, min_n_views=40, inplane_step=60):
        cam_in_obs = sample_views_fibonacci(n_views=min_n_views)
        mask = cam_in_obs[:, 2, 3] >= 0.35
        cam_in_obs = cam_in_obs[mask]

        rot_grid = []
        for i in range(len(cam_in_obs)):
            for angle in np.deg2rad(np.arange(0, 360, inplane_step)):
                cam = cam_in_obs[i]
                R = euler_matrix(0, 0, angle)
                cam_rot = cam @ R
                rot_grid.append(np.linalg.inv(cam_rot))

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
        vs, us = np.where(mask > 0)
        if len(us) == 0:
            return np.zeros(3)
        uc = (us.min() + us.max()) / 2.0
        vc = (vs.min() + vs.max()) / 2.0
        valid = (mask.astype(bool)) & (depth >= 0.001)
        if not valid.any():
            return np.zeros(3)
        zc = np.median(depth[valid])
        center = (np.linalg.inv(K) @ np.array([uc, vc, 1]).reshape(3, 1)) * zc
        return center.reshape(3)

    # ======================================================
    # Main inference API
    # ======================================================
    def register(self, K, rgb, depth, ob_mask, ob_id=None, iteration=5):
        depth = erode_depth(depth, radius=2, device="cuda")
        depth = bilateral_filter_depth(depth, radius=2, device="cuda")

        valid = (depth >= 0.001) & (ob_mask > 0)
        if valid.sum() < 4:
            pose = np.eye(4)
            pose[:3, 3] = self._estimate_translation(depth, ob_mask, K)
            return pose

        poses = self._sample_pose_hypotheses(K, rgb, depth, ob_mask)
        center = self._estimate_translation(depth, ob_mask, K)
        poses[:, :3, 3] = torch.tensor(center, device="cuda")

        xyz_map = depth2xyzmap(depth, K)

        poses, _ = self.refiner.predict(
            rgb=rgb, depth=depth, K=K,
            ob_in_cams=poses.cpu().numpy(),
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