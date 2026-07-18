# Copyright (c) 2026 [OneViewAll].
# Licensed under the MIT License.
#
# Note: This module is designed to interface with systems licensed under 
# the NVIDIA Source Code License and maintains non-commercial compatibility.

from Utils import *
import numpy as np
import torch
import torch.nn as nn

from learning.training.pose_score import PoseScore
from learning.training.pose_refinement import PoseRefinement


class OneRefPose:
    def __init__(
        self,
        model_pts,
        model_normals,
        symmetry_tfs=None,
        mesh=None,
        scorer: PoseScore = None,
        refiner: PoseRefinement = None,
    ):
        self.gt_pose = None
        self.ignore_normal_flip = True

        self._init_object_geometry(model_pts, model_normals, mesh=mesh)
        self._build_rotation_hypotheses(min_n_views=40, inplane_step=90)

        self.scorer = scorer if scorer is not None else PoseScore()
        self.refiner = refiner if refiner is not None else PoseRefinement()

        self.pose_last = None

    # ======================================================
    # Object initialization
    # ======================================================
    def _init_object_geometry(self, model_pts, model_normals, mesh=None, ob_id=-1):
        """Initialize object center and diameter."""
        max_xyz = mesh.vertices.max(axis=0)
        min_xyz = mesh.vertices.min(axis=0)

        self.model_center = (min_xyz + max_xyz) / 2

        if mesh is not None:
            mesh = mesh.copy()
            mesh.vertices = mesh.vertices - self.model_center.reshape(1, 3)

        self.diameter = compute_mesh_diameter(mesh.vertices, n_sample=10000)

    def load_object_params(self, symmetry_tfs=None, mesh=None, ob_id=-1):
        """Load cached object parameters."""
        path = f"reference_database/linemod/{ob_id}/object_params.pt"
        data = torch.load(path, map_location="cpu")

        self.diameter = data["diameter"]
        self.model_center = data["model_center"]

    # ======================================================
    # Coordinate transform
    # ======================================================
    def _get_center_transform(self):
        tf = torch.eye(4, dtype=torch.float, device="cuda")
        tf[:3, 3] = -torch.as_tensor(self.model_center, device="cuda", dtype=torch.float)
        return tf

    def to_device(self, device="cuda:0"):
        for k in self.__dict__:
            v = self.__dict__[k]
            if torch.is_tensor(v) or isinstance(v, nn.Module):
                self.__dict__[k] = v.to(device)

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

        self.rot_grid = torch.as_tensor(
            np.asarray(rot_grid),
            device="cuda",
            dtype=torch.float,
        )
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
    # Main inference API (KEEP INTERFACE)
    # ======================================================
    def register(self, K, rgb, depth, ob_mask, ob_id=None, iteration=5):
        """Full pose estimation pipeline."""

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
            rgb=rgb,
            depth=depth,
            K=K,
            ob_in_cams=poses.cpu().numpy(),
            xyz_map=xyz_map,
            mesh_diameter=self.diameter,
            iteration=iteration,
            ob_id=ob_id,
        )

        scores, _ = self.scorer.predict(
            rgb=rgb,
            depth=depth,
            K=K,
            ob_in_cams=poses,
            xyz_map=xyz_map,
            mesh_diameter=self.diameter,
            ob_id=ob_id,
        )

        ids = torch.as_tensor(scores).argsort(descending=True)

        poses = torch.as_tensor(poses)[ids]
        best_pose = poses[0] @ self._get_center_transform()

        self.pose_last = poses[0]

        return best_pose.cpu().numpy()

    # ======================================================
    # placeholder
    # ======================================================
    def compute_add_err_to_gt_pose(self, poses):
        return -torch.ones(len(poses), device="cuda")