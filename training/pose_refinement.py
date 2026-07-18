# Copyright (c) 2026 [OneViewAll].
# Licensed under the MIT License.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software.

import functools
import os, sys, kornia
import time
import numpy as np
import torch
from omegaconf import OmegaConf
from models.refine_network import RefineNet
from datasets.h5_dataset import *
from Utils import *
from datareader import *
import torchvision.transforms as T
from torchvision.utils import save_image
from torchvision.transforms import Resize, Normalize
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_mirrored_xyz_map(symmetry_axis, xyz_map, PoseA):
    """
    Mirroring logic: Transform points to object space, mirror along symmetry axis, 
    then transform back to camera space. Supports (3, H, W) or (H, W, 3).
    """
    if xyz_map.shape[0] == 3:
        xyz_map = xyz_map.permute(1, 2, 0)
    
    device = xyz_map.device
    H, W, _ = xyz_map.shape
    mask = xyz_map[..., 2] > 1e-5
    xyz_v = xyz_map[mask]

    if xyz_v.shape[0] == 0:
        return torch.zeros((H, W, 3), device=device, dtype=xyz_map.dtype)

    R_A = PoseA[:3, :3]
    t_A = PoseA[:3, 3:4]
    
    # Cam -> Obj -> Mirror -> Cam
    xyz_obj = torch.mm(R_A.t(), xyz_v.t() - t_A)
    xyz_obj[symmetry_axis, :] *= -1 
    xyz_cam_mirrored = (torch.mm(R_A, xyz_obj) + t_A).t()

    mirrored_map = torch.zeros((H, W, 3), device=device, dtype=xyz_map.dtype)
    mirrored_map[mask] = xyz_cam_mirrored
    return mirrored_map


def process_and_save_pc_data_batched_v2(
        symmetry_axis,
        xyz_mapA,
        rgbA,
        xyz_mapB=None,
        rgbB=None,
        Pose_target=None,
        PoseA=None,
        K=None,
        output_dir="output_results"):

    os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "xyz"), exist_ok=True)

    device = xyz_mapA.device
    dtype = xyz_mapA.dtype

    # 1. Parameter preprocessing
    PoseA = torch.as_tensor(PoseA, dtype=dtype, device=device)
    Pose_target = torch.as_tensor(Pose_target, dtype=dtype, device=device)
    K = torch.as_tensor(K, dtype=dtype, device=device)

    if PoseA.ndim == 2: PoseA = PoseA.unsqueeze(0)
    B = Pose_target.shape[0]
    if PoseA.shape[0] == 1 and B > 1: PoseA = PoseA.expand(B, -1, -1)
    if K.ndim == 2: K = K.unsqueeze(0).expand(B, -1, -1)

    # 2. Transformation setup
    R_A, t_A = PoseA[:, :3, :3], PoseA[:, :3, 3:] 
    R_T, t_T = Pose_target[:, :3, :3], Pose_target[:, :3, 3:]
    rel_R = torch.bmm(R_T, R_A.transpose(1, 2))

    # 3. Preprocess & Mirror Fusion
    def preprocess_map(x):
        if x.ndim == 4: x = x.permute(0, 2, 3, 1).squeeze(0)
        elif x.ndim == 3 and x.shape[0] <= 4: x = x.permute(1, 2, 0)
        return torch.where(torch.isfinite(x), x, torch.zeros_like(x))

    xyz_mapA_cl = preprocess_map(xyz_mapA)
    rgbA_cl = preprocess_map(rgbA)
    H, W, _ = xyz_mapA_cl.shape

    # Mirror symmetry logic
    xyz_mirrored_cl = create_mirrored_xyz_map(symmetry_axis, xyz_mapA_cl, PoseA[0])
    xyz_combined = torch.cat([xyz_mapA_cl.reshape(-1, 3), xyz_mirrored_cl.reshape(-1, 3)], dim=0)

    # Color merge: use grayscale for mirrored parts
    gray = rgbA_cl.mean(dim=2, keepdim=True)
    avg_color_expanded = gray.expand_as(rgbA_cl)
    rgb_combined = torch.cat([rgbA_cl.reshape(-1, 3), avg_color_expanded.reshape(-1, 3)], dim=0)

    if rgb_combined.dtype != torch.uint8:
        if rgb_combined.max() <= 1.01: rgb_combined = rgb_combined * 255.0
        rgb_combined = rgb_combined.clamp(0, 255).to(torch.uint8)

    # 4. Centralized transformation for numerical stability
    # Formula: X_new = rel_R @ (X_old - t_A) + t_T
    xyz_flat = xyz_combined.unsqueeze(0).expand(B, -1, -1) 
    centered_xyz = xyz_flat - t_A.transpose(1, 2)
    xyz_rot = torch.bmm(centered_xyz, rel_R.transpose(1, 2)) + t_T.transpose(1, 2)

    z = xyz_rot[..., 2]
    valid_mask = (torch.norm(xyz_flat, dim=-1) > 1e-6) & (z > 1e-6)

    # 5. Projection
    fx, fy = K[:, 0, 0].view(B, 1), K[:, 1, 1].view(B, 1)
    cx, cy = K[:, 0, 2].view(B, 1), K[:, 1, 2].view(B, 1)

    u = (fx * (xyz_rot[..., 0] / (z + 1e-8)) + cx).long()
    v = (fy * (xyz_rot[..., 1] / (z + 1e-8)) + cy).long()

    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & valid_mask

    # 6. Initialize outputs
    rgb_res = torch.zeros((B, H * W, 3), dtype=torch.uint8, device=device)
    xyz_res = torch.zeros((B, H * W, 3), dtype=dtype, device=device)

    # 7. Z-buffer rendering with Weighted Splatting for hole filling
    weight_kernel = torch.tensor([
        0.002, 0.001, 0.002,
        0.001, 0.000, 0.001,
        0.002, 0.001, 0.002
    ], device=device, dtype=dtype)

    for b in range(B):
        m = in_bounds[b]
        if not m.any(): continue

        bu, bv, bz = u[b][m], v[b][m], z[b][m]
        brgb = rgb_combined[m]
        bxyz = xyz_rot[b][m]

        # 3x3 Neighborhood expansion
        offsets = torch.tensor([-1, 0, 1], device=device)
        bu_9 = (bu.unsqueeze(0) + offsets.view(3, 1, 1)).clamp(0, W - 1)
        bv_9 = (bv.unsqueeze(0) + offsets.view(1, 3, 1)).clamp(0, H - 1)
        
        bu_flat = bu_9.expand(3, 3, -1).reshape(-1)
        bv_flat = bv_9.expand(3, 3, -1).reshape(-1)
        idx = (bv_flat * W + bu_flat).long()

        # Apply depth weights to prioritize center pixels
        bz_weighted = bz.repeat(9) + weight_kernel.repeat_interleave(bz.shape[0])
        brgb_flat = brgb.repeat(9, 1)
        bxyz_flat = bxyz.repeat(9, 1)

        # Depth buffer test
        depth_buffer = torch.full((H * W,), float('inf'), device=device, dtype=dtype)
        depth_buffer.scatter_reduce_(0, idx, bz_weighted, reduce="amin", include_self=True)

        visible = (bz_weighted <= depth_buffer[idx])
        rgb_res[b, idx[visible]] = brgb_flat[visible]
        xyz_res[b, idx[visible]] = bxyz_flat[visible]

    return rgb_res.view(B, H, W, 3), xyz_res.view(B, H, W, 3), Pose_target


def process_and_save_pc_data_batched(
        xyz_mapA,
        rgbA,
        xyz_mapB=None,
        rgbB=None,
        Pose_target=None,
        PoseA=None,
        K=None,
        output_dir="output_results"):

    os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "xyz"), exist_ok=True)

    device = xyz_mapA.device
    dtype = xyz_mapA.dtype

    # 1. Parameter preprocessing
    PoseA = torch.as_tensor(PoseA, dtype=dtype, device=device)
    Pose_target = torch.as_tensor(Pose_target, dtype=dtype, device=device)
    K = torch.as_tensor(K, dtype=dtype, device=device)

    if PoseA.ndim == 2: PoseA = PoseA.unsqueeze(0)
    B = Pose_target.shape[0]
    if PoseA.shape[0] == 1 and B > 1: PoseA = PoseA.expand(B, -1, -1)
    if K.ndim == 2: K = K.unsqueeze(0).expand(B, -1, -1)

    # 2. Transformation setup
    R_A, t_A = PoseA[:, :3, :3], PoseA[:, :3, 3:] 
    R_T, t_T = Pose_target[:, :3, :3], Pose_target[:, :3, 3:]
    rel_R = torch.bmm(R_T, R_A.transpose(1, 2))

    # 3. Preprocess single-view point cloud
    def preprocess_map(x):
        if x.ndim == 4: x = x.permute(0, 2, 3, 1).squeeze(0)
        elif x.ndim == 3 and x.shape[0] <= 4: x = x.permute(1, 2, 0)
        return torch.where(torch.isfinite(x), x, torch.zeros_like(x))

    xyz_map_cl = preprocess_map(xyz_mapA)
    rgb_map_cl = preprocess_map(rgbA)
    H, W, _ = xyz_map_cl.shape

    xyz_flat_base = xyz_map_cl.reshape(-1, 3)
    rgb_flat_base = rgb_map_cl.reshape(-1, 3)

    if rgb_flat_base.dtype != torch.uint8:
        if rgb_flat_base.max() <= 1.01: rgb_flat_base = rgb_flat_base * 255.0
        rgb_flat_base = rgb_flat_base.clamp(0, 255).to(torch.uint8)

    # 4. Centralized transformation
    xyz_flat = xyz_flat_base.unsqueeze(0).expand(B, -1, -1)
    centered_xyz = xyz_flat - t_A.transpose(1, 2)
    xyz_rot = torch.bmm(centered_xyz, rel_R.transpose(1, 2)) + t_T.transpose(1, 2)

    z = xyz_rot[..., 2]
    valid_mask = (torch.norm(xyz_flat, dim=-1) > 1e-6) & (z > 1e-6)

    # 5. Projection calculation
    fx, fy = K[:, 0, 0].view(B, 1), K[:, 1, 1].view(B, 1)
    cx, cy = K[:, 0, 2].view(B, 1), K[:, 1, 2].view(B, 1)

    u = (fx * (xyz_rot[..., 0] / (z + 1e-8)) + cx).long()
    v = (fy * (xyz_rot[..., 1] / (z + 1e-8)) + cy).long()

    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & valid_mask

    # 6. Initialize output
    rgb_res = torch.zeros((B, H * W, 3), dtype=torch.uint8, device=device)
    xyz_res = torch.zeros((B, H * W, 3), dtype=dtype, device=device)

    # 7. Z-buffer rendering + Weighted Splatting
    weight_kernel = torch.tensor([
        0.002, 0.001, 0.002,
        0.001, 0.000, 0.001,
        0.002, 0.001, 0.002
    ], device=device, dtype=dtype)

    for b in range(B):
        m = in_bounds[b]
        if not m.any(): continue

        bu, bv, bz = u[b][m], v[b][m], z[b][m]
        brgb, bxyz = rgb_flat_base[m], xyz_rot[b][m]

        # 3x3 Splatting logic
        offsets = torch.tensor([-1, 0, 1], device=device)
        bu_9 = (bu.unsqueeze(0) + offsets.view(3, 1, 1)).clamp(0, W - 1)
        bv_9 = (bv.unsqueeze(0) + offsets.view(1, 3, 1)).clamp(0, H - 1)
        idx = (bv_9.expand(3, 3, -1).reshape(-1) * W + bu_9.expand(3, 3, -1).reshape(-1)).long()

        bz_weighted = bz.repeat(9) + weight_kernel.repeat_interleave(bz.shape[0])
        brgb_9 = brgb.repeat(9, 1)
        bxyz_9 = bxyz.repeat(9, 1)

        depth_buffer = torch.full((H * W,), float('inf'), device=device, dtype=dtype)
        depth_buffer.scatter_reduce_(0, idx, bz_weighted, reduce="amin", include_self=True)

        visible = (bz_weighted <= depth_buffer[idx])
        rgb_res[b, idx[visible]] = brgb_9[visible]
        xyz_res[b, idx[visible]] = bxyz_9[visible]

    return rgb_res.view(B, H, W, 3), xyz_res.view(B, H, W, 3), Pose_target

@torch.inference_mode()
def make_project_data_batch_init(Ref_pose, rgb_r, depth_r, xyz_map_rs, is_symmetric, symmetry_axis, render_size, batch_size, ob_in_cams, rgb, depth, K, crop_ratio, xyz_map=None, mesh_diameter=None, cfg=None, dataset:PoseRefinePairH5Dataset=None, iteration_iter = 5, ob_id=None, gt_pose=None):
    H, W = depth.shape[:2]
    B = len(ob_in_cams)
    method = 'box_3d'

    tf_to_crops = compute_crop_window_tf_batch(H=H, W=W, poses=ob_in_cams, K=K, crop_ratio=crop_ratio, out_size=(render_size[1], render_size[0]), method=method, mesh_diameter=mesh_diameter)

    poseA = torch.as_tensor(ob_in_cams, dtype=torch.float, device='cuda')
    
    if isinstance(K, np.ndarray):
        K_tensor = torch.tensor(K, dtype=torch.float32, device='cuda')
    else:
        K_tensor = K.to(device='cuda', dtype=torch.float32)

    if K_tensor.ndim == 2:
        K_tensor = K_tensor[None].expand(B, -1, -1)

    bbox2d_crop = torch.as_tensor(np.array([0, 0, cfg['input_resize'][0]-1, cfg['input_resize'][1]-1]).reshape(2,2), device='cuda', dtype=torch.float)
    bbox2d_ori = transform_pts(bbox2d_crop, tf_to_crops.inverse()).reshape(-1,4)
    
    if rgb_r.shape[-1] == 3:
        rgb_rs = rgb_r.permute(0, 3, 1, 2)
    else:
        rgb_rs = rgb_r
    
    depth_rs = depth_r.unsqueeze(1)
    Ks = K_tensor

    # Warp RGB and XYZ maps to crop window
    rgbBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(rgb, dtype=torch.float, device='cuda').permute(2,0,1)[None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
    xyz_mapBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(xyz_map, device='cuda', dtype=torch.float).permute(2,0,1)[None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)

    Ref_xyz_mapA = xyz_map_rs
    Ref_rgb_A = rgb_rs

    def update_K_with_tf(K, tf):
        return tf @ K

    K_crop = update_K_with_tf(K_tensor, tf_to_crops)
    
    # Reprojection for symmetric or non-symmetric objects
    if (is_symmetric):
      rgb_projA, xyz_projA, updated_poseA = process_and_save_pc_data_batched_v2(symmetry_axis, Ref_xyz_mapA, Ref_rgb_A, xyz_mapBs[0], rgbBs[0], poseA, Ref_pose, K_crop, output_dir="output_results_2")
    else:
       rgb_projA, xyz_projA, updated_poseA = process_and_save_pc_data_batched(Ref_xyz_mapA, Ref_rgb_A, xyz_mapBs[0], rgbBs[0], poseA, Ref_pose, K_crop, output_dir="output_results_2")

    rgbAs = rgb_projA.permute(0, 3, 1, 2)
    xyz_mapAs = xyz_projA.permute(0, 3, 1, 2)

    mesh_diameters = torch.ones((len(rgbAs)), dtype=torch.float, device='cuda')*mesh_diameter
    pose_data = BatchPoseData(rgbAs=rgbAs, rgbBs=rgbBs, depthAs=None, depthBs=None, normalAs=None, normalBs=None, poseA=poseA, poseB=None, xyz_mapAs=xyz_mapAs, xyz_mapBs=xyz_mapBs, tf_to_crops=tf_to_crops, Ks=Ks, mesh_diameters=mesh_diameters)
    pose_data = dataset.transform_batch(batch=pose_data, H_ori=H, W_ori=W, bound=1)

    return pose_data


class PoseRefinement:
  def __init__(self,):
    self.amp = True
    self.run_name = "2023-10-28-18-33-37"
    model_name = 'model_best.pth'
    code_dir = os.path.dirname(os.path.realpath(__file__))
    ckpt_dir = f'{code_dir}/../weights/{self.run_name}/{model_name}'

    self.cfg = OmegaConf.load(f'{code_dir}/../weights/{self.run_name}/config.yml')
    self.cfg['ckpt_dir'] = ckpt_dir
    self.cfg['enable_amp'] = True
    self.cfg['batch_size'] = 28
    self.cfg['input_resize'] = [160, 160]

    # Defaults for backward compatibility
    if 'use_normal' not in self.cfg: self.cfg['use_normal'] = False
    if 'use_mask' not in self.cfg: self.cfg['use_mask'] = False
    if 'use_BN' not in self.cfg: self.cfg['use_BN'] = False
    if 'c_in' not in self.cfg: self.cfg['c_in'] = 6
    if 'crop_ratio' not in self.cfg or self.cfg['crop_ratio'] is None: self.cfg['crop_ratio'] = 1.2
    if 'n_view' not in self.cfg: self.cfg['n_view'] = 1
    if 'trans_rep' not in self.cfg: self.cfg['trans_rep'] = 'tracknet'
    if 'rot_rep' not in self.cfg: self.cfg['rot_rep'] = 'axis_angle'
    if 'zfar' not in self.cfg: self.cfg['zfar'] = 3
    if 'normalize_xyz' not in self.cfg: self.cfg['normalize_xyz'] = False
    if isinstance(self.cfg['zfar'], str) and 'inf' in self.cfg['zfar'].lower(): self.cfg['zfar'] = np.inf
    if 'normal_uint8' not in self.cfg: self.cfg['normal_uint8'] = False

    self.dataset = PoseRefinePairH5Dataset(cfg=self.cfg, h5_file='', mode='test')

    self.model = RefineNet(cfg=self.cfg, c_in=self.cfg['c_in']).cuda()
    ckpt = torch.load(ckpt_dir)
    if 'model' in ckpt: ckpt = ckpt['model']
    self.model.load_state_dict(ckpt)
    self.model.cuda().eval()

    self.last_trans_update = None
    self.last_rot_update = None
    self.cadmodel_cache = None

  @torch.inference_mode()
  def predict(self, rgb, depth, K, ob_in_cams, xyz_map, mesh_diameter=None, iteration=5, ob_id=None):
    """
    Predict refined pose
    @rgb: image (H,W,3)
    @ob_in_cams: initial pose (N,4,4)
    """ 
    import os
    import logging
    import torchvision.utils as vutils

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    tf_to_center = np.eye(4)
    ob_centered_in_cams = ob_in_cams
    
    if not self.cfg.use_normal:
      normal_map = None

    bs = self.cfg.batch_size
    device = 'cuda'

    B_in_cams = torch.as_tensor(ob_centered_in_cams, device=device, dtype=torch.float)

    rgb_tensor = torch.as_tensor(rgb, device=device, dtype=torch.float)
    depth_tensor = torch.as_tensor(depth, device=device, dtype=torch.float)
    xyz_map_tensor = torch.as_tensor(xyz_map, device=device, dtype=torch.float)
    K = torch.as_tensor(K, dtype=torch.float, device=device)
    
    trans_normalizer = self.cfg['trans_normalizer']
    if not isinstance(trans_normalizer, float):
      trans_normalizer = torch.as_tensor(list(trans_normalizer), device=device, dtype=torch.float).reshape(1,3) 
    
    iteration = 3
    render_size = self.cfg['input_resize']

    save_dir = f"reference_database/linemod_real/{ob_id}"
    save_path = os.path.join(save_dir, "ref_data.pt")

    # Load reference database example
    ref_data = torch.load(save_path)
    Ref_xyz_mapA = ref_data["xyz_map"].to(device)
    Ref_rgb_A = ref_data["rgb"].to(device)
    Ref_pose = ref_data["pose"].to(device)
    depth_r = ref_data["depth"].to(device)
    
    obj_meta = ref_data.get("meta", {})
    crop_ratio = obj_meta.get("crop_ratio", self.cfg['crop_ratio'])

    configs = [
        (True, 0),    # X 轴对称
        (True, 1),    # Y 轴对称
        (True, 2),    # Z 轴对称
    ]

    test_pose = Ref_pose.unsqueeze(0)  # [1,4,4]
    best_loss = float('inf')
    best_sym = False
    best_axis = 0

    # debug_root = f"symmetry_check/ob{ob_id}"
    # os.makedirs(debug_root, exist_ok=True)
    # os.makedirs("output_results_2", exist_ok=True)
    pose_tmp_base = make_project_data_batch_init(
        Ref_pose, Ref_rgb_A, depth_r, Ref_xyz_mapA,
        False, -1,
        self.cfg.input_resize, 1,            
        test_pose, rgb_tensor, depth_tensor, K,
        crop_ratio=crop_ratio,               
        xyz_map=xyz_map_tensor,
        cfg=self.cfg,
        dataset=self.dataset,
        mesh_diameter=mesh_diameter,
        iteration_iter=0,
        ob_id=ob_id
    )
    xyzA0 = pose_tmp_base.xyz_mapAs[0]  # [3, H, W]

    # 根据要求，使用 ref_xyzA0 的前两个通道获取 mask，形状为 [H, W]
    mask = (xyzA0[:2] > 0).any(dim=0)

    # 2. 遍历可能的对称性轴
    for is_sym, sym_axis in configs:
        pose_tmp = make_project_data_batch_init(
            Ref_pose, Ref_rgb_A, depth_r, Ref_xyz_mapA,
            is_sym, sym_axis,
            self.cfg.input_resize, 1,            
            test_pose, rgb_tensor, depth_tensor, K,
            crop_ratio=crop_ratio,               
            xyz_map=xyz_map_tensor,
            cfg=self.cfg,
            dataset=self.dataset,
            mesh_diameter=mesh_diameter,
            iteration_iter=0,
            ob_id=ob_id
        )
        rgbA = pose_tmp.rgbAs[0]      # [3, H, W] 合成渲染
        xyzA = pose_tmp.xyz_mapAs[0]  # [3, H, W]

        # 辅助保存图片
        img = rgbA.detach().float()
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        # vutils.save_image(img, os.path.join("output_results_2", f"rgbA_{ob_id}_{sym_axis}_{is_sym}.png"))
        
        if mask.sum() < 10:
            loss = 10.0
        else:
            # 严格在 mask 区域内计算 L1 误差 (3通道上的总偏差 / 有效像素数)
            diff = (xyzA - xyzA0).abs().sum() / mask.sum().float()
            loss = diff.item()

        logging.info(f"Symmetry candidate: is_sym={is_sym}, axis={sym_axis}, loss={loss:.4f}")

        if loss < best_loss:
            best_loss = loss
            best_sym = is_sym
            best_axis = sym_axis

    # 3. 选定最优对称性
    # if best_loss > 0.01:
    if best_loss > 0.3:
      is_symmetric = False
      symmetry_axis = -1
    else:
      is_symmetric = best_sym
      symmetry_axis = best_axis
    
    logging.info(f"ob_id: {ob_id}, Selected symmetry: is_symmetric={is_symmetric}, axis={symmetry_axis} (loss={best_loss:.4f})")
    # ==========================================================

    # 进入主优化循环，应用刚刚检测出的 is_symmetric 和 symmetry_axis
    for ii in range(iteration):
      # Prepare projected crop data
      pose_data = make_project_data_batch_init(
          Ref_pose, Ref_rgb_A, depth_r, Ref_xyz_mapA, 
          is_symmetric, symmetry_axis, 
          self.cfg.input_resize, self.cfg.batch_size, 
          B_in_cams, rgb_tensor, depth_tensor, K, 
          crop_ratio=crop_ratio, xyz_map=xyz_map_tensor, 
          cfg=self.cfg, dataset=self.dataset, 
          mesh_diameter=mesh_diameter, iteration_iter=ii, ob_id=ob_id
      )
      B_in_cams = []
      
      bs = pose_data.rgbAs.shape[0]
      for b in range(0, pose_data.rgbAs.shape[0], bs):
        A = torch.cat([pose_data.rgbAs[b:b+bs].cuda(), pose_data.xyz_mapAs[b:b+bs].cuda()], dim=1).float()
        B = torch.cat([pose_data.rgbBs[b:b+bs].cuda(), pose_data.xyz_mapBs[b:b+bs].cuda()], dim=1).float()
        
        with torch.no_grad():
          with torch.cuda.amp.autocast(enabled=self.amp):
            output = self.model(A, B)
        
        for k in output:
          output[k] = output[k].float()

        # Translation update
        if self.cfg['trans_rep']=='tracknet':
          if not self.cfg['normalize_xyz']:
            trans_delta = torch.tanh(output["trans"])*trans_normalizer
          else:
            trans_delta = output["trans"]
        elif self.cfg['trans_rep']=='deepim':
          rot_delta = output["rot"]
          z_pred = output['trans'][:,2]*pose_data.poseA[b:b+bs][...,2,3]
          uvA_crop = -(pose_data.poseA[b:b+bs][...,:3,3])
          uv_pred_crop = uvA_crop + output['trans'][:,:2]*self.cfg['input_resize'][0]
          uv_pred = transform_pts(uv_pred_crop, pose_data.tf_to_crops[b:b+bs].inverse().cuda())
          center_pred = torch.cat([uv_pred, torch.ones((len(rot_delta),1), dtype=torch.float, device='cuda')], dim=-1)
          center_pred = (pose_data.Ks[b:b+bs].inverse().cuda()@center_pred.reshape(len(rot_delta),3,1)).reshape(len(rot_delta),3) * z_pred.reshape(len(rot_delta),1)
          trans_delta = center_pred-pose_data.poseA[b:b+bs][...,:3,3]
        else:
          trans_delta = output["trans"]
        
        # Rotation update
        if self.cfg['rot_rep']=='axis_angle':
          rot_mat_delta = torch.tanh(output["rot"])*self.cfg['rot_normalizer']
          rot_mat_delta = so3_exp_map(rot_mat_delta).permute(0,2,1)
        elif self.cfg['rot_rep']=='6d':
          rot_mat_delta = rotation_6d_to_matrix(output['rot']).permute(0,2,1)
        else:
          raise RuntimeError

        if self.cfg['normalize_xyz']:
          trans_delta *= (mesh_diameter/2)

        # Apply delta pose
        B_in_cam = egocentric_delta_pose_to_pose(pose_data.poseA[b:b+bs], trans_delta=trans_delta, rot_mat_delta=rot_mat_delta)
        B_in_cams.append(B_in_cam)

      B_in_cams = torch.cat(B_in_cams, dim=0).reshape(len(B_in_cams[0]),4,4)

    B_in_cams_out = B_in_cams@torch.tensor(tf_to_center[None], device='cuda', dtype=torch.float)
    torch.cuda.empty_cache()
    return B_in_cams_out, None