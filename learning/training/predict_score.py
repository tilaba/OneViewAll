# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import functools
import os,sys,kornia
import time
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../../../')
from learning.datasets.h5_dataset import *
from learning.models.score_network import *
from learning.datasets.pose_dataset import *
from Utils import *
from datareader import *
from scipy.spatial.transform import Rotation as R
from torchvision.utils import save_image
from PIL import Image, ImageDraw
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def vis_batch_data_scores(pose_data, ids, scores, pad_margin=5):
  assert len(scores)==len(ids)
  canvas = []
  for id in ids:
    rgbA_vis = (pose_data.rgbAs[id]*255).permute(1,2,0).data.cpu().numpy()
    rgbB_vis = (pose_data.rgbBs[id]*255).permute(1,2,0).data.cpu().numpy()
    H,W = rgbA_vis.shape[:2]
    zmin = pose_data.depthAs[id].data.cpu().numpy().reshape(H,W).min()
    zmax = pose_data.depthAs[id].data.cpu().numpy().reshape(H,W).max()
    depthA_vis = depth_to_vis(pose_data.depthAs[id].data.cpu().numpy().reshape(H,W), zmin=zmin, zmax=zmax, inverse=False)
    depthB_vis = depth_to_vis(pose_data.depthBs[id].data.cpu().numpy().reshape(H,W), zmin=zmin, zmax=zmax, inverse=False)
    if pose_data.normalAs is not None:
      pass
    pad = np.ones((rgbA_vis.shape[0],pad_margin,3))*255
    if pose_data.normalAs is not None:
      pass
    else:
      row = np.concatenate([rgbA_vis, pad, depthA_vis, pad, rgbB_vis, pad, depthB_vis], axis=1)
    s = 100/row.shape[0]
    row = cv2.resize(row, fx=s, fy=s, dsize=None)
    row = cv_draw_text(row, text=f'id:{id}, score:{scores[id]:.3f}', uv_top_left=(10,10), color=(0,255,0), fontScale=0.5)
    canvas.append(row)
    pad = np.ones((pad_margin, row.shape[1], 3))*255
    canvas.append(pad)
  canvas = np.concatenate(canvas, axis=0).astype(np.uint8)
  return canvas

def create_mirrored_xyz_map(symmetry_axis, xyz_map, PoseA):
    """
    镜像逻辑：将点云变换到物体坐标系，进行镜像翻转，再转回相机空间。
    """
    # 兼容 (3, H, W) 格式
    if xyz_map.shape[0] == 3 and xyz_map.ndim == 3:
        xyz_map = xyz_map.permute(1, 2, 0)
    
    device = xyz_map.device
    dtype = xyz_map.dtype
    H, W, _ = xyz_map.shape
    
    # 1. 提取有效点（避免对零点或无穷大进行变换）
    mask = (torch.isfinite(xyz_map).all(dim=-1)) & (xyz_map[..., 2] > 1e-5)
    xyz_v = xyz_map[mask] # [N, 3]

    if xyz_v.shape[0] == 0:
        return torch.zeros((H, W, 3), device=device, dtype=dtype)

    # 2. 提取 Pose 参数 (修正索引错误)
    R_A = PoseA[:3, :3]  # [3, 3]
    t_A = PoseA[:3, 3:4] # [3, 1]
    
    # 3. 变换：相机系 -> 物体系
    # (X_cam - t) = R @ X_obj  =>  X_obj = R^T @ (X_cam - t)
    xyz_obj = torch.mm(R_A.t(), xyz_v.t() - t_A) # [3, N]

    # 4. 镜像翻转逻辑
    # 绕 Y 轴镜像对称平面是 X-Z 平面，操作是 X 轴取反
    # 如果你想镜像 X-Y 平面（补全前后），则操作 Z 轴
    xyz_obj[symmetry_axis, :] *= -1  # 这里反转 X 轴，实现左右/前后补全（视物体朝向而定）

    # 5. 变换回相机系：物体系 -> 相机系
    xyz_cam_mirrored = (torch.mm(R_A, xyz_obj) + t_A).t() # [N, 3]

    # 6. 写回 map
    mirrored_map = torch.zeros((H, W, 3), device=device, dtype=dtype)
    mirrored_map[mask] = xyz_cam_mirrored
    
    return mirrored_map



import torch
import os

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

    device = xyz_mapA.device
    dtype = xyz_mapA.dtype
    B = Pose_target.shape[0]

    # 1. 参数预处理
    PoseA = torch.as_tensor(PoseA, dtype=dtype, device=device)
    Pose_target = torch.as_tensor(Pose_target, dtype=dtype, device=device)
    K = torch.as_tensor(K, dtype=dtype, device=device)

    if PoseA.ndim == 2: PoseA = PoseA.unsqueeze(0)
    if PoseA.shape[0] == 1: PoseA = PoseA.expand(B, -1, -1)
    if K.ndim == 2: K = K.unsqueeze(0).expand(B, -1, -1)

    # 2. 变换矩阵准备
    R_A, t_A = PoseA[:, :3, :3], PoseA[:, :3, 3:] 
    R_T, t_T = Pose_target[:, :3, :3], Pose_target[:, :3, 3:]
    rel_R = torch.bmm(R_T, R_A.transpose(1, 2))

    # 3. 图像预处理与镜像融合 (向量化处理)
    def preprocess_map(x):
        if x.ndim == 4: x = x.squeeze(0).permute(1, 2, 0)
        elif x.ndim == 3 and x.shape[0] <= 4: x = x.permute(1, 2, 0)
        return torch.where(torch.isfinite(x), x, torch.zeros_like(x))

    xyz_mapA_cl = preprocess_map(xyz_mapA)
    rgbA_cl = preprocess_map(rgbA)
    H, W, _ = xyz_mapA_cl.shape

    # 镜像逻辑: 假设 create_mirrored_xyz_map 已经支持
    xyz_mirrored_cl = create_mirrored_xyz_map(symmetry_axis, xyz_mapA_cl, PoseA[0])
    
    # 合并点云 [N_total, 3], 其中 N_total = 2 * H * W
    xyz_combined = torch.cat([xyz_mapA_cl.reshape(-1, 3), xyz_mirrored_cl.reshape(-1, 3)], dim=0)

    # 颜色合并: 原图 RGB + 镜像灰度图
    gray = rgbA_cl.float().mean(dim=2, keepdim=True).to(rgbA_cl.dtype)
    rgb_combined = torch.cat([rgbA_cl.reshape(-1, 3), gray.expand(-1, -1, 3).reshape(-1, 3)], dim=0)

    if rgb_combined.dtype != torch.uint8:
        if rgb_combined.max() <= 1.01: rgb_combined = rgb_combined * 255.0
        rgb_combined = rgb_combined.clamp(0, 255).to(torch.uint8)

    # 4. 批量坐标变换 [B, N_total, 3]
    xyz_flat = xyz_combined.unsqueeze(0).expand(B, -1, -1)
    centered_xyz = xyz_flat - t_A.transpose(1, 2)
    xyz_rot = torch.bmm(centered_xyz, rel_R.transpose(1, 2)) + t_T.transpose(1, 2)

    z = xyz_rot[..., 2]
    valid_mask = (torch.norm(xyz_flat, dim=-1) > 1e-6) & (z > 1e-6)

    # 5. 投影计算
    fx, fy = K[:, 0, 0].view(B, 1), K[:, 1, 1].view(B, 1)
    cx, cy = K[:, 0, 2].view(B, 1), K[:, 1, 2].view(B, 1)
    u = (fx * (xyz_rot[..., 0] / (z + 1e-8)) + cx).long()
    v = (fy * (xyz_rot[..., 1] / (z + 1e-8)) + cy).long()

    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & valid_mask

    # 6. 全局并行 Z-buffer Splatting
    b_idx, n_idx = torch.where(in_bounds)
    if b_idx.numel() == 0:
        return torch.zeros((B, H, W, 3), dtype=torch.uint8, device=device), \
               torch.zeros((B, H, W, 3), dtype=dtype, device=device), Pose_target

    active_u, active_v, active_z = u[b_idx, n_idx], v[b_idx, n_idx], z[b_idx, n_idx]
    active_rgb, active_xyz = rgb_combined[n_idx], xyz_rot[b_idx, n_idx]

    # 3x3 邻域扩散与权重
    offsets = torch.tensor([-1, 0, 1], device=device)
    dv, du = torch.meshgrid(offsets, offsets, indexing='ij')
    dv, du = dv.flatten(), du.flatten()
    weight_kernel = torch.tensor([0.002, 0.001, 0.002, 0.001, 0.0, 0.001, 0.002, 0.001, 0.002], device=device, dtype=dtype)

    u_9 = (active_u.unsqueeze(1) + du).clamp(0, W - 1)
    v_9 = (active_v.unsqueeze(1) + dv).clamp(0, H - 1)
    z_9 = active_z.unsqueeze(1) + weight_kernel
    
    flat_idx = (b_idx.unsqueeze(1) * (H * W) + v_9 * W + u_9).reshape(-1)
    z_9_flat = z_9.reshape(-1)

    # 深度竞争
    total_pix = B * H * W
    depth_buffer = torch.full((total_pix,), float('inf'), device=device, dtype=dtype)
    depth_buffer.scatter_reduce_(0, flat_idx, z_9_flat, reduce="amin", include_self=True)

    # 7. 写入最终结果 [B, H, W, 3]
    winner_mask = (z_9_flat <= depth_buffer[flat_idx])
    final_indices = flat_idx[winner_mask]
    
    rgb_res_flat = torch.zeros((total_pix, 3), dtype=torch.uint8, device=device)
    xyz_res_flat = torch.zeros((total_pix, 3), dtype=dtype, device=device)
    
    # 这里使用 repeat_interleave(9) 是因为每个 active 点扩散成了 9 个点
    rgb_res_flat[final_indices] = active_rgb.repeat_interleave(9, dim=0)[winner_mask]
    xyz_res_flat[final_indices] = active_xyz.repeat_interleave(9, dim=0)[winner_mask]

    rgb_res = rgb_res_flat.view(B, H, W, 3)
    xyz_res = xyz_res_flat.view(B, H, W, 3)

    return rgb_res, xyz_res, Pose_target


def process_and_save_pc_data_batched(
        xyz_mapA,
        rgbA,
        xyz_mapB=None,
        rgbB=None,
        Pose_target=None,
        PoseA=None,
        K=None,
        output_dir="output_results"):

    device = xyz_mapA.device
    dtype = xyz_mapA.dtype
    B = Pose_target.shape[0]

    # 1. 参数预处理 (保持逻辑不变)
    PoseA = torch.as_tensor(PoseA, dtype=dtype, device=device)
    Pose_target = torch.as_tensor(Pose_target, dtype=dtype, device=device)
    K = torch.as_tensor(K, dtype=dtype, device=device)

    if PoseA.ndim == 2: PoseA = PoseA.unsqueeze(0)
    if PoseA.shape[0] == 1: PoseA = PoseA.expand(B, -1, -1)
    if K.ndim == 2: K = K.unsqueeze(0).expand(B, -1, -1)

    R_A, t_A = PoseA[:, :3, :3], PoseA[:, :3, 3:] 
    R_T, t_T = Pose_target[:, :3, :3], Pose_target[:, :3, 3:]
    rel_R = torch.bmm(R_T, R_A.transpose(1, 2))

    # 2. 维度对齐与预处理
    def preprocess_map(x):
        if x.ndim == 4: x = x.squeeze(0).permute(1, 2, 0)
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

    # 3. 批量坐标变换 [B, N, 3]
    xyz_flat = xyz_flat_base.unsqueeze(0).expand(B, -1, -1)
    centered_xyz = xyz_flat - t_A.transpose(1, 2)
    xyz_rot = torch.bmm(centered_xyz, rel_R.transpose(1, 2)) + t_T.transpose(1, 2)

    z = xyz_rot[..., 2]
    valid_mask = (torch.norm(xyz_flat, dim=-1) > 1e-6) & (z > 1e-6)

    # 4. 投影计算
    fx, fy = K[:, 0, 0].view(B, 1), K[:, 1, 1].view(B, 1)
    cx, cy = K[:, 0, 2].view(B, 1), K[:, 1, 2].view(B, 1)
    u = (fx * (xyz_rot[..., 0] / (z + 1e-8)) + cx).long()
    v = (fy * (xyz_rot[..., 1] / (z + 1e-8)) + cy).long()

    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & valid_mask

    # 5. 全局 Z-buffer Splatting (关键优化点)
    # 获取所有 Batch 中有效点的索引
    b_idx, n_idx = torch.where(in_bounds)
    if b_idx.numel() == 0:
        return torch.zeros((B, 3, H, W), dtype=torch.uint8, device=device), \
               torch.zeros((B, 3, H, W), dtype=dtype, device=device), Pose_target

    # 提取有效点的坐标、深度、RGB和旋转后的XYZ
    active_u, active_v, active_z = u[b_idx, n_idx], v[b_idx, n_idx], z[b_idx, n_idx]
    active_rgb, active_xyz = rgb_flat_base[n_idx], xyz_rot[b_idx, n_idx]

    # 3x3 邻域偏移
    offsets = torch.tensor([-1, 0, 1], device=device)
    dv, du = torch.meshgrid(offsets, offsets, indexing='ij')
    dv, du = dv.flatten(), du.flatten() # [9]
    weight_kernel = torch.tensor([0.002, 0.001, 0.002, 0.001, 0.0, 0.001, 0.002, 0.001, 0.002], device=device, dtype=dtype)

    # 扩散到 3x3 邻域: [M] -> [M, 9]
    u_9 = (active_u.unsqueeze(1) + du).clamp(0, W - 1)
    v_9 = (active_v.unsqueeze(1) + dv).clamp(0, H - 1)
    z_9 = active_z.unsqueeze(1) + weight_kernel # 应用精度补偿权重
    
    # 计算全局扁平化索引: Batch_ID * (H*W) + V * W + U
    flat_idx = (b_idx.unsqueeze(1) * (H * W) + v_9 * W + u_9).reshape(-1)
    z_9_flat = z_9.reshape(-1)

    # 并行 Z-buffer Amin 测试
    total_pix = B * H * W
    depth_buffer = torch.full((total_pix,), float('inf'), device=device, dtype=dtype)
    depth_buffer.scatter_reduce_(0, flat_idx, z_9_flat, reduce="amin", include_self=True)

    # 找出获胜的像素位置
    winner_mask = (z_9_flat <= depth_buffer[flat_idx])
    final_indices = flat_idx[winner_mask]
    
    # 6. 结果组装
    # 直接初始化目标形状，避免后续再次 reshape
    rgb_res_flat = torch.zeros((total_pix, 3), dtype=torch.uint8, device=device)
    xyz_res_flat = torch.zeros((total_pix, 3), dtype=dtype, device=device)
    
    # 映射回原始点的数据
    rgb_res_flat[final_indices] = active_rgb.repeat_interleave(9, dim=0)[winner_mask]
    xyz_res_flat[final_indices] = active_xyz.repeat_interleave(9, dim=0)[winner_mask]

    # 7. 格式转换 (按 FoundationPose 要求转为 [B, 3, H, W])
    rgb_res = rgb_res_flat.view(B, H, W, 3)
    xyz_res = xyz_res_flat.view(B, H, W, 3)

    return rgb_res, xyz_res, Pose_target

tf_to_crops = None
@torch.inference_mode()
def make_project_data_batch_init(Ref_pose, rgb_r, depth_r, xyz_map_rs, is_symmetric, symmetry_axis, render_size, ob_in_cams, rgb, depth, K, crop_ratio, xyz_map=None, ob_mask=None, mesh_diameter=None, cfg=None, dataset:PoseRefinePairH5Dataset=None, iteration_iter = 5, ob_id=None):
    H,W = depth_r.shape[:2]
    B = len(ob_in_cams)
    args = []
    method = 'box_3d'
    tf_to_crops = compute_crop_window_tf_batch(H=H, W=W, poses=ob_in_cams, K=K, crop_ratio=crop_ratio, out_size=(render_size[1], render_size[0]), method=method, mesh_diameter=mesh_diameter)
    # logging.info("make tf_to_crops done")
    poseA = torch.as_tensor(ob_in_cams, dtype=torch.float, device='cuda')
    rgb_rs = []
    depth_rs = []
    normal_rs = []
    B = len(poseA)
    extra = {}
    if isinstance(K, np.ndarray):
        K_tensor = torch.tensor(K, dtype=torch.float32, device='cuda')
    else:
        K_tensor = K.to(device='cuda', dtype=torch.float32)

    if K_tensor.ndim == 2:
        K_tensor = K_tensor[None].expand(B, -1, -1)  # [B, 3, 3]

    bbox2d_crop = torch.as_tensor(np.array([0, 0, cfg['input_resize'][0]-1, cfg['input_resize'][1]-1]).reshape(2,2), device='cuda', dtype=torch.float)
    bbox2d_ori = transform_pts(bbox2d_crop, tf_to_crops.inverse()).reshape(-1,4)
    #   rgb_r, depth_r, normal_r = renderer.render(ob_in_cams=poseA, bbox2d=bbox2d_ori, get_normal=cfg['use_normal'], extra=extra)
    start_time0 = time.time()
    index = ob_in_cams.shape[0] - 1
    end_time = time.time()
    logging.info(f"rendering time: {end_time - start_time0:.3f} seconds")

    rgb_rs = rgb_r.permute(0, 3, 1, 2) * 255           # (B, 3, H, W)
    depth_rs = depth_r.unsqueeze(1)                   # (B, 1, H, W)
    Ks = K_tensor

    Ref_xyz_mapA = xyz_map_rs
    Ref_rgb_A = rgb_rs
    rgbBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(rgb, dtype=torch.float, device='cuda').permute(2,0,1)[None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
    xyz_mapBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(xyz_map, device='cuda', dtype=torch.float).permute(2,0,1)[None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)  #(B,3,H,W)


    rgbAs = rgb_rs
    xyz_mapAs = xyz_map_rs
    def update_K_with_tf(K, tf):
        return tf @ K

    K_crop = update_K_with_tf(K_tensor, tf_to_crops)
    if (is_symmetric):
      rgb_projA, xyz_projA, updated_poseA = process_and_save_pc_data_batched_v2(symmetry_axis, Ref_xyz_mapA, Ref_rgb_A, xyz_mapBs[0], rgbBs[0], poseA,  Ref_pose, K_crop, output_dir="output_results_2")
    else:
      rgb_projA, xyz_projA, updated_poseA = process_and_save_pc_data_batched(Ref_xyz_mapA, Ref_rgb_A, xyz_mapBs[0], rgbBs[0], poseA,  Ref_pose, K_crop, output_dir="output_results_2")
    rgbAs = rgb_projA.permute(0, 3, 1, 2)
    xyz_mapAs = xyz_projA.permute(0, 3, 1, 2)

    mesh_diameters = torch.ones((len(rgbAs)), dtype=torch.float, device='cuda')*mesh_diameter
    pose_data = BatchPoseData(rgbAs=rgbAs, rgbBs=rgbBs, depthAs=None, depthBs=None, normalAs=None, normalBs=None, poseA=poseA, poseB=None, xyz_mapAs=xyz_mapAs, xyz_mapBs=xyz_mapBs, tf_to_crops=tf_to_crops, Ks=Ks, mesh_diameters=mesh_diameters)
    pose_data = dataset.transform_batch(batch=pose_data, H_ori=H, W_ori=W, bound=1)

    # torch.cuda.synchronize()
    end_time0 = time.time()
    logging.info(f"compute_crop_window_tf_batch: {end_time0 - start_time0:.3f} seconds")

    return pose_data


@torch.no_grad()
def make_crop_data_batch(render_size, ob_in_cams, mesh, rgb, depth, K, crop_ratio, normal_map=None, mesh_diameter=None, glctx=None, mesh_tensors=None, dataset:TripletH5Dataset=None, cfg=None):
  H,W = depth.shape[:2]

  args = []
  method = 'box_3d'
  tf_to_crops = compute_crop_window_tf_batch(pts=mesh.vertices, H=H, W=W, poses=ob_in_cams, K=K, crop_ratio=crop_ratio, out_size=(render_size[1], render_size[0]), method=method, mesh_diameter=mesh_diameter)
  #logging.info("make tf_to_crops done")

  B = len(ob_in_cams)
  extra = {}
  poseAs = torch.as_tensor(ob_in_cams, dtype=torch.float, device='cuda')

  bs = min(512, ob_in_cams.shape[0])
  rgb_rs = []
  depth_rs = []
  xyz_map_rs = []

  bbox2d_crop = torch.as_tensor(np.array([0, 0, cfg['input_resize'][0]-1, cfg['input_resize'][1]-1]).reshape(2,2), device='cuda', dtype=torch.float)
  tf_to_crops = tf_to_crops.to(device='cuda')
  bbox2d_ori = transform_pts(bbox2d_crop, tf_to_crops.inverse()[:,None]).reshape(-1,4)
    # Ks: (B, 3, 3) if needed, 否则广播
  if isinstance(K, np.ndarray):
      K_tensor = torch.tensor(K, dtype=torch.float32, device='cuda')
  else:
      K_tensor = K.to(device='cuda', dtype=torch.float32)

  if K_tensor.ndim == 2:
      K_tensor = K_tensor[None].expand(B, -1, -1)  # [B, 3, 3]


  start_time = time.time()
  for b in range(0,len(ob_in_cams),bs):
    extra = {}
    rgb_r, depth_r, normal_r = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=poseAs[b:b+bs], context='cuda', get_normal=cfg['use_normal'], glctx=glctx, mesh_tensors=mesh_tensors, output_size=cfg['input_resize'], bbox2d=bbox2d_ori[b:b+bs], use_light=True, extra=extra)
    rgb_rs.append(rgb_r)
    depth_rs.append(depth_r[...,None])
    xyz_map_rs.append(extra['xyz_map'])

  end_time = time.time()
  logging.info(f"predict_score rendering time: {end_time - start_time:.3f} seconds")

  rgb_rs = torch.cat(rgb_rs, dim=0).permute(0,3,1,2) * 255
  depth_rs = torch.cat(depth_rs, dim=0).permute(0,3,1,2)
  xyz_map_rs = torch.cat(xyz_map_rs, dim=0).permute(0,3,1,2)  #(B,3,H,W)
  #logging.info("render done")

  rgbBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(rgb, dtype=torch.float, device='cuda').permute(2,0,1)[None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
  # save_image(rgbBs[0]/255, f"output_results_2/rgba_original_B.png")
  depthBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(depth, dtype=torch.float, device='cuda')[None,None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  if rgb_rs.shape[-2:]!=cfg['input_resize']:
    rgbAs = kornia.geometry.transform.warp_perspective(rgb_rs, tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
    depthAs = kornia.geometry.transform.warp_perspective(depth_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  else:
    rgbAs = rgb_rs
    depthAs = depth_rs

  if xyz_map_rs.shape[-2:]!=cfg['input_resize']:
    xyz_mapAs = kornia.geometry.transform.warp_perspective(xyz_map_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  else:
    xyz_mapAs = xyz_map_rs

  normalAs = None
  normalBs = None

  Ks = torch.as_tensor(K, dtype=torch.float).reshape(1,3,3).expand(len(rgbAs),3,3)
  mesh_diameters = torch.ones((len(rgbAs)), dtype=torch.float, device='cuda')*mesh_diameter

  pose_data = BatchPoseData(rgbAs=rgbAs, rgbBs=rgbBs, depthAs=depthAs, depthBs=depthBs, normalAs=normalAs, normalBs=normalBs, poseA=poseAs, xyz_mapAs=xyz_mapAs, tf_to_crops=tf_to_crops, Ks=Ks, mesh_diameters=mesh_diameters)
  pose_data = dataset.transform_batch(pose_data, H_ori=H, W_ori=W, bound=1)

  #logging.info("pose batch data done")

  return pose_data


class ScorePredictor:
  def __init__(self, amp=True):
    self.amp = amp
    self.run_name = "2024-01-11-20-02-45"

    model_name = 'model_best.pth'
    code_dir = os.path.dirname(os.path.realpath(__file__))
    ckpt_dir = f'{code_dir}/../../weights/{self.run_name}/{model_name}'

    self.cfg = OmegaConf.load(f'{code_dir}/../../weights/{self.run_name}/config.yml')

    self.cfg['ckpt_dir'] = ckpt_dir
    self.cfg['enable_amp'] = True

    ########## Defaults, to be backward compatible
    if 'use_normal' not in self.cfg:
      self.cfg['use_normal'] = False
    if 'use_BN' not in self.cfg:
      self.cfg['use_BN'] = False
    if 'zfar' not in self.cfg:
      self.cfg['zfar'] = np.inf
    if 'c_in' not in self.cfg:
      self.cfg['c_in'] = 6
    if 'normalize_xyz' not in self.cfg:
      self.cfg['normalize_xyz'] = False
    if 'crop_ratio' not in self.cfg or self.cfg['crop_ratio'] is None:
      self.cfg['crop_ratio'] = 1.2

    #logging.info(f"self.cfg: \n {OmegaConf.to_yaml(self.cfg)}")

    self.dataset = ScoreMultiPairH5Dataset(cfg=self.cfg, mode='test', h5_file=None, max_num_key=1)
    self.model = ScoreNetMultiPair(cfg=self.cfg, c_in=self.cfg['c_in']).cuda()

    #logging.info(f"Using pretrained model from {ckpt_dir}")
    ckpt = torch.load(ckpt_dir)
    if 'model' in ckpt:
      ckpt = ckpt['model']
    self.model.load_state_dict(ckpt)
    self.model.cuda().eval()
    #logging.info("init done")


  @torch.inference_mode()
  def predict(self, rgb, depth, K, ob_in_cams, ob_mask, ob_id, normal_map=None, xyz_map = None, get_vis=False, mesh=None, mesh_tensors=None, glctx=None, mesh_diameter=None):
    '''
    @rgb: np array (H,W,3)
    '''
    ob_in_cams = torch.as_tensor(ob_in_cams, dtype=torch.float, device='cuda')

    #logging.info(f'self.cfg.use_normal:{self.cfg.use_normal}')
    if not self.cfg.use_normal:
      normal_map = None

    #logging.info("making cropped data")

    if mesh_tensors is None:
      mesh_tensors = make_mesh_tensors(mesh)

    rgb = torch.as_tensor(rgb, device='cuda', dtype=torch.float)
    depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)
    ob_mask_tensor = torch.as_tensor(ob_mask, device='cuda', dtype=torch.float)
    xyz_map_tensor = torch.as_tensor(xyz_map, device='cuda', dtype=torch.float)
    render_size = self.cfg['input_resize']

    # new_pose1 = ob_in_cams[0:1].clone()
    # R_delta = R.from_euler('xyz', [120, 0, 350], degrees=True).as_matrix()
    # new_pose1[0][:3,:3] = torch.tensor(R_delta, dtype=ob_in_cams.dtype, device=ob_in_cams.device) 
    # new_pose1[0][:3, 3] = torch.tensor([-0.01, 0.0, 1.0], dtype=ob_in_cams.dtype, device=ob_in_cams.device)
    # H,W = depth.shape[:2]  
    # tf_to_crops = compute_crop_window_tf_batch(pts=mesh.vertices, H=H, W=W, poses=new_pose1, K=K, crop_ratio=crop_ratio, out_size=(render_size[1], render_size[0]), method='box_3d', mesh_diameter=mesh_diameter)
    # bbox2d_crop = torch.as_tensor(np.array([0, 0, self.cfg['input_resize'][0]-1, self.cfg['input_resize'][1]-1]).reshape(2,2), device='cuda', dtype=torch.float)
    # bbox2d_ori = transform_pts(bbox2d_crop, tf_to_crops.inverse()).reshape(-1,4)
    # if isinstance(K, np.ndarray):
    #   K_tensor = torch.tensor(K, dtype=torch.float32, device='cuda')
    # else:
    #     K_tensor = K.to(device='cuda', dtype=torch.float32)
    # extra = {}
    
    # rgb_r, depth_r, _ = nvdiffrast_render(
    #     K=K_tensor,                     # [B, 3, 3]
    #     H=H,
    #     W=W,
    #     ob_in_cams=new_pose1,              # [B, 4, 4]
    #     context='cuda',
    #     get_normal=self.cfg['use_normal'],
    #     glctx=glctx,
    #     mesh_tensors=mesh_tensors,
    #     output_size=self.cfg['input_resize'],
    #     bbox2d=bbox2d_ori,            # [B, 4]
    #     use_light=True,
    #     extra=extra)
    # xyz_map_rs = extra['xyz_map'].permute(0, 3, 1, 2) # (B, 3, H, W)
   
    # save_dir = f"reference_database/linemod/{ob_id}"
    # save_path = os.path.join(save_dir, "ref_data.pt")
    # data = torch.load(save_path)
    # Ref_xyz_mapA = data["xyz_map"].to(device)
    # Ref_rgb_A = data["rgb"].to(device)
    # Ref_pose = data["pose"].to(device)
    # depth_r = data["depth"].to(device)
    # obj_meta = data.get("meta", {})
    # is_symmetric = obj_meta.get("is_symmetric")
    # symmetry_axis = obj_meta.get("symmetry_axis")
    # crop_ratio = obj_meta.get("crop_ratio")
    crop_ratio = 1.2
    if (ob_id == 1):
      crop_ratio = 1.0
    start_time = time.time()
    # pose_data = make_crop_data_batch(self.cfg.input_resize, ob_in_cams, mesh, rgb, depth, K, crop_ratio=crop_ratio, glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset, cfg=self.cfg, mesh_diameter=mesh_diameter)
    pose_data = make_project_data_batch_init(
        Ref_pose, Ref_rgb_A, depth_r, Ref_xyz_mapA,is_symmetric, symmetry_axis,
        render_size, ob_in_cams, rgb, depth, K, crop_ratio, xyz_map, 
        ob_mask, mesh_diameter=mesh_diameter, cfg=self.cfg, dataset=self.dataset, iteration_iter=5, ob_id=None
    )

    end_time = time.time()
    start_time0 = time.time()
    def find_best_among_pairs(pose_data:BatchPoseData):
      #logging.info(f'pose_data.rgbAs.shape[0]: {pose_data.rgbAs.shape[0]}')
      ids = []
      scores = []
      bs = pose_data.rgbAs.shape[0]
      for b in range(0, pose_data.rgbAs.shape[0], bs):
        A = torch.cat([pose_data.rgbAs[b:b+bs].cuda(), pose_data.xyz_mapAs[b:b+bs].cuda()], dim=1).float()
        B = torch.cat([pose_data.rgbBs[b:b+bs].cuda(), pose_data.xyz_mapBs[b:b+bs].cuda()], dim=1).float()
        if pose_data.normalAs is not None:
          A = torch.cat([A, pose_data.normalAs.cuda().float()], dim=1)
          B = torch.cat([B, pose_data.normalBs.cuda().float()], dim=1)
        with torch.cuda.amp.autocast(dtype=torch.float16):
          output = self.model(A, B, L=len(A))
        scores_cur = output["score_logit"].float().reshape(-1)
        ids.append(scores_cur.argmax()+b)
        scores.append(scores_cur)
      ids = torch.stack(ids, dim=0).reshape(-1)
      scores = torch.cat(scores, dim=0).reshape(-1)
      return ids, scores

    current_B = pose_data.rgbAs.shape[0]
    pose_data_iter = pose_data
    global_ids = torch.arange(current_B, device='cuda', dtype=torch.long)
    scores_global = torch.zeros(current_B, dtype=torch.float, device='cuda')

    while 1:
      ids, scores = find_best_among_pairs(pose_data_iter)
      if len(ids)==1:
        scores_global[global_ids] = scores + 100
        break
      global_ids = global_ids[ids]
      pose_data_iter = pose_data.select_by_indices(global_ids)

    ids, scores = find_best_among_pairs(pose_data_iter)

    # print("scores", scores)
    #logging.info(f'forward done')
    torch.cuda.empty_cache()

    end_time0 = time.time()
    logging.info(f"predict_score find_best_among_pairs time: {end_time0 - start_time0:.3f} seconds")

    if get_vis:
      #logging.info("get_vis...")
      canvas = []
      ids = scores.argsort(descending=True)
      canvas = vis_batch_data_scores(pose_data, ids=ids, scores=scores)
      return scores, canvas

    return scores, None