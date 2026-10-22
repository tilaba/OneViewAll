
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
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../../')
import numpy as np
import torch
from omegaconf import OmegaConf
from learning.models.refine_network import RefineNet
from learning.datasets.h5_dataset import *
from Utils import *
from datareader import *
import torchvision.transforms as T
from torchvision.utils import save_image
from torchvision.transforms import Resize, Normalize
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from kornia.feature import LoFTR
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 加载 DINOv2 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dino_model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
# dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
# dino_model = AutoModel.from_pretrained("facebook/dino-vits16")

# facebook/dinov2-small (维度: 384)
# facebook/dinov2-base (维度: 768)
# facebook/dinov2-large (维度: 1024)
# facebook/dinov2-giant (维度: 1536)

dino_model.eval()

import os
import math
import torch
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R




def create_mirrored_xyz_map(symmetry_axis, xyz_map, PoseA):
    """
    镜像逻辑：将点云变换到物体坐标系，绕 Y 轴镜像，再转回相机空间。
    输入兼容 (3, H, W) 或 (H, W, 3)
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
    
    # 相机 -> 物体 -> 镜像 -> 相机
    xyz_obj = torch.mm(R_A.t(), xyz_v.t() - t_A)
    xyz_obj[symmetry_axis, :] *= -1  # 镜像 Y 轴
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
  
def minmax_norm(x: torch.Tensor, eps=1e-5):
    min_val = float(x.min().item())
    max_val = float(x.max().item())
    return (x - min_val) / (max_val - min_val + eps)

def get_semi_dense_correspondences_dino(
    rgbA, rgbB, top_k=16, topk_patch=48, alpha=0.5, p=0.81, k_selection=12, ob_mask=None, cadmodel_cache=None):
    B = rgbA.size(0)
    assert rgbA.shape == rgbB.shape and B >= 1
    
    # chunk_size = int((252 + top_k - 1) / top_k)
    chunk_size = 252
    device = rgbA.device
    rgbA = ((rgbA.float() / 255.0) * 2 - 1).to(device)
    rgbB = ((rgbB.float() / 255.0) * 2 - 1).to(device)

    # === 提取 DINO 特征 ===
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputsA = dino_model(pixel_values=rgbA).last_hidden_state  # [B, 1+N, D]
            outputsB = dino_model(pixel_values=rgbB[0:1]).last_hidden_state  # [1, 1+N, D]

    featsA = outputsA[:, 1:]  # [B, N, D]
    featsB = outputsB[:, 1:].expand(B, -1, -1)  # [B, N, D]
    # featsA = featsB
    N = featsA.shape[1]
    patch_size = 14
    H, W = rgbA.shape[2], rgbA.shape[3]
    patch_h, patch_w = H // patch_size, W // patch_size
    assert patch_h * patch_w == N
    num_positive = (ob_mask > 0).sum()
    total_elements = ob_mask.numel()
    topk_patch = max(int(patch_h * patch_w * num_positive.float() / total_elements) + 1, 10)
    # topk_patch = int(patch_h * patch_w * num_positive.float() / total_elements) + 4
    # logging.info(f"Welcome make_crop_data_batch is {topk_patch}")
    # topk_patch = 50

    # === 坐标网格 ===
    y_coords, x_coords = torch.meshgrid(
        torch.arange(patch_h, device=device),
        torch.arange(patch_w, device=device),
        indexing="ij"
    )
    coords = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=-1).float().to(device)  # [N, 2]

    # === 标准化函数 ===
    def minmax_norm(x, eps=1e-6):
        min_val = x.min(dim=0, keepdim=True).values
        max_val = x.max(dim=0, keepdim=True).values
        return (x - min_val) / (max_val - min_val + eps)

    # # === TopK 匹配相似度打分 ===
    # def weighted_score(sim_matrix, coords, p=0.8):
    #   """
    #   优化后的 weighted_score 函数，采用中位数偏移和双向一致性检查。
    #   """
    #   b, N, _ = sim_matrix.shape
    #   coords_exp = coords.unsqueeze(0).expand(b, -1, -1)  # [b, N, 2]
    #   # A -> B 匹配
    #   max_vals_A2B, max_indices_A2B = sim_matrix.max(dim=2)  # [b, N]
    #   topk_vals_A2B, topk_idx_A2B = torch.topk(max_vals_A2B, min(topk_patch, N), dim=1)  # [b, topk]
    #   # 提取 A 的 topk 坐标
    #   coords_A = torch.gather(coords_exp, 1, topk_idx_A2B.unsqueeze(-1).expand(-1, -1, 2))  # [b, topk, 2]
    #   # 提取 B 的对应坐标
    #   idx_B = max_indices_A2B.gather(1, topk_idx_A2B) # [b, topk]
    #   coords_B = torch.gather(coords_exp, 1, idx_B.unsqueeze(-1).expand(-1, -1, 2))  # [b, topk, 2]
    #   # 计算中位数偏移
    #   median_offset = (coords_A - coords_B).median(dim=1, keepdim=True).values  # [b, 1, 2]
    #   # 计算相对位移和距离
    #   relative_disp = coords_A - coords_B - median_offset  # [b, topk, 2]
    #   dists = torch.norm(relative_disp, dim=2) + 1e-6  # [b, topk]

    #   # 基于距离的权重
    #   weights = torch.exp(-dists**2 * p)
    #   # 融合相似度值和权重
    #   weighted_sim = (topk_vals_A2B * weights).sum(dim=1)  # [b]
    #   return weighted_sim


    def weighted_score(sim_matrix, coords, p=0.81, mode="mean"):
        """
        改进版 weighted_score:
        - A→B & B→A 双向匹配
        - 最终分数支持 "mean" (平均) 或 "min" (取交集保守策略)
        """
        b, N, _ = sim_matrix.shape
        coords_exp = coords.unsqueeze(0).expand(b, -1, -1)  # [b, N, 2]

        # ========= A -> B =========
        max_vals_A2B, max_indices_A2B = sim_matrix.max(dim=2)  # [b, N]
        topk_vals_A2B, topk_idx_A2B = torch.topk(max_vals_A2B, min(topk_patch, N), dim=1)  # [b, topk]
        coords_A = torch.gather(coords_exp, 1, topk_idx_A2B.unsqueeze(-1).expand(-1, -1, 2))  # [b, topk, 2]
        idx_B = max_indices_A2B.gather(1, topk_idx_A2B)
        coords_B = torch.gather(coords_exp, 1, idx_B.unsqueeze(-1).expand(-1, -1, 2))  # [b, topk, 2]

        median_offset_A2B = (coords_A - coords_B).median(dim=1, keepdim=True).values
        relative_disp_A2B = coords_A - coords_B - median_offset_A2B
        dists_A2B = torch.norm(relative_disp_A2B, dim=2)
        mask = topk_vals_A2B > 0.2
        topk_vals_A2B = topk_vals_A2B * mask 
        weights_A2B = torch.exp(-dists_A2B**2 * 0.81) * topk_vals_A2B 
        score_A2B = (1 * weights_A2B).sum(dim=1)
        # weights_A2B = torch.exp(-dists_A2B**2 * p)
        # score_A2B = (topk_vals_A2B**2 * weights_A2B).sum(dim=1)

        # ========= B -> A =========
        max_vals_B2A, max_indices_B2A = sim_matrix.max(dim=1)  # [b, N]
        topk_vals_B2A, topk_idx_B2A = torch.topk(max_vals_B2A, min(topk_patch, N), dim=1)  # [b, topk]
        coords_B_ = torch.gather(coords_exp, 1, topk_idx_B2A.unsqueeze(-1).expand(-1, -1, 2))  # [b, topk, 2]
        idx_A_ = max_indices_B2A.gather(1, topk_idx_B2A)
        coords_A_ = torch.gather(coords_exp, 1, idx_A_.unsqueeze(-1).expand(-1, -1, 2))  # [b, topk, 2]
        median_offset_B2A = (coords_B_ - coords_A_).median(dim=1, keepdim=True).values
        relative_disp_B2A = coords_B_ - coords_A_ - median_offset_B2A
        dists_B2A = torch.norm(relative_disp_B2A, dim=2)
        mask = topk_vals_B2A > 0.2
        topk_vals_B2A = topk_vals_B2A * mask 
        weights_B2A = torch.exp(-dists_B2A**2 * 0.81) * topk_vals_B2A 
        score_B2A = (1 * weights_B2A).sum(dim=1)
        # weights_B2A = torch.exp(-dists_B2A**2 * p )
        # score_B2A = (topk_vals_B2A**2 * weights_B2A).sum(dim=1)
        # ========= 融合 =========
        score = 0.5 * (score_A2B)
        return score



    sim_scores = []
    sim_scores1 = []
    for i in range(0, B, chunk_size):
        fa = featsA[i:i+chunk_size]  # [b, N, D]
        fb = featsB[i:i+chunk_size]  # [b, N, D]
        # === Cosine 相似度 ===
        fa_cos = F.normalize(fa.float(), dim=-1)
        fb_cos = F.normalize(fb.float(), dim=-1)
        sim_cos_matrix = torch.bmm(fa_cos, fb_cos.transpose(1, 2))  # [b, N, N]

        sim_cos_A2B = weighted_score(sim_cos_matrix, coords)
        sim_cos_B2A = weighted_score(sim_cos_matrix.transpose(1, 2), coords)
        # sim_cos_score = 0.5 * (sim_cos_A2B + sim_cos_B2A)
        sim_cos_score = 0.5 * (sim_cos_A2B)

        # === 融合 ===
        sim_cos_norm = minmax_norm(sim_cos_score)
        sim_fused = sim_cos_norm
        sim_fused1 = sim_cos_score
        sim_scores.append(sim_fused)
        sim_scores1.append(sim_fused1)

    # patch_sim_scores = torch.cat(sim_scores, dim=0)  # [B]
    # topk_vals, topk_indices = torch.topk(patch_sim_scores, top_k)
    # patch_sim_scores1 = torch.cat(sim_scores1, dim=0)  # [B]
    # refined_scores = patch_sim_scores1[topk_indices]  # 取出对应 KL 分数
    # refined_vals, refined_indices_local = torch.topk(refined_scores, k=k_selection)
    # refined_indices = topk_indices[refined_indices_local]  # 映射回原始索引
    # return refined_indices, cadmodel_cache

    patch_sim_scores1 = torch.cat(sim_scores1, dim=0)  # [B]
    refined_vals, refined_indices_local = torch.topk(patch_sim_scores1, k=k_selection)
    return refined_indices_local, cadmodel_cache


def get_semi_dense_correspondences_dino_3d(
    rgbA, rgbB, xyz_mapA, xyz_mapB, ob_maskB,
    object_diameter=1.0, top_k=16, topk_patch=48, k_selection=12, is_svd = False
):
    B, _, H, W = rgbA.shape
    device = rgbA.device

    # 1. DINO 特征提取 (保持不变)
    rgbA_dino = ((rgbA.float() / 255.0) * 2 - 1).to(device)
    rgbB_dino = ((rgbB.float() / 255.0) * 2 - 1).to(device)

    with torch.no_grad():
        input_combined = torch.cat([rgbA_dino, rgbB_dino[0:1]], dim=0)
        all_feats = dino_model(pixel_values=input_combined).last_hidden_state[:, 1:]
        featsA = all_feats[:B]          
        featsB = all_feats[B:].expand(B, -1, -1) 

    N = featsA.shape[1]
    patch_h = patch_w = int(math.sqrt(N))

    # 2. 稳健的 Patch XYZ 均值计算 (仅针对 B 使用 mask)
    def xyz_patch_avg_safe(xyz_maps, ph, pw, diameter, extra_mask=None):
      b, c, h, w = xyz_maps.shape
      # 1. 像素级有效性 (深度 > 0 且非 NaN)
      valid = torch.isfinite(xyz_maps).all(dim=1, keepdim=True) & (xyz_maps[:, 2:3] > 0)
      
      # 如果有同分辨率的 mask (如 ob_maskB)，在此处结合
      if extra_mask is not None and extra_mask.shape[-2:] == (h, w):
          valid = valid & (extra_mask > 0.5)
      
      area_scale = (h * w) / (ph * pw)
      # 计算每个 patch 内有效像素的坐标总和
      patch_sum = F.adaptive_avg_pool2d(xyz_maps * valid.float(), (ph, pw)) * area_scale
      # 计算每个 patch 内有效像素的数量
      patch_count = F.adaptive_avg_pool2d(valid.float(), (ph, pw)) * area_scale
      
      patch_xyz = patch_sum / (patch_count + 1e-6)
      return (patch_xyz / diameter).permute(0, 2, 3, 1).reshape(b, -1, 3)

    # A 是 CAD 渲染图，默认全有效；B 是观测图，使用 ob_maskB
    maskA = (rgbA.sum(dim=1, keepdim=True) > 10).float()
    Mask_patchA = F.adaptive_avg_pool2d(maskA, (patch_h, patch_w)).view(B, -1)
   # A 是渲染图：直接算 patch xyz，不需要传 Mask_patchA 进去做位运算
    xyz_patchA = xyz_patch_avg_safe(xyz_mapA, patch_h, patch_w, object_diameter)

    # B 是观测图：传入原始分辨率的 ob_maskB 进行像素级过滤
    xyz_patchB = xyz_patch_avg_safe(
        xyz_mapB[:1], 
        patch_h, 
        patch_w, 
        object_diameter, 
        extra_mask=ob_maskB[:1] # 确保这是 [1, 1, H, W]
    ).expand(B, -1, -1)

    # 获取 Patch 级别的掩码用于后续评分 (这一步保持不变)
    mask_patchB = F.adaptive_avg_pool2d(ob_maskB[:1].float(), (patch_h, patch_w)).view(1, -1).expand(B, -1)

    def weighted_score_3d_bidirectional_with_mask(sim_matrix, xyzA_all, xyzB_all, mask_patchB, topk):
      """
      基于中值偏移校验的双向评分函数 (带 MaskB 过滤)
      mask_patchB 维度: [B, N_patches_B] 或 [1, N_patches_B]
      """
      b, n, _ = sim_matrix.shape
      k = min(topk, n)
      device = sim_matrix.device
      batch_idx = torch.arange(b, device=device).view(b, 1)

      # ========= A -> B 方向 (从模板找场景) =========
      # 1. 基础匹配
      
      max_vals_A2B, max_indices_A2B = sim_matrix.max(dim=2)  
      
      # 2. 引入 MaskB 过滤：如果 A 匹配到了 B 的背景区域，置信度降低
      # mask_patchB 展平后的索引对应 sim_matrix 的 dim=2
      m_B_gathered = torch.gather(mask_patchB.expand(b, -1), 1, max_indices_A2B)
      combined_conf_A2B = max_vals_A2B * m_B_gathered
      
      # 3. 选出最强匹配点对
      topk_vals_A2B, topk_idx_A2B = torch.topk(combined_conf_A2B, k, dim=1)
      curr_xyz_A = xyzA_all[batch_idx, topk_idx_A2B]  
      idx_B = max_indices_A2B.gather(1, topk_idx_A2B)
      curr_xyz_B = xyzB_all[batch_idx, idx_B] 

      def normalize_pc(pc):
        centroid = pc.mean(dim=1, keepdim=True)
        pc = pc - centroid
        # 算平均模长作为尺度因子
        dist = torch.norm(pc, dim=2, keepdim=True).mean(dim=1, keepdim=True)
        return pc / (dist + 1e-6)

        # 在计算 diff 之前
        curr_xyz_A = normalize_pc(curr_xyz_A)
        curr_xyz_B = normalize_pc(curr_xyz_B)

      # 4. 鲁棒几何校验 (Median Offset)
      diff_A2B = curr_xyz_A - curr_xyz_B
      # 只根据 Mask 内的可靠点计算中值位移
      median_offset_A2B = diff_A2B.median(dim=1, keepdim=True).values * (topk_vals_A2B > 0.1).unsqueeze(-1)
      # median_offset_A2B = diff_A2B.median(dim=1, keepdim=True).values
      
      # 计算每个点偏离整体位移的距离 (乘 5 、 9 、16 、20 、25 是为了放大残差敏感度)
      alpha = 7
      dists_A2B = torch.norm((diff_A2B - median_offset_A2B) * alpha, dim=2)
      
      # 最终 A2B 分数：几何对齐权重 * 组合置信度
      weights_A2B = torch.exp(-dists_A2B) * (topk_vals_A2B > 0.1)
      score_A2B = weights_A2B.sum(dim=1)
      
      # ========= B -> A 方向 (从场景找模板) =========
      # 1. 基础匹配 (场景中的每个点找模板中最像的点)
      max_vals_B2A, max_indices_B2A = sim_matrix.max(dim=1)  
      
      # 2. 引入 MaskB 过滤：如果起点 B 就在背景里，这个匹配直接作废
      # mask_patchB 对应 sim_matrix 的 dim=1 (如果 sim 形状是 [B, Na, Nb]) 
      # 注意：如果 sim 是 [B, Na, Nb], B->A 的 max 是在 Na 上做，对应的 B 点索引就是 dim=2 的索引
      # 这里我们直接用 mask_patchB 过滤起点
      combined_conf_B2A = max_vals_B2A * mask_patchB.expand(b, -1)
      topk_vals_B2A, topk_idx_B2A = torch.topk(combined_conf_B2A, k, dim=1)
      
      # 注意维度对应：B2A 时，topk_idx 指向的是场景 B 中的点
      curr_xyz_B_rev = xyzB_all[batch_idx, topk_idx_B2A]
      idx_A_rev = max_indices_B2A.gather(1, topk_idx_B2A)
      curr_xyz_A_rev = xyzA_all[batch_idx, idx_A_rev]

      # 3. 几何校验
      diff_B2A = curr_xyz_B_rev - curr_xyz_A_rev
      median_offset_B2A = diff_B2A.median(dim=1, keepdim=True).values * (topk_vals_B2A > 0.1).unsqueeze(-1)
      median_offset_B2A = diff_B2A.median(dim=1, keepdim=True).values
      dists_B2A = torch.norm((diff_B2A - median_offset_B2A) * alpha, dim=2)
      weights_B2A = torch.exp(-dists_B2A) * (topk_vals_B2A > 0.1)
      score_B2A = weights_B2A.sum(dim=1)
      return 0.5 * (score_A2B + score_B2A)

    def weighted_score_3d_bidirectional_svd_mask(sim_matrix, xyzA_all, xyzB_all, mask_patchB, topk=20):
      """
      双向加权 SVD + Median Offset，输出旋转角度绝对值（度）
      MaskB 过滤，top-k 点加权，Median Offset 只做一次平移补偿。

      参数:
          sim_matrix: [B, N_A, N_B] 相似度矩阵
          xyzA_all: [B, N_A, 3] 模板点云
          xyzB_all: [B, N_B, 3] 场景点云
          mask_patchB: [B, N_B] 或 [1, N_B] B端 mask (0/1)
          topk: 每个 batch 取最可靠 topk 点做 SVD

      返回:
          angle_bidir: [B] 双向旋转角度绝对值（度）
      """
      B, N_A, N_B = sim_matrix.shape
      device = sim_matrix.device
      batch_idx = torch.arange(B, device=device).view(B,1)
      k_curr = min(topk, N_A)

      def normalize_pc(pc):
        centroid = pc.mean(dim=1, keepdim=True)
        pc = pc - centroid
        dist = torch.norm(pc, dim=2, keepdim=True).mean(dim=1, keepdim=True)
        return pc / (dist + 1e-6)

      def compute_angle(P, Q, weights):
          # Median Offset 平移补偿
          median_offset = (Q - P).median(dim=1, keepdim=True).values  # Q - P
          P_centered = P + median_offset
          Q_centered = Q

          # 加权中心化
          w = weights.unsqueeze(-1) + 1e-8
          P_bar = (P_centered * w).sum(dim=1, keepdim=True) / w.sum(dim=1, keepdim=True)
          Q_bar = (Q_centered * w).sum(dim=1, keepdim=True) / w.sum(dim=1, keepdim=True)
          P_tilde = P_centered - P_bar
          Q_tilde = Q_centered - Q_bar

          # SVD 旋转计算
          H = torch.matmul((P_tilde * w).transpose(1,2), Q_tilde)
          U, S, V = torch.linalg.svd(H)
          R = torch.matmul(V, U.transpose(1,2))
          det = torch.linalg.det(R).unsqueeze(-1).unsqueeze(-1)
          R = torch.matmul(V * det.sign(), U.transpose(1,2))

          # 旋转角度绝对值 (度)
          trace = R[:,0,0] + R[:,1,1] + R[:,2,2]
          cos_theta = torch.clamp((trace - 1)/2, -1.0, 1.0)
          angle = torch.rad2deg(torch.acos(cos_theta))
          return angle

      # --------- A -> B ---------
      max_vals_A2B, max_idx_A2B = sim_matrix.max(dim=2)
      m_B_gathered = torch.gather(mask_patchB.expand(B, -1), 1, max_idx_A2B)
      weights_A2B = max_vals_A2B * m_B_gathered
      topk_vals_A2B, topk_idx_A2B = torch.topk(weights_A2B, k_curr, dim=1)

      P_A = xyzA_all[batch_idx, topk_idx_A2B]
      idx_B = max_idx_A2B.gather(1, topk_idx_A2B)
      Q_A = xyzB_all[batch_idx, idx_B]
      P_A = normalize_pc(P_A)
      Q_A = normalize_pc(Q_A)

      angle_A2B = compute_angle(P_A, Q_A, topk_vals_A2B)

      # --------- B -> A ---------
      max_vals_B2A, max_idx_B2A = sim_matrix.max(dim=1)
      weights_B2A = max_vals_B2A * mask_patchB.expand(B, -1)
      topk_vals_B2A, topk_idx_B2A = torch.topk(weights_B2A, k_curr, dim=1)

      P_B = xyzB_all[batch_idx, topk_idx_B2A]
      idx_A = max_idx_B2A.gather(1, topk_idx_B2A)
      Q_B = xyzA_all[batch_idx, idx_A]
      P_B = normalize_pc(P_B)
      Q_B = normalize_pc(Q_B)

      angle_B2A = compute_angle(P_B, Q_B, topk_vals_B2A)
      # --------- 双向平均角度 ---------
      angle_bidir = 0.5 * (angle_A2B + angle_B2A) 
      # print("angle_A2B is ",  angle_A2B)
      score = torch.exp(-(angle_bidir))
      return score
    # 4. 分块计算
    chunk_size = 128
    sim_scores_list = []
    for i in range(0, B, chunk_size):
        end = min(i + chunk_size, B)
        fa = F.normalize(featsA[i:end].float(), dim=-1)
        fb = F.normalize(featsB[i:end].float(), dim=-1)
        sim_matrix = torch.bmm(fa, fb.transpose(1, 2))
        if (is_svd == False):
          scores = weighted_score_3d_bidirectional_with_mask(
              sim_matrix, 
              xyz_patchA[i:end], 
              xyz_patchB[i:end], 
              mask_patchB[i:end],
              topk_patch
          )
          sim_scores_list.append(scores)
        else:
          scores = weighted_score_3d_bidirectional_svd_mask(
              sim_matrix, 
              xyz_patchA[i:end], 
              xyz_patchB[i:end], 
              mask_patchB[i:end],
              topk_patch
          )
          sim_scores_list.append(scores)

    sim_scores = torch.cat(sim_scores_list, dim=0)
    _, refined_indices = torch.topk(sim_scores, k=k_selection)
    
    return refined_indices



def get_semi_dense_correspondences_ultra(
    rgbA, rgbB, xyz_mapA, xyz_mapB, ob_maskB,
    object_diameter=1.0, topk_patch=48, k_selection=12, is_svd=False, chunk_size=128
):
    """
    将特征提取、Patch下采样、双向匹配、几何校验和 SVD 旋转计算全部集成在一个计算流中。
    支持 AMP 混合精度和 Tensor 并行化处理。
    """
    B, _, H, W = rgbA.shape
    device = rgbA.device
    alpha = 7.0  # 几何残差敏感度系数

    # --- 1. DINO 特征提取与归一化 (FP16 加速) ---
    with torch.cuda.amp.autocast(), torch.no_grad():
        # 归一化输入图像
        rgbA_dino = ((rgbA.float() / 255.0) * 2 - 1)
        rgbB_dino = ((rgbB[0:1].float() / 255.0) * 2 - 1)
        
        # 批量提取特征
        input_combined = torch.cat([rgbA_dino, rgbB_dino], dim=0)
        all_feats = dino_model(pixel_values=input_combined).last_hidden_state[:, 1:]
        
        # 提取后立即做 L2 Normalization，为后面的矩阵乘法做准备
        featsA = F.normalize(all_feats[:B].float(), dim=-1)
        featsB = F.normalize(all_feats[B:].expand(B, -1, -1).float(), dim=-1)

    # --- 2. 稳健的 Patch XYZ 均值计算 ---
    N = featsA.shape[1]
    ph = pw = int(math.sqrt(N))
    
    def get_patch_xyz(maps, mask=None):
        # 过滤无效深度点 (NaN 或非正值)
        valid = torch.isfinite(maps).all(dim=1, keepdim=True) & (maps[:, 2:3] > 0)
        if mask is not None:
            valid = valid & (mask > 0.5)
        
        # 面积加权平均，处理 Patch 内包含背景点的情况
        patch_sum = F.adaptive_avg_pool2d(maps * valid.float(), (ph, pw))
        patch_count = F.adaptive_avg_pool2d(valid.float(), (ph, pw))
        p_xyz = (patch_sum / (patch_count + 1e-6)) / object_diameter
        return p_xyz.flatten(2).transpose(1, 2)

    xyz_patchA = get_patch_xyz(xyz_mapA)
    # 场景图 B 使用传入的 mask 进行像素级过滤
    xyz_patchB = get_patch_xyz(xyz_mapB[:1], ob_maskB[:1]).expand(B, -1, -1)
    # Patch 级别的 Mask 用于后续评分过滤
    mask_patchB = F.adaptive_avg_pool2d(ob_maskB[:1].float(), (ph, pw)).view(1, -1).expand(B, -1)

    # --- 3. 核心计算块 (双向匹配 + 几何校验) ---
    all_scores = []
    batch_idx = torch.arange(chunk_size, device=device).view(chunk_size, 1)

    for i in range(0, B, chunk_size):
        curr_b = min(chunk_size, B - i)
        if curr_b < chunk_size: batch_idx = torch.arange(curr_b, device=device).view(curr_b, 1)
        
        # 计算相似度矩阵 [Chunk, N, N]
        sim_matrix = torch.bmm(featsA[i:i+curr_b], featsB[i:i+curr_b].transpose(1, 2))
        
        # A -> B 方向
        max_v_A2B, max_i_A2B = sim_matrix.max(dim=2)
        conf_A2B = max_v_A2B * torch.gather(mask_patchB[i:i+curr_b], 1, max_i_A2B)
        top_v_A, top_i_A = torch.topk(conf_A2B, topk_patch, dim=1)
        
        # B -> A 方向
        max_v_B2A, max_i_B2A = sim_matrix.max(dim=1)
        conf_B2A = max_v_B2A * mask_patchB[i:i+curr_b]
        top_v_B, top_i_B = torch.topk(conf_B2A, topk_patch, dim=1)

        if not is_svd:
            # --- 快速几何校验模式 (Median Offset) ---
            def quick_score(p_raw, q_raw, weights):
                # 局部点云中心化
                p = p_raw - p_raw.mean(dim=1, keepdim=True)
                q = q_raw - q_raw.mean(dim=1, keepdim=True)
                # 计算残差的中位数偏移
                diff = p - q
                median_off = diff.median(dim=1, keepdim=True).values
                dists = torch.norm((diff - median_off) * alpha, dim=2)
                return (torch.exp(-dists) * (weights > 0.1)).sum(dim=1)

            def quick_score_v2(p_raw, q_raw, weights, alpha=7.0):
                p = p_raw - p_raw.mean(dim=1, keepdim=True)
                q = q_raw - q_raw.mean(dim=1, keepdim=True)               
                diff = p - q
                median_off = diff.median(dim=1, keepdim=True).values              
                dists = torch.norm(diff - median_off, dim=2)
                score_internal = torch.exp(-dists * alpha)                
                off_norm = torch.norm(median_off, dim=2)
                score_global = torch.exp(-off_norm * 0.1)     
                point_scores = score_internal * (weights > 0.1)
                final_score = (point_scores.sum(dim=1) / (weights.count_nonzero(dim=1) + 1e-6)) * score_global.squeeze(-1)
                return final_score
          

            pA = xyz_patchA[i:i+curr_b][batch_idx, top_i_A]
            qA = xyz_patchB[i:i+curr_b][batch_idx, max_i_A2B.gather(1, top_i_A)]
            score_A = quick_score(pA, qA, top_v_A)

            pB = xyz_patchB[i:i+curr_b][batch_idx, top_i_B]
            qB = xyz_patchA[i:i+curr_b][batch_idx, max_i_B2A.gather(1, top_i_B)]
            score_B = quick_score(pB, qB, top_v_B)
            
            all_scores.append(0.5 * (score_A + score_B))

    # --- 4. 汇总评分并输出结果 ---
    final_scores = torch.cat(all_scores, dim=0)
    _, refined_indices = torch.topk(final_scores, k=k_selection)
    
    return refined_indices



tf_to_crops = None
@torch.inference_mode()
def make_crop_data_batch(render_size, batch_size, ob_in_cams, mesh, rgb, depth, K, crop_ratio, xyz_map, ob_mask, normal_map=None, mesh_diameter=None, cfg=None, glctx=None, mesh_tensors=None, dataset:PoseRefinePairH5Dataset=None, iteration_iter = 5, ob_id=None, cadmodel_cache=None):
  ##logging.info("Welcome make_crop_data_batch")
  # print('ob_in_cams shape is', ob_in_cams.size())
  H,W = depth.shape[:2]
  B = len(ob_in_cams)
  args = []
  method = 'box_3d'
  # torch.cuda.synchronize()
  # if (iteration_iter == 0):
  # crop_ratio = 1.2
  
  tf_to_crops = compute_crop_window_tf_batch(H=H, W=W, poses=ob_in_cams, K=K, crop_ratio=crop_ratio, out_size=(render_size[1], render_size[0]), method=method, mesh_diameter=mesh_diameter)
  ob_maskBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(ob_mask.unsqueeze(2), device='cuda', dtype=torch.float).permute(2,0,1)[None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)  #(B,3,H,W)
  mask = ob_maskBs[0, 0] > 0  # 取 batch 里的第一个，shape = (H, W)
  # 获取非零点坐标
  
  # logging.info("make tf_to_crops done")
  poseA = torch.as_tensor(ob_in_cams, dtype=torch.float, device='cuda')

  bs = 512
  rgb_rs = []
  depth_rs = []
  normal_rs = []
  xyz_map_rs = []
  
  bbox2d_crop = torch.as_tensor(np.array([0, 0, cfg['input_resize'][0]-1, cfg['input_resize'][1]-1]).reshape(2,2), device='cuda', dtype=torch.float)
  bbox2d_ori = transform_pts(bbox2d_crop, tf_to_crops.inverse()).reshape(-1,4)

  B = len(poseA)
  extra = {}

  # Ks: (B, 3, 3) if needed, 否则广播
  if isinstance(K, np.ndarray):
      K_tensor = torch.tensor(K, dtype=torch.float32, device='cuda')
  else:
      K_tensor = K.to(device='cuda', dtype=torch.float32)

  if K_tensor.ndim == 2:
      K_tensor = K_tensor[None].expand(B, -1, -1)  # [B, 3, 3]

  # bbox2d_ori shape: [B, 4]
  # Ensure it's torch tensor
  bbox2d_ori = bbox2d_ori.to(device='cuda', dtype=torch.float32)
  # rgb_r, depth_r, normal_r = renderer.render(ob_in_cams=poseA, bbox2d=bbox2d_ori, get_normal=cfg['use_normal'], extra=extra)
  start_time0 = time.time()
  rgb_r, depth_r, normal_r = nvdiffrast_render(
  K=K_tensor,                     # [B, 3, 3]
  H=H,
  W=W,
  ob_in_cams=poseA,              # [B, 4, 4]
  context='cuda',
  get_normal=cfg['use_normal'],
  glctx=glctx,
  mesh_tensors=mesh_tensors,
  output_size=cfg['input_resize'],
  bbox2d=bbox2d_ori,             # [B, 4]
  use_light=True,
  extra=extra)

  rgb_rs = rgb_r.permute(0, 3, 1, 2) * 255           # (B, 3, H, W)
  depth_rs = depth_r.unsqueeze(1)                   # (B, 1, H, W)
  xyz_map_rs = extra['xyz_map'].permute(0, 3, 1, 2) # (B, 3, H, W)
  Ks = K_tensor
  rgbBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(rgb, dtype=torch.float, device='cuda').permute(2,0,1)[None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
  
  if rgb_rs.shape[-2:]!=cfg['input_resize']:
    rgbAs = kornia.geometry.transform.warp_perspective(rgb_rs, tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
  else:
    rgbAs = rgb_rs
  if xyz_map_rs.shape[-2:]!=cfg['input_resize']:
    xyz_mapAs = kornia.geometry.transform.warp_perspective(xyz_map_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  else:
    xyz_mapAs = xyz_map_rs
  xyz_mapBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(xyz_map, device='cuda', dtype=torch.float).permute(2,0,1)[None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)  #(B,3,H,W)

  if cfg['use_normal']:
    normalAs = kornia.geometry.transform.warp_perspective(normal_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
    normalBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(normal_map, dtype=torch.float, device='cuda').permute(2,0,1)[None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  else:
    normalAs = None
    normalBs = None

  end_time = time.time()
  logging.info(f"rendering time: {end_time - start_time0:.3f} seconds")


  # save_image(maskAs[13], f"rgbbs/masks/mask_b0.png"
  if(B > batch_size):
    start_time = time.time()
    topk_indices = get_semi_dense_correspondences_ultra(rgbAs, rgbBs, xyz_mapAs, xyz_mapBs, ob_maskB = ob_maskBs[0], object_diameter = mesh_diameter, k_selection=4) 
    rgbBs = rgbBs[topk_indices]
    xyz_mapBs = xyz_mapBs[topk_indices]
    poseA = poseA[topk_indices]
    tf_to_crops = tf_to_crops[topk_indices]
    rgbAs = rgbAs[topk_indices]
    xyz_mapAs = xyz_mapAs[topk_indices]

  # # if(B > batch_size):
  # rgbAs = xyz_mapAs
  # rgbBs = xyz_mapBs

    
  #   torch.cuda.synchronize()
  #   end_time = time.time()
  #   logging.info(f"get_semi_dense_correspondences_dino: {end_time - start_time:.3f} seconds")
    
  if cfg['use_normal']:
      normal_rs = normal_r.permute(0, 3, 1, 2)      # (B, 3, H, W)

  mesh_diameters = torch.ones((len(rgbAs)), dtype=torch.float, device='cuda')*mesh_diameter
  pose_data = BatchPoseData(rgbAs=rgbAs, rgbBs=rgbBs, depthAs=None, depthBs=None, normalAs=normalAs, normalBs=normalBs, poseA=poseA, poseB=None, xyz_mapAs=xyz_mapAs, xyz_mapBs=xyz_mapBs, tf_to_crops=tf_to_crops, Ks=Ks, mesh_diameters=mesh_diameters)
  pose_data = dataset.transform_batch(batch=pose_data, H_ori=H, W_ori=W, bound=1)

  print("shape of rgbAs", rgbAs.shape)

  # torch.cuda.synchronize()
  end_time0 = time.time()
  logging.info(f"compute_crop_window_tf_batch: {end_time0 - start_time0:.3f} seconds")

  # logging.info("pose batch data done")

  return pose_data, cadmodel_cache


import torch
import kornia
import time
import logging
import numpy as np

@torch.inference_mode()
def make_project_data_batch_init(
    Ref_pose,
    rgb_r,
    depth_r,
    xyz_map_rs,
    is_symmetric,
    symmetry_axis,
    render_size,
    batch_size,
    ob_in_cams,
    rgb,
    depth,
    K,
    crop_ratio,
    xyz_map=None,
    ob_mask=None,
    mesh_diameter=None,
    cfg=None,
    dataset=None,
    iteration_iter=5,
    ob_id=None
):
    device = 'cuda'

    H, W = depth.shape[:2]
    B = len(ob_in_cams)
    method = 'box_3d'

    # ===================== 1. crop transform =====================
    t0 = time.time()

    tf_to_crops = compute_crop_window_tf_batch(
        H=H,
        W=W,
        poses=ob_in_cams,
        K=K,
        crop_ratio=crop_ratio,
        out_size=(render_size[1], render_size[0]),
        method=method,
        mesh_diameter=mesh_diameter
    )

    logging.info(f"compute_crop_window_tf_batch: {time.time() - t0:.3f}s")

    # ===================== 2. pose & K =====================
    poseA = torch.as_tensor(ob_in_cams, dtype=torch.float32, device=device)

    if isinstance(K, np.ndarray):
        K_tensor = torch.from_numpy(K).to(device=device, dtype=torch.float32)
    else:
        K_tensor = K.to(device=device, dtype=torch.float32)

    if K_tensor.ndim == 2:
        K_tensor = K_tensor.unsqueeze(0).expand(B, -1, -1)

    # ===================== 3. bbox =====================
    bbox2d_crop = torch.tensor(
        [0, 0, cfg['input_resize'][0]-1, cfg['input_resize'][1]-1],
        device=device,
        dtype=torch.float32
    ).reshape(2, 2)

    tf_inv = torch.inverse(tf_to_crops)
    bbox2d_ori = transform_pts(bbox2d_crop, tf_inv).reshape(-1, 4)

    # ===================== 4. 统一 tensor 输入（关键修复点） =====================

    def prepare_image_tensor(img, B):
        if isinstance(img, np.ndarray):
            tensor = torch.from_numpy(img)
        else:
            tensor = img

        tensor = tensor.to(device=device, dtype=torch.float32)

        # (H,W,3) -> (3,H,W)
        if tensor.ndim == 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)

        # (3,H,W) -> (B,3,H,W)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        if tensor.shape[0] != B:
            tensor = tensor.repeat(B, 1, 1, 1)

        return tensor

    rgb_tensor = prepare_image_tensor(rgb, B)
    xyz_tensor = prepare_image_tensor(xyz_map, B)

    # ===================== 5. warp（优化核心） =====================
    t_warp = time.time()

    rgbBs = kornia.geometry.transform.warp_perspective(
        rgb_tensor,
        tf_to_crops,
        dsize=render_size,
        mode='bilinear',
        align_corners=False
    )

    xyz_mapBs = kornia.geometry.transform.warp_perspective(
        xyz_tensor,
        tf_to_crops,
        dsize=render_size,
        mode='nearest',
        align_corners=False
    )

    # rgbBs = rgbBs[0].expand(B,-1,-1,-1)
    # xyz_mapBs = xyz_mapBs[0].expand(B,-1,-1,-1)

    logging.info(f"warp_perspective optimized: {time.time() - t_warp:.3f}s")

    # ===================== 6. render inputs =====================
    rgb_rs = rgb_r.permute(0, 3, 1, 2) * 255
    depth_rs = depth_r.unsqueeze(1)

    Ref_rgb_A = rgb_rs
    Ref_xyz_mapA = xyz_map_rs

    mask = (Ref_xyz_mapA.abs().sum(dim=1, keepdim=True) == 0) # 结果形状 [1, H, W] 
    Ref_rgb_white_bg = Ref_rgb_A.clone()
    Ref_rgb_white_bg[mask.expand_as(Ref_rgb_white_bg)] = 255

    import torchvision.utils as vutils
    # 如果你想循环保存每一张图：
    vutils.save_image(Ref_rgb_white_bg, f'output_results_2/Ref_rgb_A_{ob_id}.png', normalize=True)

    # ===================== 7. update intrinsics =====================
    K_crop = tf_to_crops @ K_tensor

    # ===================== 8. point cloud =====================
    t_pc = time.time()

    if is_symmetric:
        rgb_projA, xyz_projA, updated_poseA = process_and_save_pc_data_batched_v2(
            symmetry_axis,
            Ref_xyz_mapA,
            Ref_rgb_A,
            xyz_mapBs[0],
            rgbBs[0],
            poseA,
            Ref_pose,
            K_crop,
            output_dir="output_results_2"
        )
    else:
        rgb_projA, xyz_projA, updated_poseA = process_and_save_pc_data_batched(
            Ref_xyz_mapA,
            Ref_rgb_A,
            xyz_mapBs[0],
            rgbBs[0],
            poseA,
            Ref_pose,
            K_crop,
            output_dir="output_results_2"
        )

    logging.info(f"process_pc: {time.time() - t_pc:.3f}s")

    # ===================== 9. pack =====================
    rgbAs = rgb_projA.permute(0, 3, 1, 2)
    xyz_mapAs = xyz_projA.permute(0, 3, 1, 2)
    mesh_diameters = torch.ones((B,), dtype=torch.float32, device=device) * mesh_diameter

    pose_data = BatchPoseData(
        rgbAs=rgbAs,
        rgbBs=rgbBs,
        depthAs=None,
        depthBs=None,
        normalAs=None,
        normalBs=None,
        poseA=poseA,
        poseB=None,
        xyz_mapAs=xyz_mapAs,
        xyz_mapBs=xyz_mapBs,
        tf_to_crops=tf_to_crops,
        Ks=K_tensor,
        mesh_diameters=mesh_diameters
    )

    pose_data = dataset.transform_batch(
        batch=pose_data,
        H_ori=H,
        W_ori=W,
        bound=1
    )

    return pose_data


class PoseRefinePredictor:
  def __init__(self,):
    ##logging.info("welcome")
    self.amp = True
    self.run_name = "2023-10-28-18-33-37"
    model_name = 'model_best.pth'
    code_dir = os.path.dirname(os.path.realpath(__file__))
    ckpt_dir = f'{code_dir}/../../weights/{self.run_name}/{model_name}'

    self.cfg = OmegaConf.load(f'{code_dir}/../../weights/{self.run_name}/config.yml')

    self.cfg['ckpt_dir'] = ckpt_dir
    self.cfg['enable_amp'] = True
    self.cfg['batch_size'] = 4
    self.cfg['input_resize'] = [160, 160]

    ########## Defaults, to be backward compatible
    if 'use_normal' not in self.cfg:
      self.cfg['use_normal'] = False
    if 'use_mask' not in self.cfg:
      self.cfg['use_mask'] = False
    if 'use_BN' not in self.cfg:
      self.cfg['use_BN'] = False
    if 'c_in' not in self.cfg:
      self.cfg['c_in'] = 6
    if 'crop_ratio' not in self.cfg or self.cfg['crop_ratio'] is None:
      self.cfg['crop_ratio'] = 1.2
    if 'n_view' not in self.cfg:
      self.cfg['n_view'] = 1
    if 'trans_rep' not in self.cfg:
      self.cfg['trans_rep'] = 'tracknet'
    if 'rot_rep' not in self.cfg:
      self.cfg['rot_rep'] = 'axis_angle'
    if 'zfar' not in self.cfg:
      self.cfg['zfar'] = 3
    if 'normalize_xyz' not in self.cfg:
      self.cfg['normalize_xyz'] = False
    if isinstance(self.cfg['zfar'], str) and 'inf' in self.cfg['zfar'].lower():
      self.cfg['zfar'] = np.inf
    if 'normal_uint8' not in self.cfg:
      self.cfg['normal_uint8'] = False
    ##logging.info(f"self.cfg: \n {OmegaConf.to_yaml(self.cfg)}")

    self.dataset = PoseRefinePairH5Dataset(cfg=self.cfg, h5_file='', mode='test')

    # 初始化模型
    self.model = RefineNet(cfg=self.cfg, c_in=self.cfg['c_in']).cuda()
    ##logging.info(f"Using pretrained model from {ckpt_dir}")
    ckpt = torch.load(ckpt_dir)
    if 'model' in ckpt:
      ckpt = ckpt['model']
    self.model.load_state_dict(ckpt)
    self.model.cuda().eval()

    logging.info("==== init done ======")
    self.last_trans_update = None
    self.last_rot_update = None
    self.cadmodel_cache = None


 #收敛的角度求置信度。
  @torch.inference_mode()
  def predict(self, rgb, depth, K, ob_in_cams, xyz_map, ob_mask, normal_map=None, get_vis=False, mesh=None, mesh_tensors=None, glctx=None, mesh_diameter=None, iteration=5, ob_id=None):
    '''
    @rgb: np array (H,W,3)
    @ob_in_cams: np array (N,4,4)
    ''' 
    
    # ob_in_cams = ob_in_cams[0:15]
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    ##logging.info(f'ob_in_cams:{ob_in_cams.shape}')
    tf_to_center = np.eye(4)
    ob_centered_in_cams = ob_in_cams
    mesh_centered = mesh

    ##logging.info(f'self.cfg.use_normal:{self.cfg.use_normal}')
    if not self.cfg.use_normal:
      normal_map = None

    crop_ratio = self.cfg['crop_ratio']
    ##logging.info(f"trans_normalizer:{self.cfg['trans_normalizer']}, rot_normalizer:{self.cfg['rot_normalizer']}")
    bs = self.cfg.batch_size

    B_in_cams = torch.as_tensor(ob_centered_in_cams, device='cuda', dtype=torch.float)

    if mesh_tensors is None:
      mesh_tensors = make_mesh_tensors(mesh_centered)

    rgb_tensor = torch.as_tensor(rgb, device='cuda', dtype=torch.float)
    depth_tensor = torch.as_tensor(depth, device='cuda', dtype=torch.float)
    xyz_map_tensor = torch.as_tensor(xyz_map, device='cuda', dtype=torch.float)
    ob_mask_tensor = torch.as_tensor(ob_mask, device='cuda', dtype=torch.float)
    trans_normalizer = self.cfg['trans_normalizer']
    if not isinstance(trans_normalizer, float):
      trans_normalizer = torch.as_tensor(list(trans_normalizer), device='cuda', dtype=torch.float).reshape(1,3) 
    
    # iteration = 2
    render_size = self.cfg['input_resize']

    OBJ_META_DB = {
    1:  ["ape",         [162, 12, 102],       False, -1, 1.15],
    2:  ["benchwise",   [120, 0, 350],         True,  1, 1.05],
    4:  ["cam",         [10, 160, 0],          False,  -1, 1.15],
    5:  ["can",         [161, -60.4, -71.4],   True,  0, 1.20],
    6:  ["cat",         [200, 58, 90],         True,  0, 1.15],
    8:  ["drill",       [-121, -18, 70.4],     True,  1, 1.05],
    9:  ["duck",        [-140, 0, 170],        True,  1, 1.20],
    10: ["eggbox",      [180, 0, 0],           True,  0, 1.15],
    11: ["glue",        [180, 25, 135],        True,  0, 1.05],
    12: ["holepuncher", [100, -15, 0],         True,  1, 1.15],
    13: ["iron",        [160, -18, 0],         True,  1, 1.15],
    14: ["lamp",        [148.3, 0, -120],      True,  1, 1.15],
    15: ["phone",     [-148, 0, 60],         True,  1, 1.15],
    }

    # # # 2. 获取当前 ob_id 对应的元数据
    meta = OBJ_META_DB.get(ob_id, ["unknown", [0, 0, 0], False, -1, 1.15])
    obj_name, euler_deg, is_sym, axis_idx, obj_crop_ratio = meta
    new_pose1 = B_in_cams[0:1].clone()
    R_delta = R.from_euler('xyz', euler_deg, degrees=True).as_matrix()
    new_pose1[0][:3,:3] = torch.tensor(R_delta, dtype=B_in_cams.dtype, device=B_in_cams.device) 
    new_pose1[0][:3, 3] = torch.tensor([-0.01, 0.0, 1.0], dtype=B_in_cams.dtype, device=B_in_cams.device)
    H,W = depth_tensor.shape[:2]  
    tf_to_crops = compute_crop_window_tf_batch(pts=mesh.vertices, H=H, W=W, poses=new_pose1, K=K, crop_ratio=crop_ratio, out_size=(render_size[1], render_size[0]), method='box_3d', mesh_diameter=mesh_diameter)
    bbox2d_crop = torch.as_tensor(np.array([0, 0, self.cfg['input_resize'][0]-1, self.cfg['input_resize'][1]-1]).reshape(2,2), device='cuda', dtype=torch.float)
    bbox2d_ori = transform_pts(bbox2d_crop, tf_to_crops.inverse()).reshape(-1,4)
    if isinstance(K, np.ndarray):
      K_tensor = torch.tensor(K, dtype=torch.float32, device='cuda')
    else:
        K_tensor = K.to(device='cuda', dtype=torch.float32)
    extra = {}
    
    Ref_rgb_A, depth_r, _ = nvdiffrast_render(
        K=K_tensor,                     # [B, 3, 3]
        H=H,
        W=W,
        ob_in_cams=new_pose1,              # [B, 4, 4]
        context='cuda',
        get_normal=self.cfg['use_normal'],
        glctx=glctx,
        mesh_tensors=mesh_tensors,
        output_size=self.cfg['input_resize'],
        bbox2d=bbox2d_ori,            # [B, 4]
        use_light=True,
        extra=extra)

    Ref_xyz_mapA = extra['xyz_map'].permute(0, 3, 1, 2) # (B, 3, H, W)
    Ref_pose = new_pose1[0]


    save_dir = f"reference_database/ycb/{ob_id}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "ref_data.pt")

    # torch.save({
    #     "xyz_map": Ref_xyz_mapA.cpu(),
    #     "rgb": Ref_rgb_A.cpu(),
    #     "pose": Ref_pose.cpu(),
    #     "depth": depth_r.cpu(),
    #     "meta": {
    #         "id": ob_id,
    #         "name": obj_name,
    #         "is_symmetric": is_sym,
    #         "symmetry_axis": axis_idx,
    #         "crop_ratio": obj_crop_ratio
    #     }
    # }, save_path)

    # # --- 加载示例 ---
    data = torch.load(save_path)
    Ref_xyz_mapA = data["xyz_map"].to(device)
    Ref_rgb_A = data["rgb"].to(device)
    Ref_pose = data["pose"].to(device)
    depth_r = data["depth"].to(device)
    obj_meta = data.get("meta", {})
    is_symmetric = obj_meta.get("is_symmetric")
    symmetry_axis = obj_meta.get("symmetry_axis")
    crop_ratio = obj_meta.get("crop_ratio")
    iteration = 4
    # if (ob_id ==1 ):
    #   crop_ratio = 1.0

    start_time0 = time.time()
    for ii in range(iteration):
      ##logging.info("making cropped data")
      start_time2 = time.time()
      pose_data = make_project_data_batch_init(Ref_pose, Ref_rgb_A, depth_r, Ref_xyz_mapA, is_symmetric, symmetry_axis, self.cfg.input_resize, self.cfg.batch_size, B_in_cams, rgb_tensor, depth_tensor, K, crop_ratio=crop_ratio, xyz_map=xyz_map_tensor, ob_mask=ob_mask_tensor, cfg=self.cfg, dataset=self.dataset, mesh_diameter=mesh_diameter, iteration_iter=ii, ob_id=ob_id)
      # pose_data,  _ = make_crop_data_batch(self.cfg.input_resize, self.cfg.batch_size, B_in_cams, mesh_centered, rgb_tensor, depth_tensor, K, crop_ratio=crop_ratio, normal_map=normal_map, xyz_map=xyz_map_tensor, ob_mask=ob_mask_tensor, cfg=self.cfg, glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset, mesh_diameter=mesh_diameter, iteration_iter=ii, ob_id=ob_id, cadmodel_cache=self.cadmodel_cache)

      B_in_cams = []
      end_time2 = time.time()
      logging.info(f"make_crop_data_batch22 time: {end_time2 - start_time2:.3f} seconds")
      
      start_time3 = time.time()
      bs = pose_data.rgbAs.shape[0]
      for b in range(0, pose_data.rgbAs.shape[0], bs):
        A = torch.cat([pose_data.rgbAs[b:b+bs].cuda(), pose_data.xyz_mapAs[b:b+bs].cuda()], dim=1).float()
        B = torch.cat([pose_data.rgbBs[b:b+bs].cuda(), pose_data.xyz_mapBs[b:b+bs].cuda()], dim=1).float()
        # logging.info(f"A shape is:{A.shape}")
        # logging.info(f"B shape is:{B.shape}")
        with torch.no_grad():
          with torch.cuda.amp.autocast(enabled=self.amp):
            output = self.model(A,B)
            
        for k in output:
          output[k] = output[k].float()

        if self.cfg['trans_rep']=='tracknet':
          if not self.cfg['normalize_xyz']:
            trans_delta = torch.tanh(output["trans"])*trans_normalizer
          else:
            trans_delta = output["trans"]

        elif self.cfg['trans_rep']=='deepim':
          def project_and_transform_to_crop(centers):
            uvs = (pose_data.Ks[b:b+bs]@centers.reshape(-1,3,1)).reshape(-1,3)
            uvs = uvs/uvs[:,2:3]
            uvs = (pose_data.tf_to_crops[b:b+bs]@uvs.reshape(-1,3,1)).reshape(-1,3)
            return uvs[:,:2]

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
        
        if self.cfg['rot_rep']=='axis_angle':
          rot_mat_delta = torch.tanh(output["rot"])*self.cfg['rot_normalizer']
          rot_mat_delta = so3_exp_map(rot_mat_delta).permute(0,2,1)
        elif self.cfg['rot_rep']=='6d':
          rot_mat_delta = rotation_6d_to_matrix(output['rot']).permute(0,2,1)
        else:
          raise RuntimeError

        if self.cfg['normalize_xyz']:
          trans_delta *= (mesh_diameter/2)

        B_in_cam = egocentric_delta_pose_to_pose(pose_data.poseA[b:b+bs], trans_delta=trans_delta, rot_mat_delta=rot_mat_delta)
        B_in_cams.append(B_in_cam)

      B_in_cams = torch.cat(B_in_cams, dim=0).reshape(len(B_in_cams[0]),4,4)
      torch.cuda.synchronize()
      end_time3 = time.time()
      logging.info(f"model running time per iteration: {end_time3 - start_time3:.3f} seconds")

    # torch.cuda.synchronize()
        # logging.info("forward done")
      end_time0 = time.time()
      logging.info(f"model running time: {end_time0 - start_time0:.3f} seconds")

    B_in_cams_out = B_in_cams@torch.tensor(tf_to_center[None], device='cuda', dtype=torch.float)
    torch.cuda.empty_cache()
    end_time2 = time.time()
    logging.info(f"full model running time: {end_time2 - start_time0:.3f} seconds")

    # torch.cuda.synchronize()
    #     # logging.info("forward done")
    if get_vis:
      ##logging.info("get_vis...")
      canvas = []
      padding = 2
      pose_data, _ = make_crop_data_batch(self.cfg.input_resize, self.cfg.batch_size, torch.as_tensor(ob_centered_in_cams), mesh_centered, rgb, depth, K, crop_ratio=crop_ratio, normal_map=normal_map, xyz_map=xyz_map_tensor, ob_mask=ob_mask_tensor, cfg=self.cfg, glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset, mesh_diameter=mesh_diameter)
      for id in range(0, len(B_in_cams)):
        rgbA_vis = (pose_data.rgbAs[id]*255).permute(1,2,0).data.cpu().numpy()
        rgbB_vis = (pose_data.rgbBs[id]*255).permute(1,2,0).data.cpu().numpy()
        row = [rgbA_vis, rgbB_vis]
        H,W = rgbA_vis.shape[:2]
        if pose_data.depthAs is not None:
          depthA = pose_data.depthAs[id].data.cpu().numpy().reshape(H,W)
          depthB = pose_data.depthBs[id].data.cpu().numpy().reshape(H,W)
        elif pose_data.xyz_mapAs is not None:
          depthA = pose_data.xyz_mapAs[id][2].data.cpu().numpy().reshape(H,W)
          depthB = pose_data.xyz_mapBs[id][2].data.cpu().numpy().reshape(H,W)
        zmin = min(depthA.min(), depthB.min())
        zmax = max(depthA.max(), depthB.max())
        depthA_vis = depth_to_vis(depthA, zmin=zmin, zmax=zmax, inverse=False)
        depthB_vis = depth_to_vis(depthB, zmin=zmin, zmax=zmax, inverse=False)
        row += [depthA_vis, depthB_vis]
        if pose_data.normalAs is not None:
          pass
        row = make_grid_image(row, nrow=len(row), padding=padding, pad_value=255)
        row = cv_draw_text(row, text=f'id:{id}', uv_top_left=(10,10), color=(0,255,0), fontScale=0.5)
        canvas.append(row)
      canvas = make_grid_image(canvas, nrow=1, padding=padding, pad_value=255)

      pose_data , _ = make_crop_data_batch(self.cfg.input_resize, self.cfg.batch_size, B_in_cams, mesh_centered, rgb, depth, K, crop_ratio=crop_ratio, normal_map=normal_map, xyz_map=xyz_map_tensor, ob_mask=ob_mask_tensor, cfg=self.cfg, glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset, mesh_diameter=mesh_diameter)
      canvas_refined = []
      for id in range(0, len(B_in_cams)):
        rgbA_vis = (pose_data.rgbAs[id]*255).permute(1,2,0).data.cpu().numpy()
        rgbB_vis = (pose_data.rgbBs[id]*255).permute(1,2,0).data.cpu().numpy()
        row = [rgbA_vis, rgbB_vis]
        H,W = rgbA_vis.shape[:2]
        if pose_data.depthAs is not None:
          depthA = pose_data.depthAs[id].data.cpu().numpy().reshape(H,W)
          depthB = pose_data.depthBs[id].data.cpu().numpy().reshape(H,W)
        elif pose_data.xyz_mapAs is not None:
          depthA = pose_data.xyz_mapAs[id][2].data.cpu().numpy().reshape(H,W)
          depthB = pose_data.xyz_mapBs[id][2].data.cpu().numpy().reshape(H,W)
        zmin = min(depthA.min(), depthB.min())
        zmax = max(depthA.max(), depthB.max())
        depthA_vis = depth_to_vis(depthA, zmin=zmin, zmax=zmax, inverse=False)
        depthB_vis = depth_to_vis(depthB, zmin=zmin, zmax=zmax, inverse=False)
        row += [depthA_vis, depthB_vis]
        row = make_grid_image(row, nrow=len(row), padding=padding, pad_value=255)
        canvas_refined.append(row)

      canvas_refined = make_grid_image(canvas_refined, nrow=1, padding=padding, pad_value=255)
      canvas = make_grid_image([canvas, canvas_refined], nrow=2, padding=padding, pad_value=255)
      torch.cuda.empty_cache()
      return B_in_cams_out, canvas
    return B_in_cams_out, None
