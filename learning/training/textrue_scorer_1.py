import os
import torch
import numpy as np
from learning.datasets.h5_dataset import *
from learning.datasets.pose_dataset import *
from Utils import *
from datareader import *
from learning.datasets.pose_dataset import *
from transformers import AutoModel
import torch.nn.functional as F
from torchvision import transforms


@torch.no_grad()
def _compute_dinov2_similarity(self, rgbA, rgbB, mask=None):
    """
    rgbA, rgbB: [B, H, W, 3]  uint8 或 float [0,255]
    mask: [B, H, W] bool (可选)，若提供则只对有效区域平均特征
    返回: [B] 余弦相似度
    """
    B = rgbA.shape[0]
    # 转换为 float [0,1] 并调整尺寸到 [B, 3, 224, 224]
    def preprocess(x):
        # x: [B, H, W, 3] uint8 or float [0,255]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.clamp(0, 255) / 255.0
        # 转换为 [B, 3, H, W]
        x = x.permute(0, 3, 1, 2)
        # resize 到 224x224
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # 应用 ImageNet 归一化
        x = self.dino_transform(x)
        return x

    rgbA_proc = preprocess(rgbA)  # [B, 3, 224, 224]
    rgbB_proc = preprocess(rgbB)  # [B, 3, 224, 224]

    # 提取特征（CLS token 或平均池化）
    featA = self.dino_model(rgbA_proc).last_hidden_state[:, 0, :]  # [B, D] 使用 CLS token
    featB = self.dino_model(rgbB_proc).last_hidden_state[:, 0, :]

    # 计算余弦相似度
    sim = F.cosine_similarity(featA, featB, dim=-1)  # [B]
    return sim


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


class GeometricScorer:
    def __init__(self, distance_type='point_to_point', sym_tfs=None,
                 ref_data_dir="reference_database/linemod", use_dinov2=True):
        self.distance_type = distance_type
        self.sym_tfs = sym_tfs.cuda().float() if sym_tfs is not None else None
        self.ref_data_dir = ref_data_dir
        self._ref_cache = {}
        self.use_dinov2 = use_dinov2
        
        if self.use_dinov2:
            # 加载 DINOv2 模型（small 版本，可根据需要更换为 base/large）
            self.dino_model = AutoModel.from_pretrained("facebook/dinov2-small").cuda().eval()
            # 预处理：调整为 224x224，归一化到 [0,1]，并应用 ImageNet 均值和标准差
            self.dino_transform = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    # ---------- 参考数据加载 ----------
    def _load_ref_data(self, ob_id):
        if ob_id in self._ref_cache:
            return self._ref_cache[ob_id]

        ref_path = os.path.join(self.ref_data_dir, str(ob_id), "ref_data.pt")
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"ref_data not found at {ref_path}")
        data = torch.load(ref_path)

        # pose
        pose = data["pose"]
        if isinstance(pose, np.ndarray):
            pose = torch.from_numpy(pose)
        pose = pose.float().cuda()
        if pose.dim() == 3 and pose.shape[0] == 1:
            pose = pose[0]

        # xyz_map -> [1, 3, H, W]
        xyz_map = data["xyz_map"]
        if isinstance(xyz_map, np.ndarray):
            xyz_map = torch.from_numpy(xyz_map)
        xyz_map = xyz_map.float().cuda()
        if xyz_map.dim() == 4:
            xyz_map = xyz_map[0]
        if xyz_map.shape[0] == 3 and xyz_map.ndim == 3:
            pass  # 已经是 [3, H, W]
        elif xyz_map.shape[-1] == 3:
            xyz_map = xyz_map.permute(2, 0, 1)  # [H,W,3] -> [3,H,W]
        xyz_map = xyz_map.unsqueeze(0)          # [1, 3, H, W]

        # rgb -> [1, H, W, 3], 值域 [0,1]
        rgb = data["rgb"]
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb)
        rgb = rgb.float().cuda()
        if rgb.ndim == 4:
            rgb = rgb[0]
        if rgb.shape[0] == 3:
            rgb = rgb.permute(1, 2, 0)          # [3,H,W] -> [H,W,3]
        if rgb.max() > 1.01:
            rgb = rgb / 255.0
        rgb = rgb.unsqueeze(0)                  # [1, H, W, 3]

        # depth -> [H, W]
        depth = data["depth"]
        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth)
        depth = depth.float().cuda()
        while depth.ndim > 2:
            depth = depth.squeeze(0)

        meta = data.get("meta", {})
        info = {
            'ref_pose': pose,
            'ref_xyz_map': xyz_map,   # [1, 3, H, W]
            'ref_rgb': rgb,           # [1, H, W, 3]  值域 0-1
            'ref_depth': depth,       # [H, W]
            'meta': meta,
        }
        self._ref_cache[ob_id] = info
        return info
    
    def _compute_local_ncc(self, img1, img2, mask, window_size=5):
        """
        img1, img2: (H, W) float
        mask: (H, W) bool
        return: 标量相似度
        """
        # 将图像转为 [1, 1, H, W]
        I1 = img1.float().unsqueeze(0).unsqueeze(0)
        I2 = img2.float().unsqueeze(0).unsqueeze(0)
        mask = mask.float().unsqueeze(0).unsqueeze(0)
        
        # 计算局部均值和方差（使用平均池化）
        kernel = torch.ones(1, 1, window_size, window_size, device=I1.device) / (window_size*window_size)
        mean1 = F.conv2d(I1, kernel, padding=window_size//2)
        mean2 = F.conv2d(I2, kernel, padding=window_size//2)
        var1 = F.conv2d(I1**2, kernel, padding=window_size//2) - mean1**2
        var2 = F.conv2d(I2**2, kernel, padding=window_size//2) - mean2**2
        cov = F.conv2d(I1*I2, kernel, padding=window_size//2) - mean1*mean2
        
        # 计算局部 NCC
        eps = 1e-8
        ncc = cov / (torch.sqrt(var1*var2) + eps)  # [1,1,H,W]
        
        # 只考虑 mask 内有效的像素（mask 膨胀一下确保边缘）
        valid_mask = mask > 0.5
        if valid_mask.sum() < 100:
            return 0.0
        return ncc[valid_mask].mean().item()

    def _compute_ncc(self, img1, img2, mask):
        """
        img1, img2: (H, W) 灰度图，值域无所谓，会被归一化
        mask: (H, W) bool，True 表示有效像素
        返回标量 NCC 值，越接近 1 越相似
        """
        if mask.sum() < 100:
            return 0.0
        v1 = img1[mask].float()
        v2 = img2[mask].float()
        v1 = (v1 - v1.mean()) / (v1.std() + 1e-8)
        v2 = (v2 - v2.mean()) / (v2.std() + 1e-8)
        return (v1 * v2).mean().item()

    # ---------- 自定义裁剪变换（无需 pts，避免 Utils 版本的类型错误） ----------
    @staticmethod
    def _compute_crop_window_tf_batch(H, W, poses, K, crop_ratio=1.2,
                                      out_size=(128, 128), mesh_diameter=None):
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses).cuda().float()
        if isinstance(K, np.ndarray):
            K = torch.from_numpy(K).cuda().float()
        poses = torch.as_tensor(poses, dtype=torch.float32, device='cuda')
        K = torch.as_tensor(K, dtype=torch.float32, device='cuda')
        B = poses.shape[0]
        if K.ndim == 2:
            K = K.unsqueeze(0).expand(B, -1, -1)

        t = poses[:, :3, 3]
        fx, fy = K[:, 0, 0], K[:, 1, 1]
        cx, cy = K[:, 0, 2], K[:, 1, 2]
        u = fx * t[:, 0] / t[:, 2].clamp(min=1e-8) + cx
        v = fy * t[:, 1] / t[:, 2].clamp(min=1e-8) + cy

        if mesh_diameter is not None:
            if isinstance(mesh_diameter, (int, float)):
                diameter = torch.full((B,), mesh_diameter, device='cuda', dtype=torch.float32)
            else:
                diameter = torch.as_tensor(mesh_diameter, device='cuda', dtype=torch.float32)
        else:
            diameter = torch.full((B,), 0.1, device='cuda', dtype=torch.float32)

        size = crop_ratio * diameter / t[:, 2].clamp(min=1e-8) * fx
        half = size / 2.0
        left   = (u - half).clamp(min=0)
        top    = (v - half).clamp(min=0)
        right  = (u + half).clamp(max=W-1)
        bottom = (v + half).clamp(max=H-1)

        crop_w = (right - left).clamp(min=1)
        crop_h = (bottom - top).clamp(min=1)

        out_H, out_W = out_size[0], out_size[1]   # out_size 格式 (H, W)
        scale_w = (out_W - 1) / crop_w
        scale_h = (out_H - 1) / crop_h

        tf = torch.eye(3, device='cuda', dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1)
        tf[:, 0, 0] = scale_w
        tf[:, 1, 1] = scale_h
        tf[:, 0, 2] = -left * scale_w
        tf[:, 1, 2] = -top * scale_h
        return tf

    # ---------- 辅助采样与距离 ----------
    def _sample_points(self, pts, max_points=2000):
        if pts.shape[0] > max_points:
            idx = torch.randperm(pts.shape[0], device=pts.device)[:max_points]
            return pts[idx]
        return pts

    def _robust_mean(self, dist, inlier_ratio=0.85):
        k = max(int(dist.size(0) * inlier_ratio), 1)
        return torch.topk(dist, k, largest=False).values.mean()

    def _knn_dist(self, src, dst, chunk_size=10000):
        M = src.shape[0]
        min_dists = torch.empty(M, device=src.device, dtype=src.dtype)
        for i in range(0, M, chunk_size):
            end = min(i + chunk_size, M)
            dist_mat = torch.cdist(src[i:end], dst)
            min_dists[i:end] = dist_mat.min(dim=1)[0]
        return min_dists

    @torch.no_grad()
    def predict(self, mesh, rgb, depth, K, ob_in_cams, ob_mask,
                normal_map=None, xyz_map=None, mesh_tensors=None,
                glctx=None, mesh_diameter=None, get_vis=False, ob_id=None):
        """
        评分接口，完全保持原有签名。
        """
        if ob_id is None:
            raise ValueError("ob_id must be provided")
        if xyz_map is None:
            raise ValueError("xyz_map is required")

        # 统一为 tensor（float32）
        ob_in_cams = torch.as_tensor(ob_in_cams, device='cuda', dtype=torch.float32)
        K_tensor = torch.as_tensor(K, device='cuda', dtype=torch.float32)
        rgb_tensor = torch.as_tensor(rgb, device='cuda', dtype=torch.float32)
        depth_tensor = torch.as_tensor(depth, device='cuda', dtype=torch.float32)
        xyz_map_tensor = torch.as_tensor(xyz_map, device='cuda', dtype=torch.float32)
        ob_mask_tensor = torch.as_tensor(ob_mask, device='cuda', dtype=torch.float32)

        if mesh_diameter is not None:
            mesh_diameter = float(mesh_diameter)

        # ---------- 加载参考数据 ----------
        ref_info = self._load_ref_data(ob_id)
        ref_pose = ref_info['ref_pose']            # [4,4]
        ref_rgb = ref_info['ref_rgb']              # [1, Hr, Wr, 3]  0-1
        ref_xyz = ref_info['ref_xyz_map']          # [1, 3, Hr, Wr]
        ref_depth = ref_info['ref_depth']          # [Hr, Wr]
        meta = ref_info['meta']
        
        obj_meta = ref_info.get("meta", {})
        # is_symmetric = obj_meta.get("is_symmetric")
        is_symmetric = False
        # symmetry_axis = obj_meta.get("symmetry_axis")

        # 如果需要对称处理，可从 meta 中读取，这里暂时设为 False
        crop_ratio = float(meta.get('crop_ratio', 1.2))

        Hr, Wr = ref_depth.shape
        render_size = (Hr, Wr)

        B = ob_in_cams.shape[0]

        # ---------- 计算裁剪变换 ----------
        tf_to_crops = self._compute_crop_window_tf_batch(
            H=depth_tensor.shape[0], W=depth_tensor.shape[1],
            poses=ob_in_cams, K=K_tensor,
            crop_ratio=crop_ratio, out_size=render_size,
            mesh_diameter=mesh_diameter)

        # 更新内参
        K_crop = torch.bmm(tf_to_crops, K_tensor.unsqueeze(0).expand(B, -1, -1) if K_tensor.ndim == 2 else K_tensor)

        # ---------- 将 ob_mask 变换到裁剪空间 ----------
        if ob_mask_tensor.ndim == 2:
            ob_mask_tensor = ob_mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        mask_crop = kornia.geometry.transform.warp_perspective(
            ob_mask_tensor.expand(B, -1, -1, -1),   # [B, 1, H, W]
            tf_to_crops,
            dsize=render_size,
            mode='nearest',
            align_corners=False
        )  # [B, 1, Hr, Wr]
        mask_crop_bool = (mask_crop > 0.5).squeeze(1)  # [B, Hr, Wr]

        # ---------- 参考帧准备 ----------
        # ref_rgb 是 [1, Hr, Wr, 3] 值域 [0,1]，需要乘 255 转为 [0,255]
        Ref_rgb_A = ref_rgb.permute(0, 3, 1, 2) * 255          # [1, 3, Hr, Wr], 0-255
        Ref_xyz_mapA = ref_xyz                                   # [1, 3, Hr, Wr]

        # ---------- 批量投影 ----------
        if is_symmetric:
            rgb_projA, xyz_projA, _ = process_and_save_pc_data_batched_v2(
                symmetry_axis,
                Ref_xyz_mapA, Ref_rgb_A,
                None, None,
                ob_in_cams, ref_pose, K_crop)
        else:
            rgb_projA, xyz_projA, _ = process_and_save_pc_data_batched(
                Ref_xyz_mapA, Ref_rgb_A,
                None, None,
                ob_in_cams, ref_pose, K_crop)

        # ---------- 计算裁剪后的场景 RGB 图 (rgbBs) 和 XYZ 图 ----------
        rgbBs = kornia.geometry.transform.warp_perspective(
            rgb_tensor.permute(2, 0, 1)[None].expand(B, -1, -1, -1),
            tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False
        )   # [B, 3, Hr, Wr]

        xyz_mapBs = kornia.geometry.transform.warp_perspective(
            xyz_map_tensor.permute(2, 0, 1)[None].expand(B, -1, -1, -1),
            tf_to_crops, dsize=render_size, mode='nearest', align_corners=False
        )   # [B, 3, Hr, Wr]

        # ---------- 几何评分 ----------
        xyz_mapAs = xyz_projA.permute(0, 2, 3, 1).contiguous()   # [B, Hr, Wr, 3]
        xyz_mapBs_perm = xyz_mapBs.permute(0, 2, 3, 1).contiguous()

        scores = torch.zeros(B, device='cuda')
        for i in range(B):
            pred_xyz = xyz_mapAs[i].reshape(-1, 3)
            valid_pred = (pred_xyz[:, 2] > 1e-6) & torch.isfinite(pred_xyz).all(dim=-1)

            scene_xyz = xyz_mapBs_perm[i].reshape(-1, 3)
            mask_flat = mask_crop_bool[i].reshape(-1)            # [Hr*Wr]
            valid_scene = (scene_xyz[:, 2] > 1e-6) & torch.isfinite(scene_xyz).all(dim=-1) & mask_flat

            pred_pts = pred_xyz[valid_pred]
            scene_pts = scene_xyz[valid_scene]

            if pred_pts.shape[0] < 10 or scene_pts.shape[0] < 10:
                scores[i] = -float('inf')
                continue

            pred_pts = self._sample_points(pred_pts)
            scene_pts = self._sample_points(scene_pts)

            dist_m2s = self._knn_dist(pred_pts, scene_pts)
            dist_s2m = self._knn_dist(scene_pts, pred_pts)
            mean_m2s = self._robust_mean(dist_m2s)
            mean_s2m = self._robust_mean(dist_s2m)
            scores[i] = -(mean_m2s + mean_s2m).item()

        # ---------- 外观评分 ----------
        rgb_projA_gray = rgb_projA.float().mean(dim=-1)            # [B, Hr, Wr]
        rgbBs_gray = rgbBs.permute(0, 2, 3, 1).float().mean(dim=-1)  # [B, Hr, Wr]

        # 有效外观区域：投影非黑 + 场景在物体掩码内 + 场景非黑
        valid_app = (rgb_projA_gray > 5) & mask_crop_bool & (rgbBs_gray > 5)

        app_scores = torch.zeros(B, device='cuda')
        for i in range(B):
            app_scores[i] = self._compute_ncc(
                rgb_projA_gray[i], rgbBs_gray[i], valid_app[i]
            )

        # 融合几何与外观分数
        alpha = 1   # 外观权重，可根据需要调节
        scores = (1 - alpha) * scores + alpha * app_scores * alpha

        return scores, None