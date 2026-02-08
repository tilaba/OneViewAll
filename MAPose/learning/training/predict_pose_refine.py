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

# 加载 DINOv2 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
dino_model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
# dino_model = AutoModel.from_pretrained("facebook/dino-vits16")
dino_model.eval()

def minmax_norm(x: torch.Tensor, eps=1e-5):
    min_val = float(x.min().item())
    max_val = float(x.max().item())
    return (x - min_val) / (max_val - min_val + eps)

def get_semi_dense_correspondences_dino(
    rgbA, rgbB, top_k=16, topk_patch=48, alpha=0.5, p=0.81, k_selection=18, ob_mask=None, cadmodel_cache=None):
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
        weights_A2B = torch.exp(-dists_A2B**2 * 1) + topk_vals_A2B  * 0.81
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
        weights_B2A = torch.exp(-dists_B2A**2 * 1) + topk_vals_B2A * 0.81
        score_B2A = (1 * weights_B2A).sum(dim=1)
        # weights_B2A = torch.exp(-dists_B2A**2 * p )
        # score_B2A = (topk_vals_B2A**2 * weights_B2A).sum(dim=1)
        # ========= 融合 =========
        score = 0.5 * (score_A2B + score_B2A)
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
  start_time0 = time.time()
  # if (iteration =0):
  crop_ratio = 1.0
  tf_to_crops = compute_crop_window_tf_batch(pts=mesh.vertices, H=H, W=W, poses=ob_in_cams, K=K, crop_ratio=crop_ratio, out_size=(render_size[1], render_size[0]), method=method, mesh_diameter=mesh_diameter)
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

  # maskAs = (xyz_mapbs[:, 2, :, :] > 0.001).float().unsqueeze(1)

  # save_image(maskAs[13], f"rgbbs/masks/mask_b0.png"
  end_time = time.time()
  logging.info(f"rendering time: {end_time - start_time0:.3f} seconds")
  if(B > batch_size):
    start_time = time.time()
    def dilate_mask(mask, ksize=7):
      pad = (ksize - 1) // 2
      dilated = F.avg_pool2d(mask, kernel_size=ksize, stride=1, padding=pad)
      return dilated

    from scipy.ndimage import binary_fill_holes
    def fill_mask_holes(mask: torch.Tensor) -> torch.Tensor:
      device = mask.device
      dtype = mask.dtype
      # 转 numpy 处理
      mask_np = mask.detach().cpu().numpy() > 0
      filled_list = []
      for m in mask_np:
          if m.ndim == 3:   # [1,H,W]
              m = m[0]
          filled = binary_fill_holes(m).astype(np.uint8)
          filled_list.append(filled[None, ...])  # [1,H,W]
      filled_np = np.stack(filled_list, axis=0)  # [B,1,H,W]
      filled_torch = torch.from_numpy(filled_np).to(device=device, dtype=dtype)
      return filled_torch

    
    mask = dilate_mask(ob_maskBs)
    mask_3c = mask.repeat(1, 3, 1, 1)
    alpha = 0.5
    rgbBs1 = rgbBs * mask_3c + (1 - mask_3c) * (alpha * rgbBs + (1 - alpha) * 255)
    xyz_mapBs = xyz_mapBs * mask_3c
    ob_maskBs1 = fill_mask_holes(ob_maskBs)
    mask_3c_1 = ob_maskBs1.repeat(1, 3, 1, 1)

    save_image(rgbBs[0]/255, f"rgbbs/rgbBs0.png")
    save_image(rgbBs1[0]/255, f"rgbbs/rgbBs1.png")

    # xyz_mapBs[mask_3c == 0] = float('nan')
    topk_indices, cadmodel_cache = get_semi_dense_correspondences_dino(rgbAs, rgbBs1, top_k=batch_size, alpha=0, k_selection=12, ob_mask=ob_maskBs[0, 0], cadmodel_cache=cadmodel_cache) 
    rgbBs = rgbBs[topk_indices]
    xyz_mapBs = xyz_mapBs[topk_indices]
    poseA = poseA[topk_indices]
    tf_to_crops = tf_to_crops[topk_indices]
    rgbAs = rgbAs[topk_indices]
    xyz_mapAs = xyz_mapAs[topk_indices]

    mask_3c_1 = mask_3c_1[0:12]
    rgbBs1 = rgbBs * mask_3c_1
    topk_indices, cadmodel_cache = get_semi_dense_correspondences_dino(rgbAs, rgbBs1, top_k=4, alpha=0, k_selection=4, ob_mask=ob_maskBs[0, 0], cadmodel_cache=cadmodel_cache) 
    rgbBs = rgbBs[topk_indices]
    xyz_mapBs = xyz_mapBs[topk_indices]
    poseA = poseA[topk_indices]
    tf_to_crops = tf_to_crops[topk_indices]
    rgbAs = rgbAs[topk_indices]
    xyz_mapAs = xyz_mapAs[topk_indices]
    save_image(rgbBs1[0]/255, f"rgbbs/rgbBs2.png")

    
    # torch.cuda.synchronize()
    end_time = time.time()
    logging.info(f"get_semi_dense_correspondences_dino: {end_time - start_time:.3f} seconds")
    
  if cfg['use_normal']:
      normal_rs = normal_r.permute(0, 3, 1, 2)      # (B, 3, H, W)

  mesh_diameters = torch.ones((len(rgbAs)), dtype=torch.float, device='cuda')*mesh_diameter
  pose_data = BatchPoseData(rgbAs=rgbAs, rgbBs=rgbBs, depthAs=None, depthBs=None, normalAs=normalAs, normalBs=normalBs, poseA=poseA, poseB=None, xyz_mapAs=xyz_mapAs, xyz_mapBs=xyz_mapBs, tf_to_crops=tf_to_crops, Ks=Ks, mesh_diameters=mesh_diameters)
  pose_data = dataset.transform_batch(batch=pose_data, H_ori=H, W_ori=W, bound=1)

  # torch.cuda.synchronize()
  end_time0 = time.time()
  logging.info(f"compute_crop_window_tf_batch: {end_time0 - start_time0:.3f} seconds")

  # logging.info("pose batch data done")

  return pose_data, cadmodel_cache



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
    self.cfg['batch_size'] = 28
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

    # mesh_diameter = mesh_diameter
    #scale_factor for linmod
    # scale_factor = [0.91, 0.79, 0.79, 0.79, 
    #                 0.79, 0.79, 0.79, 0.85,
    #                 0.85, 0.79, 0.79, 0.79,
    #                 0.79, 0.79, 0.85, 0.85]
    
    # assert(ob_id>=1)
    # mesh_diameter = mesh_diameter * scale_factor[ob_id - 1]

    ##logging.info(f'self.cfg.use_normal:{self.cfg.use_normal}')
    if not self.cfg.use_normal:
      normal_map = None

    crop_ratio = self.cfg['crop_ratio']
    # crop_ratio = 1.0
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
    
    start_time0 = time.time()
    iteration = 4
    for ii in range(iteration):
      ##logging.info("making cropped data")
      start_time2 = time.time()
      pose_data, self.cadmodel_cache = make_crop_data_batch(self.cfg.input_resize, self.cfg.batch_size, B_in_cams, mesh_centered, rgb_tensor, depth_tensor, K, crop_ratio=crop_ratio, normal_map=normal_map, xyz_map=xyz_map_tensor, ob_mask=ob_mask_tensor, cfg=self.cfg, glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset, mesh_diameter=mesh_diameter, iteration_iter=ii, ob_id=ob_id, cadmodel_cache=self.cadmodel_cache)
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
    self.last_trans_update = trans_delta
    self.last_rot_update = rot_mat_delta

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

