# Copyright (c) 2026 [OneViewAll].
# Licensed under the MIT License.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software.

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
from datasets.h5_dataset import *
from models.score_network import *
from datasets.pose_dataset import *
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
    if xyz_map.shape[0] == 3 and xyz_map.ndim == 3:
        xyz_map = xyz_map.permute(1, 2, 0)
    
    device = xyz_map.device
    dtype = xyz_map.dtype
    H, W, _ = xyz_map.shape
    
    mask = (torch.isfinite(xyz_map).all(dim=-1)) & (xyz_map[..., 2] > 1e-5)
    xyz_v = xyz_map[mask] # [N, 3]

    if xyz_v.shape[0] == 0:
        return torch.zeros((H, W, 3), device=device, dtype=dtype)

    R_A = PoseA[:3, :3]  # [3, 3]
    t_A = PoseA[:3, 3:4] # [3, 1]
    
    xyz_obj = torch.mm(R_A.t(), xyz_v.t() - t_A) # [3, N]
    xyz_obj[symmetry_axis, :] *= -1 

    xyz_cam_mirrored = (torch.mm(R_A, xyz_obj) + t_A).t() # [N, 3]

    mirrored_map = torch.zeros((H, W, 3), device=device, dtype=dtype)
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

    # ----------------------------------------------------
    # ----------------------------------------------------
    PoseA = torch.as_tensor(PoseA, dtype=dtype, device=device)
    Pose_target = torch.as_tensor(Pose_target, dtype=dtype, device=device)
    K = torch.as_tensor(K, dtype=dtype, device=device)

    if PoseA.ndim == 2: PoseA = PoseA.unsqueeze(0)
    B = Pose_target.shape[0]
    if PoseA.shape[0] == 1 and B > 1: PoseA = PoseA.expand(B, -1, -1)
    if K.ndim == 2: K = K.unsqueeze(0).expand(B, -1, -1)

    # ----------------------------------------------------
    R_A, t_A = PoseA[:, :3, :3], PoseA[:, :3, 3:] # t_A: [B, 3, 1]
    R_T, t_T = Pose_target[:, :3, :3], Pose_target[:, :3, 3:]
    
    rel_R = torch.bmm(R_T, R_A.transpose(1, 2))

    # ----------------------------------------------------
    # ----------------------------------------------------
    def preprocess_map(x):
        if x.ndim == 4: x = x.permute(0, 2, 3, 1).squeeze(0)
        elif x.ndim == 3 and x.shape[0] <= 4: x = x.permute(1, 2, 0)
        return torch.where(torch.isfinite(x), x, torch.zeros_like(x))

    xyz_mapA_cl = preprocess_map(xyz_mapA)
    rgbA_cl = preprocess_map(rgbA)
    H, W, _ = xyz_mapA_cl.shape


    xyz_mirrored_cl = create_mirrored_xyz_map(symmetry_axis, xyz_mapA_cl, PoseA[0])
    xyz_combined = torch.cat([xyz_mapA_cl.reshape(-1, 3), xyz_mirrored_cl.reshape(-1, 3)], dim=0)

    gray = rgbA_cl.mean(dim=2, keepdim=True)
    avg_color_expanded = gray.expand_as(rgbA_cl)
    rgb_combined = torch.cat([rgbA_cl.reshape(-1, 3), avg_color_expanded.reshape(-1, 3)], dim=0)

    if rgb_combined.dtype != torch.uint8:
        if rgb_combined.max() <= 1.01: rgb_combined = rgb_combined * 255.0
        rgb_combined = rgb_combined.clamp(0, 255).to(torch.uint8)

    # ----------------------------------------------------
    xyz_flat = xyz_combined.unsqueeze(0).expand(B, -1, -1) # [B, N, 3]
    centered_xyz = xyz_flat - t_A.transpose(1, 2)
    xyz_rot = torch.bmm(centered_xyz, rel_R.transpose(1, 2)) + t_T.transpose(1, 2)

    z = xyz_rot[..., 2]
    valid_mask = (torch.norm(xyz_flat, dim=-1) > 1e-6) & (z > 1e-6)

    # ----------------------------------------------------
    # ----------------------------------------------------
    fx, fy = K[:, 0, 0].view(B, 1), K[:, 1, 1].view(B, 1)
    cx, cy = K[:, 0, 2].view(B, 1), K[:, 1, 2].view(B, 1)

    u = (fx * (xyz_rot[..., 0] / (z + 1e-8)) + cx).long()
    v = (fy * (xyz_rot[..., 1] / (z + 1e-8)) + cy).long()

    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & valid_mask

    # ----------------------------------------------------
    # ----------------------------------------------------
    rgb_res = torch.zeros((B, H * W, 3), dtype=torch.uint8, device=device)
    xyz_res = torch.zeros((B, H * W, 3), dtype=dtype, device=device)

    # ----------------------------------------------------
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

        offsets = torch.tensor([-1, 0, 1], device=device)
        bu_9 = (bu.unsqueeze(0) + offsets.view(3, 1, 1)).clamp(0, W - 1)
        bv_9 = (bv.unsqueeze(0) + offsets.view(1, 3, 1)).clamp(0, H - 1)
        
        bu_flat = bu_9.expand(3, 3, -1).reshape(-1)
        bv_flat = bv_9.expand(3, 3, -1).reshape(-1)
        idx = (bv_flat * W + bu_flat).long()

        bz_weighted = bz.repeat(9) + weight_kernel.repeat_interleave(bz.shape[0])
        
        brgb_flat = brgb.repeat(9, 1)
        bxyz_flat = bxyz.repeat(9, 1)

        depth_buffer = torch.full((H * W,), float('inf'), device=device, dtype=dtype)
        depth_buffer.scatter_reduce_(0, idx, bz_weighted, reduce="amin", include_self=True)

        visible = (bz_weighted <= depth_buffer[idx])


        rgb_res[b, idx[visible]] = brgb_flat[visible]
        xyz_res[b, idx[visible]] = bxyz_flat[visible]

    rgb_res = rgb_res.view(B, H, W, 3)
    xyz_res = xyz_res.view(B, H, W, 3)

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

    os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "xyz"), exist_ok=True)

    device = xyz_mapA.device
    dtype = xyz_mapA.dtype

    # ----------------------------------------------------
    # ----------------------------------------------------
    PoseA = torch.as_tensor(PoseA, dtype=dtype, device=device)
    Pose_target = torch.as_tensor(Pose_target, dtype=dtype, device=device)
    K = torch.as_tensor(K, dtype=dtype, device=device)

    if PoseA.ndim == 2: PoseA = PoseA.unsqueeze(0)
    B = Pose_target.shape[0]
    if PoseA.shape[0] == 1 and B > 1: PoseA = PoseA.expand(B, -1, -1)
    if K.ndim == 2: K = K.unsqueeze(0).expand(B, -1, -1)

    # ----------------------------------------------------
    # ----------------------------------------------------
    R_A, t_A = PoseA[:, :3, :3], PoseA[:, :3, 3:] 
    R_T, t_T = Pose_target[:, :3, :3], Pose_target[:, :3, 3:]
    
    rel_R = torch.bmm(R_T, R_A.transpose(1, 2))

    # ----------------------------------------------------
    # ----------------------------------------------------
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

    # ----------------------------------------------------
    # ----------------------------------------------------
    xyz_flat = xyz_flat_base.unsqueeze(0).expand(B, -1, -1) # [B, N, 3]
    # ：X_target = rel_R @ (X_source - t_A) + t_target
    centered_xyz = xyz_flat - t_A.transpose(1, 2)
    xyz_rot = torch.bmm(centered_xyz, rel_R.transpose(1, 2)) + t_T.transpose(1, 2)

    z = xyz_rot[..., 2]
    valid_mask = (torch.norm(xyz_flat, dim=-1) > 1e-6) & (z > 1e-6)

    # ----------------------------------------------------
    # ----------------------------------------------------
    fx, fy = K[:, 0, 0].view(B, 1), K[:, 1, 1].view(B, 1)
    cx, cy = K[:, 0, 2].view(B, 1), K[:, 1, 2].view(B, 1)

    u = (fx * (xyz_rot[..., 0] / (z + 1e-8)) + cx).long()
    v = (fy * (xyz_rot[..., 1] / (z + 1e-8)) + cy).long()

    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & valid_mask

    # ----------------------------------------------------
    # ----------------------------------------------------
    rgb_res = torch.zeros((B, H * W, 3), dtype=torch.uint8, device=device)
    xyz_res = torch.zeros((B, H * W, 3), dtype=dtype, device=device)

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

        # --- 3x3 Splatting ---
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

    # ----------------------------------------------------
    # ----------------------------------------------------
    rgb_res = rgb_res.view(B, H, W, 3)
    xyz_res = xyz_res.view(B, H, W, 3)

    return rgb_res, xyz_res, Pose_target

tf_to_crops = None
@torch.inference_mode()
def make_project_data_batch_init(Ref_pose, rgb_r, depth_r, xyz_map_rs, is_symmetric, symmetry_axis, render_size, ob_in_cams, rgb, depth, K, crop_ratio, xyz_map=None, mesh_diameter=None, cfg=None, dataset:PoseRefinePairH5Dataset=None, iteration_iter = 5, ob_id=None):
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
    index = ob_in_cams.shape[0] - 1
  
    rgb_rs = rgb_r
    depth_rs = depth_r.unsqueeze(1)                   # (B, 1, H, W)
    Ks = K_tensor

    Ref_xyz_mapA = xyz_map_rs
    Ref_rgb_A = rgb_rs
    rgbBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(rgb, dtype=torch.float, device='cuda').permute(2,0,1)[None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)

    rgbAs = rgb_rs
    xyz_mapAs = xyz_map_rs
    xyz_mapBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(xyz_map, device='cuda', dtype=torch.float).permute(2,0,1)[None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)  #(B,3,H,W)

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

    return pose_data



class PoseScore:
  def __init__(self, amp=True):
    self.amp = amp
    self.run_name = "2024-01-11-20-02-45"

    model_name = 'model_best.pth'
    code_dir = os.path.dirname(os.path.realpath(__file__))
    ckpt_dir = f'{code_dir}/../weights/{self.run_name}/{model_name}'

    self.cfg = OmegaConf.load(f'{code_dir}/../weights/{self.run_name}/config.yml')

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
  def predict(self, rgb, depth, K, ob_in_cams, ob_id, xyz_map=None, mesh_diameter=None):
    """
    @rgb: np array (H,W,3)
    """

    device = 'cuda'

    ob_in_cams = torch.as_tensor(ob_in_cams, dtype=torch.float, device=device)

    if not self.cfg.use_normal:
        normal_map = None

    rgb = torch.as_tensor(rgb, device=device, dtype=torch.float)
    depth = torch.as_tensor(depth, device=device, dtype=torch.float)
    render_size = self.cfg['input_resize']
    K = torch.as_tensor(K, dtype=torch.float, device=device)

    save_dir = f"reference_database/linemod_real/{ob_id}"
    save_path = os.path.join(save_dir, "ref_data.pt")
    data = torch.load(save_path)

    Ref_xyz_mapA = data["xyz_map"].to(device)
    Ref_rgb_A = data["rgb"].to(device)
    Ref_pose = data["pose"].to(device)
    depth_r = data["depth"].to(device)

    obj_meta = data.get("meta", {})
    is_symmetric = obj_meta.get("is_symmetric")
    symmetry_axis = obj_meta.get("symmetry_axis")
    # symmetry_axis = 2
    crop_ratio = obj_meta.get("crop_ratio")


    pose_data = make_project_data_batch_init(
        Ref_pose, Ref_rgb_A, depth_r, Ref_xyz_mapA,
        is_symmetric, symmetry_axis,
        render_size, ob_in_cams, rgb, depth, K, crop_ratio, xyz_map,
        mesh_diameter=mesh_diameter,
        cfg=self.cfg, dataset=self.dataset,
        iteration_iter=5, ob_id=None
    )

    def build_AB(pose_data):
        A = torch.cat([pose_data.rgbAs, pose_data.xyz_mapAs], dim=1)
        B = torch.cat([pose_data.rgbBs, pose_data.xyz_mapBs], dim=1)

        if pose_data.normalAs is not None:
            A = torch.cat([A, pose_data.normalAs], dim=1)
            B = torch.cat([B, pose_data.normalBs], dim=1)

        return A.float(), B.float()


    A_all, B_all = build_AB(pose_data)

    with torch.cuda.amp.autocast(dtype=torch.float16):
        output = self.model(A_all, B_all, L=len(A_all))

    scores = output["score_logit"].float().reshape(-1)

    if len(scores) > 32:
        topk = max(1, len(scores) // 2)
        ids = scores.topk(topk).indices

        A_sub = A_all[ids]
        B_sub = B_all[ids]

        with torch.cuda.amp.autocast(dtype=torch.float16):
            output = self.model(A_sub, B_sub, L=len(A_sub))

        scores_sub = output["score_logit"].float().reshape(-1)

        scores_new = torch.full_like(scores, -1e9)
        scores_new[ids] = scores_sub
        scores = scores_new

    torch.cuda.empty_cache()


    return scores, None