# Copyright (c) 2026 [OneViewAll].
# Licensed under the MIT License.

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

from training.pose_score import (
    RefData, _dump_ref_probe,
    create_mirrored_xyz_map, _resolve_out_size,
    process_and_save_pc_data_batched,
    process_and_save_pc_data_batched_v2,
    CROP_RATIO_DEFAULT,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def make_project_data_batch_init(Ref_pose, rgb_r, xyz_map_rs, is_symmetric, symmetry_axis,
                                 render_size, batch_size, ob_in_cams, rgb, depth, K,
                                 crop_ratio=CROP_RATIO_DEFAULT,
                                 xyz_map=None, mesh_diameter=None, cfg=None,
                                 dataset:PoseRefinePairH5Dataset=None, iteration_iter=5,
                                 ob_id=None, gt_pose=None, save_ref_probe_dir=None):
    H, W = depth.shape[:2]
    B = len(ob_in_cams)
    tf_to_crops = compute_crop_window_tf_batch(
        H=H, W=W, poses=ob_in_cams, K=K, crop_ratio=crop_ratio,
        out_size=(render_size[1], render_size[0]), method='box_3d',
        mesh_diameter=mesh_diameter)

    poseA = torch.as_tensor(ob_in_cams, dtype=torch.float, device='cuda')
    K_tensor = (torch.tensor(K, dtype=torch.float32, device='cuda')
                if isinstance(K, np.ndarray) else K.to(device='cuda', dtype=torch.float32))
    if K_tensor.ndim == 2:
        K_tensor = K_tensor[None].expand(B, -1, -1)

    rgb_rs = rgb_r.permute(0, 3, 1, 2) if rgb_r.shape[-1] == 3 else rgb_r
    Ref_xyz_mapA = xyz_map_rs
    Ref_rgb_A = rgb_rs
    Ks = K_tensor

    rgbBs = kornia.geometry.transform.warp_perspective(
        torch.as_tensor(rgb, dtype=torch.float, device='cuda').permute(2, 0, 1)[None].expand(B, -1, -1, -1),
        tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
    xyz_mapBs = kornia.geometry.transform.warp_perspective(
        torch.as_tensor(xyz_map, device='cuda', dtype=torch.float).permute(2, 0, 1)[None].expand(B, -1, -1, -1),
        tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)

    K_crop = tf_to_crops @ K_tensor

    is_symmetric = False

    if is_symmetric:
        rgb_projA, xyz_projA, _ = process_and_save_pc_data_batched_v2(
            symmetry_axis, Ref_xyz_mapA, Ref_rgb_A, xyz_mapBs[0], rgbBs[0],
            poseA, Ref_pose, K_crop, output_dir="output_results_2")
    else:
        rgb_projA, xyz_projA, _ = process_and_save_pc_data_batched(
            Ref_xyz_mapA, Ref_rgb_A, xyz_mapBs[0], rgbBs[0],
            poseA, Ref_pose, K_crop, output_dir="output_results_2")

    rgbAs = rgb_projA.permute(0, 3, 1, 2)
    xyz_mapAs = xyz_projA.permute(0, 3, 1, 2)

    mesh_diameters = torch.ones((len(rgbAs),), dtype=torch.float, device='cuda') * mesh_diameter
    pose_data = BatchPoseData(
        rgbAs=rgbAs, rgbBs=rgbBs, depthAs=None, depthBs=None,
        normalAs=None, normalBs=None, poseA=poseA, poseB=None,
        xyz_mapAs=xyz_mapAs, xyz_mapBs=xyz_mapBs,
        tf_to_crops=tf_to_crops, Ks=Ks, mesh_diameters=mesh_diameters)
    pose_data = dataset.transform_batch(batch=pose_data, H_ori=H, W_ori=W, bound=1)

    if save_ref_probe_dir is not None:
        _dump_ref_probe(
            save_ref_probe_dir,
            Ref_pose=Ref_pose, Ref_rgb_A=Ref_rgb_A, Ref_xyz_mapA=Ref_xyz_mapA,
            rgbAs=rgbAs, xyz_mapAs=xyz_mapAs, rgbBs=rgbBs, xyz_mapBs=xyz_mapBs,
            tag=f"iter{iteration_iter}")
    return pose_data


class PoseRefinement:
    def __init__(self, ref_data_root=None):
        self.amp = True
        self.ref_data_root = ref_data_root
        self.ref_data: Optional[RefData] = None
        self.run_name = "2023-10-28-18-33-37"
        self.save_ref_probe_dir = None
        self._probe_saved = False
        self._sym_cache = {}

        code_dir = os.path.dirname(os.path.realpath(__file__))
        ckpt_dir = f'{code_dir}/../weights/{self.run_name}/model_best.pth'
        self.cfg = OmegaConf.load(f'{code_dir}/../weights/{self.run_name}/config.yml')
        self.cfg['ckpt_dir'] = ckpt_dir
        self.cfg['enable_amp'] = True
        self.cfg['batch_size'] = 28
        self.cfg['input_resize'] = [160, 160]

        for k, v in [('use_normal', False), ('use_mask', False), ('use_BN', False),
                     ('c_in', 6), ('crop_ratio', CROP_RATIO_DEFAULT),
                     ('n_view', 1), ('trans_rep', 'tracknet'), ('rot_rep', 'axis_angle'),
                     ('zfar', 3), ('normalize_xyz', False), ('normal_uint8', False),
                     ('trans_normalizer', 0.1)]:
            if k not in self.cfg or self.cfg[k] is None:
                self.cfg[k] = v
        self.cfg['crop_ratio'] = CROP_RATIO_DEFAULT

        if isinstance(self.cfg['zfar'], str) and 'inf' in self.cfg['zfar'].lower():
            self.cfg['zfar'] = np.inf

        self.dataset = PoseRefinePairH5Dataset(cfg=self.cfg, h5_file='', mode='test')

        self.model = RefineNet(cfg=self.cfg, c_in=self.cfg['c_in']).cuda()
        ckpt = torch.load(ckpt_dir)
        if 'model' in ckpt: ckpt = ckpt['model']
        self.model.load_state_dict(ckpt)
        self.model.cuda().eval()

        self.last_trans_update = None
        self.last_rot_update = None
        self.cadmodel_cache = None

    def set_ref_data(self, ref_data: Optional[RefData]):
        self.ref_data = ref_data
        self._probe_saved = False
        self._sym_cache = {}

    def reset_probe_flag(self):
        self._probe_saved = False

    def _get_ref_data(self, ob_id, device) -> RefData:
        ref_data = self.ref_data
        if ref_data is None or (ref_data.ob_id is not None and ref_data.ob_id != ob_id):
            if self.ref_data_root is None:
                raise ValueError(
                    f"PoseRefinement: ref for ob_id={ob_id} not set, "
                    "ref_data_root is None. Call use_frame_as_ref() first.")
            from training.pose_score import load_ref_data_from_dir
            save_dir = os.path.join(self.ref_data_root, str(ob_id))
            ref_data = load_ref_data_from_dir(save_dir, ob_id=ob_id)
            self.ref_data = ref_data
        return ref_data.to(device)

    @torch.inference_mode()
    def predict(self, rgb, depth, K, ob_in_cams, xyz_map, mesh_diameter=None,
                iteration=5, ob_id=None):
        import logging
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        tf_to_center = np.eye(4)

        device = 'cuda'
        B_in_cams = torch.as_tensor(ob_in_cams, device=device, dtype=torch.float)
        rgb_tensor = torch.as_tensor(rgb, device=device, dtype=torch.float)
        depth_tensor = torch.as_tensor(depth, device=device, dtype=torch.float)
        xyz_map_tensor = torch.as_tensor(xyz_map, device=device, dtype=torch.float)
        K = torch.as_tensor(K, dtype=torch.float, device=device)

        trans_normalizer = self.cfg['trans_normalizer']
        if not isinstance(trans_normalizer, float):
            trans_normalizer = torch.as_tensor(list(trans_normalizer),
                                               device=device, dtype=torch.float).reshape(1, 3)

        render_size = self.cfg['input_resize']

        ref_data = self._get_ref_data(ob_id, device)
        Ref_xyz_mapA = ref_data.xyz_map
        Ref_rgb_A = ref_data.rgb
        Ref_pose = ref_data.pose
        
        crop_ratio = ref_data.meta.get("crop_ratio", CROP_RATIO_DEFAULT)

        sym_key = ob_id
        if sym_key in self._sym_cache:
            is_symmetric, symmetry_axis = self._sym_cache[sym_key]
        else:
            test_pose = Ref_pose.unsqueeze(0)
            best_loss = float('inf')
            best_sym, best_axis = False, 0

            pose_tmp_base = make_project_data_batch_init(
                Ref_pose, Ref_rgb_A, Ref_xyz_mapA, False, -1,
                self.cfg.input_resize, 1, test_pose, rgb_tensor, depth_tensor, K,
                crop_ratio=crop_ratio,
                xyz_map=xyz_map_tensor, cfg=self.cfg,
                dataset=self.dataset, mesh_diameter=mesh_diameter,
                iteration_iter=0, ob_id=ob_id)
            xyzA0 = pose_tmp_base.xyz_mapAs[0]
            mask = xyzA0[2] > 1e-5

            for is_sym, sym_axis in [(True, 0), (True, 1), (True, 2)]:
                pose_tmp = make_project_data_batch_init(
                    Ref_pose, Ref_rgb_A, Ref_xyz_mapA, is_sym, sym_axis,
                    self.cfg.input_resize, 1, test_pose, rgb_tensor, depth_tensor, K,
                    crop_ratio=crop_ratio,
                    xyz_map=xyz_map_tensor, cfg=self.cfg,
                    dataset=self.dataset, mesh_diameter=mesh_diameter,
                    iteration_iter=0, ob_id=ob_id)
                xyzA = pose_tmp.xyz_mapAs[0]
                loss = 10.0 if mask.sum() < 10 else ((xyzA - xyzA0).abs().sum() / mask.sum().float()).item()
                if loss < best_loss:
                    best_loss, best_sym, best_axis = loss, is_sym, sym_axis

            is_symmetric = best_loss <= 0.1
            symmetry_axis = best_axis if is_symmetric else -1
            self._sym_cache[sym_key] = (is_symmetric, symmetry_axis)
            logging.info(f"[refine] ob_id={ob_id} sym={is_symmetric} axis={symmetry_axis} loss={best_loss:.4f}")

        probe_dir = None
        if self.save_ref_probe_dir is not None and not self._probe_saved:
            probe_dir = self.save_ref_probe_dir
            print(f"[refine] first iter, will dump probe to {probe_dir}")

        for ii in range(iteration):
            pose_data = make_project_data_batch_init(
                Ref_pose, Ref_rgb_A, Ref_xyz_mapA, is_symmetric, symmetry_axis,
                self.cfg.input_resize, self.cfg.batch_size, B_in_cams,
                rgb_tensor, depth_tensor, K,
                crop_ratio=crop_ratio,
                xyz_map=xyz_map_tensor, cfg=self.cfg, dataset=self.dataset,
                mesh_diameter=mesh_diameter, iteration_iter=ii, ob_id=ob_id,
                save_ref_probe_dir=(probe_dir if ii == 0 else None))
            if probe_dir is not None and ii == 0:
                self._probe_saved = True

            new_B_in_cams = []
            bs = pose_data.rgbAs.shape[0]
            for b in range(0, pose_data.rgbAs.shape[0], bs):
                A = torch.cat([pose_data.rgbAs[b:b+bs].cuda(),
                               pose_data.xyz_mapAs[b:b+bs].cuda()], dim=1).float()
                B = torch.cat([pose_data.rgbBs[b:b+bs].cuda(),
                               pose_data.xyz_mapBs[b:b+bs].cuda()], dim=1).float()
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.amp):
                    output = self.model(A, B)
                for k in output:
                    output[k] = output[k].float()

                if self.cfg['trans_rep'] == 'tracknet':
                    trans_delta = (torch.tanh(output["trans"]) * trans_normalizer
                                   if not self.cfg['normalize_xyz'] else output["trans"])
                elif self.cfg['trans_rep'] == 'deepim':
                    rot_delta = output["rot"]
                    z_pred = output['trans'][:, 2] * pose_data.poseA[b:b+bs][..., 2, 3]
                    uvA_crop = -(pose_data.poseA[b:b+bs][..., :3, 3])
                    uv_pred_crop = uvA_crop + output['trans'][:, :2] * self.cfg['input_resize'][0]
                    uv_pred = transform_pts(uv_pred_crop,
                                            pose_data.tf_to_crops[b:b+bs].inverse().cuda())
                    center_pred = torch.cat([uv_pred,
                                             torch.ones((len(rot_delta), 1), dtype=torch.float, device='cuda')],
                                            dim=-1)
                    center_pred = (pose_data.Ks[b:b+bs].inverse().cuda()
                                   @ center_pred.reshape(len(rot_delta), 3, 1)).reshape(len(rot_delta), 3) \
                                  * z_pred.reshape(len(rot_delta), 1)
                    trans_delta = center_pred - pose_data.poseA[b:b+bs][..., :3, 3]
                else:
                    trans_delta = output["trans"]

                if self.cfg['rot_rep'] == 'axis_angle':
                    rot_mat_delta = torch.tanh(output["rot"]) * self.cfg['rot_normalizer']
                    rot_mat_delta = so3_exp_map(rot_mat_delta).permute(0, 2, 1)
                elif self.cfg['rot_rep'] == '6d':
                    rot_mat_delta = rotation_6d_to_matrix(output['rot']).permute(0, 2, 1)
                else:
                    raise RuntimeError

                if self.cfg['normalize_xyz']:
                    trans_delta *= (mesh_diameter / 2)

                B_in_cam = egocentric_delta_pose_to_pose(
                    pose_data.poseA[b:b+bs], trans_delta=trans_delta,
                    rot_mat_delta=rot_mat_delta)
                new_B_in_cams.append(B_in_cam)

            B_in_cams = torch.cat(new_B_in_cams, dim=0).reshape(len(new_B_in_cams[0]), 4, 4)

        B_in_cams_out = B_in_cams @ torch.tensor(tf_to_center[None], device='cuda', dtype=torch.float)
        torch.cuda.empty_cache()
        return B_in_cams_out, None