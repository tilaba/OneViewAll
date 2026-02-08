# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# 1. 相似度修改
# 2. iter次数修改
# python run_linemodOcclusion.py --linemod_dir  /home/yluo/GSPose/dataspace/bop_dataset/lmo --use_reconstructed_mesh 0

from Utils import *
import numpy as np
from bop_toolkit_lib import pose_error, renderer, misc
import json,uuid,joblib,os,sys
import scipy.spatial as spatial
from multiprocessing import Pool
import multiprocessing
from functools import partial
from itertools import repeat
import itertools
from datareader import *
from estimater import *
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/mycpp/build')
import yaml
import time

import json
import trimesh
from pathlib import Path
from scipy.spatial import cKDTree
from bop_toolkit_lib import renderer_vispy
# 在现有代码框架中添加以下关键模块
# from bop_toolkit_lib.renderer_pyrender import RendererPyrender as Renderer


class PoseEvaluator:
    def __init__(self, dataset_root):
        self.meta_data = self._load_linemod_meta(dataset_root)
        
    def _load_linemod_meta(self, path):
      """从 models_info.json + PLY 文件加载元数据"""
      meta = {}
      models_dir = Path(path) / "lm_models/models"
      
      # 1. 加载预定义的模型信息
      with open(models_dir / "models_info.json") as f:
          models_info = json.load(f)
      
      # 2. 遍历所有PLY模型文件
      for ply_file in models_dir.glob("obj_*.ply"):
          # 解析物体ID (e.g. obj_000002.ply -> 2)
          ob_id = int(ply_file.stem.split("_")[1])
          
          # 3. 从JSON获取基础信息
          if str(ob_id) not in models_info:
              raise ValueError(f"Object {ob_id} not found in models_info.json")
          
          info = models_info[str(ob_id)]
          
          # 4. 加载PLY模型验证数据
          mesh = trimesh.load(ply_file)
          computed_diameter = self._compute_mesh_diameter(mesh)
          
          # 5. 数据一致性校验
          if abs(info['diameter'] - computed_diameter) > 1e-3:
              print(f"Warning: Diameter mismatch for obj_{ob_id}: "
                    f"json={info['diameter']}, computed={computed_diameter}")

          syms = []
          if 'symmetries_discrete' in info and isinstance(info['symmetries_discrete'], list) and len(info['symmetries_discrete']) > 0:
              for sym in info['symmetries_discrete']:
                  arr = np.array(sym, dtype=np.float32)
                  if arr.size == 9:  # 3x3 旋转矩阵
                      R = arr.reshape(3, 3)
                  elif arr.size == 16:  # 4x4 齐次矩阵
                      R = arr.reshape(4, 4)[:3, :3]
                  else:
                      raise ValueError(f"Unexpected symmetry matrix size: {arr.size}")
                  syms.append({"R": R, "t": np.zeros((3, 1), dtype=np.float32)})
          else:
              syms.append({"R": np.eye(3, dtype=np.float32), "t": np.zeros((3, 1), dtype=np.float32)})
          
          # 6. 构建元数据
          meta[ob_id] = {
              'diameter': info['diameter'],
              'symmetry': syms,
              'mesh': mesh,
              'model_path': str(ply_file)
          }
      
      return meta

    def _compute_mesh_diameter(self, mesh):
        """基于PLY模型计算直径（凸包顶点间最大距离）"""
        hull = mesh.convex_hull
        vertices = hull.vertices
        return np.max(np.linalg.norm(vertices - vertices.mean(axis=0), axis=1)) * 2

    def calculate_pose_error(self, pose_est, pose_gt, model_pts, ob_id):
        """计算位姿误差（自动处理对称性）"""
        diameter = self.meta_data[ob_id]['diameter']
        
        # 对称物体使用ADD-S
        if self.meta_data[ob_id]['symmetry'] != 'none':
            return self._add_s_error(pose_est, pose_gt, model_pts, diameter)
        
        # 非对称物体使用ADD
        return self._add_error(pose_est, pose_gt, model_pts, diameter)

    def render_depth_map(self, model, pose, K, img_shape):
      """
      Simulate rendering a depth map for an object model in a given pose.
      Args:
          model: 3D object model (e.g., mesh vertices)
          pose: 4x4 transformation matrix [R | t]
          K: 3x3 camera intrinsic matrix
          img_shape: Tuple (height, width) of the image
      Returns:
          depth_map: 2D numpy array with depth values
      """
      # Placeholder: Simulate depth map (replace with actual rendering)
      height, width = img_shape
      depth_map = np.zeros((height, width), dtype=np.float32)
      # Example: Fill with random depth values for demonstration
      depth_map += np.random.rand(height, width) * 100  # Replace with actual rendering
      return depth_map

    def _unwrap_pose(self, pose):
      """从嵌套结构中提取 4x4 numpy pose 矩阵"""
      import numpy as np
      if isinstance(pose, np.ndarray):
          return pose
      # 如果是 list / tuple，递归取第一个元素
      if isinstance(pose, (list, tuple)):
          return _unwrap_pose(pose[0])
      # 如果是 dict，可能 key 是 'pose' 或其他
      if isinstance(pose, dict):
          for k in ['pose', 'R', 't', 'mat']:
              if k in pose:
                  return _unwrap_pose(pose[k])
      raise TypeError(f"未知的 pose 数据结构: {type(pose)}, 内容: {pose}")


    def compute_ar_for_dataset(self, model_pts, est_poses, gt_poses, cam_K, ob_id, width=640, height=480, visib_fract=1.0):
        """
        计算一个数据集上 AR_VSD、AR_MSSD、AR_MSPD 及 AR（BOP Challenge 定义）
        """
        assert len(est_poses) == len(gt_poses), "预测与GT数量不一致"

        diameter = self.meta_data[ob_id]['diameter'] / 1000
        model_path = self.meta_data[ob_id]['model_path']
        
        rnd = renderer_vispy.RendererVispy(width, height, mode='depth')
        rnd.add_object(ob_id, model_path)

        # --- 阈值设置 ---
        # VSD 的三个深度阈值
        vsd_taus = [0.05, 0.1, 0.15]
        vsd_taus_mm = [50, 100, 150]
        # vsd_taus = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        # VSD 的多个误差阈值，归一化后
        vsd_thetas = np.arange(0.05, 0.51, 0.05)
        # MSSD 的多个误差阈值，归一化后
        mssd_thetas = np.arange(0.05, 0.51, 0.05)

        # MSPD 的多个误差阈值，单位：像素
        r = width / 640.0
        mspd_thetas = np.arange(5 * r, 51 * r, 5 * r)

        syms = self.meta_data[ob_id]['symmetry']

        # --- VSD 计算 ---
        delta = 15
        vsd_recalls = []
        for theta in vsd_thetas:
            correct_flags = []
            for est_pose, gt_pose in zip(est_poses, gt_poses):
                R_est, t_est = est_pose[:3, :3], est_pose[:3, 3]
                R_gt, t_gt = gt_pose[:3, :3], gt_pose[:3, 3]

                t_gt_mm = (gt_pose[:3, 3] * 1000).reshape(3, 1)   # GT 位姿，单位 mm
                t_est_mm = (est_pose[:3, 3] * 1000).reshape(3, 1) # 预测位姿，单位 mm

                R_gt = gt_pose[:3, :3]
                R_est = est_pose[:3, :3]

                # 渲染 GT 深度
                gt_render = rnd.render_object(
                    ob_id, R_gt, t_gt_mm,
                    cam_K[0,0], cam_K[1,1], cam_K[0,2], cam_K[1,2]
                )
                gt_depth = gt_render['depth']

                # 渲染预测深度
                est_render = rnd.render_object(
                    ob_id, R_est, t_est_mm,
                    cam_K[0,0], cam_K[1,1], cam_K[0,2], cam_K[1,2]
                )
                est_depth = est_render['depth']

                # print("GT depth sum:", np.sum(gt_depth))
                if np.sum(gt_depth) == 0:
                  logging.warning(f"Skip ob_id={ob_id} due to empty GT depth.")
                  continue

                # VSD 计算也要用毫米
                e_vsd_list = pose_error.vsd(
                    R_est, t_est_mm, R_gt, t_gt_mm, gt_depth, cam_K, delta, vsd_taus_mm,
                    True, diameter * 1000, rnd, ob_id, cost_type='step'
                )

                is_correct = all(e_vsd < theta for e_vsd in e_vsd_list)
                correct_flags.append(is_correct)
                
            vsd_recalls.append(np.mean(correct_flags))
        ar_vsd = np.mean(vsd_recalls)

        # --- MSSD 计算 ---
        mssd_recalls = []
        for theta in mssd_thetas:
            correct_flags = []
            for est_pose, gt_pose in zip(est_poses, gt_poses):
                R_est, t_est = est_pose[:3, :3], est_pose[:3, 3].reshape(3, 1)
                R_gt, t_gt = gt_pose[:3, :3], gt_pose[:3, 3].reshape(3, 1)
                
                e_mssd = pose_error.mssd(R_est, t_est, R_gt, t_gt, model_pts, syms)
                normalized_e_mssd = e_mssd / diameter
                correct_flags.append(normalized_e_mssd < theta)
                
            mssd_recalls.append(np.mean(correct_flags))
        ar_mssd = np.mean(mssd_recalls)

        # --- MSPD 计算 ---
        mspd_recalls = []
        for theta in mspd_thetas:
            correct_flags = []
            for est_pose, gt_pose in zip(est_poses, gt_poses):
                R_est, t_est = est_pose[:3, :3], est_pose[:3, 3].reshape(3, 1)
                R_gt, t_gt = gt_pose[:3, :3], gt_pose[:3, 3].reshape(3, 1)
                
                e_mspd = pose_error.mspd(R_est, t_est, R_gt, t_gt, cam_K, model_pts, syms)
                correct_flags.append(e_mspd < theta)
                
            mspd_recalls.append(np.mean(correct_flags))
        ar_mspd = np.mean(mspd_recalls)

        ar = (ar_vsd + ar_mssd + ar_mspd) / 3.0
        return ar_vsd, ar_mssd, ar_mspd, ar

    def calculate_add_0_1d_success(self, pose_est, pose_gt, model_pts, ob_id):
        diameter = self.meta_data[ob_id]['diameter'] / 1000

        # if self.meta_data[ob_id]['symmetry'] != 'none':
        # error = self.adds_err(pose_est, pose_gt, model_pts, diameter)
        # else:
        error = self.add_err(pose_est, pose_gt, model_pts, diameter)
        
        return error# 是否小于0.1 × diameter（因为误差已归一化）


    # def _add_error(self, pose_est, pose_gt, model_pts, diameter):
    #     est_pts = pose_est[:3, :3] @ model_pts.T + pose_est[:3, [3]]
    #     gt_pts = pose_gt[:3, :3] @ model_pts.T + pose_gt[:3, [3]]
    #     error = np.linalg.norm(est_pts - gt_pts, axis=0).mean()
    #     return error / diameter  # 归一化到物体直径

    # def _add_s_error(self, pose_est, pose_gt, model_pts, diameter):
    #     min_error = float('inf')
    #     # 遍历所有对称变换（需预先定义对称变换矩阵）
    #     for sym_tf in self._get_symmetry_transforms(ob_id):  
    #         pose_est_sym = pose_est @ np.linalg.inv(sym_tf)
    #         error = self._add_error(pose_est_sym, pose_gt, model_pts, diameter)
    #         min_error = min(min_error, error)
    #     return min_error
    def transform_pts(self, pts, tf):
        if len(tf.shape) >= 3 and tf.shape[-3] != pts.shape[-2]:
            tf = tf[..., None, :, :]
        return (tf[..., :-1, :-1] @ pts[..., None] + tf[..., :-1, -1:])[..., 0]

    def add_err(self, pred, gt, model_pts, diameter, symetry_tfs=np.eye(4)[None]):
        pred_pts = self.transform_pts(model_pts, pred)
        gt_pts = self.transform_pts(model_pts, gt)
        e = np.linalg.norm(pred_pts - gt_pts, axis=-1).mean()
        return e / diameter

    def adds_err(self, pred, gt, model_pts, diameter):
        pred_pts = self.transform_pts(model_pts, pred)
        gt_pts = self.transform_pts(model_pts, gt)
        nn_index = cKDTree(pred_pts)
        nn_dists, _ = nn_index.query(gt_pts, k=1, workers=-1)
        return nn_dists.mean() / diameter


def get_mask(reader, i_frame, ob_id, detect_type):
  if detect_type=='box':
    mask = reader.get_mask(i_frame, ob_id)
    H,W = mask.shape[:2]
    vs,us = np.where(mask>0)
    umin = us.min()
    umax = us.max()
    vmin = vs.min()
    vmax = vs.max()
    valid = np.zeros((H,W), dtype=bool)
    valid[vmin:vmax,umin:umax] = 1
  elif detect_type=='mask':
    mask = reader.get_mask(i_frame, ob_id)
    if mask is None:
      return None
    valid = mask>0
  elif detect_type=='detected':
    mask = cv2.imread(reader.color_files[i_frame].replace('rgb','mask_cosypose'), -1)
    valid = mask==ob_id
  else:
    raise RuntimeError
  return valid

def run_pose_estimation_worker(reader, i_frames, est:FoundationPose=None, debug=0, ob_id=None, device='cuda:0', evaluator=None, model_pts=None, error_dict=None):
  torch.cuda.set_device(device)
  
  est.to_device(device)
  est.glctx = dr.RasterizeCudaContext(device=device)

  result = NestDict()
  result_gt = NestDict()
  if (ob_id != 6):
    return result, result_gt, error_dict
  
  for i, i_frame in enumerate(i_frames):
    # if (i_frame > 100):
    #     break
    
    logging.info(f"{i}/{len(i_frames)}, i_frame:{i_frame}, ob_id:{ob_id}")
    video_id = reader.get_video_id()
    color = reader.get_color(i_frame)
    depth = reader.get_depth(i_frame)
    id_str = reader.id_strs[i_frame]
    H,W = color.shape[:2]
    debug_dir =est.debug_dir

    # save_dir = "rgb_color"
    # os.makedirs(save_dir, exist_ok=True)
    # filename = f"ob_{i_frame:03d}.png"  # ob_005.png
    # save_path = os.path.join(save_dir, filename)
    # cv2.imwrite(save_path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    # est.to_device(device)
    ob_mask = get_mask(reader, i_frame, ob_id, detect_type=detect_type)
    if ob_mask is None:
      logging.info("ob_mask not found, skip")
      result[video_id][id_str][ob_id] = np.eye(4)
      result_gt[video_id][id_str][ob_id] = np.eye(4)
      return result, result_gt, error_dict

    est.gt_pose = reader.get_gt_pose(i_frame, ob_id)
    pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=ob_mask, ob_id=ob_id)
    logging.info(f"pose:\n{pose}")

    error = evaluator.calculate_add_0_1d_success(pose,  est.gt_pose, model_pts, ob_id)
    logging.info(f"error:\n{error}")
    error_dict[ob_id].append(error)

    # def draw_3d_bbox(image, pose, K, model_pts, diameter=None, scale_factor=0.9,
    #                       color=(0, 255, 0), thickness=2, crop_size=224, crop_scale=1.2):
    #   min_xyz = model_pts.min(axis=0)
    #   max_xyz = model_pts.max(axis=0)
    #   center = (min_xyz + max_xyz) / 2.0

    #   # 缩放
    #   half_size = (max_xyz - min_xyz) / 2.0 * scale_factor
    #   min_xyz = center - half_size
    #   max_xyz = center + half_size

    #   if diameter is not None:
    #       radius = (diameter / 2) * scale_factor
    #       min_xyz = center - radius
    #       max_xyz = center + radius

    #   # 8 个角点
    #   bbox_3d = np.array([
    #       [min_xyz[0], min_xyz[1], min_xyz[2]],
    #       [max_xyz[0], min_xyz[1], min_xyz[2]],
    #       [max_xyz[0], max_xyz[1], min_xyz[2]],
    #       [min_xyz[0], max_xyz[1], min_xyz[2]],
    #       [min_xyz[0], min_xyz[1], max_xyz[2]],
    #       [max_xyz[0], min_xyz[1], max_xyz[2]],
    #       [max_xyz[0], max_xyz[1], max_xyz[2]],
    #       [min_xyz[0], max_xyz[1], max_xyz[2]],
    #   ])

    #   # 相机坐标
    #   ones = np.ones((bbox_3d.shape[0], 1))
    #   bbox_h = np.hstack([bbox_3d, ones])
    #   bbox_cam = (pose @ bbox_h.T).T[:, :3]

    #   # 投影到 2D
    #   pts_2d = (K @ bbox_cam.T).T
    #   pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]
    #   pts_2d = pts_2d.astype(int)

    #   # 画框
    #   edges = [
    #       (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
    #       (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
    #       (0, 4), (1, 5), (2, 6), (3, 7)   # 立柱
    #   ]
    #   for i, j in edges:
    #       cv2.line(image, tuple(pts_2d[i]), tuple(pts_2d[j]), color, thickness)

    #   # ====== 新增：根据中心和直径比例 crop ======
    #   x_min, y_min = pts_2d.min(axis=0)
    #   x_max, y_max = pts_2d.max(axis=0)

    #   cx = (x_min + x_max) / 2
    #   cy = (y_min + y_max) / 2
    #   box_size = max(x_max - x_min, y_max - y_min) * crop_scale

    #   x1 = int(cx - box_size / 2)
    #   y1 = int(cy - box_size / 2)
    #   x2 = int(cx + box_size / 2)
    #   y2 = int(cy + box_size / 2)

    #   # 边界裁剪
    #   h, w = image.shape[:2]
    #   x1 = max(0, x1)
    #   y1 = max(0, y1)
    #   x2 = min(w, x2)
    #   y2 = min(h, y2)

    #   cropped = image[y1:y2, x1:x2]
    #   cropped_resized = cv2.resize(cropped, (crop_size, crop_size))
    #   return cropped_resized

    # if (error < 0.05):
    #   # 画框
    #   image = draw_3d_bbox(color, pose, reader.K, model_pts)
    #   save_dir = "error_img"
    #   os.makedirs(save_dir, exist_ok=True)
    #   filename = f"ob_{ob_id:03d}_i_frame{i_frame:03d}.png"  # ob_005.png
    #   save_path = os.path.join(save_dir, filename)
    #   cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    #   est.to_device(device)

    if debug>=3:
      m = est.mesh_ori.copy()
      tmp = m.copy()
      tmp.apply_transform(pose)
      tmp.export(f'{debug_dir}/model_tf.obj')

    result[video_id][id_str][ob_id] = pose
    result_gt[video_id][id_str][ob_id] = est.gt_pose 

  
  # logging.info(f"len(result):{len(result)}, len(result_gt):{len(result_gt)}")

  return result, result_gt, error_dict


def compute_auc_sklearn(errs, max_val=0.1, step=0.001):
  from sklearn import metrics
  errs = np.sort(np.array(errs))
  X = np.arange(0, max_val+step, step)
  Y = np.ones(len(X))
  for i,x in enumerate(X):
    y = (errs<=x).sum()/len(errs)
    Y[i] = y
    if y>=1:
      break
  auc = metrics.auc(X, Y) / (max_val*1)
  return auc

def run_pose_estimation():
  manager = multiprocessing.Manager()
  error_dict = manager.dict()
  wp.force_load(device='cuda')
  reader_tmp = LinemodOcclusionReader(f'{opt.linemod_dir}/lm_test_all/test/000002')

  evaluator = PoseEvaluator(opt.linemod_dir)

  debug = opt.debug
  use_reconstructed_mesh = opt.use_reconstructed_mesh
  debug_dir = opt.debug_dir

  res = NestDict()
  res_gt = NestDict()
  glctx = dr.RasterizeCudaContext()
  mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)).to_mesh()
  est = FoundationPose(model_pts=mesh_tmp.vertices.copy(), model_normals=mesh_tmp.vertex_normals.copy(), symmetry_tfs=None, mesh=mesh_tmp, scorer=None, refiner=None, glctx=glctx, debug_dir=debug_dir, debug=debug)
  
  ar_values = []
  for ob_id in reader_tmp.ob_ids:
    all_est_poses = []
    all_gt_poses = []
    error_dict[ob_id] = manager.list()
    ob_id = int(ob_id)
    if use_reconstructed_mesh:
      mesh = reader_tmp.get_reconstructed_mesh(ob_id, ref_view_dir=opt.ref_view_dir)
    else:
      mesh = reader_tmp.get_gt_mesh(ob_id)
    symmetry_tfs = reader_tmp.symmetry_tfs[ob_id]
    
    args = []

    video_dir = f'{opt.linemod_dir}/lm_test_all/test/{2:06d}'
    reader = LinemodOcclusionReader(video_dir)
    video_id = reader.get_video_id()
    est.reset_object(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(), symmetry_tfs=symmetry_tfs, mesh=mesh)
    model_pts=mesh.vertices.copy()
    scene_ob_ids = reader.make_scene_ob_ids_dict()
    # for i in range(len(reader.color_files)):
    # for im_id in scene_ob_ids:
    #   args.append((reader, im_id, est, debug, ob_id, "cuda:0", evaluator, model_pts, error_dict))
    scene_ob_ids = reader.make_scene_ob_ids_dict()
    for im_id_str in scene_ob_ids:
        i_frame = int(im_id_str)  # 转回整数索引
        args.append((reader, [i_frame], est, debug, ob_id, "cuda:0", evaluator, model_pts, error_dict))

    outs = []
    outs_gt = []
    for arg in args:
      out, out_gt, error_dict = run_pose_estimation_worker(*arg)
      outs.append(out)
      outs_gt.append(out_gt)
    
    # for out in outs:
    #   for video_id in out:
    #     for id_str in out[video_id]:
    #       for ob_id in out[video_id][id_str]:
    #         res[video_id][id_str][ob_id] = out[video_id][id_str][ob_id]
    #         res_gt[video_id][id_str][ob_id] = out[video_id][id_str][ob_id]

    for out, out_gt in zip(outs, outs_gt):
        for video_id in out:
            for id_str in out[video_id]:
                for obj_id in out[video_id][id_str]:
                    all_est_poses.append(out[video_id][id_str][obj_id])
                    all_gt_poses.append(out_gt[video_id][id_str][obj_id])
    
    ar_vsd, ar_mssd, ar_mspd, ar = evaluator.compute_ar_for_dataset(model_pts, all_est_poses, all_gt_poses, reader.K, ob_id)
    obj_num = len(all_gt_poses)
    ar_values.append((ob_id, obj_num, ar_vsd, ar_mssd, ar_mspd, ar))


  logging.info("=== 所有 ob_id 的 AR 结果 ===")
  for ob_id, obj_num, ar_vsd, ar_mssd, ar_mspd, ar in ar_values:
      logging.info(f"ob_id:{ob_id}, obj_num:{obj_num}")
      logging.info(f"AR_VSD={ar_vsd:.3f}, AR_MSSD={ar_mssd:.3f}, AR_MSPD={ar_mspd:.3f}, AR={ar:.3f}")
        
    # with open(f'{opt.debug_dir}/linemod_res.yml','w') as ff:
    #   yaml.safe_dump(make_yaml_dumpable(res), ff)


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--linemod_dir', type=str, default="/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/LINEMOD", help="linemod root dir")
  parser.add_argument('--use_reconstructed_mesh', type=int, default=0)
  parser.add_argument('--ref_view_dir', type=str, default="/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/bowen_addon/ref_views_16")
  parser.add_argument('--debug', type=int, default=0)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  opt = parser.parse_args()
  set_seed(0)

  detect_type = 'mask'   # mask / box / detected
  
  start_time = time.time()  #
  run_pose_estimation()
  end_time = time.time()
  total_time = end_time - start_time
  hours, rem = divmod(total_time, 3600)
  minutes, seconds = divmod(rem, 60)
  
  print("\n===== Time Statistics =====")
  print(f"Total execution time: {int(hours):02d}h {int(minutes):02d}m {seconds:.2f}s")
