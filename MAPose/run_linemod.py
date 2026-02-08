# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# 1. 相似度修改
# 2. iter次数修改
# python run_linemod.py --linemod_dir  /home/yluo/GSPose/dataspace/bop_dataset/Linmod  --use_reconstructed_mesh 0


from scipy.spatial.transform import Rotation as R
from Utils import *
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

# 在现有代码框架中添加以下关键模块


class PoseEvaluator:
    def __init__(self, dataset_root):
        self.meta_data = self._load_linemod_meta(dataset_root)
        
    def _load_linemod_meta(self, path):
      """从 models_info.json + PLY 文件加载元数据"""
      meta = {}
      models_dir = Path(path) / "models"
      
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
          
          # 6. 构建元数据
          meta[ob_id] = {
              'diameter': info['diameter'],  # 优先使用标注值
              'symmetry': info.get('symmetry', 'none'),
              'mesh': mesh
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

    def calculate_rot_trans_error(self, pred, gt):
      # 提取旋转和平移部分
      R_pred = pred[:3, :3]
      R_gt = gt[:3, :3]
      t_pred = pred[:3, 3]
      t_gt = gt[:3, 3]

      # 计算旋转误差（角度）
      delta_R = R_pred @ R_gt.T
      rot_vector = R.from_matrix(delta_R).as_rotvec()
      rot_err_deg = np.linalg.norm(rot_vector) * 180.0 / np.pi

      # trans_err_vec = t_pred - t_gt
      # trans_err_x = trans_err_vec[0]
      # trans_err_y = trans_err_vec[1]
      # trans_err_z = trans_err_vec[2]
      # logging.info(f"平移误差分量:")
      # logging.info(f"  x 轴: {trans_err_x*1000:.2f} 毫米")
      # logging.info(f"  y 轴: {trans_err_y*1000:.2f} 毫米")
      # logging.info(f"  z 轴: {trans_err_z*1000:.2f} 毫米")

      # 计算平移误差（欧式距离）
      trans_err_m = np.linalg.norm(t_pred - t_gt)
      return rot_err_deg, trans_err_m


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
  
  for i, i_frame in enumerate(i_frames):
    if (ob_id != 13):
      break
    
    # # # # # # logging.info(f"i_frame is: {i_frame} ")   
    # if (i_frame < 8):
      #  break
    # if (i_frame > 8):
        # break
    
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
      return result

    est.gt_pose = reader.get_gt_pose(i_frame, ob_id)

    pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=ob_mask, ob_id=ob_id)
    logging.info(f"pose:\n{pose}")
    logging.info(f"gt pose:\n{est.gt_pose}")

    error = evaluator.calculate_add_0_1d_success(pose, est.gt_pose, model_pts, ob_id)
    logging.info(f"error:\n{error}")
    error_dict[ob_id].append(error)
    rot_err, trans_err = evaluator.calculate_rot_trans_error(pose, est.gt_pose)
    
    logging.info(f"旋转误差:\n{rot_err:.2f}°")
    logging.info(f"°，平移误差:\n{trans_err*1000:.2f}毫米")

    if debug>=3:
      m = est.mesh_ori.copy()
      tmp = m.copy()
      tmp.apply_transform(pose)
      tmp.export(f'{debug_dir}/model_tf.obj')

    result[video_id][id_str][ob_id] = pose

  return result, error_dict


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
  reader_tmp = LinemodReader(f'{opt.linemod_dir}/test/000002', split=None)

  evaluator = PoseEvaluator(opt.linemod_dir)

  debug = opt.debug
  use_reconstructed_mesh = opt.use_reconstructed_mesh
  debug_dir = opt.debug_dir

  res = NestDict()
  glctx = dr.RasterizeCudaContext()
  mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)).to_mesh()
  est = FoundationPose(model_pts=mesh_tmp.vertices.copy(), model_normals=mesh_tmp.vertex_normals.copy(), symmetry_tfs=None, mesh=mesh_tmp, scorer=None, refiner=None, glctx=glctx, debug_dir=debug_dir, debug=debug)

  # for ob_id in reader_tmp.ob_ids:
  for ob_id in [13]:
    error_dict[ob_id] = manager.list()
    ob_id = int(ob_id)
    
    # if (ob_id != 12):
    #   continue

    if use_reconstructed_mesh:
      mesh = reader_tmp.get_reconstructed_mesh(ob_id, ref_view_dir=opt.ref_view_dir)
    else:
      mesh = reader_tmp.get_gt_mesh(ob_id)

    # mesh.visual.vertex_colors = None
    # mesh.visual.face_colors = None
    # mesh.visual.material = None
    # 
    symmetry_tfs = reader_tmp.symmetry_tfs[ob_id]
    
    args = []

    video_dir = f'{opt.linemod_dir}/test/{ob_id:06d}'
    reader = LinemodReader(video_dir, split=None)
    video_id = reader.get_video_id()
    est.reset_object(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(), symmetry_tfs=symmetry_tfs, mesh=mesh)
    model_pts=mesh.vertices.copy()
    for i in range(len(reader.color_files)):
      args.append((reader, [i], est, debug, ob_id, "cuda:0", evaluator, model_pts, error_dict))

    outs = []
    for arg in args:
      out, error_dict = run_pose_estimation_worker(*arg)
      outs.append(out)
    
    for out in outs:
      for video_id in out:
        for id_str in out[video_id]:
          for ob_id in out[video_id][id_str]:
            res[video_id][id_str][ob_id] = out[video_id][id_str][ob_id]
    #break
  
  thresholds = np.linspace(0, 0.1, 100)  # 定义阈值范围0-0.1
  all_errors = []
  for ob_id in error_dict:
      errors = list(error_dict[ob_id])  # 转换为普通列表
      all_errors.extend(errors)
  all_errors = np.array(all_errors)

  np.savetxt("all_errors.txt", all_errors, fmt="%.6f")

  for ob_id in error_dict:
    success_flags = list(error_dict[ob_id])  # 0 or 1
    recall = np.mean(success_flags)
    thresholds_auc = np.linspace(0, 1, 100)
    correct_rates_auc = [np.mean(success_flags <= t) for t in thresholds_auc]
    recall = np.trapz(correct_rates_auc, thresholds_auc)
    print(f"Object {ob_id} AUC recall: {recall:.3f}")

  # 收集全部误差
  all_errors = []
  for ob_id in error_dict:
      errors = list(error_dict[ob_id])
      all_errors.extend(errors)
  all_errors = np.array(all_errors)

  # AUC@0.1
  print(all_errors)
  auc = compute_auc_sklearn(all_errors)
  print(f"AUC: {auc:.4f}")

  # Mean ADD Error
  mean_error = np.mean(all_errors)
  print(f"Mean ADD Error: {mean_error:.6f}")

  # Median Error
  median_error = np.median(all_errors)
  print(f"Median ADD Error: {median_error:.6f}")

  success_thresh = 0.1
  all_success_flags = []


  for ob_id in error_dict:
    errors = np.array(error_dict[ob_id])
    add_success_rate = np.mean(errors <= success_thresh)
    print(f"[Object {ob_id}] ADD(-S) @ 0.1d: {add_success_rate:.4f}")
    all_success_flags.extend(errors <= success_thresh)

  # Overall accuracy across all objects
  overall_add_success = np.mean(all_success_flags)
  print(f"[ALL] ADD(-S) @ 0.1d: {overall_add_success:.4f}")


  with open(f'{opt.debug_dir}/linemod_res.yml','w') as ff:
    yaml.safe_dump(make_yaml_dumpable(res), ff)


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
