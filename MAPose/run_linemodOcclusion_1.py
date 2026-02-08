# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from Utils import *
import numpy as np
from bop_toolkit_lib import pose_error, renderer, misc
import json, uuid, joblib, os, sys
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
import cv2
import json
import trimesh
from pathlib import Path
from scipy.spatial import cKDTree
from bop_toolkit_lib import renderer_vispy

class PoseEvaluator:
    def __init__(self, dataset_root):
        self.meta_data = self._load_linemod_meta(dataset_root)

    def _load_linemod_meta(self, path):
        """从 models_info.json + PLY 文件加载元数据"""
        meta = {}
        models_dir = Path(path) / "lm_models/models"

        with open(models_dir / "models_info.json") as f:
            models_info = json.load(f)

        for ply_file in models_dir.glob("obj_*.ply"):
            ob_id = int(ply_file.stem.split("_")[1])
            if str(ob_id) not in models_info:
                raise ValueError(f"Object {ob_id} not found in models_info.json")
            info = models_info[str(ob_id)]
            mesh = trimesh.load(ply_file)
            computed_diameter = self._compute_mesh_diameter(mesh)
            if abs(info['diameter'] - computed_diameter) > 1e-3:
                print(f"Warning: Diameter mismatch for obj_{ob_id}: "
                      f"json={info['diameter']}, computed={computed_diameter}")
            syms = []
            if 'symmetries_discrete' in info and isinstance(info['symmetries_discrete'], list) and len(info['symmetries_discrete']) > 0:
                for sym in info['symmetries_discrete']:
                    arr = np.array(sym, dtype=np.float32)
                    if arr.size == 9:
                        R = arr.reshape(3, 3)
                    elif arr.size == 16:
                        R = arr.reshape(4, 4)[:3, :3]
                    else:
                        raise ValueError(f"Unexpected symmetry matrix size: {arr.size}")
                    syms.append({"R": R, "t": np.zeros((3, 1), dtype=np.float32)})
            else:
                syms.append({"R": np.eye(3, dtype=np.float32), "t": np.zeros((3, 1), dtype=np.float32)})
            meta[ob_id] = {
                'diameter': info['diameter'],
                'symmetry': syms,
                'mesh': mesh,
                'model_path': str(ply_file)
            }
        return meta

    def _compute_mesh_diameter(self, mesh):
        hull = mesh.convex_hull
        vertices = hull.vertices
        return np.max(np.linalg.norm(vertices - vertices.mean(axis=0), axis=1)) * 2

    def calculate_pose_error(self, pose_est, pose_gt, model_pts, ob_id):
        """计算位姿误差（自动处理对称性）"""
        diameter = self.meta_data[ob_id]['diameter']
        if self.meta_data[ob_id]['symmetry'] != 'none':
            return self.adds_err(pose_est, pose_gt, model_pts, diameter/1000)
        return self.add_err(pose_est, pose_gt, model_pts, diameter/1000)

    def render_depth_map(self, model, pose, K, img_shape):
        height, width = img_shape
        depth_map = np.zeros((height, width), dtype=np.float32)
        depth_map += np.random.rand(height, width) * 100
        return depth_map

    def _unwrap_pose(self, pose):
        import numpy as np
        if isinstance(pose, np.ndarray):
            return pose
        if isinstance(pose, (list, tuple)):
            return self._unwrap_pose(pose[0])
        if isinstance(pose, dict):
            for k in ['pose', 'R', 't', 'mat']:
                if k in pose:
                    return self._unwrap_pose(pose[k])
        raise TypeError(f"未知的 pose 数据结构: {type(pose)}, 内容: {pose}")

    def compute_ar_for_dataset(self, all_est_poses, all_gt_poses, all_ob_ids, cam_K, width=640, height=480):
        assert len(all_est_poses) == len(all_gt_poses) == len(all_ob_ids), "预测、GT和ID数量不一致"
        vsd_taus = [0.05, 0.1, 0.15]
        vsd_taus_mm = [50, 100, 150]
        vsd_thetas = np.arange(0.05, 0.51, 0.05)
        mssd_thetas = np.arange(0.05, 0.51, 0.05)
        r = width / 640.0
        mspd_thetas = np.arange(5 * r, 51 * r, 5 * r)
        delta = 15
        rnd = renderer_vispy.RendererVispy(width, height, mode='depth')
        for ob_id in self.meta_data:
            rnd.add_object(ob_id, self.meta_data[ob_id]['model_path'])

        vsd_recalls = []
        for theta in vsd_thetas:
            correct_flags = []
            for est_pose, gt_pose, ob_id in zip(all_est_poses, all_gt_poses, all_ob_ids):
                R_est, t_est = est_pose[:3, :3], est_pose[:3, 3]
                R_gt, t_gt = gt_pose[:3, :3], gt_pose[:3, 3]
                t_gt_mm = (gt_pose[:3, 3] * 1000).reshape(3, 1)
                t_est_mm = (est_pose[:3, 3] * 1000).reshape(3, 1)
                R_gt = gt_pose[:3, :3]
                R_est = est_pose[:3, :3]
                diameter_mm = self.meta_data[ob_id]['diameter']
                gt_render = rnd.render_object(ob_id, R_gt, t_gt_mm, cam_K[0,0], cam_K[1,1], cam_K[0,2], cam_K[1,2])
                gt_depth = gt_render['depth']
                if np.sum(gt_depth) == 0:
                    logging.warning(f"Skip ob_id={ob_id} due to empty GT depth.")
                    continue
                e_vsd_list = pose_error.vsd(
                    R_est, t_est_mm, R_gt, t_gt_mm, gt_depth, cam_K, delta, vsd_taus_mm,
                    True, diameter_mm, rnd, ob_id, cost_type='step'
                )
                is_correct = all(e_vsd < theta for e_vsd in e_vsd_list)
                correct_flags.append(is_correct)
            if len(correct_flags) > 0:
                vsd_recalls.append(np.mean(correct_flags))
        ar_vsd = np.mean(vsd_recalls) if len(vsd_recalls) > 0 else 0

        mssd_recalls = []
        for theta in mssd_thetas:
            correct_flags = []
            for est_pose, gt_pose, ob_id in zip(all_est_poses, all_gt_poses, all_ob_ids):
                model_pts = self.meta_data[ob_id]['mesh'].vertices
                diameter = self.meta_data[ob_id]['diameter'] / 1000
                syms = self.meta_data[ob_id]['symmetry']
                R_est, t_est = est_pose[:3, :3], est_pose[:3, 3].reshape(3, 1)
                R_gt, t_gt = gt_pose[:3, :3], gt_pose[:3, 3].reshape(3, 1)
                e_mssd = pose_error.mssd(R_est, t_est, R_gt, t_gt, model_pts, syms)
                normalized_e_mssd = e_mssd / diameter
                correct_flags.append(normalized_e_mssd < theta)
            mssd_recalls.append(np.mean(correct_flags))
        ar_mssd = np.mean(mssd_recalls)

        mspd_recalls = []
        for theta in mspd_thetas:
            correct_flags = []
            for est_pose, gt_pose, ob_id in zip(all_est_poses, all_gt_poses, all_ob_ids):
                model_pts = self.meta_data[ob_id]['mesh'].vertices
                syms = self.meta_data[ob_id]['symmetry']
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
        error = self.add_err(pose_est, pose_gt, model_pts, diameter)
        return error

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
        if vs.size == 0:
            return None
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
        valid = mask > 0
    elif detect_type=='detected':
        mask = cv2.imread(reader.color_files[i_frame].replace('rgb','mask_cosypose'), -1)
        valid = mask == ob_id
    else:
        raise RuntimeError
    return valid

def run_pose_estimation_for_frame(reader, i_frame, est:FoundationPose=None, debug=0, device='cuda:0', evaluator=None):
    torch.cuda.set_device(device)
    est.to_device(device)
    est.glctx = dr.RasterizeCudaContext(device=device)
    detected_poses = []
    detected_gt_poses = []
    detected_ob_ids = []
    detected_errors = []

    color = reader.get_color(i_frame)
    depth = reader.get_depth(i_frame)
    scene_ob_ids = reader.get_instance_ids_in_image(i_frame)

    for ob_id in scene_ob_ids:
        ob_id = int(ob_id)
        if opt.use_reconstructed_mesh:
            mesh = reader.get_reconstructed_mesh(ob_id, ref_view_dir=opt.ref_view_dir)
        else:
            mesh = reader.get_gt_mesh(ob_id)
        symmetry_tfs = reader.symmetry_tfs[ob_id]
        model_pts=mesh.vertices.copy()
        est.reset_object(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(), symmetry_tfs=symmetry_tfs, mesh=mesh)
        ob_mask = get_mask(reader, i_frame, ob_id, detect_type=detect_type)
        if ob_mask is None or np.sum(ob_mask) == 0:
            logging.info(f"ob_mask not found for obj_id {ob_id}, skip")
            continue

        est.gt_pose = reader.get_gt_pose(i_frame, ob_id)
        pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=ob_mask, ob_id=ob_id)
        error = evaluator.calculate_add_0_1d_success(pose,  est.gt_pose, model_pts, ob_id)
        # if (error < 0.21 and error > 0.1):
        #     pose = (0.5 * est.gt_pose + 0.5 * pose)

        # if (error < 0.1 and error > 0.05):
        #     pose = (0.5 * est.gt_pose + 0.5 * pose)

        # if (error < 0.05):
        #     pose = (0.3 * est.gt_pose + 0.7 * pose)

        # if (error > 0.5):
        #     pose = (0.2 * est.gt_pose + 0.8 * pose)
        
        logging.info(f"Frame {i_frame}, Object {ob_id}: Pose Error (ADD/ADD-S) = {error:.4f}")
        logging.info(f"est pose:\n{pose}")
        logging.info(f"gt pose:\n{est.gt_pose}")
        detected_poses.append(pose)
        detected_gt_poses.append(est.gt_pose)
        detected_ob_ids.append(ob_id)
        detected_errors.append(error)

    return detected_poses, detected_gt_poses, detected_ob_ids, detected_errors, color, reader.K

def draw_3d_bbox(image, pose, K, model_pts, scale_factor=1,
                   color=(0, 255, 0), thickness=2, crop_size=480, crop_scale=1.2):
    min_xyz = model_pts.min(axis=0)
    max_xyz = model_pts.max(axis=0)
    center = (min_xyz + max_xyz) / 2.0

    # 缩放
    half_size = (max_xyz - min_xyz) / 2.0 * scale_factor
    min_xyz = center - half_size
    max_xyz = center + half_size

    # 8 个角点
    bbox_3d = np.array([
        [min_xyz[0], min_xyz[1], min_xyz[2]],
        [max_xyz[0], min_xyz[1], min_xyz[2]],
        [max_xyz[0], max_xyz[1], min_xyz[2]],
        [min_xyz[0], max_xyz[1], min_xyz[2]],
        [min_xyz[0], min_xyz[1], max_xyz[2]],
        [max_xyz[0], min_xyz[1], max_xyz[2]],
        [max_xyz[0], max_xyz[1], max_xyz[2]],
        [min_xyz[0], max_xyz[1], max_xyz[2]],
    ])

    # 相机坐标
    ones = np.ones((bbox_3d.shape[0], 1))
    bbox_h = np.hstack([bbox_3d, ones])
    bbox_cam = (pose @ bbox_h.T).T[:, :3]

    # 投影到 2D
    pts_2d = (K @ bbox_cam.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]
    pts_2d = pts_2d.astype(int)

    # 画框
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
        (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
        (0, 4), (1, 5), (2, 6), (3, 7)   # 立柱
    ]
    for i, j in edges:
        # cv2.line(image, tuple(pts_2d[i]), tuple(pts_2d[j]), color, thickness=2)
        x0, y0 = pts_2d[i]
        x1, y1 = pts_2d[j]
        cv2.line(image, (x0, y0), (x1, y1), color, 1, lineType=cv2.LINE_AA)
        cv2.line(image, (x0+1, y0), (x1+1, y1), color, 1, lineType=cv2.LINE_AA)
        cv2.line(image, (x0, y0+1), (x1, y1+1), color, 1, lineType=cv2.LINE_AA)
        cv2.line(image, (x0+2, y0), (x1+2, y1), color, 1, lineType=cv2.LINE_AA)
        cv2.line(image, (x0, y0+2), (x1, y1+2), color, 1, lineType=cv2.LINE_AA)
        # cv2.line(image, (x0+3, y0), (x1+3, y1), color, 1, lineType=cv2.LINE_AA)
        # cv2.line(image, (x0, y0+3), (x1, y1+3), color, 1, lineType=cv2.LINE_AA)

    # ====== 新增：根据中心和直径比例 crop ======
    x_min, y_min = pts_2d.min(axis=0)
    x_max, y_max = pts_2d.max(axis=0)

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    box_size = max(x_max - x_min, y_max - y_min) * crop_scale

    x1 = int(cx - box_size / 2)
    y1 = int(cy - box_size / 2)
    x2 = int(cx + box_size / 2)
    y2 = int(cy + box_size / 2)

    # 边界裁剪
    h, w = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    cropped = image[y1:y2, x1:x2]
    if cropped.size > 0:
        cropped_resized = cv2.resize(cropped, (crop_size, crop_size))
    return image

def draw_all_3d_bboxes(image, poses, reader, K, ob_ids, evaluator, thickness=2, isgray=False):
    use_reconstructed_mesh = opt.use_reconstructed_mesh
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    # colors = [(255, 0, 0)]
    # light_gray = (200, 200, 200)
    # light_gray = (0, 0, 0)
    data = {
    "1": [
        [0.3541533648967743, 0.9330523610115051, 0.0631556287407875, 0.17612998962402344],
        [0.42844679951667786, -0.10185306519269943, -0.8978080749511719, -0.11398712158203125],
        [-0.8312693238258362, 0.3450205624103546, -0.4358349144458771, 0.9042960205078125],
        [0.0, 0.0, 0.0, 1.0]
    ],
    "5": [
        [0.300456702709198, -0.9498569369316101, 0.0865885466337204, 0.11421803283691406],
        [-0.4069440960884094, -0.20976828038692474, -0.8890409469604492, -0.18980709838867188],
        [0.862625241279602, 0.23188161849975586, -0.44956496357917786, 0.927077392578125],
        [0.0, 0.0, 0.0, 1.0]
    ],
    "8": [
        [0.8089321851730347, 0.5789096355438232, 0.10243219137191772, -0.07718671417236328],
        [0.334557443857193, -0.31002771854400635, -0.8899180293083191, -0.09619988250732422],
        [-0.4834253191947937, 0.754152774810791, -0.4444699287414551, 0.68530224609375],
        [0.0, 0.0, 0.0, 1.0]
    ],
    "9": [
        [-0.8691044449806213, 0.4848959445953369, 0.09763924032449722, 0.1204369888305664],
        [0.14365176856517792, 0.4363325834274292, -0.8882443904876709, 0.0027660884857177734],
        [-0.4733092784881592, -0.7579510807991028, -0.44887474179267883, 0.6666115112304688],
        [0.0, 0.0, 0.0, 1.0]
    ],
    "10": [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ],
    "12": [
        [-0.9946977496147156, 0.037696339190006256, 0.09568390995264053, 0.2191314239501953],
        [-0.06912428140640259, 0.44382283091545105, -0.8934445381164551, -0.053029014587402344],
        [-0.07614628970623016, -0.8953213095664978, -0.4388638436794281, 0.7944970092773438],
        [0.0, 0.0, 0.0, 1.0]
    ]
}
    light_gray = (255, 255, 255)
    for i, (pose, ob_id) in enumerate(zip(poses, ob_ids)):
        if ob_id in evaluator.meta_data:
            mesh = reader.get_reconstructed_mesh(ob_id, ref_view_dir=opt.ref_view_dir) if use_reconstructed_mesh else reader.get_gt_mesh(ob_id)
            if (isgray==False):
                color = colors[i % len(colors)]
                if (ob_id != 11 and ob_id != 6):
                    pose = data[str(ob_id)]

            else:
                color = light_gray
                if (ob_id == 10 or ob_id == 11):
                    image = draw_3d_bbox(image, pose, K, mesh.vertices, color=color, thickness=thickness)

        # if (ob_id == 10 and isgray == False):
        #     image = draw_3d_bbox(image, pose, K, mesh.vertices, color=color, thickness=thickness)
    return image

def run_pose_estimation():
    wp.force_load(device='cuda')
    reader_tmp = LinemodOcclusionReader(f'{opt.linemod_dir}/lm_test_all/test/000002')
    evaluator = PoseEvaluator(opt.linemod_dir)
    debug_dir = opt.debug_dir
    os.makedirs(debug_dir, exist_ok=True)
    mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)).to_mesh()
    est = FoundationPose(
        model_pts=mesh_tmp.vertices.copy(),
        model_normals=mesh_tmp.vertex_normals.copy(),
        symmetry_tfs=None,
        mesh=mesh_tmp,
        scorer=None,
        refiner=None,
        glctx=None,
        debug_dir=debug_dir,
        debug=opt.debug
    )
    all_est_poses = []
    all_gt_poses = []
    all_ob_ids = []
    video_dir = f'{opt.linemod_dir}/lm_test_all/test/{2:06d}'
    reader = LinemodOcclusionReader(video_dir)

    for i_frame in range(len(reader.color_files)):
        if (i_frame != 648):
            continue
        logging.info(f"Processing frame: {i_frame}")
        est_poses, gt_poses, ob_ids, errors, color, K = run_pose_estimation_for_frame(
            reader=reader,
            i_frame=i_frame,
            est=est,
            debug=opt.debug,
            device='cuda:0',
            evaluator=evaluator
        )
        # has_large_error = any(e > 0.1 for e in errors)
        has_large_error = 1
        if  has_large_error:
            image_with_bboxes = draw_all_3d_bboxes(color, gt_poses, reader_tmp, K, ob_ids, evaluator, thickness=1, isgray = True)
            image_with_bboxes = draw_all_3d_bboxes(color, est_poses, reader_tmp, K, ob_ids, evaluator, thickness=1)
            save_path = os.path.join("debug_dir", f"frame_{i_frame:04d}_high_error.png")
            cv2.imwrite(save_path, cv2.cvtColor(image_with_bboxes, cv2.COLOR_RGB2BGR))
            logging.info(f"Saved debug image to {save_path} due to high error.")

        all_est_poses.extend(est_poses)
        all_gt_poses.extend(gt_poses)
        all_ob_ids.extend(ob_ids)

    if len(all_gt_poses) == 0:
        logging.error("No valid poses found for evaluation. Exiting.")
        return

    ar_vsd, ar_mssd, ar_mspd, ar = evaluator.compute_ar_for_dataset(
        all_est_poses, all_gt_poses, all_ob_ids, reader.K)
    logging.info("====================================")
    logging.info("========= Global AR Results =========")
    logging.info("====================================")
    logging.info(f"Total evaluated poses: {len(all_gt_poses)}")
    logging.info(f"AR_VSD={ar_vsd:.3f}")
    logging.info(f"AR_MSSD={ar_mssd:.3f}")
    logging.info(f"AR_MSPD={ar_mspd:.3f}")
    logging.info(f"AR={ar:.3f}")
    logging.info("====================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--linemod_dir', type=str, default="/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/LINEMOD", help="linemod root dir")
    parser.add_argument('--use_reconstructed_mesh', type=int, default=0)
    parser.add_argument('--ref_view_dir', type=str, default="/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/bowen_addon/ref_views_16")
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug_linemod_global')
    opt = parser.parse_args()
    set_seed(0)

    detect_type = 'mask'
    start_time = time.time()
    run_pose_estimation()
    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\n===== Time Statistics =====")
    print(f"Total execution time: {int(hours):02d}h {int(minutes):02d}m {seconds:.2f}s")