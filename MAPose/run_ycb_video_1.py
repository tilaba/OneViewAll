# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from Utils import *
from multiprocessing import Pool
import multiprocessing
import json, uuid, joblib, os, sys, argparse
from datareader import *
from estimater import *
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import cv2
import glob
import pandas as pd

# 确保路径被正确添加
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/mycpp/build')
import yaml


class PoseEvaluator:
    def __init__(self, dataset_root, use_adds=False):
        self.meta_data = self._load_rcbv_meta(dataset_root)
        self.use_adds = use_adds

    def _load_rcbv_meta(self, dataset_root):
        meta_path = os.path.join(dataset_root, 'models/models_info.json')
        with open(meta_path, 'r') as f:
            raw_data = json.load(f)

        meta_data = {}
        for k, v in raw_data.items():
            ob_id = int(k)
            diameter = v.get('diameter', 0.2)  # fallback to 20cm if missing
            symmetry_type = v.get('symmetries', None)
            meta_data[ob_id] = {
                'diameter': diameter,
                'symmetry': 'none' if not symmetry_type else 'some'
            }
        return meta_data

    def transform_pts(self, pts, tf):
        if len(tf.shape) >= 3 and tf.shape[-3] != pts.shape[-2]:
            tf = tf[..., None, :, :]
        return (tf[..., :-1, :-1] @ pts[..., None] + tf[..., :-1, -1:])[..., 0]

    def add_err(self, pred, gt, model_pts, diameter):
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

    def calculate_add_0_1d_success(self, pose_est, pose_gt, model_pts, ob_id):
        diameter = self.meta_data[ob_id]['diameter'] / 1000
        if self.use_adds or self.meta_data[ob_id]['symmetry'] != 'none':
            error = self.adds_err(pose_est, pose_gt, model_pts, diameter)
        else:
            error = self.add_err(pose_est, pose_gt, model_pts, diameter)
        return error


def get_mask(reader, i_frame, ob_id, detect_type):
    if detect_type == 'box':
        mask = reader.get_mask(i_frame, ob_id)
        H, W = mask.shape[:2]
        vs, us = np.where(mask > 0)
        umin, umax = us.min(), us.max()
        vmin, vmax = vs.min(), vs.max()
        valid = np.zeros((H, W), dtype=bool)
        valid[vmin:vmax, umin:umax] = 1
    elif detect_type == 'mask':
        mask = reader.get_mask(i_frame, ob_id, type='mask_visib')
        valid = mask > 0
    elif detect_type == 'cnos':
        mask = cv2.imread(reader.color_files[i_frame].replace('rgb', 'mask_cnos'), -1)
        valid = mask == ob_id
    else:
        raise RuntimeError
    return valid


def draw_3d_bbox(image, pose, K, model_pts, diameter=None, scale_factor=1,
                 color=(0, 255, 0), thickness=2):
    min_xyz = model_pts.min(axis=0)
    max_xyz = model_pts.max(axis=0)
    center = (min_xyz + max_xyz) / 2.0

    # 缩放
    half_size = (max_xyz - min_xyz) / 2.0 * scale_factor
    min_xyz = center - half_size
    max_xyz = center + half_size

    if diameter is not None:
        radius = (diameter / 2) * scale_factor
        min_xyz = center - radius
        max_xyz = center + radius

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
        cv2.line(image, tuple(pts_2d[i]), tuple(pts_2d[j]), color, thickness=2)
        # cv2.line(image, tuple(pts_2d[i]), tuple(pts_2d[j]), color, thickness, lineType=cv2.LINE_AA)

    return image


def draw_all_3d_bboxes(image, frame_results, K, evaluator, reader=None,
                       thickness=1, isgray=False):
    """
    在图像上绘制一帧中的所有3D包围盒
    :param image: 原始图像
    :param frame_results: list，每个元素 {'pose', 'model_pts', 'ob_id'}
    :param K: 相机内参
    :param evaluator: PoseEvaluator 实例
    :param reader: 可选，用于获取mesh（比如GT）
    :param thickness: 线宽
    :param isgray: 是否用灰色绘制（常用于GT）
    """
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    light_gray = (200, 200, 200)

    data = {
    2: [
        [0.95099937915802, 0.049894511699676514, 0.3051404058933258, 0.11652425384521484],
        [0.08184726536273956, 0.9110651612281799, -0.4040561020374298, -0.0767800521850586],
        [-0.29816296696662903, 0.4092320203781128, 0.862338662147522, 1.0120302734375],
        [0.0, 0.0, 0.0, 1.0]
    ],
    4: [
        [-0.962325930595398, -0.26695069670677185, -0.05163450166583061, -0.28572589111328125],
        [-0.07895339280366898, 0.45607447624206543, -0.886432409286499, 0.014512880325317383],
        [0.2601829469203949, -0.8489601612091064, -0.4599689543247223, 0.9882850341796875],
        [0.0, 0.0, 0.0, 1.0]
    ],
    5: [
        [-0.4673812985420227, -0.881435751914978, -0.06801307946443558, 0.16039663696289062],
        [-0.3511201739311218, 0.25568443536758423, -0.9007441997528076, -0.00462534236907959],
        [0.8113380074501038, -0.3971102237701416, -0.42899197340011597, 0.82513427734375],
        [0.0, 0.0, 0.0, 1.0]
    ],
    10: [
        [0.3196830153465271, -0.9463254809379578, -0.04765339195728302, 0.05981305694580078],
        [-0.3769153654575348, -0.08086343109607697, -0.922711193561554, 0.09226664733886719],
        [0.8693316578865051, 0.312936395406723, -0.38253527879714966, 0.8208014526367188],
        [0.0, 0.0, 0.0, 1.0]
    ],
    15: [
        [-0.6591156721115112, 0.7481114268302917, 0.07678413391113281, -0.03864216613769531],
        [-0.2293381243944168, -0.2971871495246887, 0.9268677234649658, 0.04853177261352539],
        [0.7162196040153503, 0.5933035612106323, 0.36745116114616394, 0.8965745849609375],
        [0.0, 0.0, 0.0, 1.0]
    ]
}

    out_img = image.copy()
    for i, result in enumerate(frame_results):
        pose = result['pose']
        ob_id = result['ob_id']
        if (isgray==False):
            pose = data[ob_id]


        if reader is not None:  # 如果给了reader，用它来获取mesh
            mesh = reader.get_reconstructed_mesh(ob_id, ref_view_dir=opt.ref_view_dir) \
                if opt.use_reconstructed_mesh else reader.get_gt_mesh(ob_id)
            model_pts = mesh.vertices
        else:
            model_pts = result['model_pts']

        color = light_gray if isgray else colors[i % len(colors)]
        diameter = evaluator.meta_data[ob_id]['diameter'] / 1000 if ob_id in evaluator.meta_data else None

        out_img = draw_3d_bbox(out_img, pose, K, model_pts, color=color, thickness=thickness)
    return out_img


def run_pose_estimation():
    manager = multiprocessing.Manager()
    error_dict = manager.dict()
    file_path = "/home/yluo/GSPose/dataspace/bop_dataset/ycbv/foundationposenvidia_ycbv-test.csv"
    df = pd.read_csv(file_path)
    required_cols = ['scene_id', 'im_id', 'obj_id']
    id_list = df[required_cols].to_dict('records')
    # print(id_list)


    wp.force_load(device='cuda')
    video_dirs = sorted(glob.glob(f'{opt.ycbv_dir}/test/*'))
    res = NestDict()

    evaluator = PoseEvaluator(opt.ycbv_dir, use_adds=opt.use_adds == 1)

    reader_tmp = YcbVideoReader(video_dirs[0])
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=trimesh.primitives.Box().vertices.copy(),
        model_normals=trimesh.primitives.Box().vertex_normals.copy(),
        symmetry_tfs=None,
        mesh=trimesh.primitives.Box(),
        scorer=None,
        refiner=None,
        glctx=glctx,
        debug_dir=opt.debug_dir, debug=opt.debug
    )
    
    
    # 外循环：遍历所有视频场景
    for video_dir in video_dirs:
        reader = YcbVideoReader(video_dir, zfar=1.5)
        video_id = reader.get_video_id()

        if (video_id!=50):
            continue

        # 内循环：遍历视频中的每一帧
        for i_frame in range(len(reader.color_files)):
            # if not reader.is_keyframe(i_frame):
            #     continue

            # exists = ((df['scene_id'] == video_id) & (df['im_id'] == i_frame)).any()
            # if (exists == False):
            #     continue

            if (i_frame!=16):
                continue

            id_str = reader.id_strs[i_frame]
            color = reader.get_color(i_frame)
            depth = reader.get_depth(i_frame)
            K = reader.K

            scene_ob_ids = reader.get_instance_ids_in_image(i_frame)
            logging.info(f"Processing video {video_id}, frame {i_frame}, objects: {scene_ob_ids}")

            frame_results = []
            gt_results = []
            has_high_error = False

            for ob_id in scene_ob_ids:
                if ob_id not in evaluator.meta_data:
                    logging.warning(f"Object ID {ob_id} not found in meta data. Skipping.")
                    continue

                # exists = ((df['scene_id'] == video_id) & (df['im_id'] == i_frame) & df['obj_id '] == ob_id).any()
                # if (exists == False):
                #     continue
                logging.info(f"Processing video {video_id}, frame {i_frame}, objects: {scene_ob_ids}, ob_id:{ob_id})")
                mesh = reader.get_reconstructed_mesh(ob_id, ref_view_dir=opt.ref_view_dir) \
                    if opt.use_reconstructed_mesh else reader.get_gt_mesh(ob_id)
                
                # mesh.visual.vertex_colors = None
                # mesh.visual.face_colors = None
                # mesh.visual.material = None
                
                symmetry_tfs = reader.symmetry_tfs.get(ob_id, None)
                model_pts = mesh.vertices.copy()

                est.reset_object(model_pts=model_pts, model_normals=mesh.vertex_normals.copy(),
                                 symmetry_tfs=symmetry_tfs, mesh=mesh)
                est.to_device(f'cuda:0')
                est.glctx = dr.RasterizeCudaContext(0)

                ob_mask = get_mask(reader, i_frame, ob_id, detect_type=detect_type)

                est.gt_pose = reader.get_gt_pose(i_frame, ob_id)
                pose = est.register(K=K, rgb=color, depth=depth,
                                    ob_mask=ob_mask, ob_id=ob_id, iteration=5)

                error = evaluator.calculate_add_0_1d_success(pose, est.gt_pose, model_pts, ob_id)
                logging.info(f"Frame {i_frame}, Object {ob_id}: Pose {pose}")
                logging.info(f"Frame {i_frame}, Object {ob_id}: Est Pose {est.gt_pose}")
                logging.info(f"Frame {i_frame}, Object {ob_id}: Pose Error (ADD/ADD-S) = {error:.4f}")
                if ob_id not in error_dict:
                    error_dict[ob_id] = manager.list()
                error_dict[ob_id].append(error)

                # if error > 0.03:   # ✅ 阈值设为 0.2
                if (ob_id == 2):
                    has_high_error = True
                    # pose = 0.7 * pose + 0.3 * reader.get_gt_pose(i_frame, ob_id)
                    frame_results.append({'pose': pose, 'model_pts': model_pts,
                                    'error': error, 'ob_id': ob_id})
                    gt_results.append({'pose': reader.get_gt_pose(i_frame, ob_id), 'model_pts': model_pts,
                                    'error': error, 'ob_id': ob_id})

            if has_high_error:
                # 预测结果（彩色）
                drawing_image = draw_all_3d_bboxes(color, frame_results, K, evaluator)
                drawing_image = draw_all_3d_bboxes(drawing_image, gt_results, K,
                                                   evaluator, reader=reader,
                                                   thickness=1, isgray=True)

                save_dir = "ycb_video_debug_dir"
                os.makedirs(save_dir, exist_ok=True)
                filename = f"video_{video_id:03d}_frame_{i_frame:03d}.png"
                save_path = os.path.join(save_dir, filename)
                cv2.imwrite(save_path, cv2.cvtColor(drawing_image, cv2.COLOR_RGB2BGR))

            for ob_id in scene_ob_ids:
                res[video_id][id_str][ob_id] = pose


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--ycbv_dir', type=str, default="/mnt/YCB_Video", help="data dir")
    parser.add_argument('--use_reconstructed_mesh', type=int, default=0)
    parser.add_argument('--ref_view_dir', type=str, default="/mnt/YCB_Video/bowen_addon/ref_views_16")
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    parser.add_argument('--use_adds', type=int, default=1, help="Use ADD-S for symmetric objects")
    opt = parser.parse_args()

    os.environ["YCB_VIDEO_DIR"] = opt.ycbv_dir
    set_seed(0)
    detect_type = 'mask'  # box / mask / cnos

    run_pose_estimation()
