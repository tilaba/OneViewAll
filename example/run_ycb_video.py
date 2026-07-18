# Copyright (c) 2026 [OneViewAll].
# Licensed under the MIT License.
#
# Note: This module is designed to interface with systems licensed under 
#python run_ycb_video.py  --ycbv_dir /home/yluo/GSPose/dataspace/bop_dataset/ycbv --use_reconstructed_mesh 0\

from Utils import *
from multiprocessing import Pool
import multiprocessing
import json, uuid, joblib, os, sys, argparse
from datareader import *
from pose_calculation import *
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

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

    def compute_auc(self, errors, max_threshold=0.2, num_steps=1000):
        thresholds = np.linspace(0, max_threshold, num_steps)
        recalls = np.array([(errors < t).mean() for t in thresholds])
        auc = np.trapz(recalls, thresholds) / max_threshold
        return auc, thresholds, recalls


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


def run_pose_estimation_worker(reader, i_frames, est: OneRefPose, debug=False, ob_id=None, device: int = 0, evaluator=None, model_pts=None, error_dict=None):
    result = NestDict()
    torch.cuda.set_device(device)
    est.to_device(f'cuda:{device}')
    est.glctx = dr.RasterizeCudaContext(device)
    debug_dir = est.debug_dir

    for i in range(len(i_frames)):      
        i_frame = i_frames[i]
        # if (i_frame > 10):
        #     break

        id_str = reader.id_strs[i_frame]
        color = reader.get_color(i_frame)
        depth = reader.get_depth(i_frame)
        H, W = color.shape[:2]
        scene_ob_ids = reader.get_instance_ids_in_image(i_frame)
        video_id = reader.get_video_id()

        if ob_id not in scene_ob_ids:
            continue

        ob_mask = get_mask(reader, i_frame, ob_id, detect_type=detect_type)
        est.gt_pose = reader.get_gt_pose(i_frame, ob_id)
        pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=ob_mask, ob_id=ob_id, iteration=5)
        logging.info(f"pose:\n{pose}")

        error = evaluator.calculate_add_0_1d_success(pose, est.gt_pose, model_pts, ob_id)
        error_dict[ob_id].append(error)
        logging.info(f"error:\n{error}")

        def draw_3d_bbox(image, pose, K, model_pts, diameter=None, scale_factor=0.95,
                          color=(0, 255, 0), thickness=2, crop_size=224, crop_scale=1.2):
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
                cv2.line(image, tuple(pts_2d[i]), tuple(pts_2d[j]), color, thickness)

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
            cropped_resized = cv2.resize(cropped, (crop_size, crop_size))
            return image

        if (error < 0.05):
            # 画框
            image = draw_3d_bbox(color, pose, reader.K, model_pts)
            save_dir = "error_img"
            os.makedirs(save_dir, exist_ok=True)
            filename = f"ob_{ob_id:03d}_i_frame{i_frame:03d}.png"  # ob_005.png
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            est.to_device(device)

        if debug >= 3:
            tmp = est.mesh_ori.copy()
            tmp.apply_transform(pose)
            tmp.export(f'{debug_dir}/model_tf.obj')

        result[video_id][id_str][ob_id] = pose

    return result, error_dict


def run_pose_estimation():
    manager = multiprocessing.Manager()
    error_dict = manager.dict()

    wp.force_load(device='cuda')
    video_dirs = sorted(glob.glob(f'{opt.ycbv_dir}/test/*'))
    res = NestDict()

    debug = opt.debug
    use_reconstructed_mesh = opt.use_reconstructed_mesh
    debug_dir = opt.debug_dir

    evaluator = PoseEvaluator(opt.ycbv_dir, use_adds=opt.use_adds == 1)
    reader_tmp = YcbVideoReader(video_dirs[0])
    glctx = dr.RasterizeCudaContext()
    mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4))
    est = OneRefPose(model_pts=mesh_tmp.vertices.copy(), model_normals=mesh_tmp.vertex_normals.copy(),
                         symmetry_tfs=None, mesh=mesh_tmp, scorer=None, refiner=None, glctx=glctx,
                         debug_dir=debug_dir, debug=debug)
    ob_ids = reader_tmp.ob_ids

    for ob_id in ob_ids:
        #if ob_id != 1:  # TODO: remove this to test all objects
        #    break
        error_dict[ob_id] = manager.list()

        mesh = reader_tmp.get_reconstructed_mesh(ob_id, ref_view_dir=opt.ref_view_dir) if use_reconstructed_mesh else reader_tmp.get_gt_mesh(ob_id)


        symmetry_tfs = reader_tmp.symmetry_tfs[ob_id]
        est.reset_object(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(),
                         symmetry_tfs=symmetry_tfs, mesh=mesh)
        
        model_pts = mesh.vertices.copy()


        args = []
        for video_dir in video_dirs:
            reader = YcbVideoReader(video_dir, zfar=1.5)
            scene_ob_ids = reader.get_instance_ids_in_image(0)
            if ob_id not in scene_ob_ids:
                continue

            for i in range(len(reader.color_files)):
                if not reader.is_keyframe(i):
                    continue
                args.append((reader, [i], est, debug, ob_id, 0, evaluator, model_pts, error_dict))

        outs = []
        for arg in args:
            out, error_dict = run_pose_estimation_worker(*arg)
            outs.append(out)

        for out in outs:
            for video_id in out:
                for id_str in out[video_id]:
                    res[video_id][id_str][ob_id] = out[video_id][id_str][ob_id]

    with open(f'{opt.debug_dir}/ycbv_res.yml', 'w') as ff:
        yaml.safe_dump(make_yaml_dumpable(res), ff)

    # === AUC Evaluation ===
    print("\n===== ADD(-S) AUC EVALUATION =====")
    all_aucs = []

    for ob_id, errors in error_dict.items():
        errors_np = np.array(errors)
        auc, thresholds, recalls = evaluator.compute_auc(errors_np)
        print(f"Object {ob_id:02d} - AUC (0.1 threshold): {auc:.4f}")
        all_aucs.append(auc)

        if debug >= 1:
            plt.plot(thresholds, recalls, label=f'obj {ob_id:02d} (AUC={auc:.3f})')

    if debug >= 1:
        plt.xlabel("Threshold (normalized ADD(-S))")
        plt.ylabel("Recall")
        plt.title("ADD(-S) Recall Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{debug_dir}/add_auc_curve.png")

    mean_auc = np.mean(all_aucs)
    print(f"\nMean AUC over all objects: {mean_auc:.4f}")

    # === Recall Evaluation ===
    print("\n===== ADD(-S) RECALL @ 0.1d EVALUATION =====")
    all_recalls = []

    for ob_id, errors in error_dict.items():
        errors_np = np.array(errors)  # 归一化误差到相对直径
        thresholds = np.linspace(0, 0.1, 100)  # 0~0.1比例区间
        recalls = [np.mean(errors_np < t) for t in thresholds]
        ar_score = np.mean(recalls)
        print(f"Object {ob_id:02d} - Recall (error < 0.1 × diameter): {ar_score * 100:.2f}% "
        f"({(errors_np < 0.1).sum()}/{len(errors_np)})")

    mean_recall = np.mean(all_recalls)
    print(f"\nMean Recall over all objects: {mean_recall * 100:.2f}%")


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
