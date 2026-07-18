# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from Utils import *
import json,uuid,joblib,os,sys
import scipy.spatial as spatial
from multiprocessing import Pool, cpu_count
import multiprocessing
from functools import partial
from itertools import repeat
import itertools
from datareader import *
from pose_calculation import *
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/mycpp/build')
import yaml

def get_mask(reader, i_frame, ob_id, detect_type):
    # [保持原有实现不变]
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

def run_pose_estimation_worker(reader, i_frames, est:FoundationPose=None, debug=0, ob_id=None, device='cuda:0'):
    # [保持原有实现不变]
    torch.cuda.set_device(device)
    est.to_device(device)
    est.glctx = dr.RasterizeCudaContext(device=device)

    result = NestDict()

    for i, i_frame in enumerate(i_frames):
        logging.info(f"{i}/{len(i_frames)}, i_frame:{i_frame}, ob_id:{ob_id}")
        video_id = reader.get_video_id()
        color = reader.get_color(i_frame)
        depth = reader.get_depth(i_frame)
        id_str = reader.id_strs[i_frame]
        H,W = color.shape[:2]

        debug_dir = est.debug_dir

        ob_mask = get_mask(reader, i_frame, ob_id, detect_type=detect_type)
        if ob_mask is None:
            logging.info("ob_mask not found, skip")
            result[video_id][id_str][ob_id] = np.eye(4)
            return result

        est.gt_pose = reader.get_gt_pose(i_frame, ob_id)

        pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=ob_mask, ob_id=ob_id)
        logging.info(f"pose:\n{pose}")

        if debug>=3:
            m = est.mesh_ori.copy()
            tmp = m.copy()
            tmp.apply_transform(pose)
            tmp.export(f'{debug_dir}/model_tf.obj')

        result[video_id][id_str][ob_id] = pose

    return result

def process_object_group(args_group):
    """处理一组物体的入口函数"""
    # 解包参数
    (ob_ids, device, linemod_dir, use_reconstructed_mesh, 
     ref_view_dir, debug_dir, debug, detect_type) = args_group
    
    # 初始化当前设备的资源
    wp.force_load(device=device)
    torch.cuda.set_device(device)
    
    # 创建临时结果存储
    local_res = NestDict()
    
    # 初始化基准Reader
    base_reader = LinemodReader(f'{linemod_dir}/lm_test_all/test/000002', split=None)
    
    # 为每个物体创建独立的estimater实例
    mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)).to_mesh()
    glctx = dr.RasterizeCudaContext(device=device)
    
    # 需要确保scorer和refiner正确初始化（根据实际实现）
    est = FoundationPose(
        model_pts=mesh_tmp.vertices.copy(),
        model_normals=mesh_tmp.vertex_normals.copy(),
        symmetry_tfs=None,
        mesh=mesh_tmp,
        scorer=ScorePredictor(),   # 需替换实际初始化方法
        refiner=PoseRefinePredictor(), # 需替换实际初始化方法
        glctx=glctx,
        debug_dir=debug_dir,
        debug=debug
    )

    for ob_id in ob_ids:
        ob_id = int(ob_id)
        try:
            # 初始化物体特定资源
            if use_reconstructed_mesh:
                mesh = base_reader.get_reconstructed_mesh(ob_id, ref_view_dir=ref_view_dir)
            else:
                mesh = base_reader.get_gt_mesh(ob_id)
            symmetry_tfs = base_reader.symmetry_tfs[ob_id]
            
            # 初始化视频Reader
            video_dir = f'{linemod_dir}/lm_test_all/test/{ob_id:06d}'
            reader = LinemodReader(video_dir, split=None)
            
            # 配置estimater
            est.reset_object(
                model_pts=mesh.vertices.copy(),
                model_normals=mesh.vertex_normals.copy(),
                symmetry_tfs=symmetry_tfs,
                mesh=mesh
            )
            
            # 处理所有帧
            for i in range(len(reader.color_files)):
                args = (reader, [i], est, debug, ob_id, device)
                out = run_pose_estimation_worker(*args)
                
                # 合并结果
                for vid in out:
                    for frame in out[vid]:
                        for obj in out[vid][frame]:
                            local_res[vid][frame][obj] = out[vid][frame][obj]
                            
        except Exception as e:
            logging.error(f"Error processing ob_id {ob_id} on {device}: {str(e)}")
    
    return local_res

def run_pose_estimation():
    # 参数解析
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--linemod_dir', type=str, default="/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/LINEMOD", help="linemod root dir")
    parser.add_argument('--use_reconstructed_mesh', type=int, default=0)
    parser.add_argument('--ref_view_dir', type=str, default="/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/bowen_addon/ref_views_16")
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    opt = parser.parse_args()
    set_seed(0)
    detect_type = 'mask'  # mask / box / detected

    # 初始化基准Reader获取物体列表
    base_reader = LinemodReader(f'{opt.linemod_dir}/lm_test_all/test/000002', split=None)
    all_ob_ids = [int(ob_id) for ob_id in base_reader.ob_ids]
    
    # 分割物体到两个GPU
    num_gpus = 2
    split_ob_ids = np.array_split(all_ob_ids, num_gpus)
    
    # 准备进程参数
    process_args = []
    for gpu_id in range(num_gpus):
        args = (
            split_ob_ids[gpu_id].tolist(),  # 当前GPU处理的物体ID列表
            f'cuda:{gpu_id}',               # 设备ID
            opt.linemod_dir,
            opt.use_reconstructed_mesh,
            opt.ref_view_dir,
            opt.debug_dir,
            opt.debug,
            detect_type
        )
        process_args.append(args)
    
    # 创建进程池
    pool = Pool(processes=num_gpus)
    results = pool.map(process_object_group, process_args)
    
    # 合并结果
    final_res = NestDict()
    for res in results:
        for vid in res:
            for frame in res[vid]:
                for obj in res[vid][frame]:
                    final_res[vid][frame][obj] = res[vid][frame][obj]
    
    # 保存最终结果
    with open(f'{opt.debug_dir}/linemod_res.yml','w') as ff:
        yaml.safe_dump(make_yaml_dumpable(final_res), ff)

if __name__=='__main__':
    # 设置多进程启动方法
    multiprocessing.set_start_method('spawn', force=True)
    run_pose_estimation()