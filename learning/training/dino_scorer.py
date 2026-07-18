# geometric_scorer.py
import torch
import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
from Utils import *

class GeometricScorer:
    def __init__(self, use_icp_refine=True, icp_max_iter=30, icp_tolerance=1e-6,
                 distance_type='point_to_point', sym_tfs=None):
        """
        基于几何距离的评分器，用模型点云与场景点云的距离衡量姿态质量。
        
        Args:
            use_icp_refine: 是否对每个假设先用ICP微调再计算距离（精度更高但更慢）
            icp_max_iter: ICP迭代次数
            icp_tolerance: ICP收敛容差
            distance_type: 'point_to_point' 或 'point_to_plane'（仅对 ICP 有效）
            sym_tfs: 对称变换矩阵列表 [N_sym, 4, 4]，用于计算对称物体的最小距离
        """
        self.use_icp_refine = use_icp_refine
        self.icp_max_iter = icp_max_iter
        self.icp_tolerance = icp_tolerance
        self.distance_type = distance_type
        self.sym_tfs = sym_tfs

    @torch.no_grad()
    def predict(self, mesh, rgb, depth, K, ob_in_cams, ob_mask,
                normal_map=None, xyz_map=None, mesh_tensors=None,
                glctx=None, mesh_diameter=None, get_vis=False, ob_id=None):
        """
        参数与 ScorePredictor.predict 完全相同。
        返回:
            scores: Tensor (N,) 分数，越大表示姿态越好
            vis: None
        """
        if isinstance(ob_in_cams, np.ndarray):
            ob_in_cams = torch.as_tensor(ob_in_cams, dtype=torch.float, device='cuda')
        elif isinstance(ob_in_cams, torch.Tensor):
            ob_in_cams = ob_in_cams.cuda().float()

        # ---------- 1. 准备场景点云 ----------
        if xyz_map is None:
            # 如果外部未提供xyz_map，就从depth和K生成
            if isinstance(depth, np.ndarray):
                depth_t = torch.as_tensor(depth, device='cuda', dtype=torch.float)
            else:
                depth_t = depth.cuda().float()
            xyz_map = depth2xyzmap(depth_t, K)

        # xyz_map 现在是 (H, W, 3) 或 (3, H, W)
        if xyz_map.shape[0] == 3 and xyz_map.ndim == 3:
            xyz_map = xyz_map.permute(1, 2, 0)  # -> (H, W, 3)

        ob_mask_t = torch.as_tensor(ob_mask, device='cuda', dtype=torch.bool)
        valid_mask = ob_mask_t & (xyz_map[..., 2] > 1e-6)
        scene_pts = xyz_map[valid_mask].cpu().numpy()  # (M, 3)

        if scene_pts.shape[0] < 10:
            # 场景点太少，返回全零分数
            return torch.zeros(len(ob_in_cams), device='cuda'), None

        # ---------- 2. 准备模型点云（居中后的） ----------
        # mesh.vertices 已经是居中后的，与 pose 对应
        model_pts = np.asarray(mesh.vertices, dtype=np.float32)  # (N_m, 3)
        if self.sym_tfs is not None and len(self.sym_tfs) > 1:
            # 如果有对称性，将模型点云扩展到对称组（简化：仅计算对称姿态的最小距离）
            pass  # 这里我们采用在计算距离时考虑多个对称姿态（见步骤3）

        # 转换到 numpy，便于使用 Open3D ICP
        ob_in_cams_np = ob_in_cams.cpu().numpy()

        # ---------- 3. 逐个姿态计算距离 ----------
        scores = []
        for i in range(len(ob_in_cams_np)):
            pose = ob_in_cams_np[i]
            # 将模型点云变换到相机坐标系
            transformed_pts = (pose[:3, :3] @ model_pts.T + pose[:3, 3:4]).T  # (N_m, 3)

            if self.use_icp_refine:
                # 用 Open3D 进行点到点/点到面的 ICP 微调
                source = o3d.geometry.PointCloud()
                source.points = o3d.utility.Vector3dVector(transformed_pts)
                target = o3d.geometry.PointCloud()
                target.points = o3d.utility.Vector3dVector(scene_pts)

                # 估计法线（如需点到面ICP）
                if self.distance_type == 'point_to_plane':
                    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

                # 执行ICP
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    source, target, max_correspondence_distance=0.02,
                    init=np.eye(4),
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=self.icp_max_iter,
                        relative_fitness=self.icp_tolerance,
                        relative_rmse=self.icp_tolerance)
                )
                # ICP 后的变换矩阵
                icp_trans = reg_p2p.transformation
                # 应用ICP变换更新模型点云
                transformed_pts = (icp_trans[:3, :3] @ transformed_pts.T + icp_trans[:3, 3:4]).T
                # 使用 ICP 的 fitness（重叠比例）和 RMSE 综合打分
                fitness = reg_p2p.fitness
                inlier_rmse = reg_p2p.inlier_rmse
                # 分数 = 内点率 - α * 均方根误差（归一化）
                score = fitness - 0.5 * inlier_rmse / mesh_diameter if mesh_diameter else fitness
            else:
                # 不使用ICP，直接计算 Chamfer 距离（双向最近点平均距离）
                tree_scene = KDTree(scene_pts)
                dist_model_to_scene, _ = tree_scene.query(transformed_pts, k=1)
                tree_model = KDTree(transformed_pts)
                dist_scene_to_model, _ = tree_model.query(scene_pts, k=1)
                chamfer_dist = np.mean(dist_model_to_scene) + np.mean(dist_scene_to_model)
                # 分数 = -chamfer_dist（越大越好）
                score = -chamfer_dist

            # 若有对称性，对每个对称姿态也计算距离，取最小距离（对应最大分数）
            if self.sym_tfs is not None and len(self.sym_tfs) > 1:
                best_sym_score = score
                for sym_tf in self.sym_tfs[1:]:  # 第一个是单位矩阵
                    sym_pose = pose @ sym_tf.cpu().numpy()
                    sym_pts = (sym_pose[:3, :3] @ model_pts.T + sym_pose[:3, 3:4]).T
                    # 简单计算单向 Chamfer 距离（因为对称）
                    dist_sym, _ = tree_scene.query(sym_pts, k=1)
                    sym_dist = np.mean(dist_sym)
                    sym_score = -sym_dist
                    if sym_score > best_sym_score:
                        best_sym_score = sym_score
                score = best_sym_score

            scores.append(score)

        scores = torch.tensor(scores, device='cuda', dtype=torch.float)
        return scores, None