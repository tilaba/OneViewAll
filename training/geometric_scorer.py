import os
import torch
import numpy as np

class GeometricScorer:
    def __init__(self, distance_type='point_to_point', sym_tfs=None,
                 ref_data_dir="reference_database/linemod_real"):
        """
        已移除所有 ICP 相关的参数设置。
        """
        self.distance_type = distance_type
        self.sym_tfs = sym_tfs.cuda().float() if sym_tfs is not None else None
        self.ref_data_dir = ref_data_dir
        self._ref_cache = {}

    def _load_ref_data(self, ob_id):
        if ob_id in self._ref_cache:
            return self._ref_cache[ob_id]

        ref_path = os.path.join(self.ref_data_dir, str(ob_id), "ref_data.pt")
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"ref_data not found at {ref_path}")
        data = torch.load(ref_path)

        # ------ 处理 pose ------
        pose = data["pose"]
        if isinstance(pose, np.ndarray):
            pose = torch.from_numpy(pose)
        pose = pose.float().cuda()
        if pose.dim() == 3 and pose.shape[0] == 1:
            pose = pose[0]
        assert pose.shape == (4, 4), f"Unexpected pose shape: {pose.shape}"

        # ------ 处理 xyz_map ------
        xyz_map = data["xyz_map"]
        if isinstance(xyz_map, np.ndarray):
            xyz_map = torch.from_numpy(xyz_map)
        xyz_map = xyz_map.float().cuda()
        if xyz_map.dim() == 4:
            xyz_map = xyz_map[0]
        if xyz_map.shape[0] == 3 and xyz_map.dim() == 3:
            xyz_map = xyz_map.permute(1, 2, 0)
        elif xyz_map.shape[-1] != 3:
            raise ValueError(f"Unexpected xyz_map shape: {xyz_map.shape}")
        H, W, _ = xyz_map.shape

        pts_cam = xyz_map.reshape(-1, 3)
        valid = (pts_cam.norm(dim=-1) > 1e-6) & torch.isfinite(pts_cam).all(dim=-1)
        ref_pts_cam = pts_cam[valid]

        meta = data.get("meta", {})
        info = {
            'ref_pts_cam': ref_pts_cam,
            'ref_pose': pose,
            'meta': meta,
            'H': H, 'W': W,
        }
        self._ref_cache[ob_id] = info
        return info

    def _get_sym_tfs_for_ob(self, ob_id, meta):
        # 1. 如果外部直接传入了对称矩阵，优先使用
        if self.sym_tfs is not None:
            return self.sym_tfs
            
        # 2. 如果 meta 数据集(如 BOP 格式)直接提供了对称矩阵列表
        if "sym_tfs" in meta:
            tfs = torch.as_tensor(meta["sym_tfs"], dtype=torch.float32, device='cuda')
            return tfs if tfs.dim() == 3 else tfs.unsqueeze(0)

        is_symmetric = meta.get("is_symmetric", False)
        if not is_symmetric:
            return None

        # 获取对称轴（通常LINEMOD中，Z轴是2，Y轴是1）
        symmetry_axis = meta.get("symmetry_axis", 2) 
        is_continuous = meta.get("is_continuous", False) # 区分 eggbox 和 bowl

        sym_tfs_list = []

        if is_continuous:
            # 【连续对称】如 Bowl (碗)
            # 绕对称轴每隔 30 度采样一个位姿 (可以根据显存调整步长)
            steps = 12 
            angles = torch.linspace(0, 2 * np.pi, steps + 1, device='cuda')[:-1]
            
            for angle in angles:
                sym_mat = torch.eye(4, device='cuda')
                c, s = torch.cos(angle), torch.sin(angle)
                if symmetry_axis == 0:   # 绕X轴
                    sym_mat[1, 1], sym_mat[1, 2] = c, -s
                    sym_mat[2, 1], sym_mat[2, 2] = s, c
                elif symmetry_axis == 1: # 绕Y轴
                    sym_mat[0, 0], sym_mat[0, 2] = c, s
                    sym_mat[2, 0], sym_mat[2, 2] = -s, c
                elif symmetry_axis == 2: # 绕Z轴
                    sym_mat[0, 0], sym_mat[0, 1] = c, -s
                    sym_mat[1, 0], sym_mat[1, 1] = s, c
                sym_tfs_list.append(sym_mat)
        else:
            # 【离散对称】如 Eggbox (蛋盒) 
            # 绕对称轴旋转 180 度。数学上等于对称轴保持不变，另外两个轴取反。
            sym_mat = torch.eye(4, device='cuda')
            for i in range(3):
                if i != symmetry_axis:
                    sym_mat[i, i] = -1.0 # 另外两个轴取反，行列式为 1 (-1 * -1 = 1)
            sym_tfs_list.append(sym_mat)

        return torch.stack(sym_tfs_list)

    # ---------- 优化工具函数 ----------
    def _sample_points(self, pts, max_points=2000):
        """随机降采样，平衡点密度并防止显存爆炸"""
        if pts.shape[0] > max_points:
            idx = torch.randperm(pts.shape[0], device=pts.device)[:max_points]
            return pts[idx]
        return pts

    def _robust_mean(self, dist, inlier_ratio=0.85):
        """鲁棒均值：剔除距离最大的点（通常是背景噪点）"""
        k = max(int(dist.size(0) * inlier_ratio), 1)
        return torch.topk(dist, k, largest=False).values.mean()

    def _knn_dist(self, src, dst, chunk_size=10000):
        """分块计算倒角距离"""
        M = src.shape[0]
        min_dists = torch.empty(M, device=src.device, dtype=src.dtype)
        for i in range(0, M, chunk_size):
            end = min(i + chunk_size, M)
            dist_mat = torch.cdist(src[i:end], dst)
            min_dists[i:end] = dist_mat.min(dim=1)[0]
        return min_dists

    @torch.no_grad()
    def predict(self, rgb, depth, K, ob_in_cams, ob_mask,
                normal_map=None, xyz_map=None, mesh_tensors=None,
                glctx=None, mesh_diameter=None, get_vis=False, ob_id=None):
        
        # ---------- 输入转换 ----------
        ob_in_cams = torch.as_tensor(ob_in_cams, device='cuda', dtype=torch.float)
        depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)
        ob_mask = torch.as_tensor(ob_mask, device='cuda', dtype=torch.bool)
        K_tensor = torch.as_tensor(K, device='cuda', dtype=torch.float)
        if K_tensor.ndim == 2:
            K_tensor = K_tensor.unsqueeze(0).expand(len(ob_in_cams), -1, -1)

        if xyz_map is not None:
            xyz_map = torch.as_tensor(xyz_map, device='cuda', dtype=torch.float)
        if xyz_map.shape[0] == 3 and xyz_map.ndim == 3:
            xyz_map = xyz_map.permute(1, 2, 0)

        # ---------- 场景点云处理 ----------
        valid_mask = ob_mask & (xyz_map[..., 2] > 1e-6)
        scene_pts = xyz_map[valid_mask]
        
        if scene_pts.shape[0] < 10:
            return torch.zeros(len(ob_in_cams), device='cuda'), None

        # 对场景点云进行降采样
        scene_pts = self._sample_points(scene_pts, max_points=2000)

        # ---------- 加载参考模型 ----------
        if ob_id is None:
            raise ValueError("ob_id must be provided")
            
        ref_info = self._load_ref_data(ob_id)
        # 对参考模型点云也进行降采样
        ref_pts_cam = self._sample_points(ref_info['ref_pts_cam'], max_points=2000)
        ref_pose = ref_info['ref_pose']
        meta = ref_info['meta']
        sym_tfs = self._get_sym_tfs_for_ob(ob_id, meta)

        H_img, W_img = xyz_map.shape[:2]
        scores = torch.zeros(len(ob_in_cams), device='cuda')

        # ---------- 假设位姿打分 ----------
        for i, pose_i in enumerate(ob_in_cams):
            T_rel = pose_i @ torch.inverse(ref_pose)
            pts_cam_i = (T_rel[:3, :3] @ ref_pts_cam.T + T_rel[:3, 3:4]).T

            fx, fy = K_tensor[i, 0, 0], K_tensor[i, 1, 1]
            cx, cy = K_tensor[i, 0, 2], K_tensor[i, 1, 2]
            z = pts_cam_i[:, 2]
            u = (fx * pts_cam_i[:, 0] / z.clamp(min=1e-8) + cx).round().long()
            v = (fy * pts_cam_i[:, 1] / z.clamp(min=1e-8) + cy).round().long()
            valid = (z > 1e-6) & (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img)
            
            if not valid.any():
                scores[i] = -float('inf')
                continue
            pred_pts = pts_cam_i[valid]

            best_score = -float('inf')
            
            # 将主视角和对称视角（若有）统一放入列表，简化逻辑
            hypotheses_pts = [pred_pts]
            if sym_tfs is not None:
                for sym_tf in sym_tfs:
                    sym_pose = pose_i @ sym_tf
                    T_sym_rel = sym_pose @ torch.inverse(ref_pose)
                    pts_sym = (T_sym_rel[:3, :3] @ ref_pts_cam.T + T_sym_rel[:3, 3:4]).T
                    
                    z_sym = pts_sym[:, 2]
                    u_sym = (fx * pts_sym[:, 0] / z_sym.clamp(min=1e-8) + cx).round().long()
                    v_sym = (fy * pts_sym[:, 1] / z_sym.clamp(min=1e-8) + cy).round().long()
                    valid_sym = (z_sym > 1e-6) & (u_sym >= 0) & (u_sym < W_img) & (v_sym >= 0) & (v_sym < H_img)
                    
                    if valid_sym.any():
                        hypotheses_pts.append(pts_sym[valid_sym])

            # 遍历所有可能的位姿，纯使用 KNN 进行打分
            for pts_to_eval in hypotheses_pts:
                dist_m2s = self._knn_dist(pts_to_eval, scene_pts)
                dist_s2m = self._knn_dist(scene_pts, pts_to_eval)
                
                # 使用截断均值，剔除 15% 的噪点 (inlier_ratio=0.85)
                mean_m2s = self._robust_mean(dist_m2s, inlier_ratio=0.85)
                mean_s2m = self._robust_mean(dist_s2m, inlier_ratio=0.85)
                
                cur_score = -(mean_m2s + mean_s2m).item()
                    
                if cur_score > best_score:
                    best_score = cur_score

            scores[i] = best_score

        return scores, None