import pickle
import numpy as np
import os
import re
from pathlib import Path

class Real275BatchDebugger:
    def __init__(self, dataset_root):
        """
        Initialize the debugger with the dataset root path.
        dataset_root should contain 'gts/real_test' and 'real_test/scene_x'.
        """
        self.dataset_root = Path(dataset_root)
        self.gts_path = self.dataset_root / "gts" / "real_test"
        self.model_dir = self.dataset_root / "obj_models"
        
        # Standard REAL275 Camera Intrinsics (Fixed for this dataset)
        self.K = np.array([[591.0125, 0, 322.525], 
                           [0, 590.16775, 244.11084], 
                           [0, 0, 1]], dtype=np.float32)

    def _load_meta_info(self, scene_id, frame_id_str):
        """Load object categories from the corresponding _meta.txt file."""
        scene_dir = self.dataset_root / "real_test" / scene_id
        meta_path = scene_dir / f"{frame_id_str}_meta.txt"
        meta_data = []
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                for line in f:
                    p = line.strip().split()
                    # Format: inst_id, class_id, class_name
                    if len(p) >= 3: 
                        meta_data.append({'id': p[1], 'name': p[2]})
        return meta_data

    def run_batch_analysis(self):
        """Iterate through all .pkl files in the gts directory."""
        # Find all pkl files and sort them to keep the output organized
        pkl_files = sorted(list(self.gts_path.glob("results_real_test_scene_*.pkl")))
        
        if not pkl_files:
            print(f"Error: No .pkl files found in {self.gts_path}")
            return

        print(f"Found {len(pkl_files)} pkl files. Starting analysis...\n")

        for pkl_path in pkl_files:
            filename = pkl_path.stem
            
            # Extract scene index and frame index using Regex
            scene_match = re.search(r'scene_(\d+)', filename)
            frame_match = re.search(r'(\d+)$', filename)
            
            if not scene_match or not frame_match:
                continue
                
            scene_id = f"scene_{scene_match.group(1)}"
            frame_id_str = frame_match.group(1)

            # Load the pkl content
            with open(pkl_path, 'rb') as f:
                try:
                    data = pickle.load(f)
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")
                    continue

            # Extract Poses (RT matrices)
            gt_rt = np.array(data.get('gt_RTs', []))
            if gt_rt.size == 0: continue
            
            # Ensure 3D shape (N, 4, 4) even if only one object exists
            if gt_rt.ndim == 2: gt_rt = gt_rt[np.newaxis, ...]

            # Load metadata for this specific frame
            meta_data = self._load_meta_info(scene_id, frame_id_str)

            print(f"{'#'*20} {scene_id} | Frame: {frame_id_str} {'#'*20}")
            print(f"{'Inst':<5} | {'Category':<10} | {'Dist(m)':<10} | {'Translation (t)'}")
            print(f"{'-'*70}")

            for i, pose in enumerate(gt_rt):
                # Skip padding or invalid poses (all zeros)
                if np.all(pose == 0): continue
                
                cat_name = meta_data[i]['name'] if i < len(meta_data) else "Unknown"
                t = pose[:3, 3]
                distance = np.linalg.norm(t)
                
                print(f"{i:<5} | {cat_name:<10} | {distance:<10.3f} | {t}")
            
            print(f"{'#'*70}\n")

if __name__ == "__main__":
    # Path configuration
    # Ensure this directory contains 'gts/real_test' and 'real_test/'
    dataset_root_dir = "/home/yluo/GSPose/dataspace/data/nocs"
    
    debugger = Real275BatchDebugger(dataset_root_dir)
    debugger.run_batch_analysis()