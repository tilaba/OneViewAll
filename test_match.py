# Copyright (c) 2026 [OneViewAll].
# Licensed under the MIT License.
# 
# Tool: Find the original Linemod frame that matches the cropped reference image.

import os
import glob
import argparse
import cv2
import numpy as np
import json
import torch
import csv
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from datareader import LinemodReader
from training.pose_score import capture_ref_data_from_frame


def match_frame(ref_img, target_img, detector, matcher, min_inliers=4):
    """
    Match features using SIFT/ORB and calculate the number of inliers.
    """
    kp1, des1 = detector.detectAndCompute(ref_img, None)
    kp2, des2 = detector.detectAndCompute(target_img, None)
    
    if des1 is None or des2 is None:
        return 0, 999
    if len(kp1) < 4 or len(kp2) < 4:
        return 0, 999
        
    try:
        matches = matcher.knnMatch(des1, des2, k=2)
    except Exception:
        return 0, 999

    # Lowe's Ratio Test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    if len(good_matches) < min_inliers:
        return 0, 999
        
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Geometric validation using RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is None:
        return 0, 999
        
    inliers = int(np.sum(mask))
    mean_dist = float(np.mean([m.distance for i, m in enumerate(good_matches) if mask[i]]))
    return inliers, mean_dist


def main():
    parser = argparse.ArgumentParser(description="Find original frames from cropped reference images.")
    parser.add_argument("--linemod_dir", type=str, required=True, help="Root of Linemod dataset")
    parser.add_argument("--ref_root", type=str, default="reference_database/linemod_real", help="Root of reference database")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"], help="Which split to search")
    parser.add_argument("--method", type=str, default="sift", choices=["sift", "orb"], help="Feature extractor")
    parser.add_argument("--crop_ratio", type=float, default=1.2, help="Crop ratio used to generate the reference image")
    parser.add_argument("--max_search_frames", type=int, default=0, help="Max frames to search per object (0 = all)")
    parser.add_argument("--out_dir", type=str, default="matched_frames", help="Directory to save matched Top-1 images and CSV")
    args = parser.parse_args()

    ref_root = Path(args.ref_root)
    ob_dirs = sorted([d for d in ref_root.iterdir() if d.is_dir() and d.name.isdigit()], key=lambda x: int(x.name))
    
    if not ob_dirs:
        print("No object directories found.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    if args.method == "sift":
        detector = cv2.SIFT_create()
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    else:
        detector = cv2.ORB_create(nfeatures=2000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    print(f"Using {args.method.upper()} feature matcher (Top 3)\n")
    print(f"{'Object':<6} | {'Meta':<6} | {'Top 1 (Inliers)':<16} | {'Top 2 (Inliers)':<16} | {'Top 3 (Inliers)':<16} | {'Status'}")
    print("-" * 95)

    all_rows = []

    for d in ob_dirs:
        ob_id = int(d.name)
        ref_img_path = d / "rgb.png"
        meta_path = d / "meta.json"
        
        if not ref_img_path.is_file():
            print(f"{ob_id:<6} | {'-':<6} | {'-':<16} | {'-':<16} | {'-':<16} | No rgb.png")
            continue
            
        ref_img = cv2.imread(str(ref_img_path), cv2.IMREAD_GRAYSCALE)
        if ref_img is None:
            print(f"{ob_id:<6} | {'-':<6} | {'-':<16} | {'-':<16} | {'-':<16} | Failed to read rgb.png")
            continue

        meta_frame = "-"
        mesh_diameter = None
        if meta_path.is_file():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    meta_frame = str(meta.get("source_frame", "-"))
            except Exception:
                pass

        ob_id_padded = f"{ob_id:06d}"
        video_dir = Path(args.linemod_dir) / args.split / ob_id_padded
        if not video_dir.is_dir():
            print(f"{ob_id:<6} | {meta_frame:<6} | {'-':<16} | {'-':<16} | {'-':<16} | Dir not found")
            continue

        try:
            reader = LinemodReader(str(video_dir))
            if hasattr(reader, "get_gt_mesh") and mesh_diameter is None:
                mesh = reader.get_gt_mesh(ob_id)
                if mesh is not None:
                    verts = mesh.vertices
                    mesh_diameter = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
        except Exception as e:
            print(f"{ob_id:<6} | {meta_frame:<6} | {'-':<16} | {'-':<16} | {'-':<16} | Read Error: {e}")
            continue

        img_files = sorted(glob.glob(str(video_dir / "rgb" / "*.png"))) + sorted(glob.glob(str(video_dir / "rgb" / "*.jpg")))
        if not img_files:
            print(f"{ob_id:<6} | {meta_frame:<6} | {'-':<16} | {'-':<16} | {'-':<16} | No images")
            continue

        if args.max_search_frames > 0:
            img_files = img_files[:args.max_search_frames]

        # Store (inliers, dist, frame_idx, original_img_path, cropped_rgb)
        results = []  

        pbar = tqdm(img_files, desc=f"Obj {ob_id:02d}", leave=False)
        for img_path in pbar:
            frame_idx = int(Path(img_path).stem)
            pbar.set_postfix({"frame": f"{frame_idx}"})
            
            try:
                ref_data = capture_ref_data_from_frame(
                    reader, frame_idx, ob_id,
                    crop_ratio=args.crop_ratio,
                    device="cuda", 
                    render_size=(160, 160),
                    mesh_diameter=mesh_diameter,
                    crop_to_ref=True
                )
                cropped_rgb = ref_data.rgb.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                cropped_gray = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2GRAY)
            except Exception:
                continue
            
            inliers, dist = match_frame(ref_img, cropped_gray, detector, matcher)
            
            if inliers > 0:
                results.append((inliers, dist, frame_idx, img_path, cropped_rgb))

        # Sort by inliers descending, then by distance ascending
        results.sort(key=lambda x: (-x[0], x[1]))
        
        top_k = min(3, len(results))
        top_strs = []
        for i in range(top_k):
            inl, dist, f_idx, _, _ = results[i]
            top_strs.append(f"{f_idx:04d} ({inl})")
        while len(top_strs) < 3:
            top_strs.append("-")
            
        status = "No match"
        if top_k > 0:
            best_inl, best_dist, best_frame, best_img_path, best_cropped_rgb = results[0]
            
            # Verify if Top1 matches with the meta.json frame index
            if meta_frame != "-" and (str(best_frame).zfill(4) == meta_frame or str(best_frame) == meta_frame):
                status = "Match!"
            elif meta_frame != "-":
                status = "Mismatch!"
            else:
                status = "Top1 found"
                
            # Save Top-1 images (original + cropped) into subdirectory (e.g., matched_frames/1/0142.png)
            ob_out_dir = os.path.join(args.out_dir, str(ob_id))
            os.makedirs(ob_out_dir, exist_ok=True)
            
            orig_img = cv2.imread(best_img_path)
            if orig_img is not None:
                cv2.imwrite(os.path.join(ob_out_dir, f"{best_frame:04d}.png"), orig_img)
                
            crop_bgr = cv2.cvtColor(best_cropped_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(ob_out_dir, f"{best_frame:04d}_crop.png"), crop_bgr)
                
        print(f"{ob_id:<6} | {meta_frame:<6} | {top_strs[0]:<16} | {top_strs[1]:<16} | {top_strs[2]:<16} | {status}")
        all_rows.append([ob_id, meta_frame, top_strs[0], top_strs[1], top_strs[2], status])

    print("\n")
    
    # Save results table to CSV
    csv_path = os.path.join(args.out_dir, "match_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Object", "Meta", "Top1", "Top2", "Top3", "Status"])
        writer.writerows(all_rows)
    print(f"All results saved to {csv_path}")


if __name__ == "__main__":
    main()