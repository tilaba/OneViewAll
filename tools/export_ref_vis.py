# tools/export_ref_vis.py
# Copyright (c) 2026 [OneViewAll].
# Licensed under the MIT License.
#
# Read-only exporter for reference_database/*/ref_data.pt.
# - NEVER modifies ref_data.pt.
# - Default: writes to --out_dir (must be outside ref_data_root).
# - With --inplace: writes rgb/depth/xyz/mask/pose/meta/describe next to
#   each ref_data.pt (still never touching ref_data.pt itself).

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch

try:
    import cv2
except ImportError:
    cv2 = None

from PIL import Image


# ---------------------------------------------------------
# shape helpers
# ---------------------------------------------------------
def _to_hwc(x: torch.Tensor) -> np.ndarray:
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    elif x.ndim == 3 and x.shape[0] in (1, 3, 4):
        x = x.permute(1, 2, 0)
    return x.detach().cpu().float().numpy()


def _to_hw(x: torch.Tensor) -> np.ndarray:
    if x.ndim == 2:
        pass
    elif x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    elif x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    elif x.ndim == 3 and x.shape[0] <= 4:
        x = x[0]
    return x.detach().cpu().float().numpy()


def _jsonable(v):
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if torch.is_tensor(v):
        return v.detach().cpu().tolist()
    return v


# ---------------------------------------------------------
# per-object visualization
# ---------------------------------------------------------
def render_object(ref_data: dict, ob_id, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz_map = ref_data["xyz_map"]
    rgb     = ref_data["rgb"]
    depth   = ref_data["depth"]
    pose    = ref_data["pose"]
    meta    = ref_data.get("meta", {}) or {}

    # ---- RGB ----
    rgb_arr = _to_hwc(rgb)
    if rgb_arr.max() <= 1.01:
        rgb_arr = rgb_arr * 255.0
    rgb_arr = np.clip(rgb_arr, 0, 255).astype(np.uint8)
    if rgb_arr.shape[-1] == 1:
        rgb_arr = np.repeat(rgb_arr, 3, axis=-1)
    elif rgb_arr.shape[-1] == 4:
        rgb_arr = rgb_arr[..., :3]

    # ---- Depth（jet 伪彩色）----
    depth_arr = _to_hw(depth)
    valid_d = depth_arr > 0
    depth_vis = np.zeros((*depth_arr.shape, 3), dtype=np.uint8)
    if valid_d.any():
        dmin, dmax = float(depth_arr[valid_d].min()), float(depth_arr[valid_d].max())
        dnorm = np.clip((depth_arr - dmin) / (dmax - dmin + 1e-6), 0, 1)
        if cv2 is not None:
            color = cv2.applyColorMap((dnorm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        else:
            color = np.repeat((dnorm * 255).astype(np.uint8)[..., None], 3, axis=-1)
        depth_vis = color * valid_d[..., None]

    # ---- XYZ 伪彩色 ----
    xyz_arr = _to_hwc(xyz_map)
    if xyz_arr.shape[-1] >= 3:
        valid_xyz = xyz_arr[..., 2] > 1e-3
        pts = xyz_arr[valid_xyz][..., :3]
    else:
        valid_xyz = xyz_arr[..., 0] > 1e-3
        pts = xyz_arr[valid_xyz][..., :1]
    xyz_vis = np.zeros((*valid_xyz.shape, 3), dtype=np.uint8)
    if pts.size > 0:
        pmin = pts.min(axis=0)
        pmax = pts.max(axis=0)
        pnorm = (pts - pmin) / (pmax - pmin + 1e-6)
        xyz_vis[valid_xyz] = (pnorm * 255).astype(np.uint8)

    # ---- Mask ----
    mask_vis = valid_d.astype(np.uint8) * 255

    Image.fromarray(rgb_arr).save(out_dir / "rgb.png")
    Image.fromarray(depth_vis).save(out_dir / "depth.png")
    Image.fromarray(xyz_vis).save(out_dir / "xyz.png")
    Image.fromarray(mask_vis).save(out_dir / "mask.png")

    with open(out_dir / "pose.txt", "w") as f:
        f.write(str(pose.detach().cpu().numpy()))

    with open(out_dir / "meta.json", "w") as f:
        json.dump(_jsonable(meta), f, indent=2)

    with open(out_dir / "describe.txt", "w") as f:
        f.write(f"[RefData] ob_id={ob_id}\n")
        f.write(f"  xyz_map: shape={tuple(xyz_map.shape)}, dtype={xyz_map.dtype}\n")
        f.write(f"  rgb    : shape={tuple(rgb.shape)},    dtype={rgb.dtype}\n")
        f.write(f"  depth  : shape={tuple(depth.shape)},  dtype={depth.dtype}\n")
        f.write(f"  pose   : shape={tuple(pose.shape)}\n")
        f.write(f"  meta   : {meta}\n")

    return rgb_arr, depth_vis, xyz_vis, mask_vis


# ---------------------------------------------------------
# main
# ---------------------------------------------------------
def export_all(ref_data_root: str, out_dir: str, force: bool = False, inplace: bool = False):
    root = Path(ref_data_root).resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"ref_data_root not found: {root}")

    # 非 inplace 模式：--out_dir 必须在 ref_data_root 外部
    if not inplace:
        out_root = Path(out_dir).resolve()
        if out_root == root or root in out_root.parents:
            raise ValueError(
                f"Refusing to write inside ref_data_root ({root}). "
                f"Use --inplace to write next to each ref_data.pt, "
                f"or choose --out_dir outside of it (e.g. {root.parent / 'ref_vis'})."
            )
    else:
        out_root = root  # inplace 模式忽略 --out_dir

    entries = sorted(
        [p for p in root.iterdir() if p.is_dir()],
        key=lambda p: (int(p.name) if p.name.isdigit() else 10**9, p.name),
    )

    ok = skip = fail = 0
    for d in entries:
        pt = d / "ref_data.pt"
        if not pt.is_file():
            print(f"[skip] {d.name}: no ref_data.pt")
            skip += 1
            continue

        target = d if inplace else (out_root / d.name)

        if (not force) and (target / "rgb.png").exists():
            print(f"[skip] {d.name}: already exported ({target})")
            skip += 1
            continue

        try:
            # 只读加载，绝不写回 ref_data.pt
            data = torch.load(str(pt), map_location="cpu")
            ob_id = int(d.name) if d.name.isdigit() else None
            render_object(data, ob_id, target)
            print(f"[ok]   {d.name} -> {target}")
            ok += 1
        except Exception as e:
            print(f"[fail] {d.name}: {e}")
            fail += 1

    print(f"\nDone. ok={ok}, skip={skip}, fail={fail}")
    print(f"ref_data_root : {root}")
    print(f"mode          : {'inplace' if inplace else 'out_dir'}")
    print(f"out_root      : {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read-only visualization exporter for ref_data.pt. "
                    "Never modifies ref_data.pt."
    )
    parser.add_argument(
        "--ref_data_root",
        type=str,
        default="reference_database/linemod_real",
        help="Root dir containing <ob_id>/ref_data.pt (read-only).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="ref_vis",
        help="Where to write visualizations when not using --inplace. "
             "Must be outside ref_data_root.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Write visualizations next to each ref_data.pt (inside "
             "<ref_data_root>/<ob_id>/). ref_data.pt itself is never modified.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing visualizations.",
    )
    args = parser.parse_args()

    export_all(args.ref_data_root, args.out_dir, force=args.force, inplace=args.inplace)