# OneRefPose – Installation & Inference Guide

---

# 1. 📦 Data & Pretrained Weights

## 1.1 Download Weights

Download pretrained models:
https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i

Required checkpoints:
- Refiner: 2023-10-28-18-33-37  
- Scorer: 2024-01-11-20-02-45  

---

## 1.2 Directory Structure

weights/
├── 2023-10-28-18-33-37/   # refiner
└── 2024-01-11-20-02-45/   # scorer

---

Create directories:

mkdir -p weights/2023-10-28-18-33-37
mkdir -p weights/2024-01-11-20-02-45

---

## 1.3 Demo Data

mkdir -p demo_data/
# extract demo data into demo_data/

---

## 1.4 LINEMOD Dataset

pip install -U "huggingface_hub[cli]"

export DATASET_NAME=lm

huggingface-cli download bop-benchmark/$DATASET_NAME \
  --local-dir ./${DATASET_NAME}/ \
  --repo-type=dataset

---

## 1.5 Reference Database (`ref_data_root`)

OneRefPose requires per-object reference data under
`<ref_data_root>/<ob_id>/ref_data.pt`.

Default layout for LINEMOD:

reference_database/linemod_real/
├── 1/ref_data.pt
├── 2/ref_data.pt
├── 4/ref_data.pt
├── 5/ref_data.pt
├── 6/ref_data.pt
├── 8/ref_data.pt
├── 9/ref_data.pt
├── 10/ref_data.pt
├── 11/ref_data.pt
├── 12/ref_data.pt
├── 13/ref_data.pt
├── 14/ref_data.pt
└── 15/ref_data.pt

`ref_data_root` must be provided on the command line; the program raises
`ValueError` if it is omitted.

Optional: to make the reference data human-inspectable
(`rgb.png / xyz_map.npy / pose.txt / meta.json` next to `ref_data.pt`):

python tools/convert_ref_data.py \
  --ref_data_root reference_database/linemod_real \
  --inplace --backup

---

## 1.6 Reference Data Sources: Real vs. Rendered

Two kinds of reference databases are supported, corresponding to the
two settings reported in §5 (`Real (*)` and `Rendered (†)`):

### Real captures (`main` branch)

`reference_database/linemod_real/` is built from **real RGB-D captures**
of the target object. It contains sensor noise, imperfect masks and
partial views, which reflects the practical 1-shot setting.

### CAD-rendered (`remotes/origin/render-branch`)

The branch `remotes/origin/render-branch` provides a reference database
built from **CAD-rendered images**. Each `ref_data.pt` is produced by
rendering the object mesh under a canonical view, so the reference is
**ideal / noise-free** (no depth noise, no mask error, no
occlusion) — analogous to the rendered-reference setting used by
SinRef-6D [1].

To switch to the rendered reference database:

```bash
# Fetch the branch (read-only, no checkout needed)
git fetch origin render-branch

# Option A: checkout the branch and use its reference database
git checkout render-branch
python run_linemod.py \
  --linemod_dir /root/data/lm \
  --ref_data_root reference_database/linemod_render

# Option B: stay on your current branch and point to the render DB
# (after copying/extracting it from the branch)
python run_linemod.py \
  --linemod_dir /root/data/lm \
  --ref_data_root /path/to/linemod_render