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

```
weights/
├── 2023-10-28-18-33-37/   # refiner
└── 2024-01-11-20-02-45/   # scorer
```

Create directories:

```bash
mkdir -p weights/2023-10-28-18-33-37
mkdir -p weights/2024-01-11-20-02-45
```

---

## 1.3 Demo Data

```bash
mkdir -p demo_data/
# extract demo data into demo_data/
```

---

## 1.4 LINEMOD Dataset

```bash
pip install -U "huggingface_hub[cli]"

export DATASET_NAME=lm

huggingface-cli download bop-benchmark/$DATASET_NAME \
  --local-dir ./${DATASET_NAME}/ \
  --repo-type=dataset
```

---

## 1.5 Reference Database (`ref_data_root`)

OneRefPose requires per-object reference data under
`<ref_data_root>/<ob_id>/ref_data.pt`.

Default layout for LINEMOD:

```
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
```

`ref_data_root` must be provided on the command line; the program raises
`ValueError` if it is omitted.

Optional: to make the reference data human-inspectable
(`rgb.png / xyz_map.npy / pose.txt / meta.json` next to `ref_data.pt`):

```bash
python tools/convert_ref_data.py \
  --ref_data_root reference_database/linemod_real \
  --inplace --backup
```

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
**ideal / noise-free** (no depth noise, no mask error, no occlusion) —
analogous to the rendered-reference setting used by SinRef-6D [1].

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
```

Use the rendered database when you want to isolate the pose estimation
from reference-image quality, or when reproducing the `Rendered (†)`
numbers in §5.

> [1] SinRef-6D: Single-Reference 6D Pose Estimation, 2025.
> (Rendered-reference setting.)

---

# 2. 🛠 Installation (Conda)

## 2.1 Environment

```bash
conda create -n onerefpose python=3.9 -y
conda activate onerefpose
```

---

## 2.2 Eigen

```bash
conda install -c conda-forge eigen=3.4.0 -y
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$CONDA_PREFIX"
```

---

## 2.3 Dependencies

```bash
pip install -r requirement.txt
```

---

## 2.4 Kaolin

```bash
pip install kaolin==0.15.0 \
  -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html
```

---

## 2.5 PyTorch3D

```bash
pip install --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html
```

---

# 3. 🚀 Inference

## 3.1 Environment Variable

```bash
export BOP_DIR=/path/to/lm
```

---

## 3.2 LINEMOD

`--linemod_dir` and `--ref_data_root` are both required.

```bash
python run_linemod.py \
  --linemod_dir /root/data/lm \
  --ref_data_root reference_database/linemod_real/
```

Result: per-frame ADD(-S) / rotation / translation errors are printed,
and the overall metrics (Mean / Median / ADD(-S)@0.1d / AUC@0.1) are
written to `all_errors.txt` at the end.

To evaluate the `Rendered (†)` setting, replace `--ref_data_root` with
the rendered reference database (see §1.6).

---

# 4. ⚠️ Troubleshooting

- Ensure CUDA matches PyTorch (recommended CUDA 11.8)
- RTX 4090+ → CUDA ≥ 12.1 preferred
- If build fails:
  - check Kaolin version
  - check PyTorch3D compatibility
- `OneRefPose requires ref_data_root ...`
  → you forgot `--ref_data_root`; see §1.5
- `Failed to load reference data for ob_id X`
  → `<ref_data_root>/X/ref_data.pt` is missing or malformed
- Want the CAD-rendered (noise-free) reference instead of real captures?
  → see §1.6 (`remotes/origin/render-branch`)

---

# 5. 📊 Benchmark Results on LINEMOD

We evaluate on LINEMOD using ADD-0.1% metric.

Settings:
- RGB / RGB-D inputs
- 1-shot setting
- Hypotheses number N
- Real (*)  → real-capture reference database (`main` branch, §1.5)
- Rendered (†) → CAD-rendered reference database
  (`remotes/origin/render-branch`, §1.6), analogous to SinRef-6D [1]

---

## 5.1 Key Results Summary

| Method | Year | Modality | Ref | Mean | Time |
|--------|------|----------|-----|------|------|
| OnePose* | 2022 | RGB | 200 | 63.6 | 66 ms |
| OnePose++* | 2023 | RGB | 200 | 76.9 | 88 ms |
| FS6D* | 2022 | RGB-D | 16 | 88.9 | 72 ms |
| SinRef-6D† | 2025 | RGB-D | 1 | 90.2 | - |
| Ours (N=12)* | 2026 | RGB-D | 1 | 89.9 | 80 ms |
| Ours (N=78)* | 2026 | RGB-D | 1 | 92.5 | 375 ms |
| Ours (N=12)† | 2026 | RGB-D | 1 | 91.2 | 80 ms |
| Ours (N=78)† | 2026 | RGB-D | 1 | 99.1 | 375 ms |

> [1] SinRef-6D: Single-Reference 6D Pose Estimation, 2025.

---

## 5.2 Full LaTeX Table

```latex
\definecolor{highlightblue}{RGB}{235, 235, 255}

\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{1.8pt}
\renewcommand{\arraystretch}{1.3}

\caption{LINEMOD (ADD-0.1\%) comparison.}
\label{tab:linemod}

\begin{tabular}{lcccccccccccccccccc}
\toprule
Method & Year & Mod. & Ref. & \multicolumn{13}{c}{Object ID} & Mean & Time \\
\midrule

OnePose* & 2022 & RGB & 200 & ... \\
OnePose++* & 2023 & RGB & 200 & ... \\
FS6D* & 2022 & RGB-D & 16 & ... \\
SinRef-6D\textsuperscript{†} & 2025 & RGB-D & 1 & ... \\

\midrule
\rowcolor{highlightblue}
Ours (N=12) & 2026 & RGB-D & 1 & ... \\

\rowcolor{highlightblue}
Ours (N=78) & 2026 & RGB-D & 1 & ... \\

\bottomrule
\end{tabular}
\end{table*}
```