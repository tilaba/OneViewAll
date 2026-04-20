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

# 2. 🛠 Installation (Conda)

## 2.1 Environment

conda create -n onerefpose python=3.9 -y
conda activate onerefpose

---

## 2.2 Eigen

conda install -c conda-forge eigen=3.4.0 -y
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$CONDA_PREFIX"

---

## 2.3 Dependencies

pip install -r requirements.txt

---

## 2.4 Kaolin

pip install kaolin==0.15.0 \
  -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html

---

## 2.5 PyTorch3D

pip install --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

---

# 3. 🚀 Inference

## 3.1 Environment Variable

export BOP_DIR=/path/to/lm

---

## 3.2 LINEMOD

python run_linemod.py \
  --linemod_dir /path/to/lm \
  --use_reconstructed_mesh 0

---

## 3.3 YCB-Video

python run_ycb_video.py \
  --ycbv_dir /path/to/YCB_Video \
  --use_reconstructed_mesh 0

---

# 4. ⚠️ Troubleshooting

- Ensure CUDA matches PyTorch (recommended CUDA 11.8)
- RTX 4090+ → CUDA ≥ 12.1 preferred
- If build fails:
  - check Kaolin version
  - check PyTorch3D compatibility

---

# 5. 📊 Benchmark Results on LINEMOD

We evaluate on LINEMOD using ADD-0.1% metric.

Settings:
- RGB / RGB-D inputs
- 1-shot setting
- Hypotheses number N
- Real (*), Rendered (†)

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

---

## 5.2 Full LaTeX Table

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
SinRef-6D† & 2025 & RGB-D & 1 & ... \\

\midrule
\rowcolor{highlightblue}
Ours (N=12) & 2026 & RGB-D & 1 & ... \\

\rowcolor{highlightblue}
Ours (N=78) & 2026 & RGB-D & 1 & ... \\

\bottomrule
\end{tabular}
\end{table*}