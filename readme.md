OneRefPose – Installation & Inference Guide
===========================================

1. Data & Weights Preparation
-----------------------------

Download all network weights:
https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i

Required weights:
- Refiner: 2023-10-28-18-33-37
- Scorer: 2024-01-11-20-02-45

Directory structure:
weights/
├── 2023-10-28-18-33-37/   (refiner)
└── 2024-01-11-20-02-45/   (scorer)

Prepare directories:
mkdir -p weights/2023-10-28-18-33-37
mkdir -p weights/2024-01-11-20-02-45

Prepare demo data:
mkdir -p demo_data/
# extract demo data into demo_data/

Download LINEMOD dataset (via HuggingFace):
pip install -U "huggingface_hub[cli]"

export DATASET_NAME=lm
huggingface-cli download bop-benchmark/$DATASET_NAME \
  --local-dir ./${DATASET_NAME}/ \
  --repo-type=dataset


2. Installation (Conda)
-----------------------

conda create -n onerefpose python=3.9 -y
conda activate onerefpose

# Install Eigen3
conda install -c conda-forge eigen=3.4.0 -y
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$CONDA_PREFIX"

# Install dependencies
pip install -r requirements.txt

# Install Kaolin
pip install kaolin==0.15.0 \
  -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html

# Install PyTorch3D
pip install --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

# Build extensions
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 \
bash build_all_conda.sh


3. Inference
------------
 export BOP_DIR=/path/to/lm

# LINEMOD
python run_linemod.py \
  --linemod_dir /path/to/lm \
  --use_reconstructed_mesh 0

# YCB-Video
python run_ycb_video.py \
  --ycbv_dir /path/to/YCB_Video \
  --use_reconstructed_mesh 0


4. Troubleshooting
------------------

- Ensure CUDA matches PyTorch (e.g. CUDA 11.8)
- RTX 4090+ users: prefer CUDA >= 12.1
- Check Kaolin / PyTorch3D compatibility if build fails