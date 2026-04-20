================================================================
OneRefPose: Conda Installation and Inference Guide
================================================================

1. DATA & WEIGHTS PREPARATION
--------------------------------------------------
# Step 1: Create directories for weights
mkdir -p weights/2023-10-28-18-33-37
mkdir -p weights/2024-01-11-20-02-45

# Step 2: Download and place weights
# Place refiner weights in: weights/2023-10-28-18-33-37/
# Place scorer weights in: weights/2024-01-11-20-02-45/

# Step 3: Prepare demo data
mkdir -p demo_data/
# Extract demo data into: demo_data/


2. INSTALLATION (CONDA)
--------------------------------------------------
# Create and activate environment
conda create -n onerefpose python=3.9 -y
conda activate onerefpose

# Install Eigen3 (Required)
conda install conda-forge::eigen=3.4.0 -y
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$CONDA_PREFIX"

# Install core dependencies
python -m pip install -r requirements.txt

# Install Kaolin
python -m pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html

# Install PyTorch3D
python -m pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

# Build C++/CUDA extensions
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh


3. INFERENCE (MODEL-BASED)
--------------------------------------------------
# Run standard demo
python run_demo.py

# Inference on LINEMOD
python run_linemod.py --linemod_dir /path/to/LINEMOD --use_reconstructed_mesh 0

# Inference on YCB-Video
python run_ycb_video.py --ycbv_dir /path/to/YCB_Video --use_reconstructed_mesh 0


4. TROUBLESHOOTING
--------------------------------------------------
- Ensure your CUDA toolkit version matches your PyTorch version (e.g., CUDA 11.8).
- If using RTX 4090+, ensure you are using compatible library versions (CUDA 12.1+ recommended).
================================================================