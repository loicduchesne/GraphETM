#!/bin/bash

set -e

# 1. Install Miniconda
echo "Downloading Miniconda installer..."
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

echo "Installing Miniconda silently..."
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Initialize conda for bash
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
$HOME/miniconda3/bin/conda init

# Activate conda in this shell
. ~/.bashrc

# 2. Install required packages in base environment
echo "Installing PyTorch with CUDA 12.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "Installing other Python packages..."
pip install scikit-learn pandas plotly wandb notebook optuna ipywidgets anndata scanpy celltypist

echo "Installing PyTorch Geometric and dependencies..."
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# 3. Install Guest Agent (metrics for Lambda Labs)
curl -L https://lambdalabs-guest-agent.s3.us-west-2.amazonaws.com/scripts/install.sh | sudo bash

echo "Fetching Jupyter Notebook token:"
jupyter notebook list

echo "Setup complete! Check jupyter.log for notebook output."
