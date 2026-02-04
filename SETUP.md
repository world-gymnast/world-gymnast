# World-Gymnast Installation Guide

This guide provides step-by-step instructions for setting up the World-Gymnast environment.

#### Step 1: Install veRL

> **Note:** We recommend veRL version 0.2 or 0.3. Latest versions may have library conflicts.

Follow the official [veRL installation guide](https://verl.readthedocs.io/en/v0.3.x/start/install.html):

```bash
# Create and activate conda environment
conda create -n worldgymnast python==3.10
conda activate worldgymnast

# Install PyTorch
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Clone veRL (recommended to place at the same level as world-gymnast, not inside the world-gymnast folder)
git clone -b v0.2.x https://github.com/volcengine/verl.git
cd verl
pip3 install -e .
cd ..
```

#### Step 2: Install OpenVLA-OFT

Follow the official [OpenVLA-OFT installation guide](https://github.com/moojink/openvla-oft):

```bash
conda activate worldgymnast

# Clone OpenVLA-OFT (place at the same level as simplevla-rl, not inside the simplevla-rl folder)
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
pip install -e .

# Install Flash Attention 2 for training
# Reinstall torch 2.4.0
pip install torch==2.4.0 torchaudio torchvision
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

#### Step 3: Install world-model-eval

```bash
conda activate worldgymnast

git clone https://github.com/world-gymnast/world-model-eval.git
cd world-model-eval
pip install -e . --no-deps --no-build-isolation

# Install additional dependencies
pip install mediapy==1.2.4 opencv-python==4.11.0.86 pytorchvideo==0.1.5 fire==0.7.1 imageio==2.37.0 backports.strenum==1.3.1
```