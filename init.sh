# sudo apt-get update
#sudo apt update
#sudo apt install mbuffer pv tree
# sudo apt-get install -y clang build-essential

# git aliases
#git config --global alias.ac '!git add -u . && git commit'
#git config --global alias.fp '!git fetch && git pull'
#git config pull.rebase true

# make sure the submodules are synced with main
# git submodule foreach 'git checkout main'

# general alias
echo "alias pvc='cd /mnt/czi-sci-ai/project-scg-llm-pvc/'" >> ~/.bashrc
echo "alias pvc2='cd /mnt/czi-sci-ai/project-scg-llm-data-2/'" >> ~/.bashrc
echo "export PYTORCH_KERNEL_CACHE_PATH=~/.cache" >> ~/.bashrc

# env variables
export HYDRA_FULL_ERROR=1
export TORCH_LOGS=dynamic
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export WANDB_BASE_URL=https://czi.wandb.io

# export NCCL_MIN_NCHANNELS=8
# export NCCL_NSOCKS_PERTHREAD=4
# export NCCL_SOCKET_NTHREADS=2
# export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.8"

#multinode
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_HCA=ibp
# export UCX_NET_DEVICES=ibp0:1,ibp1:1,ibp2:1,ibp3:1,ibp4:1,ibp5:1,ibp6:1,ibp7:1

export UV_CACHE_DIR=$HOME/.cache/uv

# Remove any existing UCX variables (they might conflict)
unset UCX_NET_DEVICES
unset UCX_TLS
unset UCX_IB_GID_INDEX

# NCCL InfiniBand optimization for your ibp0-ibp7 setup
export NCCL_IB_HCA=ibp0,ibp1,ibp2,ibp3,ibp4,ibp5,ibp6,ibp7  # Your 8 ports
export NCCL_IB_GID_INDEX=3                                    # Native IB (fastest)
export NCCL_NET_GDR_LEVEL=5                                   # GPU Direct RDMA

# environments
# source ~/.bashrc
uv venv -n venv/scldm_cd4 --python 3.11
source venv/scldm_cd4/bin/activate
uv pip install --quiet  -e '.'
# install --pre torch==2.5.0.dev20240909+cu124 --index-url https://download.pytorch.org/whl/nightly/cu124
# uv pip install --quiet --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124
# uv pip install --quiet --pre torch==2.6.0.dev20241204+cu124 --index-url https://download.pytorch.org/whl/nightly/cu124
uv pip install --quiet  -e '.[dev]'
uv pip install --quiet  ipykernel
uv pip install --quiet  torch-optimizer  # Install LAMB optimizer package
pre-commit install
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch_optimizer; print(f'✅ torch-optimizer installed: {torch_optimizer.__version__}')"
printenv
