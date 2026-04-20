# scLDM.CD4 Inference & Model Training Guide

This guide explains how to run inference using a pre-trained checkpoint, as well as how to train the scLDM.CD4 autoencoder and flow matching models based on scLDM ([Palla et al. 2025](https://arxiv.org/abs/2511.02986)).

A pre-trained checkpoint for the flow matching model and associated config file can be downloaded from [Hugging Face](https://huggingface.co/biohub/scldm_cd4).

Input data consists of CD4+ T cell Perturb-seq data described in [Zhu et al 2025 preprint](https://www.biorxiv.org/content/10.64898/2025.12.23.696273v1), in h5ad format.

---
## Run inference

### 1. Clone the repository

```bash
git clone <repo_url>
cd scldm_cd4
```

---

### 2. Download Model Checkpoint and Config from Huggingface
The latest model checkpoint and `config.yaml` are available at https://huggingface.co/biohub/scldm_cd4

The repo is currently configured to expect both of them in a top level directory named `model/`, but you can update paths as needed in step 5.

---

### 3a. Initialize the environment (default, uv + venv)

Run the initialization script to set up the virtual environment and dependencies:

```bash
./init.sh
```

This will create a new virtual environment under `venv/scldm_cd4`.

---

### 3b. Alternative: NixOS / pixi

If you're on NixOS, or prefer [pixi](https://pixi.sh) for reproducible environments, this repo ships a `flake.nix` plus a `[tool.pixi.*]` block in `pyproject.toml`. From the repo root:

```bash
direnv allow           # if using direnv; else: `nix develop`
pixi install           # solves and installs (~3-5 min cold)
pixi shell             # OR invoke tasks directly via `pixi run <task>`
```

Task shortcuts — each accepts an optional positional GPU count (default 4):

- `pixi run test` — full test suite; `pixi run test-fast` — CPU-only subset
- `pixi run vae [N]` / `pixi run fm [N]` — train VAE / flow matching on `N` GPUs
- `pixi run infer [N]` — inference on `N` GPUs
- `pixi run size-factors --data_dir ... --output_dir ...` — see step 4
- `pixi run jupyter` — launch JupyterLab against `notebooks/`

Hydra overrides pass through after the positional argument:

```bash
pixi run infer 4 training.num_epochs=1
```

The Nix flake provides `pixi` plus the env vars triton needs on NixOS (`LD_LIBRARY_PATH` and `TRITON_LIBCUDA_PATH`). On non-NixOS hosts, `pixi install` works standalone without `nix develop`.

---

### 4. Compute size factors

**(These files are precomputed and available at `data/size_factors_hvg` for the quickstart notebook, but if you would like to test inference with a different dataset, you will need to perform this step.)**

We provide a script for computing log size factor statistics, which are required for performing inference and for training the flow matching model. The script calculates the mean (mu) and standard deviation (sd) of log-transformed library sizes for each combination of experimental conditions specified in the config file. In the Marson data, a combination consists of donor x time point x perturbation. For combinations with no observed cells, statistics are aggregated at a courser resolution.

To generate size factors for the Marson data, run:
```bash
python scripts/compute_log_size_factors.py \
  --data_dir /path/to/train/data/dir \
  --output_dir /path/to/size/factors/dir \
  --config_path experiments/config/datamodule/single_anndata.yaml \
  --dataset_name marson_hvg \
  --fallback_conditions experimental_perturbation_time_point donor_id
```
This will generate two size factor files: `log_size_factor_mu.pkl` and `log_size_factor_sd.pkl`.

---

### 5. Configure user paths

The shipped defaults in `experiments/config/paths/user_paths.yaml` are repo-relative (`./model`, `./quickstart_data`, `./runs`, `./output`) and work out of the box if you launch from the repo root and downloaded the HuggingFace checkpoint into `./model/`. Only edit this file if your data or checkpoint live elsewhere.

To run inference from a pretrained checkpoint, you will need to specify:
 - `data_path`
 - `experiment_path`
 - `pretrained_checkpoint_path`
 - `inference_output`

To train the models, you will need to specify:
 - `data_path`
 - `experiment_path`
 - `vae_checkpoint_path` if training VAE
 - `fm_checkpoint_path` if training flow matching

---

### 6. Configure dataset paths

To run training or inference on a single h5ad, edit the following data paths for
`marson_hvg` in `experiments/config/datamodule/single_anndata.yaml`:

```yaml
dataset_params:
  marson_hvg:
    adata_train: /path/to/train/adata_1.h5ad
    adata_test: /path/to/test/adata_1.h5ad
    mu_path: /path/to/log_size_factor_mu.pkl
    sd_path: /path/to/log_size_factor_sd.pkl
```

---

### 7. Run inference

Users can find a tutorial notebook for running inference using a pre-trained checkpoint in `notebooks/quickstart_tutorial.ipynb`. The tutorial notebook is configured to run on a single GPU in a reasonable time.  CPU-only inference has not been tested at this time.

Inference can also be run from the command line.
To run inference using a pre-trained checkpoint with 1 node and 8 GPUs:

```python
torchrun \
    --nnodes 1 \
    --nproc-per-node 8 \
    experiments/scripts/inference_ddp.py \
    --config-name=inference_fm
```

---

## Optional: Train Models

### Train Autoencoder

Train autoencoder for 60 epochs, using 1 node and 8 GPUs per node:

```python
torchrun \
    --nnodes 1 \
    --nproc-per-node 8 \
    experiments/scripts/train.py \
    --config-name=marson_vae
```

---

### Train Flow Matching Model

Train FM for 150 epochs using 1 node and 8 GPUs per node:

```python
torchrun \
    --nnodes 1 \
    --nproc-per-node 8 \
    experiments/scripts/train.py \
    --config-name=marson_fm
```

---

## Code of Conduct

This project adheres to the Contributor Covenant [code of conduct](https://github.com/chanzuckerberg/.github/blob/master/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [opensource@chanzuckerberg.com](mailto:opensource@chanzuckerberg.com).

## Reporting Security Issues

If you believe you have found a security issue, please responsibly disclose by contacting us at [security@chanzuckerberg.com](mailto:security@chanzuckerberg.com).
