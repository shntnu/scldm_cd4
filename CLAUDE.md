# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`scg_vae` is a two-stage latent diffusion model (scLDM.CD4) for CD4+ T cell Perturb-seq data. Training and inference are organized as:

1. **Stage 1 — Transformer VAE** (`scg_vae.vae.TransformerVAE`): encodes UMI counts into a per-cell set of latent tokens ("inducing points") and decodes via a negative-binomial head. Lives inside the `VAE` LightningModule (`scg_vae.models.VAE`).
2. **Stage 2 — Flow-matching DiT** (`scg_vae.diffusion.FlowMatching` + `scg_vae.nnets.DiT`): operates in the VAE latent space, conditioned on categorical labels (donor, perturbation, time point). Wrapped by the `LatentDiffusion` LightningModule (`scg_vae.models.LatentDiffusion`), which holds the (frozen) VAE as a "tokenizer" and an EMA copy of the DiT.

Input is CD4+ Perturb-seq `.h5ad` (Zhu et al. 2025 preprint). The pretrained FM checkpoint lives on HF Hub at `biohub/scldm_cd4`.

## Environment / commands

Python 3.11–3.12 only. The project uses `uv` + an editable install. Use the init script (creates `venv/scldm_cd4/`, installs deps incl. `cellarium-ml` from git, installs pre-commit, exports NCCL/UCX env vars):

```bash
./init.sh
source venv/scldm_cd4/bin/activate
```

**Alternative (NixOS or any pixi host):** `pixi install` then `pixi shell` — env defined under `[tool.pixi.*]` in `pyproject.toml` and activated via `flake.nix` devShell. Task aliases: `pixi run test` / `test-fast` / `vae [N]` / `fm [N]` / `infer [N]` (`N` = `--nproc-per-node`, default 4). README section 3b has the full flow.

Tests (pytest, CPU-only by default — `-m "not rapids"` is the default via `pyproject.toml`):

```bash
pytest                                    # full suite
pytest tests/test_diffusion.py            # single file
pytest tests/test_diffusion.py::test_diffusion -k "use_adaln-True"   # single param combo
```

Pre-commit hooks are minimal (trailing whitespace, EOF, merge-conflict, private-key detection). Ruff rules are commented out in `pyproject.toml` — there is **no enforced linter/formatter**; don't add style-only cleanups to unrelated code.

## Running training / inference

All entry points are Hydra apps launched via `torchrun`. The config-name selects the experiment (see `experiments/config/*.yaml`):

```bash
# Train VAE (stage 1)
torchrun --nnodes 1 --nproc-per-node 8 experiments/scripts/train.py --config-name=marson_vae

# Train flow-matching on frozen VAE latents (stage 2)
torchrun --nnodes 1 --nproc-per-node 8 experiments/scripts/train.py --config-name=marson_fm

# Inference from a pretrained checkpoint
torchrun --nnodes 1 --nproc-per-node 8 experiments/scripts/inference_ddp.py --config-name=inference_fm
```

The tutorial notebook `notebooks/quickstart_tutorial.ipynb` drives inference on a single GPU (CPU-only inference is untested). The size-factor precompute step (`scripts/compute_log_size_factors.py`) is required before training or inference on **new** data; quickstart data under `quickstart_data/size_factors_hvg/` is already precomputed.

## Config structure (Hydra)

Experiment entry configs live in `experiments/config/<name>.yaml` and compose groups via `defaults:`:

- `paths/user_paths.yaml` — **user-editable**: `data_path`, `experiment_path`, `pretrained_checkpoint_path`, `vae_checkpoint_path`, `fm_checkpoint_path`, `inference_output`. Edit these before running.
- `model/{vae_base,vae_tiny,ldm_base,ldm_small}.yaml` — network shapes & optimizer/scheduler. `ldm_*` configs reference VAE encoder dims via `${model.module.vae_model.encoder.*}`, so the VAE block must remain in the FM config even when the VAE is loaded from a checkpoint.
- `datamodule/single_anndata.yaml` — single-h5ad datamodule. Edit `dataset_params.marson_hvg.adata_train / adata_test / mu_path / sd_path` to point at real data; `class_vocab_sizes` (donor/guide_target_ensembl/time_point) must match the VocabularyEncoder built from the train adata.
- `datamodule/census.yaml` — alternate datamodule backed by CELLxGENE Census.
- `training/default.yaml` — trainer, CSV/W&B loggers (`entity: scg-vae`), `ModelCheckpoint` and `LearningRateMonitor` callbacks.

Custom Hydra resolver `${eval:'...'}` is registered in `main()` — use it for config-time arithmetic (e.g. `warmup_epochs: ${eval:'${training.num_epochs} // 2'}`).

## Distributed / runtime quirks to know

- **Launch-method expectation**: `train.py` reads `PET_NNODES`, `RUNAI_NUM_OF_GPUS`, and `WORLD_SIZE` env vars (set by `torchrun` and CZI's Run:AI). Running without `torchrun` will likely break DDP init.
- **LR is scaled by `WORLD_SIZE_ENV`** in `train.py`. If you change node/GPU count, the effective LR changes.
- **`torch._dynamo` flags** are set globally at script start (`optimize_ddp=False`, `cache_size_limit=1000`, `capture_scalar_outputs=True`). `torch.compile` is enabled for the VAE and DiT by default (`model.module.compile: true`, `compile_mode: "default"`).
- **`inference_ddp.py` monkey-patches `torch.load`** to force `weights_only=False`. This is intentional (PyTorch 2.6+ default changed) — preserve it when editing that script.
- **Checkpoint loading path for FM training**: `cfg.model.module.vae_as_tokenizer.load_from_checkpoint` points to a trained VAE job; `train.py` loads `{ckpt_path}/{job_name}/{epoch=N.ckpt or last.ckpt}` plus the VAE's `config.yaml`, then validates consistency via `load_validate_statedict_config`.
- **Safetensors inference path**: `inference_fm.yaml` pins `filename: "model.safetensors"` / `ckpt_file: "model.safetensors"` — the HF-released checkpoint is safetensors, not a Lightning `.ckpt`.
- **GPT gene embeddings** (optional): if the datamodule's `VocabularyEncoderSimplified` provides `gpt_gene_embeddings`, `train.py` injects them into both the VAE input layer and (if `use_gpt_for_gene_ko=true`) the DiT's gene-KO conditioning — including the EMA DiT. Don't remove this injection logic silently.

## NixOS specifics (pixi path only)

- `flake.nix` devShell sets `LD_LIBRARY_PATH=/run/opengl-driver/lib` (for `libcuda.so.1`) and `TRITON_LIBCUDA_PATH=/run/opengl-driver/lib` (bypasses triton's hardcoded `/sbin/ldconfig` probe that fails on NixOS).
- `cuda-cudart-dev` is in `[tool.pixi.dependencies]` because triton JIT-compiles a C helper on first use that `#include`s `cuda.h` — the runtime-only conda pytorch doesn't ship headers.
- After editing `flake.nix`, `direnv reload` (or exit + re-enter the shell) to pick up new env vars; `pixi run` in an already-active shell won't see them.
- `numpyro<0.20` pinned in pixi deps: 0.20.x imports `jit_p` from `jax.extend.core.primitives`, which jax 0.6 renamed to `pjit_p`.

## Code map

`src/scg_vae/`:

- `models.py` — `BaseModel`, `VAE`, `LatentDiffusion` LightningModules. `LatentDiffusion` holds `vae_model` (frozen tokenizer) + `diffusion_model` + `ema_model`, implements `shared_step`, `inference`, `sample`, and metric aggregation (MMD, Wasserstein, R²).
- `vae.py` — `TransformerVAE`: encoder → latent tokens → decoder → NB head.
- `diffusion.py` — `BaseDiffusion`, `FlowMatching` (only diffusion class actually exported today; older imports of `StraightLineDiffusion` are stale).
- `nnets.py` — `Encoder`, `Decoder`, `DiT`, plus scVI-style MLP variants (`EncoderScvi`/`DecoderScvi`).
- `layers.py` / `stochastic_layers.py` — transformer blocks, `InputTransformerVAE` (agg functions: `log1p`, `softbin`, `anscombe`, …), NB/Poisson/discretized-Gaussian/logistic decoder heads.
- `encoder.py` — `VocabularyEncoder` (full) and `VocabularyEncoderSimplified` (used by `SimplifiedDataModule`); builds class→index maps, size-factor lookups, and optional GPT gene-embedding lookups from the train adata.
- `datamodule.py` — `SimplifiedDataModule` (single-h5ad) and the Census-backed datamodule. Both emit the batch dict consumed by `VAE.shared_step`.
- `transport/` — flow-matching path/integrator utilities consumed via `scg_vae.transport.create_transport` in `ldm_base.yaml`.
- `_train_utils.py` — Hydra/Lightning glue: `setup_datamodule_and_steps`, `setup_callbacks_and_loggers_and_paths`, `setup_wandb_run`, `load_validate_statedict_config`, `maybe_fix_compiled_weights` (strips `_orig_mod.` prefixes from compiled checkpoints), `process_generation_output`, `process_inference_output`.
- `evaluations.py` / `mmd.py` — MMD kernels (Bray–Curtis, Tanimoto, Ruzicka, RBF) and Sinkhorn Wasserstein wrappers used for validation/test metrics.
- `viz_callbacks.py` — `ReconstructionVisualizationCallback` (VAE only — it reconstructs UMAPs of real vs. reconstructed cells).
- `flops.py` / `optimizers.py` (LAMB / `AdamWLegacy`) / `priors.py` / `logger.py` / `_utils.py` (incl. `wsd_schedule`, `world_info_from_env`).

`experiments/scripts/`:

- `train.py` — unified VAE or FM trainer (branches on presence of `vae_as_tokenizer`). On completion runs `trainer.predict` (if FM) to generate an adata and logs a UMAP to W&B, then runs `trainer.test`.
- `inference_ddp.py` — DDP inference from a pretrained FM checkpoint, writes results under `paths.inference_output`.
- `inference.py` — older single-process inference variant.

`tests/` — small CPU-runnable tests for encoder, layers, masks, diffusion round-trips, misc utilities. `tests/conftest.py` builds synthetic sparse AnnDatas.

## Conventions to preserve when editing

- **Don't hand-edit the VAE block out of FM configs** — its encoder dims (`n_embed`, `n_embed_latent`, `n_inducing_points`) are interpolated into DiT and the tokenizer loader.
- **Keep `class_vocab_sizes` in `datamodule/single_anndata.yaml` in sync with the train adata**; the `VocabularyEncoderSimplified` will validate and assertion-fail otherwise.
- **`norm_factor_path` / normalizer computation** in `train.py` is commented-out. If re-enabling, broadcast across ranks before use (the skeleton is already there).
- **Checkpointing on FM**: `save_weights_only: false` is required because the EMA model state is serialized alongside.
- **Run:AI YAMLs** in `experiments/config/runai/` are launcher-side manifests (not Hydra configs); they aren't loaded by `train.py`.
- **Stale-symbol diagnosis**: when a test fails with `ImportError` / `TypeError` on a symbol, run `git log --all -S '<symbol>' -- src/` before writing code — three symbols were referenced by tests but never shipped (`_random_mask`, `StraightLineDiffusion`, `DiT(use_adaln=...)`), all from an aborted refactor. `vae.py`'s `masking_prop` / `mask_token_idx` args are similar dead code (they're accepted by `TransformerVAE.forward` but the active `InputTransformerVAE.forward` takes only 2 args).
- **Default paths** in `experiments/config/paths/user_paths.yaml` are repo-relative (`./model`, `./quickstart_data`, `./runs`, `./output`) and assume launch from repo root. Hydra 1.2+ defaults `hydra.job.chdir=false` so CWD doesn't change mid-run. The `../` paths in older forks are broken.
- **Wandb** defaults to the caller's personal `api.wandb.ai` unless `WANDB_BASE_URL` is set (CZI's internal is `https://czi.wandb.io`, exported by `init.sh` but NOT by the flake/pixi path). `training/default.yaml` sets `entity: scg-vae`; `inference_fm.yaml` clears entity, so inference logs to the logged-in user's default namespace.
