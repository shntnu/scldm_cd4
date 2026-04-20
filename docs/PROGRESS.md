# Progress Log

INSTRUCTIONS: Add log at the end of the file. See "Maintenance Guidelines" and "Template for Future Entries" first

## Maintenance Guidelines

**When to add entries:**

- User explicitly asks to document progress
- Major milestone completed (pipeline working, bug resolved)
- Critical decision point requiring future reference

**Entry format (max 20 lines):**

- Brief title: what was accomplished
- Key findings or decisions (not step-by-step debugging)
- Blocking issues or next actions only if unresolved

**Monthly rollup:**

- Summarize entries older than 30 days to 1-3 lines per week
- Keep detailed recent history for active debugging
- Git history preserves all details if forensics needed

**DO NOT include:**

- Command outputs or error messages
- Step-by-step debugging notes
- Exploratory analysis details
- Duplicate information in code comments

## Template for Future Entries

```text
## YYYY-MM-DD: Brief Description

### What was done
- Key accomplishment 1
- Key accomplishment 2

### Key findings or decisions
- Finding that requires future reference

### Unresolved issues (if any)
- Issue description and next steps

### Notes (optional)
- Additional context (max 5 lines)
```

## 2026-04-19: Environment bring-up on NixOS + Nix/pixi

### What was done

- Stood up dev shell via `flake.nix` (Nix) + `pixi install`; confirmed 4× H100 NVL (96 GB each) visible to torch.
- Unblocked triton/deepspeed on NixOS by exporting `LD_LIBRARY_PATH` and `TRITON_LIBCUDA_PATH` to `/run/opengl-driver/lib`, and adding `cuda-cudart-dev` to pixi deps (triton JIT needs `cuda.h`).
- Made default Hydra paths repo-relative (`./model`, `./quickstart_data`, `./runs`, `./output`) since Hydra 1.2+ no longer chdir's into the run dir.
- Demoted the legacy `init.sh` (uv+venv) path in CLAUDE.md; pixi-in-Nix is now the primary path.

### Key findings or decisions

- `numpyro<0.20` is pinned — 0.20.x imports a primitive (`jit_p`) that jax 0.6 renamed to `pjit_p`.
- The released HF checkpoint is **safetensors**, not a Lightning `.ckpt`. `inference_ddp.py` monkey-patches `torch.load` (weights_only=False) — preserve this.
- Three stale symbols referenced by tests/args but never shipped (`StraightLineDiffusion`, `_random_mask`, `DiT(use_adaln=...)`); verify with `git log -S` before assuming a failure is yours.

### Notes

- Pretrained FM checkpoint (`biohub/scldm_cd4`, 830 MB safetensors) is downloaded to `./model/`; quickstart data + precomputed size factors are in `./quickstart_data/`.

---

## 2026-04-19: CPU test suite passing; 4-GPU inference exposes DDP init gap

### What was done

- Ran full pytest suite (CPU-only default: `-m "not rapids"`) — passing.
- Ran `pixi run infer 4` against the released safetensors checkpoint on 4× H100 NVL. Inference compiled the DiT, loaded the pretrained FM, and completed all 32 batches × 50 inference steps. Output `output/marson_hvg_generated_0_rank0.h5ad` = (4000 cells, 3699 HVGs), md5-identical to the pre-existing prior run → seed=56 path is deterministic and reproducible.
- Preserved the prior generated h5ad as `*.prior.h5ad` before the new run.

### Key findings or decisions

- **`inference_ddp.py` doesn't initialize DDP under plain `torchrun`.** Line 67 gates `dist.init_process_group()` on `WORLD_SIZE = PET_NNODES * RUNAI_NUM_OF_GPUS` (CZI Run:AI-specific vars). Locally these default to 1, so the guard never fires even when torchrun sets `WORLD_SIZE=4` env var. All 4 processes fall back to cellarium's `rank=0, num_replicas=1`, do 4× redundant work, and race to write the same `rank0.h5ad`; h5py file-lock serializes them — one succeeds, three crash with `BlockingIOError`, then `dist.get_rank()` in the except-handler raises a secondary error. Job exits non-zero despite valid output.
- Workaround without code change: `RUNAI_NUM_OF_GPUS=4 pixi run infer 4` (makes the guard fire). Proper fix: gate on `WORLD_SIZE_ENV` (what torchrun actually sets) in `inference_ddp.py`.
- Released inference path is safetensors-only: `inference_fm.yaml` pins `filename: model.safetensors` / `ckpt_file: model.safetensors`; retrained `.ckpt` won't load through this script without a config flip or safetensors export.

### Notes

- Generated adata carries `donor_id`, `guide_target_ensembl`, `experimental_perturbation_time_point`, `dataset`, `generation_idx` in `.obs`; latent is in `obsm['z']`.

---

## 2026-04-19: DDP init fix — 4-GPU inference now actually sharded

### What was done

- One-line fix in `experiments/scripts/inference_ddp.py:67`: guard changed from `if WORLD_SIZE > 1` to `if WORLD_SIZE_ENV > 1`. `WORLD_SIZE_ENV` is the env var torchrun actually sets; the old guard only fired under Run:AI (which sets `PET_NNODES`/`RUNAI_NUM_OF_GPUS`).
- Re-ran `pixi run infer 4`. Exit 0. Four rank shards written (`rank0..rank3`), 1000 cells × 3699 HVGs each = 4000 cells total. Wandb now coordinates run_id across ranks via a broadcast barrier (previously dead code).
- Wall time dropped from ~22 min (4× redundant) to ~6 min (properly sharded).

### Key findings or decisions

- Total cells generated is **unchanged** at 4000 — `dataset_generation_idx=0` is a fixed generation budget; multi-GPU shortens latency, doesn't generate more cells. For more cells, bump `dataset_generation_idx` or generation args.
- Cosmetic leak warning at shutdown: `destroy_process_group() was not called before program exit` — benign, worth a follow-up if training paths ever want cleaner teardown.
- Per-shard diversity looks healthy (~440–453 unique perturbations per 1000-cell shard, 2 donors per shard).

### Unresolved issues

- `train.py` uses the same `WORLD_SIZE = PET_NNODES * RUNAI_NUM_OF_GPUS` pattern (per CLAUDE.md); LR scaling is also keyed off it. Haven't audited whether this breaks identically under plain torchrun — defer until we actually want to train.
