import os
from typing import Any

import anndata as ad
import hydra
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig
from scipy import sparse

from scg_vae.logger import logger
from scg_vae.models import ModelEnum


def setup_datamodule_and_steps(cfg, world_size, num_epochs):
    import hydra

    datamodule = hydra.utils.instantiate(cfg.datamodule.datamodule)
    num_steps_per_epoch = int(datamodule.n_cells / (cfg.datamodule.datamodule.batch_size * world_size))
    effective_steps = num_epochs * num_steps_per_epoch
    logger.info(
        f"Effective steps: {effective_steps}, Num epochs: {num_epochs}, World size: {world_size}, Batch size: {cfg.datamodule.datamodule.batch_size}, N cells: {datamodule.n_cells}"
    )
    if cfg.model.module.vae_scheduler is not None:
        cfg.model.module.vae_scheduler.num_training_steps = effective_steps
        cfg.model.module.vae_scheduler.num_warmup_steps = int(0.1 * effective_steps)
    if hasattr(cfg.model.module, "diffusion_scheduler") and cfg.model.module.diffusion_scheduler is not None:
        cfg.model.module.diffusion_scheduler.num_training_steps = effective_steps
        cfg.model.module.diffusion_scheduler.num_warmup_steps = int(0.1 * effective_steps)
    cfg.training.trainer.max_steps = effective_steps
    return datamodule


def setup_callbacks_and_loggers_and_paths(cfg, override_dict, inference: bool = False):
    import hydra
    from pytorch_lightning.callbacks import Callback
    from pytorch_lightning.loggers import Logger

    if not inference:
        if "runai_job_name" in cfg:
            run_name = cfg.runai_job_name
        elif override_dict:
            # Use Hydra overrides as run name
            run_name = ",".join(
                f"{k}={v}" for k, v in override_dict.items()
            ).replace(" ", "-")[:250]
        else:
            # No Hydra overrides used
            run_name = "no_overrides"

        cfg.training.logger.csv.version = f"{cfg.experiment_name}/{run_name}"
        if cfg.training.logger.wandb is not None:
            cfg.training.logger.wandb.name = run_name
        cfg.training.callbacks.model_checkpoints.dirpath = os.path.join(
            cfg.paths.experiment_path, "checkpoints", cfg.experiment_name, run_name
        )

    callbacks_: dict[str, Callback] = {}

    for cb_name, cb in cfg.training.callbacks.items():
        callbacks_[cb_name] = hydra.utils.instantiate(cb)

    loggers_: dict[str, Logger] = {}
    for lg_name, lg in cfg.training.logger.items():
        if lg is not None:
            loggers_[lg_name] = hydra.utils.instantiate(lg)

    return callbacks_, loggers_, cfg


def process_generation(generation_output: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    import anndata as ad
    from scipy import sparse

    generated_counts = torch.cat(
        [
            generation_output[i][f"{ModelEnum.COUNTS.value}_generated"]
            for i in range(len(generation_output))
            if generation_output[i] is not None
        ]
    )
    true_counts = torch.cat(
        [
            generation_output[i][ModelEnum.COUNTS.value]
            for i in range(len(generation_output))
            if generation_output[i] is not None
        ]
    )

    adata_true = ad.AnnData(sparse.csr_matrix(true_counts.numpy()))
    adata_generated = ad.AnnData(sparse.csr_matrix(generated_counts.numpy()))
    return adata_true, adata_generated


def process_generation_output(
    output: list[dict[str, torch.Tensor]],
    datamodule: Any,
) -> ad.AnnData:
    logger.info("Processing generation output")
    # counts_true_sparse = sparse.vstack([sparse.csr_matrix(o[f"{ModelEnum.COUNTS.value}"].numpy()) for o in output])
    counts_generated_unconditional_sparse = sparse.vstack(
        [sparse.csr_matrix(o[f"{ModelEnum.COUNTS.value}_generated_unconditional"].numpy()) for o in output]
    )
    counts_generated_conditional_sparse = sparse.vstack(
        [sparse.csr_matrix(o[f"{ModelEnum.COUNTS.value}_generated_conditional"].numpy()) for o in output]
    )
    z_generated_unconditional = np.vstack([o["z_generated_unconditional"].numpy() for o in output])
    z_generated_conditional = np.vstack([o["z_generated_conditional"].numpy() for o in output])

    genes = output[0][f"{ModelEnum.GENES.value}"][0, :]
    var_names = datamodule.vocabulary_encoder.decode_genes(genes)

    obs = {
        k: datamodule.vocabulary_encoder.decode_metadata(torch.cat([o[k] for o in output], dim=0).numpy(), k)
        for k in datamodule.vocabulary_encoder.labels.keys()
    }

    del output

    n_cells = counts_generated_unconditional_sparse.shape[0]

    obs_generated_unconditional = pd.DataFrame(obs, index=np.arange(n_cells).astype(str))
    obs_generated_conditional = pd.DataFrame(obs, index=np.arange(n_cells, 2 * n_cells).astype(str))

    obs_generated_unconditional["dataset"] = "generated_unconditional"
    obs_generated_conditional["dataset"] = "generated_conditional"

    X_combined = sparse.vstack([counts_generated_unconditional_sparse, counts_generated_conditional_sparse])
    z_combined = np.vstack([z_generated_unconditional, z_generated_conditional])

    obs_combined = pd.concat([obs_generated_unconditional, obs_generated_conditional], axis=0)
    adata = ad.AnnData(X=X_combined, obs=obs_combined, obsm={"z": z_combined})
    adata.var_names = var_names
    return adata


def create_anndata_from_inference_output(
    output: dict[str, torch.Tensor],
    datamodule: Any,
) -> ad.AnnData:
    z_sample = output["z_sample"].numpy()
    z_sample_flat = output["z_sample_flat"].numpy()
    generated_counts = sparse.csr_matrix(output["reconstructed_counts"].numpy())
    genes = output[f"{ModelEnum.GENES.value}"][0, :]
    var_names = datamodule.vocabulary_encoder.decode_genes(genes)
    obs = {
        k: datamodule.vocabulary_encoder.decode_metadata(output[k].numpy(), k)
        for k in datamodule.vocabulary_encoder.labels.keys()
    }

    n_cells = len(z_sample)
    obs = pd.DataFrame(obs, index=np.arange(n_cells).astype(str))

    adata = ad.AnnData(X=generated_counts, obs=obs, obsm={"z_sample": z_sample, "z_sample_flat": z_sample_flat})
    adata.var_names = var_names
    adata.layers["counts"] = adata.X.copy()
    return adata


def process_inference_output(
    output: list[ad.AnnData],
    datamodule: Any,
) -> ad.AnnData:
    logger.info("Processing inference output")
    adata = ad.concat(output)

    # sc.pp.normalize_total(adata)
    # sc.pp.log1p(adata)
    # sc.pp.pca(adata)
    # sc.pp.neighbors(adata)
    # sc.tl.umap(adata)

    # # process latents
    # adata.obsm["z_sample_flat_pca"] = sc.pp.pca(adata.obsm["z_sample_flat"])

    # sc.pp.neighbors(adata, use_rep="z_sample", key_added="z_sample_neighbors")
    # sc.pp.neighbors(adata, use_rep="z_sample_flat_pca", key_added="z_sample_flat_pca_neighbors", n_neighbors=10)

    # sc.tl.umap(adata, neighbors_key="z_sample_neighbors", key_added="z_sample_neighbors_umap")
    # sc.tl.umap(adata, neighbors_key="z_sample_flat_pca_neighbors", key_added="z_sample_flat_pca_neighbors_umap")

    return adata


def load_validate_statedict_config(
    checkpoints: dict[str, Any],
    config: DictConfig,
    pretrain_config: DictConfig,
) -> tuple[dict[str, Any], DictConfig]:
    vae_state_dict = {
        k.replace("vae_model.", "", 1): v for k, v in checkpoints["state_dict"].items() if k.startswith("vae_model.")
    }
    # diffusion_state_dict = {
    #     k.replace("diffusion_model.", "", 1): v
    #     for k, v in checkpoints["state_dict"].items()
    #     if k.startswith("diffusion_model.")
    # }
    config.model.module.vae_model = pretrain_config.model.module.vae_model
    config.model.module.diffusion_model.n_embed_input = pretrain_config.model.module.vae_model.encoder.n_embed_latent
    config.model.module.diffusion_model.seq_len = pretrain_config.model.module.vae_model.encoder.n_inducing_points

    return vae_state_dict, config
    # TODO: this is not working, need to fix it
    # vae_state_dict = {
    #     k.replace("vae_model._orig_mod.", "", 1): v
    #     for k, v in checkpoints["state_dict"].items()
    #     if k.startswith("vae_model._orig_mod.")
    # }

    # config.model.module.vae_model = pretrain_config.model.module.vae_model
    # return vae_state_dict, config, None


def maybe_fix_compiled_weights(
    ckpt_path: str, diffusion_compile: bool, vae_compile: bool, module_keys: list[str]
) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    modified = False

    # Sort both lists to ensure consistent ordering
    sorted_checkpoint_items = sorted(state_dict.items())
    sorted_module_keys = sorted(module_keys)

    # assert len(sorted_checkpoint_items) == len(sorted_module_keys), "Checkpoint and module keys have different lengths"

    for i, (k, v) in enumerate(sorted_checkpoint_items):
        new_k = k

        if new_k != sorted_module_keys[i]:
            modified = True
            checkpoint_key = new_k
            module_key = sorted_module_keys[i]

            # Check if the ONLY difference is "_orig_mod." prefix
            if checkpoint_key.replace("_orig_mod.", "") == module_key:
                new_k = module_key
            # Check if the ONLY difference is "compiled." prefix
            elif checkpoint_key.replace("compiled.", "") == module_key:
                new_k = module_key
            # Check if module key with "_orig_mod." matches checkpoint key
            elif module_key.replace(".", "._orig_mod.", 1) == checkpoint_key:
                new_k = module_key
            # Check if module key with "compiled." matches checkpoint key
            elif module_key.replace(".", ".compiled.", 1) == checkpoint_key:
                new_k = module_key
            # Check if checkpoint key with "_orig_mod." inserted matches module key
            elif checkpoint_key.replace(".", "._orig_mod.", 1) == module_key:
                new_k = module_key
            # Check if module key with "_orig_mod." removed matches checkpoint key
            elif module_key.replace("_orig_mod.", "") == checkpoint_key:
                new_k = module_key
            # If none of the above patterns match, set_trace for debugging
            else:
                # raise error
                raise ValueError(f"Problem here: {new_k} != {module_key}")

        new_state_dict[new_k] = v

    # Second pass: handle missing keys in checkpoint that exist in module
    for module_key in sorted_module_keys:
        if module_key not in new_state_dict:
            # Check if removing "_compiled" from module key gives us a key in the original checkpoint
            if "_compiled" in module_key:
                checkpoint_key_without_compiled = module_key.replace("_compiled._orig_mod", "")

                if checkpoint_key_without_compiled in new_state_dict:
                    # Found the corresponding key in checkpoint, copy the weight
                    new_state_dict[module_key] = new_state_dict[checkpoint_key_without_compiled]
                    modified = True
                    # print(f"Added missing key: {module_key} from {checkpoint_key_without_compiled}")

    if modified:
        logger.info(f"Fixing compiled weights for diffusion: {diffusion_compile}, vae: {vae_compile} in {ckpt_path}")
        checkpoint["state_dict"] = new_state_dict
        torch.save(checkpoint, ckpt_path)


def setup_wandb_run(
    is_main_process: bool,
    cfg: DictConfig,
    global_rank: int,
):
    run_name = cfg.training.logger.wandb.name
    project_name = cfg.training.logger.wandb.project
    run_id_to_use = getattr(cfg.training.logger.wandb, "id", None)  # use id if provided

    # Ensure all ranks are synchronized before starting WandB setup
    if dist.is_initialized():
        dist.barrier()
        logger.info(f"Rank {global_rank} - Starting WandB setup after barrier")

    if run_id_to_use is None:
        # Only query existing runs if no ID is provided (optional)
        try:
            api = wandb.Api(timeout=19)
            path_for_runs = project_name
            logger.info(f"Querying wandb API for runs at path: {path_for_runs}")
            runs = api.runs(path=path_for_runs)

            for run in runs:
                if run.name == run_name:
                    run_id_to_use = run.id
                    logger.info(
                        f"Found existing wandb run '{run.name}' (ID: {run.id}, State: {run.state}). Will attempt to resume."
                    )
                    break

        except Exception as e:  # noqa: BLE001
            logger.error(
                f"Could not connect to W&B API or search for runs: {e}. Will proceed to init without resume ID lookup."
            )

    # Synchronize all processes before broadcast
    if dist.is_initialized():
        dist.barrier()

        # Broadcast from rank 0 to all other ranks
        object_list_to_broadcast = [run_id_to_use]
        logger.info(f"Rank {global_rank} - Broadcasting wandb run_id: {run_id_to_use}")
        dist.broadcast_object_list(object_list_to_broadcast, src=0)

        if not is_main_process:
            run_id_to_use = object_list_to_broadcast[0]
            logger.info(f"Rank {global_rank} - Received wandb run_id: {run_id_to_use}")

        # Final barrier to ensure all ranks have the run_id
        dist.barrier()
        logger.info(f"Rank {global_rank} - Completed WandB setup with run_id: {run_id_to_use}")

    return run_id_to_use


def compute_model_stats(cfg: DictConfig, module: Any) -> DictConfig:
    cfg.model.num_parameters = sum(p.numel() for p in module.vae_model.parameters() if p.requires_grad)
    if hasattr(module, "diffusion_model") and module.diffusion_model is not None:
        cfg.model.num_parameters += sum(p.numel() for p in module.diffusion_model.parameters() if p.requires_grad)

    if cfg.model.get_flops is None:
        cfg.model.flops = None
        return cfg

    cfg.model.flops = hydra.utils.instantiate(cfg.model.get_flops)
    return cfg
