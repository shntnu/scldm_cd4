import os
import pathlib
from collections.abc import Callable

import hydra
from omegaconf import DictConfig, OmegaConf

os.environ["HYDRA_FULL_ERROR"] = "1"


# Fix for PyTorch 2.6+ default weights_only=True behavior
# Import torch FIRST, patch it ONCE, then never re-import
import torch
import torch.distributed as dist

# Monkey-patch torch.load to FORCE weights_only=False
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

def train(cfg) -> None:
    import pytorch_lightning as pl
    #import torch
    #import torch.distributed as dist
    from hydra.core.hydra_config import HydraConfig
    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies import DDPStrategy

    from scg_vae._train_utils import (
        compute_model_stats,
        process_generation_output,
        process_inference_output,
        setup_callbacks_and_loggers_and_paths,
        setup_datamodule_and_steps,
        setup_wandb_run,
    )
    from scg_vae._utils import world_info_from_env
    from scg_vae.logger import logger

    torch.set_float32_matmul_precision("high")

    pl.seed_everything(cfg.seed + cfg.dataset_generation_idx)
    logger.info("Set distributed environment...")

    local_rank, global_rank, world_size = world_info_from_env()
    logger.info(f"LOCAL RANK {local_rank}, GLOBAL RANK {global_rank}, WORLD SIZE {world_size}")

    # TODO: maybe remove this eventually
    torch._dynamo.config.capture_scalar_outputs = True

    # get env variables from runai
    NUM_NODES = int(os.environ.get("PET_NNODES", 1))
    NUM_GPU_PER_DEVICE = int(os.environ.get("RUNAI_NUM_OF_GPUS", 1))
    WORLD_SIZE_ENV = int(os.environ.get("WORLD_SIZE") or "1")  # Fix: Handle None case properly
    WORLD_SIZE = NUM_NODES * NUM_GPU_PER_DEVICE
    if WORLD_SIZE_ENV is None:
        WORLD_SIZE_ENV = WORLD_SIZE
    print(f"WORLD_SIZE: {WORLD_SIZE}, WORLD_SIZE_ENV: {WORLD_SIZE_ENV}, world_size: {world_size}")
    # assert WORLD_SIZE == WORLD_SIZE_ENV == world_size, "WORLD SIZE MISMATCH"

    # Initialize the distributed environment
    # Set the device before initializing process group
    torch.cuda.set_device(local_rank)
    if WORLD_SIZE > 1:
        dist.init_process_group(backend="nccl")
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"
    is_main_process = (world_size == 1) or (global_rank == 0) #Only rank 0 does logging/saving

    logger.info("Loading original configs...")
    original_cfg = OmegaConf.load(os.path.join(cfg.training.callbacks.model_checkpoints.dirpath, "config.yaml"))

    # # specific rewrite for fm inference
    if hasattr(original_cfg.model.module, "vae_as_tokenizer"):
        original_cfg.model.module.vae_as_tokenizer = None # Don't need tokenizer
    #     original_cfg.model.module.norm_factor = cfg.model.module.norm_factor
    #     logger.info(f"norm_factor: {original_cfg.model.module.norm_factor}")
    original_cfg.model.batch_size = cfg.model.batch_size

    if hasattr(cfg.model.module, "inference_args"):
        original_cfg.model.module.inference_args = cfg.model.module.inference_args
    if hasattr(cfg.model.module, "generation_args"):
        original_cfg.model.module.generation_args = cfg.model.module.generation_args

    test_batch_size = cfg.model.test_batch_size
    batch_size = cfg.model.batch_size
    cfg.model = original_cfg.model
    cfg.model.test_batch_size = test_batch_size
    cfg.model.batch_size = batch_size
    # cfg.datamodule = original_cfg.datamodule

    logger.info("Instantiating datamodule...")
    datamodule = setup_datamodule_and_steps(cfg, WORLD_SIZE, cfg.training.num_epochs)
    logger.info(f"Effective number of steps {cfg.training.trainer.max_steps}")
    datamodule.setup()

    # Configure InputTransformerVAE for GPT mode if embeddings available
    gpt_input_layer = None
    if hasattr(datamodule, 'vocabulary_encoder'):
        vocab_encoder = datamodule.vocabulary_encoder
        if hasattr(vocab_encoder, 'gpt_gene_embeddings') and vocab_encoder.gpt_gene_embeddings is not None:
            # Set has_masked_gene_tokens based on sample_genes
            has_masked = False if datamodule.sample_genes in ("none", None) else True
            # Manually instantiate input_layer with runtime objects
            input_layer_partial = hydra.utils.instantiate(cfg.model.module.vae_model.input_layer)
            # Instantiate WITH GPT embeddings
            gpt_input_layer = input_layer_partial(
                gpt_gene_embeddings=vocab_encoder.gpt_gene_embeddings,
                gene_idx_to_name=vocab_encoder.gene_idx_to_name,
                has_masked_gene_tokens=has_masked,
            )
            logger.info("GPT gene embeddings will be used for VAE.")


    logger.info("Instantiating module...")
    module = hydra.utils.instantiate(cfg.model.module)

    # Inject GPT embeddings for VAE
    if gpt_input_layer is not None:
        module.vae_model.input_layer = gpt_input_layer
        logger.info("GPT input layer injected into model.")

    # Inject GPT embeddings for gene-KO in diffusion model (AFTER module creation)
    use_gpt_for_gene_ko = cfg.model.module.diffusion_model.nnet.get("use_gpt_for_gene_ko", False)
    if use_gpt_for_gene_ko:
        gene_ko_class_name = cfg.model.module.diffusion_model.nnet.get("gene_ko_class_name", None)

        if gene_ko_class_name and hasattr(datamodule, 'vocabulary_encoder'):
            vocab_encoder = datamodule.vocabulary_encoder
            if hasattr(vocab_encoder, 'gpt_gene_embeddings') and vocab_encoder.gpt_gene_embeddings is not None:
                gpt_embeddings = vocab_encoder.gpt_gene_embeddings
                gene_ko_idx_to_name = vocab_encoder.idx2classes[gene_ko_class_name]
                control_perturbation_name = cfg.model.module.diffusion_model.nnet.get("control_perturbation_name", None)

                # Inject into main diffusion model
                module.diffusion_model.nnet.set_gpt_gene_ko_embeddings(
                    gpt_embeddings,
                    gene_ko_idx_to_name,
                    control_perturbation_name,
                )
                logger.info(f"GPT embeddings injected into DiT for gene-KO class: {gene_ko_class_name}")

                # ALSO inject into EMA model
                if hasattr(module, 'ema_model') and hasattr(module.ema_model, 'ema_model'):
                    module.ema_model.ema_model.nnet.set_gpt_gene_ko_embeddings(
                        gpt_embeddings,
                        gene_ko_idx_to_name,
                        control_perturbation_name,
                    )
                    logger.info("GPT embeddings also injected into EMA model")


    module.vae_model.eval()  # Disable dropout, batchnorm, etc.
    module.diffusion_model.eval()  # Disable dropout, batchnorm, etc.
    if hasattr(module, 'ema_model'):
        module.ema_model.eval()

    # Compute normalizer factor for latent diffusion
    # norm_factor_path = os.path.join(cfg.training.callbacks.model_checkpoints.dirpath, "normalize_factor.pt")
    # if is_main_process and original_cfg.model.module.norm_factor is None:
    #     logger.info("Computing normalizer factor for VAE latents...")
    #     normalizer = module.compute_normalizer(
    #         dataloader=datamodule.train_dataloader(),
    #         norm_factor_path=norm_factor_path,
    #     )
    # else:
    #     normalizer = 0.0  # placeholder for non-main processes

    # # Synchronize normalizer across all processes
    # if dist.is_initialized():
    #     normalizer_tensor = torch.tensor(normalizer, device=torch.cuda.current_device())
    #     dist.broadcast(normalizer_tensor, src=0)
    #     normalizer = normalizer_tensor.item()
    #     dist.barrier()

    # Assign the synchronized normalizer to the VAE model
    # module.vae_model.norm_factor = 0.66968032905521
    # logger.info(f"Normalizer factor set to: {module.vae_model.norm_factor}")

    # first, get run name
    # get overrides if it exists
    if len(HydraConfig.get().job.override_dirname):
        import re

        # Split by commas that are not inside square brackets
        override_items = re.split(r",(?![^\[]*\])", HydraConfig.get().job.override_dirname)
        override_dict = dict(item.split("=", 1) for item in override_items)
        override_dict = {k.replace("+", ""): v for k, v in override_dict.items()}
    else:
        override_dict = {}  # if no overrides used, e.g. for testing

    logger.info("Instantiating callbacks and logger...")
    callbacks_, loggers_, cfg = setup_callbacks_and_loggers_and_paths(cfg, override_dict, inference=True)

    cfg = compute_model_stats(cfg, module)
    logger.info(f"Model flops: {cfg.model.flops} parameters: {cfg.model.num_parameters}")
    override_dict["num_parameters"] = cfg.model.num_parameters
    override_dict["flops"] = cfg.model.flops

    # Save the config to the checkpoint directory
    config_path = os.path.join(cfg.training.callbacks.model_checkpoints.dirpath, "config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f)
    logger.info(f"Configuration saved to {config_path}")

    if cfg.training.logger.wandb is not None:
        run_id_to_use = setup_wandb_run(is_main_process=is_main_process, cfg=cfg, global_rank=global_rank)
        group = override_dict.get("seed", "default")
        loggers_["wandb"] = loggers_["wandb"](
            id=run_id_to_use,
            group=group,
        )
        if is_main_process:
            loggers_["wandb"].experiment.config.update(override_dict, allow_val_change=True)
            loggers_["wandb"].experiment.save(config_path)

    trainer_: Callable[..., Trainer] = hydra.utils.instantiate(cfg.training.trainer)  # partial

    if "strategy" not in cfg.training.trainer:
        strategy: str | DDPStrategy = "ddp" if WORLD_SIZE > 1 else "auto"
        if strategy == "ddp":
            strategy = DDPStrategy(
                gradient_as_bucket_view=True,
                static_graph=False,
                find_unused_parameters=False,
            )
    else:
        strategy = cfg.training.trainer.strategy

    trainer: Trainer = trainer_(
        num_nodes=NUM_NODES,
        devices=NUM_GPU_PER_DEVICE,
        strategy=strategy,
        use_distributed_sampler=False,
        logger=list(loggers_.values()),
        callbacks=list(callbacks_.values()),
    )
    try:
        checkpoint_dir = pathlib.Path(cfg.training.callbacks.model_checkpoints.dirpath)
        last_checkpoint = checkpoint_dir / cfg.ckpt_file
        ckpt_path = last_checkpoint if last_checkpoint.exists() else None
        # if ckpt_path is not None:
        #     maybe_fix_compiled_weights(ckpt_path, diffusion_compile=cfg.model.compile, vae_compile=False)
        # trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
        if dist.is_initialized():
            dist.barrier()

        # Clear cache on all processes
        torch.cuda.empty_cache()
        torch._dynamo.config.cache_size_limit = 1000

        if cfg.model.module.generation_args is not None or cfg.model.module.inference_args is not None:
            # ALL GPUs participate in prediction
            datamodule.setup("predict")
            output = trainer.predict(
                module,
                datamodule=datamodule,
                ckpt_path=ckpt_path,
            )

            # Each GPU processes its portion of the output
            if cfg.model.module.generation_args is not None:
                local_adata = process_generation_output(output, datamodule)
            else:
                local_adata = process_inference_output(output, datamodule)

            inference_type = "generated" if cfg.model.module.generation_args is not None else "inference"
            local_adata.obs["generation_idx"] = cfg.dataset_generation_idx

            # Each GPU saves its shard independently (no concatenation)
            rank = dist.get_rank() if dist.is_initialized() else 0
            world_size_actual = dist.get_world_size() if dist.is_initialized() else 1

            shard_path = (
                pathlib.Path(cfg.inference_path)
                / f"{cfg.datamodule.dataset}_{inference_type}_{cfg.dataset_generation_idx}_rank{rank}.h5ad"
            )
            shard_path.parent.mkdir(parents=True, exist_ok=True)
            local_adata.write_h5ad(shard_path)
            logger.info(f"Rank {rank}/{world_size_actual} saved {len(local_adata)} cells to {shard_path}")

            # Wait for all GPUs to finish saving
            if dist.is_initialized():
                dist.barrier()

            if is_main_process:
                logger.info(f"All {world_size_actual} GPU shards saved successfully. No concatenation performed.")
                logger.info(f"Output files: {cfg.datamodule.dataset}_{inference_type}_{cfg.dataset_generation_idx}_rank*.h5ad")
        else:
            # TEST mode - run only on main process
            if is_main_process:
                datamodule.setup("test")
                trainer = Trainer(
                    devices=1,
                    logger=list(loggers_.values()),
                    callbacks=list(callbacks_.values()),
                )
                test_results = trainer.test(module, dataloaders=datamodule.test_dataloader(), ckpt_path=ckpt_path)
                logger.info(test_results)

    except Exception:
        import traceback

        logger.error(f"An error occurred on rank {dist.get_rank()}")
        logger.error(traceback.format_exc())
        raise


@hydra.main(
    config_path="./../config/",
    config_name="inference_fm_22.yaml",  # Override with --config-name=<your_config>
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except ValueError:
        pass
    train(cfg)


if __name__ == "__main__":
    main()

# Usage examples:
# Single GPU:  torchrun --nnodes 1 --nproc-per-node 1 experiments/scripts/inference_ddp.py --config-name=inference_fm_22
# 8 GPUs:      torchrun --nnodes 1 --nproc-per-node 8 experiments/scripts/inference_ddp.py --config-name=inference_fm_22
# Multi-node:  torchrun --nnodes 2 --nproc-per-node 8 experiments/scripts/inference_ddp.py --config-name=inference_fm_22
