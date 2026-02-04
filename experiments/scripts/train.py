import os
import pathlib
from collections.abc import Callable

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

os.environ["HYDRA_FULL_ERROR"] = "1"


def train(cfg) -> None:
    import pytorch_lightning as pl
    import torch.distributed as dist
    import wandb
    from hydra.core.hydra_config import HydraConfig
    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies import DDPStrategy

    from scg_vae._train_utils import (
        compute_model_stats,
        load_validate_statedict_config,
        maybe_fix_compiled_weights,
        process_generation_output,
        setup_callbacks_and_loggers_and_paths,
        setup_datamodule_and_steps,
        setup_wandb_run,
    )
    from scg_vae._utils import world_info_from_env
    from scg_vae.logger import logger

    # Enable useful Dynamo debug logs to trace recompiles/graph breaks
    os.environ.setdefault("TORCH_LOGS", "recompiles,graph_breaks,guards")
    torch.set_float32_matmul_precision("high")

    pl.seed_everything(cfg.seed, workers=True)
    logger.info("Set distributed environment...")

    local_rank, global_rank, world_size = world_info_from_env()
    logger.info(f"LOCAL RANK {local_rank}, GLOBAL RANK {global_rank}, WORLD SIZE {world_size}")

    # TODO: maybe remove this eventually
    torch._dynamo.config.optimize_ddp = False
    torch._dynamo.config.cache_size_limit = 1000
    torch._dynamo.config.capture_scalar_outputs = True  # TODO: this is needed but should be removed and fixed upstream
    torch._dynamo.config.verbose = True
    torch.backends.cudnn.benchmark = True

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
    dist.init_process_group(backend="nccl")
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"
    is_main_process = (world_size == 1) or (global_rank == 0)

    logger.info("Instantiating datamodule...")
    datamodule = setup_datamodule_and_steps(cfg, WORLD_SIZE, cfg.training.num_epochs)
    logger.info(f"Effective number of steps {cfg.training.trainer.max_steps}")
    datamodule.setup()
    cfg.model.module.vae_optimizer.lr = cfg.model.module.vae_optimizer.lr * int(WORLD_SIZE_ENV)
    if "diffusion_optimizer" in cfg.model.module:
        cfg.model.module.diffusion_optimizer.lr = cfg.model.module.diffusion_optimizer.lr * int(WORLD_SIZE_ENV)


    is_vae_as_tokenizer = (
        hasattr(cfg.model.module, "vae_as_tokenizer") and "load_from_checkpoint" in cfg.model.module.vae_as_tokenizer
    )

    if is_vae_as_tokenizer:
        job_path = pathlib.Path(
            f"{cfg.model.module.vae_as_tokenizer.load_from_checkpoint.ckpt_path}/{cfg.model.module.vae_as_tokenizer.load_from_checkpoint.job_name}"
        )
        checkpoint_file = (
            f"epoch={cfg.model.module.vae_as_tokenizer.load_from_checkpoint.epoch}.ckpt"
            if cfg.model.module.vae_as_tokenizer.load_from_checkpoint.epoch is not None
            else "last.ckpt"
        )
        vae_checkpoints = torch.load(job_path / checkpoint_file, weights_only=False)
        vae_config = OmegaConf.load(job_path / "config.yaml")
        vae_state_dict, cfg = load_validate_statedict_config(vae_checkpoints, cfg, vae_config)

        # labels in ldm might be different than training
        logger.info(f"Found `vae_as_tokenizer` in config, loading checkpoints from: {job_path / checkpoint_file}")


    # Configure InputTransformerVAE for GPT mode if embeddings available
    gpt_input_layer = None  # ADD THIS LINE
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
            logger.info("GPT gene embeddings will be used.")


    logger.info("Instantiating module...")
    module = hydra.utils.instantiate(cfg.model.module)

    #inject GPT embeddings for VAE
    if gpt_input_layer is not None:
        module.vae_model.input_layer = gpt_input_layer
        logger.info("GPT input layer injected into model.")

    # Inject GPT embeddings for gene-KO in flow matching if enabled
    use_gpt_for_gene_ko = cfg.model.module.diffusion_model.nnet.get("use_gpt_for_gene_ko", False) if is_vae_as_tokenizer else False
    if use_gpt_for_gene_ko:
        gene_ko_class_name = cfg.model.module.diffusion_model.nnet.get("gene_ko_class_name", None)
        assert gene_ko_class_name is not None, "gene_ko_class_name must be set"

        if hasattr(datamodule, 'vocabulary_encoder'):
            vocab_encoder = datamodule.vocabulary_encoder
            if hasattr(vocab_encoder, 'gpt_gene_embeddings') and vocab_encoder.gpt_gene_embeddings is not None:
                if gene_ko_class_name in vocab_encoder.class_vocab_sizes:
                    if hasattr(vocab_encoder, 'idx2classes') and gene_ko_class_name in vocab_encoder.idx2classes:
                        gpt_embeddings = vocab_encoder.gpt_gene_embeddings
                        gene_ko_idx_to_name = vocab_encoder.idx2classes[gene_ko_class_name]
                        control_perturbation_name = cfg.model.module.diffusion_model.nnet.get("control_perturbation_name", None)

                        module.diffusion_model.nnet.set_gpt_gene_ko_embeddings(
                            gpt_embeddings,
                            gene_ko_idx_to_name,
                            control_perturbation_name,
                        )
                        logger.info(f"GPT gene embeddings injected into DiT for gene-KO class: {gene_ko_class_name}")
                        if control_perturbation_name is not None:
                            logger.info(f"Control perturbation name: {control_perturbation_name}")
                        else:
                            raise ValueError("control_perturbation_name must be set")

                        # Also set GPT embeddings on EMA model
                        if hasattr(module, 'ema_model') and hasattr(module.ema_model, 'ema_model'):
                            module.ema_model.ema_model.nnet.set_gpt_gene_ko_embeddings(
                                gpt_embeddings,
                                gene_ko_idx_to_name,
                                control_perturbation_name,
                            )
                            logger.info("GPT gene embeddings also injected into EMA model's DiT.")
                    else:
                        raise ValueError(
                            f"idx2classes mapping not found for {gene_ko_class_name}. "
                        )
                else:
                    raise ValueError(
                        f"Gene-KO class '{gene_ko_class_name}' not found in class_vocab_sizes. "
                    )
            else:
                raise ValueError(
                    "GPT embeddings not available in vocabulary encoder. "
                )
        else:
            raise ValueError(
                "Vocabulary encoder not found in datamodule. "
            )

    # if module.module_is_compiled:
    #     logger.info(f"Compiling model with {cfg.model.module.compile_mode} mode.")
    #     if hasattr(module, "diffusion_model"):
    #         module.diffusion_model_compiled = torch.compile(module.diffusion_model, mode=cfg.model.module.compile_mode, dynamic=False)
    #     if hasattr(module, "vae_model"):
    #         module.vae_model_compiled = torch.compile(module.vae_model, mode=cfg.model.module.compile_mode, dynamic=False)

    if is_vae_as_tokenizer:
        module.vae_model.load_state_dict(vae_state_dict)
        logger.info(f"VAE model loaded from checkpoints, with train mode: {cfg.model.module.vae_as_tokenizer.train}")
        # module.vae_model.eval()

        # # Compute normalizer factor for latent diffusion
        # # norm_factor_path = os.path.join(cfg.training.callbacks.model_checkpoints.dirpath, "normalize_factor.pt")
        # # if is_main_process:
        # #     logger.info("Computing normalizer factor for VAE latents...")
        # #     normalizer = module.compute_normalizer(
        # #         dataloader=datamodule.train_dataloader(),
        # #         norm_factor_path=norm_factor_path,
        # #     )
        # # else:
        # #     normalizer = 0.0  # placeholder for non-main processes

        # # Synchronize normalizer across all processes
        # if dist.is_initialized():
        #     normalizer_tensor = torch.tensor(normalizer, device=torch.cuda.current_device())
        #     dist.broadcast(normalizer_tensor, src=0)
        #     normalizer = normalizer_tensor.item()
        #     dist.barrier()

        # # Assign the synchronized normalizer to the VAE model
        # module.vae_model.norm_factor = normalizer
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
    callbacks_, loggers_, cfg = setup_callbacks_and_loggers_and_paths(cfg, override_dict)

    os.makedirs(cfg.training.callbacks.model_checkpoints.dirpath, exist_ok=True)
    logger.info(f"Checkpoints will be saved to {cfg.training.callbacks.model_checkpoints.dirpath}")

    cfg = compute_model_stats(cfg, module)
    logger.info(f"Model flops: {cfg.model.flops} parameters: {cfg.model.num_parameters}")
    override_dict["num_parameters"] = cfg.model.num_parameters
    override_dict["flops"] = cfg.model.flops

    # Save the config to the checkpoint directory (only on main process)
    if is_main_process:
        config_path = os.path.join(cfg.training.callbacks.model_checkpoints.dirpath, "config.yaml")
        with open(config_path, "w") as f:
            OmegaConf.save(cfg, f)
        logger.info(f"Configuration saved to {config_path}")

    # Synchronize all processes after config saving
    if dist.is_initialized():
        dist.barrier()

    if cfg.training.logger.wandb is not None:
        # Debug wandb configuration

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
        strategy = hydra.utils.instantiate(cfg.training.trainer.strategy)

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
        last_checkpoint = checkpoint_dir / "last.ckpt"
        ckpt_path = last_checkpoint if last_checkpoint.exists() else None
        if ckpt_path is not None:
            maybe_fix_compiled_weights(
                ckpt_path,
                diffusion_compile=cfg.model.module.compile,
                vae_compile=False,
                module_keys=list(module.state_dict().keys()),
            )

        trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)

        if dist.is_initialized():
            dist.barrier()
        if is_main_process:
            # Store EMA model checkpoint
            if hasattr(module, "ema_model"):
                ema_checkpoint_path = pathlib.Path(cfg.training.callbacks.model_checkpoints.dirpath) / "ema_model.ckpt"
                torch.save(
                    {
                        "ema_model_state_dict": module.ema_model.state_dict(),
                        "ema_model": module.ema_model,
                    },
                    ema_checkpoint_path,
                )
                logger.info(f"EMA model checkpoint saved to {ema_checkpoint_path}")
            torch.cuda.empty_cache()
            torch._dynamo.config.cache_size_limit = 1000
            # PREDICT
            if is_vae_as_tokenizer:
                module.vae_model.eval()
                module.diffusion_model.eval()
                datamodule.setup("predict")
                output = trainer.predict(
                    module,
                    datamodule.predict_dataloader(),
                )
                adata = process_generation_output(output, datamodule)
                adata.write_h5ad(
                    pathlib.Path(cfg.training.callbacks.model_checkpoints.dirpath) / "adata_test_generated.h5ad"
                )
                # Plot UMAP and log to wandb
                import matplotlib.pyplot as plt
                import scanpy as sc

                # process adata
                # Subsample to top 10k cells for each dataset type
                adata_unconditional = adata[adata.obs["dataset"] == "generated_unconditional"]
                adata_conditional = adata[adata.obs["dataset"] == "generated_conditional"]

                # Take top 10k cells from each
                n_cells_unconditional = min(10000, adata_unconditional.n_obs)
                n_cells_conditional = min(10000, adata_conditional.n_obs)

                adata_unconditional_subset = adata_unconditional[:n_cells_unconditional]
                adata_conditional_subset = adata_conditional[:n_cells_conditional]

                # Combine subsets
                adata_subset = adata_unconditional_subset.concatenate(adata_conditional_subset)

                # Process the subset
                sc.pp.normalize_total(adata_subset)
                sc.pp.log1p(adata_subset)
                sc.pp.pca(adata_subset)
                sc.pp.neighbors(adata_subset)
                sc.tl.umap(adata_subset)

                sc.pl.umap(adata_subset, color=["dataset"], show=False)
                plt.tight_layout()
                if loggers_["wandb"] is not None:
                    loggers_["wandb"].experiment.log({"umap": wandb.Image(plt)})
                plt.close()
            # TEST
            datamodule.setup("test")
            trainer = Trainer(
                devices=1,
                logger=list(loggers_.values()),
                callbacks=list(callbacks_.values()),
            )
            test_results = trainer.test(module, dataloaders=datamodule.test_dataloader())
            logger.info(test_results)

    except Exception:
        import traceback

        logger.error(f"An error occurred on rank {dist.get_rank()}")
        logger.error(traceback.format_exc())
        raise


@hydra.main(
    config_path="./../config/",
    config_name="generation_benchmark.yaml",
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

# torchrun --nnodes 1 --nproc-per-node 4 experiments/scripts/train.py
