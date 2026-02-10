"""
Notebook-friendly inference wrapper for SCG-VAE model.

This module provides functions to run inference from a Jupyter notebook
without needing torchrun or distributed training setup.
"""

import os
import pathlib
from typing import Optional, Dict, Any
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import compose, initialize_config_dir
from pytorch_lightning import Trainer


# Fix for PyTorch 2.6+ default weights_only=True behavior
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


class NotebookInference:
    """
    A class to handle model inference from Jupyter notebooks.

    Usage:
        ```python
        from notebook_inference import NotebookInference

        # Initialize
        inferencer = NotebookInference(
            config_path="/path/to/config/dir",
            config_name="inference_fm"
        )

        # Load model and run inference
        results = inferencer.run_inference(
            checkpoint_path="/path/to/model.safetensors",
            output_dir="/path/to/output",
            generation_idx=0,
            batch_size=32
        )
        ```
    """

    def __init__(
        self,
        config_path: str,
        config_name: str = "inference_fm_22",
        overrides: Optional[list] = None
    ):
        """
        Initialize the inference handler.

        Args:
            config_path: Path to the directory containing config files
            config_name: Name of the config file (without .yaml extension)
            overrides: List of config overrides (e.g., ["model.batch_size=16"])
        """
        self.config_path = os.path.abspath(config_path)
        self.config_name = config_name
        self.overrides = overrides or []

        # Register custom resolvers
        try:
            OmegaConf.register_new_resolver("eval", eval)
        except ValueError:
            pass

        # Set environment
        os.environ["HYDRA_FULL_ERROR"] = "1"
        torch.set_float32_matmul_precision("high")

        self.cfg = None
        self.module = None
        self.datamodule = None

    def load_config(self, additional_overrides: Optional[list] = None):
        """
        Load the configuration.

        Args:
            additional_overrides: Additional config overrides for this run
        """
        all_overrides = self.overrides + (additional_overrides or [])

        # Initialize Hydra with the config directory
        with initialize_config_dir(config_dir=self.config_path, version_base="1.2"):
            self.cfg = compose(config_name=self.config_name, overrides=all_overrides)

        print(f"✓ Configuration loaded: {self.config_name}")
        return self.cfg

    def setup_model_and_data(
        self,
        checkpoint_dir: str,
        seed: int = 42,
        dataset_generation_idx: int = 0,
        batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None
    ):
        """
        Setup the model and datamodule.

        Args:
            checkpoint_dir: Directory containing the trained model checkpoint
            seed: Random seed
            dataset_generation_idx: Index for dataset generation
            batch_size: Override batch size
            test_batch_size: Override test batch size
        """
        if self.cfg is None:
            raise ValueError("Config not loaded. Call load_config() first.")

        # Set seed
        pl.seed_everything(seed + dataset_generation_idx)

        # Load original config from checkpoint directory
        original_config_path = os.path.join(checkpoint_dir, "config.yaml")
        if not os.path.exists(original_config_path):
            raise FileNotFoundError(f"Config not found at {original_config_path}")

        print(f"Loading original config from {original_config_path}")
        original_cfg = OmegaConf.load(original_config_path)

        # Merge configs
        if hasattr(original_cfg.model.module, "vae_as_tokenizer"):
            original_cfg.model.module.vae_as_tokenizer = None

        # Override batch sizes if provided
        if batch_size is not None:
            original_cfg.model.batch_size = batch_size
        elif hasattr(self.cfg.model, 'batch_size'):
            original_cfg.model.batch_size = self.cfg.model.batch_size

        if test_batch_size is not None:
            test_batch_size_val = test_batch_size
        elif hasattr(self.cfg.model, 'test_batch_size'):
            test_batch_size_val = self.cfg.model.test_batch_size
        else:
            test_batch_size_val = original_cfg.model.batch_size

        # Copy inference/generation args
        if hasattr(self.cfg.model.module, "inference_args"):
            original_cfg.model.module.inference_args = self.cfg.model.module.inference_args
        if hasattr(self.cfg.model.module, "generation_args"):
            original_cfg.model.module.generation_args = self.cfg.model.module.generation_args

        # Update config
        self.cfg.model = original_cfg.model
        self.cfg.model.test_batch_size = test_batch_size_val

        # Setup datamodule
        print("Setting up datamodule...")
        from scg_vae._train_utils import setup_datamodule_and_steps

        self.datamodule = setup_datamodule_and_steps(
            self.cfg,
            world_size=1,  # Single GPU for notebook
            num_epochs=self.cfg.training.num_epochs
        )
        self.datamodule.setup()
        print("✓ Datamodule ready")

        # Setup model
        print("Instantiating model...")

        # Handle GPT embeddings for VAE if available
        gpt_input_layer = None
        if hasattr(self.datamodule, 'vocabulary_encoder'):
            vocab_encoder = self.datamodule.vocabulary_encoder
            if hasattr(vocab_encoder, 'gpt_gene_embeddings') and vocab_encoder.gpt_gene_embeddings is not None:
                has_masked = False if self.datamodule.sample_genes in ("none", None) else True
                input_layer_partial = hydra.utils.instantiate(self.cfg.model.module.vae_model.input_layer)
                gpt_input_layer = input_layer_partial(
                    gpt_gene_embeddings=vocab_encoder.gpt_gene_embeddings,
                    gene_idx_to_name=vocab_encoder.gene_idx_to_name,
                    has_masked_gene_tokens=has_masked,
                )
                print("✓ GPT gene embeddings configured for VAE")

        # Instantiate module
        self.module = hydra.utils.instantiate(self.cfg.model.module)

        # Inject GPT embeddings for VAE
        if gpt_input_layer is not None:
            self.module.vae_model.input_layer = gpt_input_layer
            print("✓ GPT input layer injected into VAE")

        # Inject GPT embeddings for gene-KO in diffusion model
        use_gpt_for_gene_ko = self.cfg.model.module.diffusion_model.nnet.get("use_gpt_for_gene_ko", False)
        if use_gpt_for_gene_ko:
            gene_ko_class_name = self.cfg.model.module.diffusion_model.nnet.get("gene_ko_class_name", None)

            if gene_ko_class_name and hasattr(self.datamodule, 'vocabulary_encoder'):
                vocab_encoder = self.datamodule.vocabulary_encoder
                if hasattr(vocab_encoder, 'gpt_gene_embeddings') and vocab_encoder.gpt_gene_embeddings is not None:
                    gpt_embeddings = vocab_encoder.gpt_gene_embeddings
                    gene_ko_idx_to_name = vocab_encoder.idx2classes[gene_ko_class_name]
                    control_perturbation_name = self.cfg.model.module.diffusion_model.nnet.get("control_perturbation_name", None)

                    # Inject into main model
                    self.module.diffusion_model.nnet.set_gpt_gene_ko_embeddings(
                        gpt_embeddings,
                        gene_ko_idx_to_name,
                        control_perturbation_name,
                    )
                    print(f"✓ GPT embeddings injected for gene-KO class: {gene_ko_class_name}")

                    # Inject into EMA model
                    if hasattr(self.module, 'ema_model') and hasattr(self.module.ema_model, 'ema_model'):
                        self.module.ema_model.ema_model.nnet.set_gpt_gene_ko_embeddings(
                            gpt_embeddings,
                            gene_ko_idx_to_name,
                            control_perturbation_name,
                        )
                        print("✓ GPT embeddings injected into EMA model")

        # Set to eval mode
        self.module.vae_model.eval()
        self.module.diffusion_model.eval()
        if hasattr(self.module, 'ema_model'):
            self.module.ema_model.eval()

        print("✓ Model ready")

        return self.module, self.datamodule

    def run_inference(
        self,
        checkpoint_path: str,
        output_dir: str,
        dataset_generation_idx: int = 0,
        checkpoint_dir: Optional[str] = None,
        seed: int = 42,
        batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        mode: str = "predict",  # "predict" or "test"
        device: str = "cuda"
    ):
        """
        Run inference on the model.

        Args:
            checkpoint_path: Path to the model checkpoint file
            output_dir: Directory to save inference outputs
            dataset_generation_idx: Index for dataset generation
            checkpoint_dir: Directory containing config (if None, derived from checkpoint_path)
            seed: Random seed
            batch_size: Batch size for inference
            test_batch_size: Test batch size
            mode: "predict" for generation/inference, "test" for evaluation
            device: Device to run on ("cuda" or "cpu")

        Returns:
            Results from inference (AnnData object or test metrics)
        """
        # Derive checkpoint_dir if not provided
        if checkpoint_dir is None:
            checkpoint_dir = os.path.dirname(checkpoint_path)

        # Load config if not already loaded
        if self.cfg is None:
            self.load_config()

        # Setup model and data if not already done
        if self.module is None or self.datamodule is None:
            self.setup_model_and_data(
                checkpoint_dir=checkpoint_dir,
                seed=seed,
                dataset_generation_idx=dataset_generation_idx,
                batch_size=batch_size,
                test_batch_size=test_batch_size
            )

        # --- Load checkpoint ---
        ckpt_extension = pathlib.Path(checkpoint_path).suffix.lower()

        if ckpt_extension == ".safetensors":
            from safetensors.torch import load_file as load_safetensors
            print(f"Loading SafeTensors checkpoint: {checkpoint_path}")
            state_dict = load_safetensors(checkpoint_path, device=device)
            self.module.load_state_dict(state_dict)
            ckpt_path_for_trainer = None  # already loaded
        else:
            print(f"Using standard checkpoint: {checkpoint_path}")
            ckpt_path_for_trainer = checkpoint_path  # Lightning will load it internally

        # Create output directory
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Setup trainer
        print(f"Setting up trainer (mode: {mode})...")
        trainer = Trainer(
            devices=1 if device == "cuda" else 0,
            accelerator="gpu" if device == "cuda" else "cpu",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True
        )

        # Run inference
        if mode == "predict":
            is_generation = self.cfg.model.module.generation_args is not None

            self.datamodule.setup("predict")
            print(f"Running {'generation' if is_generation else 'inference'}...")

            output = trainer.predict(
                self.module,
                datamodule=self.datamodule,
                ckpt_path=ckpt_path_for_trainer,
            )

            # Process output
            from scg_vae._train_utils import process_generation_output, process_inference_output

            if is_generation:
                adata = process_generation_output(output, self.datamodule)
                inference_type = "generated"
            else:
                adata = process_inference_output(output, self.datamodule)
                inference_type = "inference"

            adata.obs["generation_idx"] = dataset_generation_idx

            # Save output
            output_path = (
                pathlib.Path(output_dir) /
                f"{self.cfg.datamodule.dataset}_{inference_type}_{dataset_generation_idx}.h5ad"
            )
            adata.write_h5ad(output_path)
            print(f"✓ Saved {len(adata)} cells to {output_path}")

            return adata

        elif mode == "test":
            self.datamodule.setup("test")
            print("Running test...")

            test_results = trainer.test(
                self.module,
                dataloaders=self.datamodule.test_dataloader(),
                ckpt_path=checkpoint_path
            )
            print("✓ Test complete")

            return test_results

        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'predict' or 'test'")


def inference(
    config_path: str,
    config_name: str,
    checkpoint_path: str,
    output_dir: str,
    dataset_generation_idx: int = 0,
    seed: int = 42,
    batch_size: int = 32,
    device: str = "cuda",
    overrides: Optional[list] = None
):
    """
    Quick inference function for simple use cases.

    Args:
        config_path: Path to config directory
        config_name: Config file name (without .yaml)
        checkpoint_path: Path to checkpoint file
        output_dir: Output directory
        dataset_generation_idx: Generation index
        seed: Random seed
        batch_size: Batch size
        device: Device ("cuda" or "cpu")
        overrides: Config overrides

    Returns:
        AnnData object with inference results
    """
    inferencer = NotebookInference(
        config_path=config_path,
        config_name=config_name,
        overrides=overrides
    )

    return inferencer.run_inference(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        dataset_generation_idx=dataset_generation_idx,
        seed=seed,
        batch_size=batch_size,
        device=device
    )
