import os

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
import umap
from pytorch_lightning.callbacks import Callback
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _create_reconstruction_visualization(model, datamodule, output_dir, max_cells=100, device="cuda"):
    """Create UMAP visualization of original vs reconstructed cells."""
    print("Creating reconstruction visualization...")
    print(f"Model type: {type(model)}")
    print(f"Has VAE model: {hasattr(model, 'vae_model')}")
    if hasattr(model, "vae_model"):
        print(f"VAE model type: {type(model.vae_model)}")
        print(f"VAE model device: {next(model.vae_model.parameters()).device}")
    # Store original training state
    original_training_state = model.training
    model.eval()
    model = model.to(device)

    # Get test dataloader
    test_dataloader = datamodule.test_dataloader()

    originals = []
    reconstructions = []
    metadata = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            if len(originals) * batch["counts"].shape[0] >= max_cells:
                break

            # Move to device
            counts = batch["counts"].to(device)
            genes = batch["genes"].to(device)

            # Get metadata
            meta = {k: v for k, v in batch.items() if k not in ["counts", "genes"]}

            # Forward pass for reconstruction
            library_size = counts.sum(1, keepdim=True)

            try:
                # Get reconstruction using VAE model (without condition parameter)
                mu, theta, z = model.vae_model.forward(counts=counts, genes=genes, library_size=library_size)

                # mu is already the mean of the reconstruction
                recon = mu

                originals.append(counts.cpu().numpy())
                reconstructions.append(recon.cpu().numpy())
                metadata.append({k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in meta.items()})

            except (RuntimeError, ValueError, TypeError) as e:
                print(f"Warning: Failed to reconstruct batch {batch_idx}: {e}")
                continue

    if not originals:
        print("Warning: No reconstructions generated")
        return

    # Combine data
    original_data = np.vstack(originals)
    # round reconstruction to nearest integer
    reconstructed_data = np.round(np.vstack(reconstructions))

    print(f"Generated reconstructions for {original_data.shape[0]} cells")

    # Normalize data
    scaler = StandardScaler()
    original_scaled = scaler.fit_transform(original_data)
    reconstructed_scaled = scaler.transform(reconstructed_data)

    # Combine both original and reconstructed data for joint UMAP fitting
    combined_data = np.vstack([original_scaled, reconstructed_scaled])

    # Create UMAP embedding fitted on both original and reconstructed data together
    reducer = umap.UMAP(n_neighbors=100, min_dist=0.1, random_state=42, n_components=2)
    combined_embedding = reducer.fit_transform(combined_data)

    # Split the embeddings back into original and reconstructed
    n_original = original_scaled.shape[0]
    original_embedding = combined_embedding[:n_original]
    reconstructed_embedding = combined_embedding[n_original:]

    # Create additional embedding approaches for comparison

    # Approach 1: UMAP fitted only on original data (consistent across epochs)
    reducer_orig = umap.UMAP(n_neighbors=100, min_dist=0.1, random_state=42, n_components=2)
    original_embedding_orig_fit = reducer_orig.fit_transform(original_scaled)
    reconstructed_embedding_orig_fit = reducer_orig.transform(reconstructed_scaled)

    # Approach 2: PCA fitted only on original data
    pca_orig = PCA(n_components=2, random_state=42)
    pca_orig.fit(original_scaled)
    original_pca_orig_fit = pca_orig.transform(original_scaled)
    reconstructed_pca_orig_fit = pca_orig.transform(reconstructed_scaled)

    # Approach 3: PCA fitted on combined data
    pca_combined = PCA(n_components=2, random_state=42)
    combined_pca_combined_fit = pca_combined.fit_transform(combined_data)
    original_pca_combined_fit = combined_pca_combined_fit[:n_original]
    reconstructed_pca_combined_fit = combined_pca_combined_fit[n_original:]

    # Create visualization with all four embedding approaches
    plt.figure(figsize=(18, 16))

    # Row 1: UMAP fitted on combined data (Joint UMAP)
    plt.subplot(4, 3, 1)
    plt.scatter(original_embedding[:, 0], original_embedding[:, 1], c="blue", alpha=0.7, s=20)
    plt.title("UMAP Joint Fit: Original")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 3, 2)
    plt.scatter(reconstructed_embedding[:, 0], reconstructed_embedding[:, 1], c="red", alpha=0.7, s=20)
    plt.title("UMAP Joint Fit: Reconstructed")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 3, 3)
    plt.scatter(original_embedding[:, 0], original_embedding[:, 1], c="blue", alpha=0.5, s=15, label="Original")
    plt.scatter(
        reconstructed_embedding[:, 0], reconstructed_embedding[:, 1], c="red", alpha=0.5, s=15, label="Reconstructed"
    )
    plt.title("UMAP Joint Fit: Overlay")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Row 2: UMAP fitted on original only
    plt.subplot(4, 3, 4)
    plt.scatter(original_embedding_orig_fit[:, 0], original_embedding_orig_fit[:, 1], c="blue", alpha=0.7, s=20)
    plt.title("UMAP Orig Fit: Original")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 3, 5)
    plt.scatter(
        reconstructed_embedding_orig_fit[:, 0], reconstructed_embedding_orig_fit[:, 1], c="red", alpha=0.7, s=20
    )
    plt.title("UMAP Orig Fit: Reconstructed")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 3, 6)
    plt.scatter(
        original_embedding_orig_fit[:, 0],
        original_embedding_orig_fit[:, 1],
        c="blue",
        alpha=0.5,
        s=15,
        label="Original",
    )
    plt.scatter(
        reconstructed_embedding_orig_fit[:, 0],
        reconstructed_embedding_orig_fit[:, 1],
        c="red",
        alpha=0.5,
        s=15,
        label="Reconstructed",
    )
    plt.title("UMAP Orig Fit: Overlay")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Row 3: PCA fitted on original only
    plt.subplot(4, 3, 7)
    plt.scatter(original_pca_orig_fit[:, 0], original_pca_orig_fit[:, 1], c="blue", alpha=0.7, s=20)
    plt.title("PCA Orig Fit: Original")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 3, 8)
    plt.scatter(reconstructed_pca_orig_fit[:, 0], reconstructed_pca_orig_fit[:, 1], c="red", alpha=0.7, s=20)
    plt.title("PCA Orig Fit: Reconstructed")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 3, 9)
    plt.scatter(original_pca_orig_fit[:, 0], original_pca_orig_fit[:, 1], c="blue", alpha=0.5, s=15, label="Original")
    plt.scatter(
        reconstructed_pca_orig_fit[:, 0],
        reconstructed_pca_orig_fit[:, 1],
        c="red",
        alpha=0.5,
        s=15,
        label="Reconstructed",
    )
    plt.title("PCA Orig Fit: Overlay")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Row 4: PCA fitted on combined data
    plt.subplot(4, 3, 10)
    plt.scatter(original_pca_combined_fit[:, 0], original_pca_combined_fit[:, 1], c="blue", alpha=0.7, s=20)
    plt.title("PCA Joint Fit: Original")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 3, 11)
    plt.scatter(reconstructed_pca_combined_fit[:, 0], reconstructed_pca_combined_fit[:, 1], c="red", alpha=0.7, s=20)
    plt.title("PCA Joint Fit: Reconstructed")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True, alpha=0.3)

    plt.subplot(4, 3, 12)
    plt.scatter(
        original_pca_combined_fit[:, 0], original_pca_combined_fit[:, 1], c="blue", alpha=0.5, s=15, label="Original"
    )
    plt.scatter(
        reconstructed_pca_combined_fit[:, 0],
        reconstructed_pca_combined_fit[:, 1],
        c="red",
        alpha=0.5,
        s=15,
        label="Reconstructed",
    )
    plt.title("PCA Joint Fit: Overlay")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle(
        f"VAE Reconstruction Quality (4 Embedding Approaches) - Epoch {output_dir.split('_')[-1] if 'epoch' in output_dir else 'Final'}",
        fontsize=16,
    )
    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "reconstruction_visualization.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Reconstruction visualization saved to: {save_path}")
    plt.close()

    # Compute metrics
    mse = np.mean((original_data - reconstructed_data) ** 2)
    correlations = []
    for i in range(original_data.shape[0]):
        corr = np.corrcoef(original_data[i], reconstructed_data[i])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)

    # Compute embedding distances for all four approaches
    umap_joint_distances = []
    umap_orig_distances = []
    pca_orig_distances = []
    pca_joint_distances = []

    for i in range(len(original_embedding)):
        # UMAP joint fit distances
        umap_joint_dist = np.linalg.norm(original_embedding[i] - reconstructed_embedding[i])
        umap_joint_distances.append(umap_joint_dist)

        # UMAP original fit distances
        umap_orig_dist = np.linalg.norm(original_embedding_orig_fit[i] - reconstructed_embedding_orig_fit[i])
        umap_orig_distances.append(umap_orig_dist)

        # PCA original fit distances
        pca_orig_dist = np.linalg.norm(original_pca_orig_fit[i] - reconstructed_pca_orig_fit[i])
        pca_orig_distances.append(pca_orig_dist)

        # PCA joint fit distances
        pca_joint_dist = np.linalg.norm(original_pca_combined_fit[i] - reconstructed_pca_combined_fit[i])
        pca_joint_distances.append(pca_joint_dist)

    print("\n" + "=" * 70)
    print("RECONSTRUCTION QUALITY METRICS")
    print("=" * 70)
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Mean Correlation: {np.mean(correlations):.4f}")
    print(f"Std Correlation: {np.std(correlations):.4f}")
    print(f"Min Correlation: {np.min(correlations):.4f}")
    print(f"Max Correlation: {np.max(correlations):.4f}")

    print("\nEmbedding Space Quality (Mean Distance orig↔recon):")
    print(f"  UMAP Joint Fit:  {np.mean(umap_joint_distances):.4f} ± {np.std(umap_joint_distances):.4f}")
    print(f"  UMAP Orig Fit:   {np.mean(umap_orig_distances):.4f} ± {np.std(umap_orig_distances):.4f}")
    print(f"  PCA Orig Fit:    {np.mean(pca_orig_distances):.4f} ± {np.std(pca_orig_distances):.4f}")
    print(f"  PCA Joint Fit:   {np.mean(pca_joint_distances):.4f} ± {np.std(pca_joint_distances):.4f}")

    print("\nExplained Variance Ratios:")
    print(f"  PCA Orig Fit:    {pca_orig.explained_variance_ratio_}")
    print(f"  PCA Joint Fit:   {pca_combined.explained_variance_ratio_}")
    print("=" * 70)

    # Restore original training state
    if original_training_state:
        model.train()
        print(f"Restored model to training mode: {model.training}")
    else:
        print(f"Model remains in eval mode: {model.training}")


def create_reconstruction_visualization(
    model, datamodule, output_dir, max_cells=100, device="cuda", target_sum=1e4, n_pcs=50, n_neighbors=15, min_dist=0.5
):
    """Create a simple UMAP visualization using scanpy preprocessing pipeline."""
    print("Creating simple UMAP visualization...")

    # Store original training state
    original_training_state = model.training
    model.eval()
    model = model.to(device)

    # Get test dataloader
    test_dataloader = datamodule.test_dataloader()

    originals = []
    reconstructions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            if len(originals) * batch["counts"].shape[0] >= max_cells:
                break

            # Move to device
            counts = batch["counts"].to(device)
            genes = batch["genes"].to(device)

            # Forward pass for reconstruction
            library_size = counts.sum(1, keepdim=True)

            try:
                # Get reconstruction using VAE model
                mu, theta, z = model.vae_model.forward(counts=counts, genes=genes, library_size=library_size)
                recon = mu

                originals.append(counts.cpu().numpy())
                reconstructions.append(recon.cpu().numpy())

            except (RuntimeError, ValueError, TypeError) as e:
                print(f"Warning: Failed to reconstruct batch {batch_idx}: {e}")
                continue

    if not originals:
        print("Warning: No reconstructions generated")
        return

    # Combine data
    original_data = np.vstack(originals)
    reconstructed_data = np.round(np.vstack(reconstructions))

    print(f"Generated reconstructions for {original_data.shape[0]} cells")

    # Create combined dataset
    combined_data = np.vstack([original_data, reconstructed_data])

    # Create labels
    n_cells = original_data.shape[0]
    labels = ["Original"] * n_cells + ["Reconstructed"] * n_cells

    # Create AnnData object
    import anndata as ad

    adata = ad.AnnData(X=combined_data)
    adata.obs["data_type"] = labels

    print("Running scanpy preprocessing pipeline...")

    # Step 1: Normalize to target sum
    print(f"  1. Normalizing to {target_sum} counts per cell...")
    sc.pp.normalize_total(adata, target_sum=target_sum)

    # Step 2: Log-transform
    print("  2. Log-transforming...")
    sc.pp.log1p(adata)

    # Step 3: Scale to unit variance and zero mean
    print("  3. Scaling...")
    sc.pp.scale(adata, max_value=10)

    # Step 4: PCA
    print(f"  4. Computing PCA ({n_pcs} components)...")
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")

    # Step 5: Compute neighborhood graph
    print(f"  5. Computing neighbor graph (n_neighbors={n_neighbors})...")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    # Step 6: Compute UMAP
    print(f"  6. Computing UMAP (min_dist={min_dist})...")
    sc.tl.umap(adata, min_dist=min_dist)

    # Step 7: Plot UMAP
    print("  7. Plotting UMAP...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set scanpy figure settings
    sc.settings.figdir = output_dir
    sc.settings.set_figure_params(dpi=300, facecolor="white")

    # Plot UMAP colored by data type
    sc.pl.umap(
        adata,
        color="data_type",
        title="VAE Reconstruction: Original vs Reconstructed",
        palette=["#1f77b4", "#ff7f0e"],  # Blue for original, orange for reconstructed
        save="_simple_reconstruction_umap.png",
        show=False,
    )

    print(f"Simple UMAP visualization saved to: {output_dir}/umap_simple_reconstruction_umap.png")

    # Restore original training state
    if original_training_state:
        model.train()
        print(f"Restored model to training mode: {model.training}")
    else:
        print(f"Model remains in eval mode: {model.training}")


class ReconstructionVisualizationCallback(Callback):
    """Callback to create reconstruction visualizations at the end of each validation epoch."""

    def __init__(self, max_cells=100, device="cuda"):
        super().__init__()
        self.max_cells = max_cells
        self.device = device

    def _create_visualization(self, trainer, pl_module, epoch_name):
        """Helper method to create visualization."""
        try:
            # Get the datamodule from the trainer
            datamodule = trainer.datamodule

            # Make sure datamodule is setup for test
            if hasattr(datamodule, "setup"):
                datamodule.setup("test")  # Use test data for reconstruction visualization

            # Get output directory from checkpoint callback
            checkpoint_callback = None
            for callback in trainer.callbacks:
                if hasattr(callback, "dirpath"):
                    checkpoint_callback = callback
                    break

            output_dir = checkpoint_callback.dirpath if checkpoint_callback else "./visualizations"

            # Create epoch-specific subdirectory
            epoch_dir = os.path.join(output_dir, epoch_name)

            print(f"Creating reconstruction visualization for {epoch_name}...")

            # Call the visualization function
            create_reconstruction_visualization(
                model=pl_module,
                datamodule=datamodule,
                output_dir=epoch_dir,
                max_cells=self.max_cells,
                device=self.device,
            )
        except (RuntimeError, ValueError, TypeError, ImportError, OSError) as e:
            print(f"Warning: Failed to create reconstruction visualization in callback: {e}")
            import traceback

            print(f"Full traceback: {traceback.format_exc()}")
        finally:
            # Always clear cache in case of errors
            torch.cuda.empty_cache()

    def on_sanity_check_end(self, trainer, pl_module):
        """Called at the end of sanity check - runs visualization once at the beginning."""
        # Only run on the main process in distributed training
        if trainer.is_global_zero:
            self._create_visualization(trainer, pl_module, "sanity_check")

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at the end of each validation epoch."""
        # Only run on the main process in distributed training
        if trainer.is_global_zero:
            self._create_visualization(trainer, pl_module, f"val_epoch_{trainer.current_epoch}")
