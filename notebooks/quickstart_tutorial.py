import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # scLDM.CD4 Quickstart Inference Tutorial
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Learning Goals
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Perform inference and generate scRNA-seq perturbation data using a pre-trained checkpoint.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prerequisites
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Compute resources:** Inference with the released pre-trained checkpoint has been tested on NVIDIA A100, H100, and A6000 GPUs. CPU-only inference is not currently supported; we recommend using at least one GPU for inference.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **General requirements:** This tutorial assumes that you have already downloaded the pre-trained checkpoint and associated config file from Hugging Face, set up a virtual environment, and updated paths in yaml files as described in the repo's README.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, import the necessary modules.
    """)
    return


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    from notebook_inference import NotebookInference, inference

    # For displaying results
    import anndata as ad
    import scanpy as sc
    import pandas as pd
    import matplotlib.pyplot as plt

    return Path, ad, inference, os, sc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Define the required paths.
    """)
    return


@app.cell
def _(Path, os):
    REPO_ROOT = Path(__file__).resolve().parent.parent
    os.chdir(REPO_ROOT)

    INFERENCE_CONFIG_PATH = str(REPO_ROOT / "experiments" / "config")
    INFERENCE_CONFIG_NAME = "inference_fm"
    CHECKPOINT_PATH = str(REPO_ROOT / "model" / "model.safetensors")
    OUTPUT_DIR = str(REPO_ROOT / "inference_outputs")
    return (
        CHECKPOINT_PATH,
        INFERENCE_CONFIG_NAME,
        INFERENCE_CONFIG_PATH,
        OUTPUT_DIR,
        REPO_ROOT,
    )


@app.cell
def _(REPO_ROOT):
    TEST_ADATA_PATH = str(
        REPO_ROOT / "quickstart_data" / "test_hvg" / "adata_1.h5ad"
    )
    return (TEST_ADATA_PATH,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run Model Inference
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Inference will generate a new adata, which will also be saved to the output directory.
    """)
    return


@app.cell
def _(
    CHECKPOINT_PATH,
    INFERENCE_CONFIG_NAME,
    INFERENCE_CONFIG_PATH,
    OUTPUT_DIR,
    inference,
):
    generated_adata = inference(
        config_path=INFERENCE_CONFIG_PATH,
        config_name=INFERENCE_CONFIG_NAME,
        checkpoint_path=CHECKPOINT_PATH,
        output_dir=OUTPUT_DIR,
        dataset_generation_idx=0,
        seed=42,
        batch_size=32,
        device="cuda",
        overrides=[
            "model.batch_size=32",
            "datamodule.datamodule.num_workers=0",  # avoid spawn pickle issue with lambdas in setup()
        ],
    )

    print(f"Generated {generated_adata.n_obs} cells")
    print(f"Features: {generated_adata.n_vars}")
    return (generated_adata,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note: if you run out of GPU memory, try reducing `batch_size`.

    2k cells run in ~4 minutes on a single H100 GPU and ~17 minutes on a single A6000 GPU
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Outputs and Visualization
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Examine the generated adata.
    """)
    return


@app.cell
def _(generated_adata):
    generated_adata
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that the generated adata contains two datasets: `generated_unconditional` and `generated_conditional`.
    """)
    return


@app.cell
def _(generated_adata):
    generated_adata.obs["dataset"].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Load test data to compare with generated data.
    """)
    return


@app.cell
def _(TEST_ADATA_PATH, sc):
    test_adata = sc.read_h5ad(TEST_ADATA_PATH)
    return (test_adata,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Add a `"dataset"` column to the test data so we can compare it with the generated data.
    """)
    return


@app.cell
def _(test_adata):
    test_adata.obs["dataset"] = "test"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Concatenate the two adatas so we can visualize the true and generated cells together.
    """)
    return


@app.cell
def _(ad, generated_adata, test_adata):
    adata = ad.concat([generated_adata, test_adata], join="outer")
    return (adata,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Run standard processing steps.
    """)
    return


@app.cell
def _(adata, sc):
    sc.pp.normalize_total(adata, target_sum=10_000)
    sc.pp.log1p(adata)
    return


@app.cell
def _(adata, sc):
    sc.pp.pca(adata)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot PCA, colored by dataset.
    """)
    return


@app.cell
def _(adata, sc):
    sc.pl.pca(adata, color="dataset", show=False, return_fig=True)
    return


@app.cell
def _(adata, sc):
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot PCA, colored by experimental time point.
    """)
    return


@app.cell
def _(adata, sc):
    sc.pl.pca(adata, color="experimental_perturbation_time_point", show=False, return_fig=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot UMAP, colored by dataset.
    """)
    return


@app.cell
def _(adata, sc):
    sc.pl.umap(adata, color="dataset", show=False, return_fig=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot UMAP, colored by experimental time point.
    """)
    return


@app.cell
def _(adata, sc):
    sc.pl.umap(adata, color="experimental_perturbation_time_point", show=False, return_fig=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot UMAP, colored by donor.
    """)
    return


@app.cell
def _(adata, sc):
    sc.pl.umap(adata, color="donor_id", show=False, return_fig=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Contact
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For issues with this tutorial, please contact Mei Knudson (knudsonm@uchicago.edu).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Special thanks to Kavita Kulkarni and Jason Perera for their consultation on this quickstart.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Responsible Use
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We are committed to advancing the responsible development and use of artificial intelligence. Please follow our [Acceptable Use Policy](https://www.google.com/url?q=https%3A%2F%2Fvirtualcellmodels.cziscience.com%2Facceptable-use-policy) when engaging with our services.
    """)
    return


if __name__ == "__main__":
    app.run()
