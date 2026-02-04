import json
import os
import pickle
import warnings
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import anndata as ad
import numpy as np
import torch
import torch.distributed as dist
from anndata import AnnData
from cellarium.ml.data import (
    DistributedAnnDataCollection,
    IterableDistributedAnnDataCollectionDataset,
)
from cellarium.ml.utilities.data import AnnDataField, convert_to_tensor
from pytorch_lightning import LightningDataModule
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader

from scg_vae._utils import get_tissue_adata_files, sort_h5ad_files
from scg_vae.constants import ModelEnum
from scg_vae.encoder import VocabularyEncoder, VocabularyEncoderSimplified
from scg_vae.logger import logger


class SimplifiedDataModule(LightningDataModule):
    def __init__(
        self,
        train_adata_path: Path,
        test_adata_path: Path,
        adata_attr: str,
        adata_key: str | None,
        vocabulary_encoder: VocabularyEncoderSimplified,
        val_as_test: bool = True,
        batch_size: int = 256,
        test_batch_size: int = 256,
        num_workers: int = 4,
        seed: int = 42,
        prefetch_factor: int = 4,
        persistent_workers: bool = True,
        drop_last_indices: bool = False,
        drop_incomplete_batch: bool = True,
        sample_genes: Literal["random", "weighted", "none"] = "none",
        genes_seq_len: int = 100,
        hvg_genes_path: str | Path | None = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        assert isinstance(vocabulary_encoder, VocabularyEncoderSimplified)

        self.vocabulary_encoder = vocabulary_encoder
        self.adata_attr = adata_attr
        self.adata_key = adata_key
        self.train_adata_path = Path(train_adata_path)
        self.test_adata_path = Path(test_adata_path)
        self.val_as_test = val_as_test
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.drop_last_indices = drop_last_indices
        self.drop_incomplete_batch = drop_incomplete_batch
        self.sample_genes = sample_genes
        self.genes_seq_len = genes_seq_len
        self.hvg_genes_path = hvg_genes_path
        self.hvg_mask = None

        # this should be done in `setup`, but we need to read the metadata to get the number of cells
        # this won't work in distributed training

        if "adata_0.h5ad" in str(self.train_adata_path):
            # Read metadata from folder
            metadata_path = os.path.join(self.train_adata_path.parent, "metadata.json")
            with open(metadata_path) as f:
                self.train_metadata = json.load(f)
            self.n_cells = self.train_metadata["n_cells"]
        else:
            train_adata = ad.read_h5ad(self.train_adata_path)
            self.n_cells = train_adata.n_obs
            self.train_metadata = None

        self._adata_inference = None

    @property
    def adata_inference(self):
        return self._adata_inference

    def setup(self, stage: str | None = None):
        if "adata_0.h5ad" in str(self.train_adata_path):
            logger.info("Using train_val_split_list from sharded train files")
            # Read metadata from folder
            train_metadata_path = os.path.join(self.train_adata_path.parent, "metadata.json")
            with open(train_metadata_path) as f:
                self.train_metadata = json.load(f)
            test_metadata_path = os.path.join(self.test_adata_path.parent, "metadata.json")
            with open(test_metadata_path) as f:
                self.test_metadata = json.load(f)
            self.train_files = sort_h5ad_files(self.train_adata_path.parent)
            self.test_files = sort_h5ad_files(self.test_adata_path.parent)
        else:
            self.train_adata = ad.read_h5ad(self.train_adata_path)
            self.train_metadata = None
            self.test_metadata = None
            self.test_adata = ad.read_h5ad(self.test_adata_path)

        if self.val_as_test:
            if self.train_metadata is None:
                self.val_adata = self.test_adata
                self.val_ann_collection = self.val_adata
                self.train_ann_collection = self.train_adata
                self.test_ann_collection = self.test_adata  # BUG fix

            else:
                self.train_ann_collection = DistributedAnnDataCollection(
                    self.train_files,
                    shard_size=self.train_metadata["shard_size"],
                    last_shard_size=self.train_metadata["last_shard_size"],
                    indices_strict=False,
                    max_cache_size=10,
                )
                self.val_ann_collection = DistributedAnnDataCollection(
                    self.test_files,
                    shard_size=self.test_metadata["shard_size"],
                    last_shard_size=self.test_metadata["last_shard_size"],
                    indices_strict=False,
                    max_cache_size=10,
                )
                self.test_ann_collection = DistributedAnnDataCollection(
                    self.test_files,
                    shard_size=self.test_metadata["shard_size"],
                    last_shard_size=self.test_metadata["last_shard_size"],
                    indices_strict=False,
                    max_cache_size=10,
                )
        else:
            # Split train files into train and validation sets
            if self.train_metadata is None:
                rng = np.random.RandomState(self.seed)
                n_cells = self.train_adata.n_obs
                n_val_cells = int(0.1 * n_cells)
                indices = np.arange(n_cells)
                resample_indices = rng.permutation(indices)

                train_indices = resample_indices[:-n_val_cells]
                val_indices = resample_indices[-n_val_cells:]

                self.val_adata = self.train_adata[val_indices]
                self.train_adata = self.train_adata[train_indices]
                self.train_ann_collection = self.train_adata
                self.val_ann_collection = self.val_adata
                self.test_ann_collection = self.test_adata
            else:
                logger.info("Using train_val_split_list from sharded train files")
                train_indices, val_indices = train_val_split_list(self.train_files, self.seed)
                self.val_files = [self.train_files[i] for i in val_indices]
                self.train_files = [self.train_files[i] for i in train_indices]
                self.train_ann_collection = DistributedAnnDataCollection(
                    self.train_files,
                    shard_size=self.train_metadata["shard_size"],
                    last_shard_size=self.train_metadata["last_shard_size"],
                    indices_strict=False,
                    max_cache_size=10,
                )
                self.val_ann_collection = DistributedAnnDataCollection(
                    self.val_files,
                    shard_size=self.train_metadata["shard_size"],
                    last_shard_size=self.test_metadata["last_shard_size"]
                    if self.val_as_test
                    else self.train_metadata["shard_size"],
                    indices_strict=False,
                    max_cache_size=10,
                )
                self.test_ann_collection = DistributedAnnDataCollection(
                    self.test_files,
                    shard_size=self.test_metadata["shard_size"],
                    last_shard_size=self.test_metadata["last_shard_size"],
                    indices_strict=False,
                    max_cache_size=10,
                )

        labels = {}
        if self.vocabulary_encoder.labels is not None and isinstance(self.vocabulary_encoder.labels, dict):
            labels = {
                label: AnnDataField(
                    attr="obs",
                    key=label,
                    convert_fn=lambda x, label=label: self.vocabulary_encoder.encode_metadata(x, label=label),
                )
                for label in self.vocabulary_encoder.labels.keys()
            }

        gene_tokens_transform = partial(
            tokenize_cells,
            encoder=self.vocabulary_encoder,
        )

        train_batch_keys = {
            ModelEnum.GENES.value: AnnDataFieldWithVarNames(
                attr=self.adata_attr,
                key=self.adata_key,
                convert_fn=cast(
                    Any,
                    lambda data_dict: gene_tokens_transform(
                        cell=data_dict["data"],
                        genes_seq_len=self.genes_seq_len,
                        var_names=data_dict["var_names"],
                        sample_genes=self.sample_genes,
                    ),
                ),
            ),
            **labels,
        }
        val_batch_keys = {
            ModelEnum.GENES.value: AnnDataFieldWithVarNames(
                attr=self.adata_attr,
                key=self.adata_key,
                convert_fn=cast(
                    Any,
                    lambda data_dict: gene_tokens_transform(
                        cell=data_dict["data"],
                        genes_seq_len=self.genes_seq_len,
                        var_names=data_dict["var_names"],
                        sample_genes=self.sample_genes,
                    ),
                ),
            ),
            **labels,
        }
        test_batch_keys = {
            ModelEnum.GENES.value: AnnDataFieldWithVarNames(
                attr=self.adata_attr,
                key=self.adata_key,
                convert_fn=cast(
                    Any,
                    lambda data_dict: gene_tokens_transform(
                        cell=data_dict["data"],
                        genes_seq_len=self.genes_seq_len,
                        var_names=data_dict["var_names"],
                        sample_genes=self.sample_genes,
                    ),
                ),
            ),
            **labels,
        }
        logger.info("Using IterableDistributedAnnDataCollectionDataset", stacklevel=2)

        dataset = partial(
            IterableDistributedAnnDataCollectionDataset,
            shuffle_seed=self.seed,
            worker_seed=None,
        )

        self.train_dataset = dataset(
            batch_keys=train_batch_keys,  # type: ignore
            dadc=self.train_ann_collection,
            shuffle=True,
            batch_size=self.batch_size,
            drop_last_indices=True,
            drop_incomplete_batch=True,
        )
        self.val_dataset = dataset(
            batch_keys=val_batch_keys,  # type: ignore
            dadc=self.val_ann_collection,
            shuffle=False,
            shuffle_seed=False,
            batch_size=self.test_batch_size,
            drop_last_indices=False,
            drop_incomplete_batch=False,
        )
        self.test_dataset = dataset(
            batch_keys=test_batch_keys,  # type: ignore
            dadc=self.test_ann_collection,
            shuffle=False,
            shuffle_seed=False,
            batch_size=self.test_batch_size,
            drop_last_indices=False,
            drop_incomplete_batch=False,
        )
        if stage == "predict":
            if self.adata_inference is not None:
                raise NotImplementedError("Inference is not supported for single anndata")
            predict_batch_keys = deepcopy(test_batch_keys)
            self.predict_dataset = dataset(
                batch_keys=predict_batch_keys,  # type: ignore
                dadc=self.test_ann_collection,
                shuffle=False,
                shuffle_seed=False,
                batch_size=self.test_batch_size,
                drop_last_indices=False,
                drop_incomplete_batch=False,
            )

        # Load HVG genes and create mask if path is provided
        if self.hvg_genes_path is not None:
            logger.info(f"Loading HVG genes from: {self.hvg_genes_path}")
            with open(self.hvg_genes_path, 'rb') as f:
                hvg_data = pickle.load(f)

            hvg_gene_names = hvg_data["gene_names"]
            all_genes = self.vocabulary_encoder.genes

            # Create boolean mask for HVG genes
            hvg_gene_set = set(hvg_gene_names)
            hvg_mask = np.array([gene in hvg_gene_set for gene in all_genes])

            # Convert to torch tensor
            self.hvg_mask = torch.tensor(hvg_mask, dtype=torch.bool)

            # Log statistics
            n_hvg_found = hvg_mask.sum()
            n_hvg_total = len(hvg_gene_names)
            n_genes_total = len(all_genes)
            if n_hvg_found < n_hvg_total:
                raise ValueError(f"Not all HVG genes found in dataset")
            logger.info(f"HVG mask created: {n_hvg_found}/{n_hvg_total} HVG genes found in dataset ({n_genes_total} total genes)")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=collate_fn,
            # batch_size=self.batch_size * 2,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=collate_fn,
            # batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            collate_fn=collate_fn,
            # batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )

    def collate_fn_annloader(
        self,
        batch,
        sample_genes: Literal["random", "weighted", "none"],
        genes_seq_len: int,
    ):
        output = tokenize_cells(batch.X, batch.var_names, self.vocabulary_encoder, genes_seq_len, sample_genes)
        output.update(
            {
                k: self.vocabulary_encoder.encode_metadata(batch.obs[k].values, label=k)
                for k in self.vocabulary_encoder.labels
            }
        )
        output = tree_map(lambda x: x.detach().clone() if torch.is_tensor(x) else torch.tensor(x), output)
        return output


class CellariumDataModule(LightningDataModule):
    """Data module for gene expression datasets."""

    """ Not currently used, but keep for large data training compatibility """

    def __init__(
        self,
        data_path: Path,
        metadata_file: str,
        vocabulary_encoder: VocabularyEncoder,
        val_as_test: bool = True,
        batch_size: int = 256,
        test_batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
        indices_strict: bool = False,
        iteration_strategy: Literal["same_order", "cache_efficient"] = "cache_efficient",
        max_cache_size: int = 10,
        index_unique: str = "-",
        prefetch_factor: int = 4,
        persistent_workers: bool = True,
        drop_last_indices: bool = False,
        drop_incomplete_batch: bool = True,
        sample_genes: Literal["random", "weighted", "none"] = "none",
        genes_seq_len: int = 100,
        cp_files: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_path = Path(data_path)
        self.metadata_file = metadata_file
        self.vocabulary_encoder = vocabulary_encoder
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.val_as_test = val_as_test
        self.seed = seed
        self.indices_strict = indices_strict
        self.iteration_strategy = iteration_strategy
        self.max_cache_size = max_cache_size
        self.index_unique = index_unique
        self.prefetch_factor = prefetch_factor
        self.cp_files = cp_files

        if "census_human" not in str(self.data_path):
            self.metadata_train = json.load(open(self.data_path / "train" / self.metadata_file))
            self.n_cells = self.metadata_train["n_cells"]
        else:
            _, self.n_cells, _ = get_tissue_adata_files(self.data_path, "train")

        self.shuffle = True  # Default value for training
        self.shuffle_seed = seed
        self.drop_last_indices = drop_last_indices
        self.drop_incomplete_batch = drop_incomplete_batch
        self.worker_seed = seed
        self.n_train = None  # Will be set in setup()
        self.persistent_workers = persistent_workers
        self._adata_inference = None

        # genes attributes
        self.sample_genes = sample_genes
        self.genes_seq_len = genes_seq_len

    @property
    def adata_inference(self):
        return self._adata_inference

    @adata_inference.setter
    def adata_inference(self, adata: AnnData):
        # Filter genes based on vocabulary_encoder.genes
        if hasattr(self.vocabulary_encoder, "genes") and self.vocabulary_encoder.genes is not None:
            available_genes = set(adata.var_names)
            required_genes = set(self.vocabulary_encoder.genes)

            missing_genes = required_genes - available_genes
            logger.info(f"Missing genes in adata_inference: {len(missing_genes)}")

            genes = list(available_genes & required_genes)
            adata = adata[:, genes].copy()
        self._adata_inference = adata

    def setup(
        self,
        stage: str | None = None,
    ):
        if self.cp_files and dist.is_available() and dist.is_initialized():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))

            # First synchronize all processes before copying data
            dist.barrier()

            # Only rank 0 copies the data and prepares the path string
            if local_rank == 0:
                new_data_path = self._copy_data_files_mbuffer()
                logger.info(f"New data path: {new_data_path}")
                data_path_str = str(new_data_path)
            else:
                data_path_str = None

            # Broadcast the new data path from rank 0 to all ranks
            data_path_list = [data_path_str]
            dist.broadcast_object_list(data_path_list, src=0)
            if data_path_list[0] is not None:
                self.data_path = Path(data_path_list[0])

            # Final sync before proceeding
            dist.barrier()

        if "census_human" not in str(self.data_path):
            self.train_files = sort_h5ad_files(self.data_path / "train")  # [-10:]
            self.test_files = sort_h5ad_files(self.data_path / "test")
            self.metadata_train = json.load(open(self.data_path / "train" / self.metadata_file))
            self.metadata_test = json.load(open(self.data_path / "test" / self.metadata_file))
        else:
            self.train_files, n_cells_train, shard_size_train = get_tissue_adata_files(self.data_path, "train")
            self.test_files, n_cells_val, shard_size_val = get_tissue_adata_files(self.data_path, "test")
            self.train_files = self.train_files
            self.test_files = self.test_files
            self.metadata_train = {
                "n_cells": n_cells_train,
                "shard_size": shard_size_train,
                "last_shard_size": shard_size_train,
            }
            self.metadata_test = {
                "n_cells": n_cells_val,
                "shard_size": shard_size_val,
                "last_shard_size": shard_size_val,
            }

        if self.val_as_test:
            self.val_files = self.test_files
        else:
            # Split train files into train and validation sets
            train_indices, val_indices = train_val_split_list(self.train_files, self.seed)
            self.val_files = [self.train_files[i] for i in val_indices]
            self.train_files = [self.train_files[i] for i in train_indices]

        if len(self.train_files) > 1:
            assert self.metadata_train["shard_size"] == self.metadata_test["shard_size"], "shard_size mismatch"
        self.last_shard_size_train = self.metadata_train["last_shard_size"]
        self.last_shard_size_test = self.metadata_test["last_shard_size"]
        self.shard_size = self.metadata_train["shard_size"]
        self.n_cells = self.metadata_train["n_cells"]
        self.n_train = self.n_cells

        self.train_ann_collection = DistributedAnnDataCollection(
            self.train_files,
            shard_size=self.shard_size,
            last_shard_size=self.last_shard_size_train,
            # convert=self.dataset_encoder,
            indices_strict=self.indices_strict,
            max_cache_size=self.max_cache_size,
        )
        self.val_ann_collection = DistributedAnnDataCollection(
            self.val_files,
            shard_size=self.shard_size,
            last_shard_size=self.last_shard_size_test if self.val_as_test else self.shard_size,
            # convert=self.dataset_encoder,
            indices_strict=self.indices_strict,
            max_cache_size=self.max_cache_size,
        )
        self.test_ann_collection = DistributedAnnDataCollection(
            self.test_files,
            shard_size=self.shard_size,
            last_shard_size=self.last_shard_size_test,
            # convert=self.dataset_encoder,
            indices_strict=self.indices_strict,
            max_cache_size=self.max_cache_size,
        )

        labels = {}
        if self.vocabulary_encoder.labels is not None and isinstance(self.vocabulary_encoder.labels, dict):
            labels = {
                label: AnnDataField(
                    attr="obs",
                    key=label,
                    convert_fn=lambda x, label=label: self.vocabulary_encoder.encode_metadata(x, label=label),
                )
                for label in self.vocabulary_encoder.labels.keys()
            }

        gene_tokens_transform = partial(
            tokenize_cells,
            encoder=self.vocabulary_encoder,
        )

        train_batch_keys = {
            ModelEnum.GENES.value: AnnDataFieldWithVarNames(
                attr="X",
                convert_fn=cast(
                    Any,
                    lambda data_dict: gene_tokens_transform(
                        cell=data_dict["data"],
                        genes_seq_len=self.genes_seq_len,
                        var_names=data_dict["var_names"],
                        sample_genes=self.sample_genes,
                    ),
                ),
            ),
            **labels,
        }
        val_batch_keys = {
            ModelEnum.GENES.value: AnnDataFieldWithVarNames(
                attr="X",
                convert_fn=cast(
                    Any,
                    lambda data_dict: gene_tokens_transform(
                        cell=data_dict["data"],
                        var_names=data_dict["var_names"],
                        genes_seq_len=self.genes_seq_len,
                        sample_genes=self.sample_genes,
                    ),
                ),
            ),
            **labels,
        }
        test_batch_keys = {
            ModelEnum.GENES.value: AnnDataFieldWithVarNames(
                attr="X",
                convert_fn=cast(
                    Any,
                    lambda data_dict: gene_tokens_transform(
                        cell=data_dict["data"],
                        var_names=data_dict["var_names"],
                        genes_seq_len=self.genes_seq_len,
                        sample_genes="none",
                    ),
                ),
            ),
            **labels,
        }
        logger.info("Using IterableDistributedAnnDataCollectionDataset", stacklevel=2)
        dataset = partial(
            IterableDistributedAnnDataCollectionDataset,
            iteration_strategy=self.iteration_strategy,
            shuffle_seed=self.seed,
            worker_seed=None,
        )

        self.train_dataset = dataset(
            batch_keys=train_batch_keys,  # type: ignore
            dadc=self.train_ann_collection,
            shuffle=True,
            batch_size=self.batch_size,
            drop_last_indices=self.drop_last_indices,
            drop_incomplete_batch=self.drop_incomplete_batch,
        )
        self.val_dataset = dataset(
            batch_keys=val_batch_keys,  # type: ignore
            dadc=self.val_ann_collection,
            shuffle=False,
            shuffle_seed=False,
            batch_size=self.batch_size,
            drop_last_indices=False,
            drop_incomplete_batch=False,
        )
        self.test_dataset = dataset(
            batch_keys=test_batch_keys,  # type: ignore
            dadc=self.test_ann_collection,
            shuffle=False,
            shuffle_seed=False,
            batch_size=self.test_batch_size,
            drop_last_indices=False,
            drop_incomplete_batch=False,
        )
        logger.info(f"Train dataset size: {len(self.train_dataset)}")
        logger.info(f"Val dataset size: {len(self.val_dataset)}")
        logger.info(f"Test dataset size: {len(self.test_dataset)}")

        # Add distributed logging
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        train_size = len(self.train_dataset)
        val_size = len(self.val_dataset)

        logger.info(f"Rank {rank}/{world_size} - Train dataset size: {train_size}")
        logger.info(f"Rank {rank}/{world_size} - Val dataset size: {val_size}")

        # Calculate steps per epoch
        train_steps = train_size // self.batch_size
        val_steps = val_size // self.batch_size

        logger.info(f"Rank {rank}/{world_size} - Train steps per epoch: {train_steps}")
        logger.info(f"Rank {rank}/{world_size} - Val steps per epoch: {val_steps}")
        self.train_steps = train_steps
        self.val_steps = val_steps

        if stage == "predict":
            if self.adata_inference is not None:
                assert isinstance(self.adata_inference, AnnData), "adata_inference must be set before calling setup"
                predict_batch_keys = {
                    # ModelEnum.COUNTS.value: AnnDataField(attr="X", convert_fn=lambda x: x.toarray()),
                    ModelEnum.GENES.value: AnnDataFieldWithVarNames(
                        attr="X",
                        convert_fn=cast(
                            Any,
                            lambda data_dict: gene_tokens_transform(
                                cell=data_dict["data"],
                                var_names=data_dict["var_names"],
                                genes_seq_len=self.genes_seq_len,
                                sample_genes="none",
                            ),
                        ),
                    ),
                    **labels,
                }
                self.predict_dataset = dataset(
                    batch_keys=predict_batch_keys,  # type: ignore
                    dadc=self.adata_inference,
                    shuffle=False,
                    shuffle_seed=False,
                    batch_size=self.batch_size,
                )
            else:
                self.predict_dataset = self.test_dataset

    def train_dataloader(self):
        if dist.is_initialized():
            rank = dist.get_rank()
            logger.info(f"Rank {rank} - Creating train dataloader, dataset size: {len(self.train_dataset)}")
        return DataLoader(
            self.train_dataset,
            collate_fn=collate_fn,
            # batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        if dist.is_initialized():
            rank = dist.get_rank()
            logger.info(f"Rank {rank} - Creating val dataloader, dataset size: {len(self.val_dataset)}")
        return DataLoader(
            self.val_dataset,
            collate_fn=collate_fn,
            # batch_size=self.batch_size * 2,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=collate_fn,
            # batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            collate_fn=collate_fn,
            # batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )

    def state_dict(self) -> dict[str, Any]:
        """Returns the state dict of the data module for checkpointing."""
        assert self.trainer is not None
        state = {
            "iteration_strategy": self.iteration_strategy,
            "num_workers": self.num_workers,
            "num_replicas": self.trainer.num_devices,
            "num_nodes": self.trainer.num_nodes,
            "batch_size": self.batch_size,
            "accumulate_grad_batches": self.trainer.accumulate_grad_batches,
            "shuffle": self.shuffle,
            "shuffle_seed": self.shuffle_seed,
            "drop_last_indices": self.drop_last_indices,
            "drop_incomplete_batch": self.drop_incomplete_batch,
            "n_train": self.n_train,
            "worker_seed": self.worker_seed,
            "epoch": self.trainer.current_epoch,
            "resume_step": self.trainer.global_step,
            "prefetch_factor": self.prefetch_factor,
            "persistent_workers": self.persistent_workers,
            "max_cache_size": self.max_cache_size,
            "indices_strict": self.indices_strict,
            "index_unique": self.index_unique,
            "vocabulary_encoder": pickle.dumps(self.vocabulary_encoder),
        }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Loads the state dict of the data module for resuming training."""
        if hasattr(self, "train_dataset"):
            assert self.trainer is not None
            if state_dict["iteration_strategy"] != self.iteration_strategy:
                raise ValueError(
                    "Cannot resume training with a different iteration strategy. "
                    f"Expected {self.iteration_strategy}, got {state_dict['iteration_strategy']}."
                )
            if state_dict["num_workers"] != self.num_workers:
                raise ValueError(
                    "Cannot resume training with a different number of workers. "
                    f"Expected {self.num_workers}, got {state_dict['num_workers']}."
                )
            if state_dict["num_replicas"] != self.trainer.num_devices:
                raise ValueError(
                    "Cannot resume training with a different number of replicas. "
                    f"Expected {self.trainer.num_devices}, got {state_dict['num_replicas']}."
                )
            if state_dict["num_nodes"] != self.trainer.num_nodes:
                raise ValueError(
                    "Cannot resume training with a different number of nodes. "
                    f"Expected {self.trainer.num_nodes}, got {state_dict['num_nodes']}."
                )
            if state_dict["batch_size"] != self.batch_size:
                raise ValueError(
                    "Cannot resume training with a different batch size. "
                    f"Expected {self.batch_size}, got {state_dict['batch_size']}."
                )
            if state_dict["accumulate_grad_batches"] != 1:
                raise ValueError("Training with gradient accumulation is not supported when resuming training.")
            if state_dict["shuffle"] != self.shuffle:
                raise ValueError(
                    "Cannot resume training with a different shuffle value. "
                    f"Expected {self.shuffle}, got {state_dict['shuffle']}."
                )
            if state_dict["shuffle_seed"] != self.shuffle_seed:
                raise ValueError(
                    "Cannot resume training with a different shuffle seed. "
                    f"Expected {self.shuffle_seed}, got {state_dict['shuffle_seed']}."
                )
            if state_dict["drop_last_indices"] != self.drop_last_indices:
                raise ValueError(
                    "Cannot resume training with a different drop_last_indices value. "
                    f"Expected {self.drop_last_indices}, got {state_dict['drop_last_indices']}."
                )
            if state_dict["drop_incomplete_batch"] != self.drop_incomplete_batch:
                raise ValueError(
                    "Cannot resume training with a different drop_incomplete_batch value. "
                    f"Expected {self.drop_incomplete_batch}, got {state_dict['drop_incomplete_batch']}."
                )
            if state_dict["n_train"] != self.n_train:
                raise ValueError(
                    "Cannot resume training with a different train size. "
                    f"Expected {self.n_train}, got {state_dict['n_train']}."
                )
            if (self.worker_seed is not None) and (state_dict["worker_seed"] == self.worker_seed):
                warnings.warn(
                    "Resuming training with the same worker seed as the previous run. "
                    "This may lead to repeated behavior in the workers upon resuming training.",
                    stacklevel=2,
                )

            # Load the dataset state dict and synchronize epoch
            # Get the saved epoch from checkpoint - this is the epoch we want to resume from
            saved_epoch = state_dict.get("epoch", 0)
            trainer_epoch = self.trainer.current_epoch
            logger.info(f"Resuming training: checkpoint epoch={saved_epoch}, trainer epoch={trainer_epoch}")

            # The checkpoint epoch is the correct epoch to resume from
            # The trainer's current_epoch will be 0 at this point since it hasn't loaded its state yet
            resume_epoch = saved_epoch

            if hasattr(self.train_dataset, "load_state_dict") and callable(self.train_dataset.load_state_dict):
                self.train_dataset.load_state_dict(state_dict)
                logger.info("Successfully loaded dataset state dict")
            else:
                logger.warning(
                    "Train dataset does not have a load_state_dict method, skipping dataset state restoration"
                )

            # Synchronize dataset epochs with the checkpoint epoch (not trainer epoch)
            # This ensures datasets are at the correct epoch to resume training
            logger.info(f"Synchronizing all dataset epochs to checkpoint epoch {resume_epoch}")

            # Synchronize train dataset epoch
            set_epoch = getattr(self.train_dataset, "set_epoch", None)
            if callable(set_epoch):
                logger.info(f"Setting train dataset epoch to {resume_epoch}")
                set_epoch(resume_epoch)
            else:
                logger.warning("Train dataset does not have set_epoch method")

            # Also reset the resume step to ensure clean state
            set_resume_step = getattr(self.train_dataset, "set_resume_step", None)
            if callable(set_resume_step):
                logger.info("Resetting train dataset resume step to None")
                set_resume_step(None)

            # Synchronize validation dataset epoch
            set_epoch_val = getattr(self.val_dataset, "set_epoch", None)
            if callable(set_epoch_val):
                logger.info(f"Setting val dataset epoch to {resume_epoch}")
                set_epoch_val(resume_epoch)
            else:
                logger.warning("Val dataset does not have set_epoch method")

            # Also reset the resume step for validation dataset
            set_resume_step_val = getattr(self.val_dataset, "set_resume_step", None)
            if callable(set_resume_step_val):
                logger.info("Resetting val dataset resume step to None")
                set_resume_step_val(None)

            # Synchronize test dataset epoch
            set_epoch_test = getattr(self.test_dataset, "set_epoch", None)
            if callable(set_epoch_test):
                logger.info(f"Setting test dataset epoch to {resume_epoch}")
                set_epoch_test(resume_epoch)
            else:
                logger.warning("Test dataset does not have set_epoch method")

            # Also reset the resume step for test dataset
            set_resume_step_test = getattr(self.test_dataset, "set_resume_step", None)
            if callable(set_resume_step_test):
                logger.info("Resetting test dataset resume step to None")
                set_resume_step_test(None)

            # Load the vocabulary encoder
            self.vocabulary_encoder = pickle.loads(state_dict["vocabulary_encoder"])

    # def _copy_data_files(self):
    #     """Copy data files to /tmp on node 0 using rsync."""
    #     import subprocess
    #     from pathlib import Path

    #     logger.info(f"Copying data from {self.data_path} to /tmp on node {self.trainer.node_rank}")

    #     # The new data path will be in /tmp. Let's use the same name as the original data path.
    #     new_data_path = Path("/tmp") / self.data_path.name

    #     if new_data_path.exists():
    #         subprocess.run(["rm", "-rf", str(new_data_path)], check=True)

    #     # We're copying the contents of self.data_path, so add a trailing slash.
    #     src_path_str = str(self.data_path) + "/"

    #     # Run rsync with progress.
    #     # Dst path is `new_data_path`. rsync will create it.
    #     cmd = [
    #         "rsync",
    #         "-axH",
    #         "--numeric-ids",
    #         "--inplace",
    #         "--no-compress",
    #         "--progress",
    #         "--info=progress2",
    #         src_path_str,
    #         str(new_data_path),
    #     ]
    #     logger.info(f"Running rsync command: {' '.join(cmd)}")

    #     try:
    #         result = subprocess.run(cmd, check=True)
    #         if result.returncode != 0:
    #             raise RuntimeError(f"rsync failed with return code {result.returncode}")
    #     except subprocess.CalledProcessError as e:
    #         raise RuntimeError(f"rsync failed: {str(e)}") from e

    #     return new_data_path

    # def _copy_data_files(self):
    #     """Ultra-fast copy for DGX: network filesystem -> local NVMe SSD."""
    #     import subprocess
    #     from pathlib import Path

    #     logger.info(f"Fast copying {self.data_path} to /tmp (network fs -> local SSD)")

    #     new_data_path = Path("/tmp") / self.data_path.name

    #     if new_data_path.exists():
    #         subprocess.run(["rm", "-rf", str(new_data_path)], check=True)

    #     # tar with progress - much faster than rsync for first-time copies
    #     cmd = f"tar -cf - -C {self.data_path.parent} {self.data_path.name} | pv -s $(du -sb {self.data_path} | cut -f1) | tar -xf - -C /tmp"

    #     logger.info(f"Running fast copy: {cmd}")

    #     try:
    #         result = subprocess.run(cmd, shell=True, check=True)
    #         logger.info("Fast copy completed successfully")
    #         return new_data_path
    #     except subprocess.CalledProcessError as e:
    #         raise RuntimeError(f"Fast copy failed: {str(e)}") from e

    def _copy_data_files_mbuffer(self):
        """Maximum performance copy using mbuffer."""
        # Install mbuffer first: apt-get install mbuffer
        import os
        from pathlib import Path

        new_data_path = Path("/tmp") / self.data_path.name
        os.makedirs(new_data_path, exist_ok=True)

        def get_all_files(directory: str | Path) -> set[str]:
            """Get all files and their relative paths in a directory."""
            files = set()
            for root, _, filenames in os.walk(directory):
                rel_root = os.path.relpath(root, directory)
                for filename in filenames:
                    rel_path = os.path.join(rel_root, filename)
                    if rel_path.startswith("./"):
                        rel_path = rel_path[2:]
                    files.add(rel_path)
            return files

        # Check if destination exists and has identical file structure
        files_match = False
        if new_data_path.exists():
            source_files = get_all_files(self.data_path)
            dest_files = get_all_files(new_data_path)
            files_match = source_files == dest_files

        # If files don't match or destination doesn't exist, do a fresh copy
        if not files_match:
            logger.info(f"Files don't match, doing a fresh copy from {self.data_path} to {new_data_path}")
        #     if new_data_path.exists():
        #         subprocess.run(["rm", "-rf", str(new_data_path)], check=True)

        #     cmd = (
        #         f"tar -cf - -C {self.data_path.parent} {self.data_path.name} | mbuffer -s 1M -m 32G | tar -xf - -C /tmp"
        #     )

        #     try:
        #         subprocess.run(cmd, shell=True, check=True)
        #     except subprocess.CalledProcessError as e:
        #         raise RuntimeError(f"mbuffer copy failed: {str(e)}") from e
        else:
            logger.info(f"Files match, skipping copy from {self.data_path} to {new_data_path}")

        return new_data_path


def collate_fn(
    batch: list[dict[str, dict[str, np.ndarray] | np.ndarray]],
) -> dict[str, dict[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor]:
    keys = batch[0].keys()
    collated_batch: dict[str, dict[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor] = {}
    if len(batch) > 1 and not all(keys == data.keys() for data in batch[1:]):
        raise ValueError("All dictionaries in the batch must have the same keys.")
    for key in keys:
        if key == ModelEnum.GENES.value:
            collated_batch[ModelEnum.COUNTS.value] = np.concatenate(
                [data[key][ModelEnum.COUNTS.value] for data in batch],
                axis=0,  # type: ignore
            )
            collated_batch[ModelEnum.GENES.value] = np.concatenate(
                [data[key][ModelEnum.GENES.value] for data in batch],
                axis=0,  # type: ignore
            )
            collated_batch[ModelEnum.LIBRARY_SIZE.value] = np.concatenate(
                [data[key][ModelEnum.LIBRARY_SIZE.value] for data in batch],
                axis=0,  # type: ignore
            )
            # Optional extras if provided by tokenizer
            if ModelEnum.GENES_SUBSET.value in batch[0][key]:  # type: ignore
                collated_batch[ModelEnum.GENES_SUBSET.value] = np.concatenate(
                    [data[key][ModelEnum.GENES_SUBSET.value] for data in batch],
                    axis=0,  # type: ignore
                )
            if ModelEnum.COUNTS_SUBSET.value in batch[0][key]:  # type: ignore
                collated_batch[ModelEnum.COUNTS_SUBSET.value] = np.concatenate(
                    [data[key][ModelEnum.COUNTS_SUBSET.value] for data in batch],
                    axis=0,  # type: ignore
                )
            continue
        if isinstance(batch[0][key], dict):
            subkeys = batch[0][key].keys()  # type: ignore
            if len(batch) > 1 and not all(subkeys == data[key].keys() for data in batch[1:]):  # type: ignore
                raise ValueError(f"All '{key}' sub-dictionaries in the batch must have the same subkeys.")
            # Concatenate all subkeys regardless of their suffix
            value = {
                subkey: np.concatenate([data[key][subkey] for data in batch], axis=0)
                for subkey in subkeys  # type: ignore
            }
        elif key.endswith("_g") or key.endswith("_categories"):
            # Check that all values are the same
            if len(batch) > 1:
                if not all(np.array_equal(batch[0][key], data[key]) for data in batch[1:]):
                    raise ValueError(f"All dictionaries in the batch must have the same {key}.")
            value = batch[0][key]
        else:
            value = np.concatenate([data[key] for data in batch], axis=0)  # type: ignore

        collated_batch[key] = value
    return tree_map(convert_to_tensor, collated_batch)


def tokenize_cells(
    cell: np.ndarray,
    var_names: Sequence[str],
    encoder: VocabularyEncoder | VocabularyEncoderSimplified,
    genes_seq_len: int,
    sample_genes: Literal["random", "weighted", "expressed", "none"],
    gene_tokens_key: str = ModelEnum.GENES.value,
    counts_key: str = ModelEnum.COUNTS.value,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    counts = cell
    gene_idx = np.tile(encoder.encode_genes(var_names), (len(counts), 1))
    library_size = counts.sum(1, keepdims=True)

    # assert len(gene_idx) == len(counts), "var_names and counts must have the same length"
    if sample_genes == "weighted":
        rng = np.random.default_rng(seed=seed)
        # divide by mean so it's weighted against dataset
        # add 1 to avoid 0 probability, which would cause an error in sampling below
        if encoder.metadata_genes is not None:
            scaled_counts = (counts + 1) / encoder.metadata_genes["means"].values
            scaled_counts = scaled_counts / scaled_counts.sum(1, keepdims=True)  # divide by sum so it's a proportion
            sampled_idx = np.stack(
                [
                    rng.choice(np.arange(gene_idx.shape[1]), size=genes_seq_len, replace=False, p=p)
                    for p in scaled_counts
                ]
            )
            gene_idx = np.take_along_axis(gene_idx, sampled_idx, axis=1)
            counts = np.take_along_axis(counts, sampled_idx, axis=1)
            return {
                gene_tokens_key: gene_idx,
                counts_key: counts,
                "library_size": library_size,
            }
        else:
            raise ValueError("encoder.metadata_genes must be set for weighted sampling")
    elif sample_genes == "expressed":
        # Take all expressed genes in order and pad to genes_seq_len with mask/zeros.
        mask_idx = encoder.mask_token_idx

        # # Single-cell vector
        # if counts.ndim == 1:
        #     n_genes = counts.shape[0]
        #     if genes_seq_len > n_genes:
        #         raise ValueError("genes_seq_len cannot exceed number of genes")

        #     expressed_idx = np.nonzero(counts > 0)[0]
        #     expr_gene_idx = gene_idx[expressed_idx]
        #     expr_counts = counts[expressed_idx]

        #     pad_len = genes_seq_len - expr_gene_idx.shape[0]
        #     if pad_len < 0:
        #         raise ValueError("genes_seq_len is smaller than number of expressed genes")

        #     padded_gene_idx = np.pad(expr_gene_idx, (0, pad_len), constant_values=mask_idx)
        #     padded_counts = np.pad(expr_counts, (0, pad_len), constant_values=0)

        #     return {
        #         gene_tokens_key: gene_idx[None, :],
        #         counts_key: counts[None, :],
        #         ModelEnum.GENES_SUBSET.value: padded_gene_idx[None, :],
        #         ModelEnum.COUNTS_SUBSET.value: padded_counts[None, :],
        #         "library_size": library_size,
        #     }

        # Batch (2D) input
        expressed = counts > 0  # (N, G) bool
        num_expressed = expressed.sum(axis=1)  # (N,)

        # Keep the same invariant as your loop (error if too many expressed genes)
        if (num_expressed > genes_seq_len).any():
            raise ValueError("genes_seq_len is smaller than number of expressed genes in a row")

        # Rank positions of expressed genes within each row: 0..k-1 where True, undefined where False
        pos_order = expressed.cumsum(axis=1) - 1  # (N, G) int, valid only where expressed is True

        # Outputs prefilled with padding
        N = counts.shape[0]
        genes_out = np.full((N, genes_seq_len), mask_idx, dtype=gene_idx.dtype)
        counts_out = np.zeros((N, genes_seq_len), dtype=counts.dtype)

        # Indices to scatter (only expressed entries)
        ii, jj = np.where(expressed)  # flat indices of True entries
        pp = pos_order[expressed]  # slot within [0, genes_seq_len)

        # Scatter gathered values into compacted layout
        genes_out[ii, pp] = gene_idx[ii, jj]
        counts_out[ii, pp] = counts[ii, jj]

        return {
            gene_tokens_key: gene_idx,
            counts_key: counts,
            ModelEnum.GENES_SUBSET.value: genes_out,
            ModelEnum.COUNTS_SUBSET.value: counts_out,
            "library_size": library_size,
        }
    elif sample_genes == "random":
        rng = np.random.default_rng(seed=seed)
        N, G = gene_idx.shape

        # sample per cell
        sampled_idx = np.stack([rng.choice(G, size=genes_seq_len, replace=False) for _ in range(N)])

        # per-cell mask on full gene set: True = unsampled
        mask = np.ones((N, G), dtype=bool)
        mask[np.arange(N)[:, None], sampled_idx] = False

        # indices of unsampled genes per cell
        # rest_idx = np.where(mask)[1].reshape(N, G - genes_seq_len)

        # gather
        # sampled_gene_idx = np.take_along_axis(gene_idx, sampled_idx, axis=1)
        # rest_gene_idx = np.take_along_axis(gene_idx, rest_idx, axis=1)

        # subset_counts = np.take_along_axis(counts, sampled_idx, axis=1)
        # remaining_counts = library_size - subset_counts.sum(1, keepdims=True)

        return {
            # gene_tokens_key: sampled_gene_idx,
            # counts_key: subset_counts,
            "library_size": library_size,
        }
    elif sample_genes == "none":
        return {
            gene_tokens_key: gene_idx,
            counts_key: counts,
            "library_size": library_size,
        }
    else:
        raise ValueError(f"Invalid sample_genes value: {sample_genes}")


@dataclass
class AnnDataFieldWithVarNames:
    """
    Custom AnnDataField that returns both the data and var_names.

    This is useful when you need both the X data and the corresponding var_names
    for processing, such as for tokenization where you need the actual gene names.
    """

    attr: str
    key: list[str] | str | None = None
    convert_fn: Callable[[dict[str, Any]], np.ndarray] | None = None

    def __call__(self, adata: AnnData) -> np.ndarray:
        from operator import attrgetter

        value = attrgetter(self.attr)(adata)
        if self.key is not None:
            value = value[self.key]

        # Create a dictionary with both the data and var_names
        # data_dict = {"data": value.toarray(), "var_names": adata.var_names}

        # Handle both sparse and dense arrays
        if hasattr(value, "toarray"):
            # It's a sparse matrix
            data_array = value.toarray()
        else:
            # It's already a dense array
            data_array = np.asarray(value)

        # Create a dictionary with both the data and var_names
        data_dict = {"data": data_array, "var_names": adata.var_names}

        if self.convert_fn is not None:
            return self.convert_fn(data_dict)
        else:
            return np.asarray(data_dict["data"])


def train_val_split_list(files: list[str], seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.RandomState(seed)
    n_files = len(files)
    n_val_files = int(0.1 * n_files)
    # Only resample from first 50% of files to avoid last file with different cell count
    n_resample = n_files // 2
    indices = np.arange(n_files)
    resample_indices = rng.permutation(n_resample)
    train_indices_arr = np.concatenate([resample_indices[:-n_val_files], indices[n_resample:]])
    val_indices_arr = resample_indices[-n_val_files:]
    return train_indices_arr.tolist(), val_indices_arr.tolist()
