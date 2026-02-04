#!/usr/bin/env python3
"""
Compute log library size statistics for an AnnData dataset split across shards.

For each combination of conditioning variables, compute mean and std of
log(total counts). If a combination has no cells, fall back to coarser
conditioning variables.
"""

import argparse
import pickle
from itertools import product
from pathlib import Path

import anndata as ad
import numpy as np
import yaml
import glob


def load_condition_names(config_path, dataset_name):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    try:
        return list(
            config["dataset_params"][dataset_name]["class_vocab_sizes"].keys()
        )
    except KeyError as e:
        raise KeyError(
            f"Could not find dataset '{dataset_name}' in config: {config_path}"
        ) from e


def load_and_concat_shards(data_dir, pattern="adata_*.h5ad"):
    shard_files = sorted(glob.glob(str(Path(data_dir) / pattern)))
    if not shard_files:
        raise ValueError(f"No shard files matching {pattern} in {data_dir}")

    adatas = [ad.read_h5ad(p) for p in shard_files]
    return ad.concat(adatas, join="outer", index_unique="-")


def compute_log_library_sizes(adata):
    lib_sizes = np.asarray(adata.X.sum(axis=1)).flatten()
    if np.any(lib_sizes <= 0):
        raise ValueError("Found non-positive library sizes")
    return np.log(lib_sizes)


def compute_fallback_stats(
    adata, log_sizes, fallback_conditions
):
    stats_mu = {}
    stats_sd = {}

    for values in product(*[adata.obs[c].unique() for c in fallback_conditions]):
        mask = np.ones(len(adata), dtype=bool)
        for c, v in zip(fallback_conditions, values):
            mask &= adata.obs[c] == v

        if mask.sum() == 0:
            raise ValueError(
                f"No cells found for fallback combination: {values}"
            )

        key = "_".join(map(str, values))
        stats_mu[key] = float(log_sizes[mask].mean())
        stats_sd[key] = float(log_sizes[mask].std())

    return stats_mu, stats_sd


def compute_stats(
    adata,
    log_sizes,
    condition_names,
    fallback_conditions,
):
    joint_key = "_".join(condition_names)
    mu = {joint_key: {}}
    sd = {joint_key: {}}

    fallback_mu, fallback_sd = compute_fallback_stats(
        adata, log_sizes, fallback_conditions
    )

    unique_vals = [adata.obs[c].unique() for c in condition_names]

    for values in product(*unique_vals):
        mask = np.ones(len(adata), dtype=bool)
        for c, v in zip(condition_names, values):
            mask &= adata.obs[c] == v

        key = "_".join(map(str, values))

        if mask.sum() > 0:
            mu[joint_key][key] = float(log_sizes[mask].mean())
            sd[joint_key][key] = float(log_sizes[mask].std())
        else:
            fb_vals = [
                str(values[condition_names.index(c)])
                for c in fallback_conditions
            ]
            fb_key = "_".join(fb_vals)

            mu[joint_key][key] = fallback_mu[fb_key]
            sd[joint_key][key] = fallback_sd[fb_key]

    return mu, sd


def main(args):
    condition_names = load_condition_names(
        args.config_path, args.dataset_name
    )

    adata = load_and_concat_shards(args.data_dir)
    log_sizes = compute_log_library_sizes(adata)

    fallback_conditions = args.fallback_conditions

    mu, sd = compute_stats(
        adata,
        log_sizes,
        condition_names,
        fallback_conditions,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "log_size_factor_mu.pkl", "wb") as f:
        pickle.dump(mu, f)

    with open(output_dir / "log_size_factor_sd.pkl", "wb") as f:
        pickle.dump(sd, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument(
        "--fallback_conditions",
        nargs="+",
        default=["experimental_perturbation_time_point", "donor_id"],
    )

    main(parser.parse_args())
