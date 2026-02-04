import pandas as pd
import numpy as np
import os
import anndata as ad
import pickle

def perturbation_fewshot_train_test_split_rnd_pert_holdout(
    adata_backed,
    test_cell_types,
    perturbation_col,
    control_perturbation_name,
    pert_holdout_ratio=0.2,
    cell_type_col="cell_type",  # assuming standard column name
    random_state=42,
):
    """
    Create train/test split by holding out random subset of perturbations for specific cell types.
    Note: Control perturbation is not held out and by default it goes to the training set but can/should be changed!

    Parameters:
    -----------
    adata_backed : AnnData (backed)
        Backed AnnData object to avoid memory issues
    test_cell_types : list
        List of cell type names to apply perturbation holdout to
    perturbation_col : str
        Column name containing perturbation information (e.g., 'cytokines')
    control_perturbation_name : str
        Name of control perturbation to exclude from holdout (e.g., 'PBS')
    pert_holdout_ratio : float, default 0.2
        Fraction of non-control perturbations to hold out for testing
    cell_type_col : str, default 'cell_type'
        Column name containing cell type information
    random_state : int, default 42
        Random seed for reproducibility

    Returns:
    --------
    train_indices : np.array
        Indices of training cells
    test_indices : np.array
        Indices of test cells
    holdout_info : dict
        Information about held-out perturbations per cell type
    """

    np.random.seed(random_state)

    print("🔬 Creating perturbation-aware train/test split...")
    print(f"   - Test cell types: {test_cell_types}")
    print(f"   - Perturbation column: {perturbation_col}")
    print(f"   - Control perturbation: {control_perturbation_name}")
    print(f"   - Holdout ratio: {pert_holdout_ratio}")

    # Get observation metadata (this is memory efficient)
    obs_df = adata_backed.obs

    test_indices = []
    holdout_info = {}

    # Process each test cell type
    for cell_type in test_cell_types:
        print(f"\\n📊 Processing cell type: {cell_type}")

        # Filter to current cell type
        cell_type_mask = obs_df[cell_type_col] == cell_type
        cell_type_obs = obs_df[cell_type_mask]

        if len(cell_type_obs) == 0:
            print(f"   ⚠️  No cells found for {cell_type}")
            continue

        print(f"   - Found {len(cell_type_obs):,} cells")

        # Get unique perturbations (excluding control)
        perturbations = cell_type_obs[perturbation_col].unique()
        non_control_perts = [p for p in perturbations if p != control_perturbation_name]

        print(f"   - Total perturbations: {len(perturbations)}")
        print(f"   - Non-control perturbations: {len(non_control_perts)}")

        if len(non_control_perts) == 0:
            print(f"   ⚠️  No non-control perturbations found for {cell_type}")
            continue

        # Calculate number to hold out
        n_holdout = int(np.ceil(len(non_control_perts) * pert_holdout_ratio))
        print(
            f"   - Holding out {n_holdout} perturbations ({n_holdout/len(non_control_perts):.1%})"
        )

        # Randomly select perturbations to hold out
        holdout_perts = np.random.choice(
            non_control_perts, size=n_holdout, replace=False
        )

        # Get indices of all cells with held-out perturbations
        holdout_mask = cell_type_obs[perturbation_col].isin(holdout_perts)
        holdout_cells = cell_type_obs[holdout_mask]

        # Convert to original indices
        holdout_indices = holdout_cells.index.values
        test_indices.extend(holdout_indices)

        print(f"   - Held-out perturbations: {list(holdout_perts)}")
        print(f"   - Held-out cells: {len(holdout_indices):,}")

        # Store info
        holdout_info[cell_type] = {
            "total_perturbations": len(perturbations),
            "non_control_perturbations": len(non_control_perts),
            "holdout_perturbations": list(holdout_perts),
            "holdout_cells": len(holdout_indices),
            "total_cells": len(cell_type_obs),
        }

    # Convert to numpy arrays
    test_indices = np.array(test_indices)
    all_indices = np.arange(adata_backed.n_obs)
    train_indices = np.setdiff1d(all_indices, test_indices)

    print("\\n✅ Split completed!")
    print(
        f"   - Training cells: {len(train_indices):,} ({len(train_indices)/adata_backed.n_obs:.1%})"
    )
    print(
        f"   - Test cells: {len(test_indices):,} ({len(test_indices)/adata_backed.n_obs:.1%})"
    )

    return train_indices, test_indices, holdout_info


def donor_cytokine_train_test_split(
    adata_backed,
    donor_cytokine_holdouts,
    perturbation_col,
    donor_col="donor",
    random_state=42,
):
    """
    Create train/test split by holding out specific cytokines from specific donors.
    Note: In this function, control perturbation can be specified to be held out or not!

    Parameters:
    -----------
    adata_backed : AnnData (backed)
        Backed AnnData object to avoid memory issues
    donor_cytokine_holdouts : dict
        Dictionary mapping donor names to list of cytokines to hold out
        e.g., {"Donor1": ["C5a", "TWEAK", ...], "Donor4": [...]}
    perturbation_col : str
        Column name containing perturbation information (e.g., 'cytokine')
    donor_col : str, default 'donor'
        Column name containing donor information
    random_state : int, default 42
        Random seed for reproducibility (for consistency with original function)

    Returns:
    --------
    train_indices : np.array
        Indices of training cells
    test_indices : np.array
        Indices of test cells
    holdout_info : dict
        Information about held-out donor-cytokine combinations
    """

    np.random.seed(random_state)  # For consistency with original function

    print("🧪 Creating donor-cytokine specific train/test split...")
    print(f"   - Perturbation column: {perturbation_col}")
    print(f"   - Donor column: {donor_col}")
    print(f"   - Donors with holdouts: {list(donor_cytokine_holdouts.keys())}")

    # Get observation metadata (memory efficient)
    obs_df = adata_backed.obs

    test_indices = []
    holdout_info = {}

    # Process each donor with holdout cytokines
    for donor, holdout_cytokines in donor_cytokine_holdouts.items():
        print(f"\\n👤 Processing donor: {donor}")

        # Filter to current donor
        donor_mask = obs_df[donor_col] == donor
        donor_obs = obs_df[donor_mask]

        if len(donor_obs) == 0:
            print(f"   ⚠️  No cells found for {donor}")
            continue

        print(f"   - Total cells for {donor}: {len(donor_obs):,}")

        # Get unique cytokines for this donor
        donor_cytokines = donor_obs[perturbation_col].unique()
        print(f"   - Available cytokines: {len(donor_cytokines)}")

        # Find which holdout cytokines are actually present for this donor
        available_holdout_cytokines = [
            cyt for cyt in holdout_cytokines if cyt in donor_cytokines
        ]
        missing_cytokines = [
            cyt for cyt in holdout_cytokines if cyt not in donor_cytokines
        ]

        print(f"   - Holdout cytokines (requested): {len(holdout_cytokines)}")
        print(f"   - Holdout cytokines (available): {len(available_holdout_cytokines)}")

        if missing_cytokines:
            print(
                f"   - Missing cytokines: {missing_cytokines[:10]}{'...' if len(missing_cytokines) > 10 else ''}"
            )

        if len(available_holdout_cytokines) == 0:
            print(f"   ⚠️  No holdout cytokines available for {donor}")
            continue

        # Get indices of cells with holdout cytokines for this donor
        holdout_mask = donor_obs[perturbation_col].isin(available_holdout_cytokines)
        holdout_cells = donor_obs[holdout_mask]

        # Convert to original indices
        holdout_indices = holdout_cells.index.values
        test_indices.extend(holdout_indices)

        print(f"   - Held-out cytokines: {len(available_holdout_cytokines)}")
        print(f"   - Held-out cells: {len(holdout_indices):,}")
        print(
            f"   - Holdout ratio for {donor}: {len(holdout_indices)/len(donor_obs):.1%}"
        )

        # Store detailed info
        holdout_info[donor] = {
            "total_cells": len(donor_obs),
            "total_cytokines": len(donor_cytokines),
            "requested_holdout_cytokines": len(holdout_cytokines),
            "available_holdout_cytokines": len(available_holdout_cytokines),
            "holdout_cytokines_list": available_holdout_cytokines,
            "missing_cytokines": missing_cytokines,
            "holdout_cells": len(holdout_indices),
            "holdout_ratio": len(holdout_indices) / len(donor_obs)
            if len(donor_obs) > 0
            else 0,
        }

    # Convert to numpy arrays
    test_indices = np.array(test_indices)
    all_indices = np.arange(adata_backed.n_obs)
    train_indices = np.setdiff1d(all_indices, test_indices)

    print("\\n✅ Split completed!")
    print(
        f"   - Training cells: {len(train_indices):,} ({len(train_indices)/adata_backed.n_obs:.1%})"
    )
    print(
        f"   - Test cells: {len(test_indices):,} ({len(test_indices)/adata_backed.n_obs:.1%})"
    )

    # Summary statistics
    print("\\n📊 Summary by donor:")
    for donor, info in holdout_info.items():
        print(
            f"   - {donor}: {info['holdout_cells']:,} cells "
            f"({info['holdout_ratio']:.1%}) "
            f"from {info['available_holdout_cytokines']} cytokines"
        )

    return train_indices, test_indices, holdout_info


"""
Example usage of perturbation_fewshot_train_test_split_rnd_pert_holdout function
"""
test_cell_types_set0 = ["B Naive", "CD4 Naive", "CD8 Naive", "CD14 Mono"]
train_idx, test_idx, info = perturbation_fewshot_train_test_split_rnd_pert_holdout(
    adata_backed = adata_backed,
    test_cell_types = test_cell_types_set0,
    perturbation_col = "cytokine",  # Your perturbation column
    control_perturbation_name = "PBS",  # Your control
    pert_holdout_ratio = 0.45,  # Hold out 45% of perturbations --> ~20% of cells
    cell_type_col = "cell_type",  # Your cell type column
    random_state = 42,
)

"""
Example usage of donor_cytokine_train_test_split function
"""
# Define the donor-cytokine holdouts as specified by STATE
test_cytokines = [
    "C5a",
    "TWEAK",
    "LIF",
    "IL-17C", ..., "PBS"]

donor_cytokine_holdouts = {
    "Donor1": test_cytokines.copy(),
    "Donor4": test_cytokines.copy(),
    "Donor9": test_cytokines.copy(),
    "Donor12": test_cytokines.copy(),
} # so all perturbations of the remaining donors, as well as the remaining pertubations of these 4 donors go to the training set

train_idx, test_idx, info = donor_cytokine_train_test_split(
    adata_backed = adata_backed,
    donor_cytokine_holdouts = donor_cytokine_holdouts,
    perturbation_col = "cytokine",
    donor_col = "donor",
    random_state = 42,
)
