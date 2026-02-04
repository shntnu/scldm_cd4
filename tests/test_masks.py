import pytest
import torch

from scg_vae.layers import _random_mask


@pytest.mark.parametrize("mask_proportion", [0.2])
@pytest.mark.parametrize("seq_len", [10])
def test_random_mask(mask_proportion: float, seq_len: int):
    # Create test inputs
    batch_size = 2
    genes = torch.ones(batch_size, seq_len, dtype=torch.long)
    counts = torch.ones(batch_size, seq_len, dtype=torch.float)
    mask_token_idx = 0

    # Apply masking
    masked_counts, masked_genes = _random_mask(
        genes,
        counts,
        mask_proportion,
        mask_token_idx,
    )

    # Test shapes
    assert masked_genes.shape == (batch_size, seq_len)
    assert masked_counts.shape == (batch_size, seq_len)

    # Test masking proportion
    num_masked = (masked_genes == mask_token_idx).sum().item() / batch_size
    expected_masked = int(seq_len * mask_proportion)
    assert num_masked == expected_masked, f"Expected {expected_masked} masked tokens, got {num_masked}"

    # Test masked values
    mask = masked_genes == mask_token_idx
    assert (masked_counts[mask] == -1.0).all(), "Masked counts should be -1.0"
    assert (masked_genes[mask] == mask_token_idx).all(), "Masked genes should be mask_token_idx"

    # Test unmasked values
    unmask = ~mask
    assert (masked_counts[unmask] == 1.0).all(), "Unmasked counts should be 1.0"
    assert (masked_genes[unmask] == 1).all(), "Unmasked genes should be 1"
