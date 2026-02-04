import pytest
import torch
import torchmetrics


def test_metrics():
    dim = 5
    batch_size = 3
    pcc = torchmetrics.regression.PearsonCorrCoef(num_outputs=dim)
    mask = torch.randint(0, 2, (batch_size, dim), dtype=torch.bool)
    mask = mask.float().masked_fill(mask, float("-inf"))
    true = torch.randn((batch_size, dim))
    pred = torch.randn((batch_size, dim))
    true[mask == float("-inf")] = torch.nan
    pred[mask == float("-inf")] = torch.nan
    out = pcc(pred, true)
    assert out.shape == (dim,)
