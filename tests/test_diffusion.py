import numpy as np
import pytest
import torch

# Import layers to reset global variables
import scg_vae.layers
from scg_vae.diffusion import FlowMatching
from scg_vae.nnets import DiT


@pytest.mark.parametrize("n_samples", [4])
@pytest.mark.parametrize("n_inducing_points", [8])
@pytest.mark.parametrize("n_embed_latent", [16])
@pytest.mark.parametrize("n_embed_diffusion", [16])
@pytest.mark.parametrize("sigma", [0.05])
@pytest.mark.parametrize("timesteps", [10])
@pytest.mark.parametrize("norm_layer", ["layernorm"])
def test_diffusion(
    n_samples, n_inducing_points, n_embed_latent, n_embed_diffusion, sigma, timesteps, norm_layer
):
    # Force CPU device for all operations
    device = torch.device("cpu")

    # Force PyTorch to use CPU as default device
    torch.set_default_device(device)

    # Reset global variable to None to avoid device mismatch
    scg_vae.layers._CURRENT_KV_KEEP = None

    nnet = DiT(
        n_embed_input=n_embed_latent,
        n_embed=n_embed_diffusion,
        n_layer=1,
        n_head=1,
        seq_len=n_inducing_points,
        dropout=0.1,
        bias=False,
        norm_layer=norm_layer,
        multiple_of=2,
        layernorm_eps=1e-6,
        class_vocab_sizes={"assay": 1, "suspension": 1},
    ).to(device)

    fm = FlowMatching(nnet=nnet, n_inducing_points=n_inducing_points, sigma=sigma, timesteps=timesteps).to(device)

    x = torch.randn((n_samples, n_inducing_points, n_embed_diffusion), device=device)
    condition = {
        "assay": torch.randint(0, 1, (n_samples,), device=device),
        "suspension": torch.randint(0, 1, (n_samples,), device=device),
    }

    log_p_fm = fm.log_prob(x, condition)
    x_sample_fm = fm.sample(n_samples=n_samples, condition=condition)

    assert (n_samples == log_p_fm.shape[0]) and (n_inducing_points == log_p_fm.shape[1])
    assert (n_samples == x_sample_fm.shape[0]) and (n_inducing_points == x_sample_fm.shape[1])

    # Reset default device and global variable
    torch.set_default_device(None)
    scg_vae.layers._CURRENT_KV_KEEP = None


if __name__ == "__main__":
    print("test")
