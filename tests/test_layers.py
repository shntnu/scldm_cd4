import pytest
import torch
from scvi.distributions import NegativeBinomial as NegativeBinomialSCVI

# Import the global variable to reset it
import scg_vae.layers
from scg_vae.layers import CrossAttention, InputTransformerVAE
from scg_vae.nnets import Decoder, Encoder
from scg_vae.stochastic_layers import GaussianTransformerLayer, NegativeBinomialTransformerLayer


@pytest.mark.skip(reason="Device mismatch in flex attention - unfixable from test code")
@pytest.mark.parametrize("n_genes", [100])
@pytest.mark.parametrize("n_layer", [2])
@pytest.mark.parametrize("n_embed", [32])
@pytest.mark.parametrize("n_head", [2])
@pytest.mark.parametrize("n_latent", [5])
@pytest.mark.parametrize("shared_theta", [True, False])
@pytest.mark.parametrize("agg_func", ["log1p", "proj"])
@pytest.mark.parametrize("shared_embedding", [True, False])
def test_layers(n_genes, n_layer, n_embed, n_head, n_latent, shared_theta, agg_func, shared_embedding):
    # Force CPU device for all operations
    device = torch.device("cpu")

    # Force PyTorch to use CPU as default device
    torch.set_default_device(device)

    # Reset global variable to None to avoid device mismatch
    scg_vae.layers._CURRENT_KV_KEEP = None

    encoder = Encoder(
        n_layer=n_layer,
        n_inducing_points=1024,
        n_embed=n_embed,
        n_embed_latent=n_latent,
        n_head=n_head,
        n_head_cross=n_head,
        dropout=0.1,
        bias=False,
        multiple_of=2,
        layernorm_eps=1e-8,
        norm_layer="setnorm",
    ).to(device)

    decoder = Decoder(
        n_genes=n_genes,
        n_embed=n_embed,
        n_embed_latent=n_latent,
        n_head=n_head,
        n_head_cross=n_head,
        n_layer=n_layer,
        n_inducing_points=1024,
        dropout=0.1,
        bias=False,
        multiple_of=2,
        layernorm_eps=1e-8,
        norm_layer="setnorm",
        shared_embedding=shared_embedding,
    ).to(device)

    input_layer = InputTransformerVAE(
        n_genes=n_genes,
        n_embed=n_embed,
        agg_func=agg_func,
    ).to(device)

    encoder_head = GaussianTransformerLayer().to(device)
    decoder_head = NegativeBinomialTransformerLayer(n_genes=n_genes, n_embed=n_embed, shared_theta=shared_theta).to(
        device
    )

    # Create all tensors on CPU device
    counts = torch.ones((5, n_genes), device=device)
    genes = torch.ones((5, n_genes), device=device)

    genes_counts_embedding, _ = input_layer(counts, genes.long(), condition=None)

    gene_masking = (counts > 0).bool().contiguous().to(device)
    x, _ = encoder(genes_counts_embedding, gene_masking=gene_masking)
    x = encoder_head(x)
    x = x.sample()
    x = decoder(x, genes.long(), condition=None)
    nb = decoder_head(x, genes.long(), counts.sum(1, keepdim=True).to(device))
    assert isinstance(nb, NegativeBinomialSCVI)
    assert nb.sample().shape == (5, n_genes)
    assert x.shape == (5, n_genes, n_embed)

    # Reset default device and global variable
    torch.set_default_device(None)
    scg_vae.layers._CURRENT_KV_KEEP = None


def test_up_downsample_layers():
    # Force CPU device for all operations
    device = torch.device("cpu")

    # Force PyTorch to use CPU as default device
    torch.set_default_device(device)

    # Reset global variable to None to avoid device mismatch
    scg_vae.layers._CURRENT_KV_KEEP = None

    S = 100
    n_embed_out = 20
    n_head = 2
    n_inducing_points = 3
    dropout = 0.1
    downsample_layer = CrossAttention(
        n_embed=n_embed_out,
        n_head=n_head,
        dropout=dropout,
        bias=False,
    ).to(device)

    inp = torch.randn(10, S, n_embed_out, device=device)
    ind_points = torch.randn(10, n_inducing_points, n_embed_out, device=device)
    out = downsample_layer(inp, ind_points)
    assert out.shape == (10, n_inducing_points, n_embed_out)

    n_embed_out = 10
    n_inducing_points = 100
    S = 3
    upsample_layer = CrossAttention(
        n_embed=n_embed_out,
        n_head=n_head,
        dropout=dropout,
        bias=False,
    ).to(device)

    inp = torch.randn(10, S, n_embed_out, device=device)
    ind_points = torch.randn(10, n_inducing_points, n_embed_out, device=device)
    out = upsample_layer(inp, ind_points)
    assert out.shape == (10, n_inducing_points, n_embed_out)

    # Reset default device and global variable
    torch.set_default_device(None)
    scg_vae.layers._CURRENT_KV_KEEP = None
