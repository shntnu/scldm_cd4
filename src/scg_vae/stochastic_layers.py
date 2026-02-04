import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from scvi.distributions import NegativeBinomial as NegativeBinomialSCVI
from torch.distributions import Distribution

from scg_vae.distributions import (
    MaskedZeroTruncatedNegativeBinomialSCVI,
)


###########################
#     GAUSSIAN LAYERS     #
###########################
class GaussianTransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        loc, log_scale = torch.chunk(x, 2, dim=-1)  # chunk over last dim
        loc = nn.functional.hardtanh(loc, min_val=-4.0, max_val=4.0)
        log_scale = nn.functional.hardtanh(log_scale, min_val=-7.0, max_val=5.0)
        scale = torch.exp(log_scale)
        return dist.Normal(loc, scale)

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        distribution = self.forward(x)
        return distribution.rsample()

    def log_prob(self, x: torch.Tensor, loc: torch.Tensor | None, scale: torch.Tensor | None) -> torch.Tensor:
        if (loc is None) or (scale is None):
            distribution = self.forward(x)
        else:
            distribution = dist.Normal(loc, scale)
        log_p = distribution.log_prob(x)
        return log_p

    def loss(self, x: torch.Tensor, loc: torch.Tensor | None, scale: torch.Tensor | None) -> torch.Tensor:
        return self.log_prob(x, loc, scale)


class GaussianLinearLayer(nn.Module):
    def __init__(
        self,
        n_hidden: int,
        n_latent: int,
    ):
        super().__init__()
        self.loc = nn.Linear(n_hidden, n_latent, bias=True)
        self.scale = nn.Linear(n_hidden, n_latent, bias=True)

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        # location
        loc = self.loc(x)
        # scale
        log_scale = self.scale(x)
        log_scale = nn.functional.hardtanh(log_scale, min_val=-7.0, max_val=5.0)
        scale = torch.exp(log_scale)
        return dist.Normal(loc, scale)

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        distribution = self.forward(x)
        return distribution.rsample()

    def log_prob(self, x: torch.Tensor, loc: torch.Tensor | None, scale: torch.Tensor | None) -> torch.Tensor:
        if (loc is None) or (scale is None):
            distribution = self.forward(x)
        else:
            distribution = dist.Normal(loc, scale)
        log_p = distribution.log_prob(x)
        return log_p

    def loss(self, x: torch.Tensor, loc: torch.Tensor | None, scale: torch.Tensor | None) -> torch.Tensor:
        return self.log_prob(x, loc, scale)


############################
#     NBinomial LAYERS     #
############################
def exp_linear(z: torch.Tensor, t: float = 10.0) -> torch.Tensor:
    exp_t = torch.exp(torch.as_tensor(t, device=z.device, dtype=z.dtype))
    # left branch
    left = torch.exp(z)
    # right branch (linear with slope exp(t), anchored at z=t)
    right = exp_t * (1.0 + (z - t))
    return torch.where(z <= t, left, right)


class RestScalarAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.att_proj = nn.Linear(d_model, 1)  # scalar score per rest token
        self.rest_head = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 1))  # pooled -> logit with nonlinearity

    def forward(self, rest_embs: torch.Tensor) -> torch.Tensor:
        scores = self.att_proj(rest_embs).squeeze(-1)  # B, notG, E
        att = torch.softmax(scores, dim=1)  # B, notG, 1
        pooled = torch.bmm(att.unsqueeze(1), rest_embs).squeeze(1)  # B, E
        logit_rest = self.rest_head(pooled)  # B, 1
        return logit_rest


class NegativeBinomialTransformerLayer(nn.Module):
    def __init__(
        self,
        *,
        n_genes: int,
        shared_theta: bool = False,
        n_embed: int | None = None,
        norm_layer: str = "layernorm",
        layernorm_eps: float = 1e-8,
        eps_: float = 1e-6,
        t: float = 10.0,
    ):
        super().__init__()
        self.shared_theta = shared_theta

        if shared_theta:
            self.theta = nn.Embedding(n_genes + 1, 1)
            torch.nn.init.ones_(self.theta.weight)
            self.params = nn.Linear(n_embed, 1, bias=True)
        else:
            self.theta = None
            self.params = nn.Linear(n_embed, 2, bias=True)

        self.eps_ = eps_
        self.t = t

    def forward(
        self,
        counts: torch.Tensor,
        genes: torch.Tensor,
        library_size: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.theta, nn.Embedding):
            mu = self.params(counts)
            theta = self.theta(genes.long())
        else:
            params = self.params(counts)
            mu, theta = torch.chunk(params, 2, dim=-1)
        mu, theta = mu.squeeze(-1), torch.exp(theta).squeeze(-1)
        mu = nn.functional.softmax(mu, dim=1) * library_size
        return mu, theta

    def log_prob(self, counts: torch.Tensor, genes: torch.Tensor, total_counts: torch.Tensor) -> torch.Tensor:
        distribution = self.forward(counts, genes, total_counts)
        return distribution.log_prob(counts)

class NegativeBinomialTransformerLayerDecoupled(nn.Module):
    """
    Goal is to module mu such that:
    E[Σμ_i] = Σ exp(log_ℓ + η_i) = ℓ · Σ exp(η_i)

    This won't necessarily equal ℓ unless Σ exp(η_i) = 1, we hope the model learns to do this (or we can impose in loss!)
    """
    def __init__(
        self,
        *,
        n_genes: int,
        shared_theta: bool = False,
        n_embed: int,
        use_gene_bias: bool = True,
        min_theta: float = 1e-5,
        eps: float = 0.0,
    ):
        super().__init__()
        self.shared_theta = shared_theta
        self.eps = eps
        self.min_theta = min_theta

        # per-gene bias for mu
        self.gene_bias = nn.Embedding(n_genes+1, 1) if use_gene_bias else None # Index 0 reserved for mask token
        if self.gene_bias is not None:
            nn.init.zeros_(self.gene_bias.weight)

        # map gene representation -> [eta_logit] or [eta_logit, theta_logit]
        if shared_theta:
            # only eta; theta comes from an embedding
            self.params = nn.Linear(n_embed, 1, bias=True)
            self.theta_embed = nn.Embedding(n_genes + 1, 1)
            nn.init.constant_(self.theta_embed.weight, 0.0)  # softplus(0) ~ 0.693
        else:
            # both eta and theta per gene
            self.params = nn.Linear(n_embed, 2, bias=True)
            self.theta_embed = None

    def forward(
        self,
        counts: torch.Tensor,   # [B, G, n_embed] – decoder representation per gene
        genes: torch.Tensor,         # [B, G] – gene indices
        library_size: torch.Tensor,  # [B] or [B, 1]
    ):
        B, G, _ = counts.shape
        log_lib = torch.log(library_size.view(B, 1) + self.eps)  # [B,1]

        if self.shared_theta:
            eta_logit = self.params(counts).squeeze(-1)     # [B,G]
            if self.gene_bias is not None:
                eta_logit = eta_logit + self.gene_bias(genes).squeeze(-1)  # [B,G]
            # theta from embedding (shared across cells, gene-specific)
            theta_logit = self.theta_embed(genes).squeeze(-1)    # [B,G]
        else:
            out = self.params(counts)                       # [B,G,2]
            eta_logit, theta_logit = out.unbind(dim=-1)          # [B,G], [B,G]
            if self.gene_bias is not None:
                eta_logit = eta_logit + self.gene_bias(genes).squeeze(-1)

        # POSITIVITY CONSTRAINTS
        # μ = exp(log ℓ + η)  (GLM offset); θ = softplus(theta_logit) + min_theta
        mu = torch.exp(log_lib + eta_logit)                      # [B,G]
        theta = F.softplus(theta_logit) + self.min_theta         # [B,G]

        return mu, theta

    def log_prob(self, counts: torch.Tensor, genes: torch.Tensor, total_counts: torch.Tensor) -> torch.Tensor:
        distribution = self.forward(counts, genes, total_counts)
        return distribution.log_prob(counts)


class NegativeBinomialLinearLayer(nn.Module):
    def __init__(
        self,
        *,
        n_genes: int,
        n_hidden: int,
        shared_theta: bool = False,
    ):
        super().__init__()
        self.shared_theta = shared_theta

        self.mu = nn.Linear(n_hidden, n_genes, bias=True)
        if self.shared_theta:
            self.theta: nn.Parameter | nn.Linear = nn.Parameter(torch.ones(n_genes), requires_grad=True)
        else:
            self.theta: nn.Linear = nn.Linear(n_hidden, n_genes, bias=True)
        self.softplus = nn.Softplus()

    def forward(
        self,
        counts: torch.Tensor,
        genes: torch.Tensor,
        library_size: torch.Tensor,
    ) -> Distribution:
        mu = self.mu(counts)
        if isinstance(self.theta, nn.Parameter):
            theta = self.softplus(self.theta)
        else:
            theta = self.softplus(self.theta(counts))
        mu = nn.functional.softmax(mu, dim=1)
        mu = mu * library_size
        return NegativeBinomialSCVI(mu=mu, theta=theta)

    def log_prob(self, x: torch.Tensor, total_counts: torch.Tensor) -> torch.Tensor:
        distribution = self.forward(x, total_counts)
        return distribution.log_prob(x)


class MaskedNegativeBinomialLinearLayer(nn.Module):
    def __init__(
        self,
        *,
        n_genes: int,
        shared_theta: bool = False,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.shared_theta = shared_theta

        self.mu = nn.Linear(n_genes, n_genes, bias=True)
        if self.shared_theta:
            # Create parameter on the same device as the model
            self.register_parameter(
                "theta", nn.Parameter(torch.ones(n_genes, device=next(self.parameters()).device), requires_grad=True)
            )
        else:
            self.theta: nn.Linear = nn.Linear(n_genes, n_genes, bias=True)
        self.logits = nn.Linear(n_genes, n_genes, bias=True)
        self.softplus = nn.Softplus()

    def forward(
        self,
        x: torch.Tensor,
        total_counts: torch.Tensor,
    ) -> Distribution:
        mu = self.mu(x)
        if isinstance(self.theta, nn.Parameter):
            theta = self.softplus(self.theta)
        else:
            theta = self.softplus(self.theta(x))
        logits = self.logits(x)
        mu = nn.functional.softmax(mu, dim=1)
        mu = mu * total_counts
        return MaskedZeroTruncatedNegativeBinomialSCVI(mu=mu, theta=theta, logits=logits)

    def log_prob(self, x: torch.Tensor, total_counts: torch.Tensor) -> torch.Tensor:
        distribution = self.forward(x, total_counts)
        return distribution.log_prob(x)
