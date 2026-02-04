import torch
import torch.nn as nn

from scg_vae.nnets import DiT


#################################
#         BASE DIFFUSION        #
#################################
class BaseDiffusion(nn.Module):
    def __init__(
        self,
        nnet: DiT,
        n_inducing_points: int,
        sigma: float = 0.025,
        timesteps: int = 50,
        min_val_clip: float = -100.0,
        max_val_clip: float = 100.0,
        **kwargs,
    ):
        super().__init__()
        self.n_inducing_points = n_inducing_points
        self.nnet = nnet
        self.timesteps = timesteps
        self.min_val_clip = min_val_clip
        self.max_val_clip = max_val_clip
        self.sigma = sigma
        self.mse_loss = nn.MSELoss(reduction="none")
        # Add seq_len from the nnet (DiT) model
        self.seq_len = nnet.seq_len

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        return self.nnet(x, t, condition)

    def _expand_time_dims(self, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """Expand time tensor to match input tensor dimensions."""
        dims = [1] * (len(x_shape) - 1)
        return t.view(len(t), *dims)

    def sample_base(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randn((batch_size, self.n_inducing_points, self.nnet.n_embed), device=device)


class FlowMatching(BaseDiffusion):
    def __init__(
        self,
        nnet: DiT,
        n_inducing_points: int,
        sigma: float = 0.025,
        timesteps: int = 50,
        min_val_clip: float = -100.0,
        max_val_clip: float = 100.0,
        cfm: bool = False,
        **kwargs,
    ):
        super().__init__(
            nnet=nnet,
            n_inducing_points=n_inducing_points,
            sigma=sigma,
            timesteps=timesteps,
            min_val_clip=min_val_clip,
            max_val_clip=max_val_clip,
            **kwargs,
        )
        self.cfm = cfm
        # Add seq_len from the nnet (DiT) model
        self.seq_len = nnet.seq_len

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        return self.nnet(x, t, condition)

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: dict[str, torch.Tensor] | None = None,
        cfg_scale: dict[str, float] | None = None,
    ) -> torch.Tensor:
        """Classifier-free guidance method that delegates to the underlying DiT model."""
        if self.nnet.condition_strategy == "joint":
            return self.nnet.forward_with_cfg_joint(x, t, condition, cfg_scale)
        else:
            return self.nnet.forward_with_cfg(x, t, condition, cfg_scale)

    def conditional_vector_field(
        self, xt: torch.Tensor, x_start: torch.Tensor, x_end: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Conditional vector field for CFM"""
        t = self._expand_time_dims(t, x_start.shape)
        if self.cfm:
            return x_end - x_start
        else:
            return (x_end - (1.0 - self.sigma) * xt) / (1.0 - (1.0 - self.sigma) * t)

    def sample_x_t(self, x_start: torch.Tensor, x_end: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = self._expand_time_dims(t, x_start.shape)
        if self.cfm:
            mu_t = (1 - t) * x_start + t * x_end
            sigma_t = self.sigma
        else:
            mu_t = t * x_end
            sigma_t = 1.0 - (1.0 - self.sigma) * t
        return mu_t + sigma_t * torch.randn_like(x_start)

    def log_prob(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        """Placeholder log_prob method for FlowMatching."""
        # This is a placeholder implementation - FlowMatching log_prob is complex
        return torch.zeros(x.shape[0], x.shape[1], device=x.device)

    @torch.no_grad()
    def sample(self, n_samples: int, condition: torch.Tensor | None = None) -> torch.Tensor:
        """Placeholder sample method for FlowMatching."""
        device = next(self.nnet.parameters()).device
        # Return a simple sample for now
        return torch.randn(n_samples, self.n_inducing_points, self.nnet.n_embed, device=device)


class torch_wrapper(torch.nn.Module):
    """Wraps diffusion model to torchdyn compatible format."""

    def __init__(
        self,
        model: nn.Module,
        conditional: bool = False,
        guidance_weight: dict[str, float] | None = None,
        condition: dict[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.model = model  # The diffusion model being wrapped
        self.conditional = conditional  # Flag for conditional generation
        self.condition = condition
        self.guidance_weight = guidance_weight  # Guidance weight dictionary

    def forward(self, t, x, *args, **kwargs):
        # Expand t to match batch size if it's a scalar
        if t.dim() == 0:
            t = t.expand(x.shape[0])

        x_t_uncond = self.model(x, t, condition=None)

        if not self.conditional:
            return x_t_uncond

        x_t = x_t_uncond.clone()

        for cov in self.condition.keys():
            x_t += self.guidance_weight[cov] * (self.model(x, t, condition={cov: self.condition[cov]}) - x_t_uncond)

        return x_t
