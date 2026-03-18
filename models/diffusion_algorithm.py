"""
Diffusion algorithm abstraction for DIRECTOR.

Each video backbone (CogVideoX, WanVideo, etc.) uses a different diffusion
formulation. This module provides a unified interface so the training loop
and inference code remain backbone-agnostic.

Supported algorithms:
  - DDPMVPrediction: CogVideoX's native DDPM with v-prediction
  - FlowMatching:    Rectified flow (x_t = t*x_1 + (1-t)*x_0, v = x_1 - x_0)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class DiffusionStepOutput:
    """Output from a single inference step."""
    latents: torch.Tensor
    state: Any = None  # Algorithm-specific state carried across steps


class DiffusionAlgorithm(ABC):
    """Base class for diffusion training/inference algorithms."""

    @abstractmethod
    def add_noise(
        self, clean: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """Forward diffusion: create noisy sample x_t from clean sample and noise."""
        ...

    @abstractmethod
    def get_target(
        self, clean: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """Compute the training target that the model should predict."""
        ...

    @abstractmethod
    def recover_clean(
        self, noisy: torch.Tensor, model_output: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """Recover the clean sample x_0 from noisy input and model prediction (single-step)."""
        ...

    @abstractmethod
    def sample_timesteps(
        self, batch_size: int, device: torch.device,
        sampling: str = "uniform", **kwargs
    ) -> torch.Tensor:
        """Sample training timesteps."""
        ...

    @abstractmethod
    def prepare_inference(self, num_steps: int, device: torch.device) -> torch.Tensor:
        """Set up inference and return the timestep schedule (descending for DDPM, ascending for FM)."""
        ...

    @abstractmethod
    def inference_step(
        self, model_output: torch.Tensor, latents: torch.Tensor,
        timestep: torch.Tensor, step_index: int, timesteps: torch.Tensor,
        state: Any = None,
    ) -> DiffusionStepOutput:
        """Single reverse step during inference."""
        ...

    def compute_loss(
        self, model_output: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss. Override for custom losses."""
        return F.mse_loss(model_output, target, reduction="mean")


class DDPMVPrediction(DiffusionAlgorithm):
    """
    DDPM v-prediction algorithm (CogVideoX native).

    Noise schedule: x_t = sqrt(α_t) * x_0 + sqrt(1-α_t) * ε
    Target:         v   = sqrt(α_t) * ε  - sqrt(1-α_t) * x_0
    Recovery:       x_0 = sqrt(α_t) * x_t - sqrt(1-α_t) * v

    Timestep convention: t ∈ {0,...,999}, higher = noisier.
    """

    def __init__(self, scheduler):
        """
        Args:
            scheduler: CogVideoXDPMScheduler instance (from diffusers).
        """
        self.scheduler = scheduler

    def add_noise(self, clean, noise, timestep):
        return self.scheduler.add_noise(clean, noise, timestep)

    def get_target(self, clean, noise, timestep):
        return self.scheduler.get_velocity(clean, noise, timestep)

    def recover_clean(self, noisy, model_output, timestep):
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(
            device=noisy.device
        )
        alphas_cumprod = self.scheduler.alphas_cumprod.to(dtype=noisy.dtype)
        alpha_t = alphas_cumprod[timestep]

        sqrt_alpha = alpha_t.sqrt()
        while len(sqrt_alpha.shape) < len(noisy.shape):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)

        sqrt_one_minus_alpha = (1.0 - alpha_t).sqrt()
        while len(sqrt_one_minus_alpha.shape) < len(noisy.shape):
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        return sqrt_alpha * noisy - sqrt_one_minus_alpha * model_output

    def sample_timesteps(self, batch_size, device, sampling="logit_normal", **kwargs):
        if sampling == "logit_normal":
            mean = kwargs.get("logit_normal_mean", 0.0)
            std = kwargs.get("logit_normal_std", 1.0)
            u = torch.randn(batch_size, device=device) * std + mean
            t_frac = torch.sigmoid(u)
            return (t_frac * 999.0).long().clamp(0, 999)
        else:
            return torch.randint(0, 1000, (batch_size,), device=device, dtype=torch.long)

    def prepare_inference(self, num_steps, device):
        self.scheduler.set_timesteps(num_steps, device=device)
        return self.scheduler.timesteps

    def inference_step(self, model_output, latents, timestep, step_index, timesteps, state=None):
        step_output = self.scheduler.step(
            model_output, timestep, latents, return_dict=False,
        )
        # step_output is (prev_sample, pred_original_sample) or just prev_sample
        if isinstance(step_output, tuple):
            latents = step_output[0]
        else:
            latents = step_output
        return DiffusionStepOutput(latents=latents, state=None)


class FlowMatching(DiffusionAlgorithm):
    """
    Rectified Flow / Flow Matching algorithm (WanVideo, Stable Diffusion 3, etc.).

    Interpolation: x_t = t * x_1 + (1-t) * ε   (t ∈ [0,1], t=0 noise, t=1 clean)
    Target:        v   = x_1 - ε
    Recovery:      x_1 = x_t + (1-t) * v

    Timestep mapping to model: timestep = t * 1000 (or backbone-specific).
    """

    def __init__(self, sigma_min: float = 0.001, timestep_max: float = 1000.0):
        self.sigma_min = sigma_min
        self.timestep_max = timestep_max

    def _t_to_scalar(self, timestep: torch.Tensor) -> torch.Tensor:
        """Convert integer timestep back to t ∈ [0, 1]."""
        return timestep.float() / self.timestep_max

    def add_noise(self, clean, noise, timestep):
        # timestep is t_frac * 1000 (integer)
        t = self._t_to_scalar(timestep)
        while len(t.shape) < len(clean.shape):
            t = t.unsqueeze(-1)
        return t * clean + (1.0 - t) * noise

    def get_target(self, clean, noise, timestep):
        return clean - noise

    def recover_clean(self, noisy, model_output, timestep):
        t = self._t_to_scalar(timestep)
        while len(t.shape) < len(noisy.shape):
            t = t.unsqueeze(-1)
        return noisy + (1.0 - t) * model_output

    def sample_timesteps(self, batch_size, device, sampling="logit_normal", **kwargs):
        sigma_min = kwargs.get("sigma_min", self.sigma_min)
        if sampling == "logit_normal":
            mean = kwargs.get("logit_normal_mean", 0.0)
            std = kwargs.get("logit_normal_std", 1.0)
            u = torch.randn(batch_size, device=device) * std + mean
            t_frac = torch.sigmoid(u).clamp(sigma_min, 1.0 - sigma_min)
        else:
            t_frac = torch.rand(batch_size, device=device) * (1.0 - 2 * sigma_min) + sigma_min
        return (t_frac * self.timestep_max).long()

    def prepare_inference(self, num_steps, device):
        # Ascending: t=0 (noise) → t=1 (clean), mapped to integer timesteps
        dt = 1.0 / num_steps
        t_values = [i * dt for i in range(num_steps)]
        return torch.tensor(
            [int(t * self.timestep_max) for t in t_values],
            device=device, dtype=torch.long,
        )

    def inference_step(self, model_output, latents, timestep, step_index, timesteps, state=None):
        dt = 1.0 / len(timesteps)
        return DiffusionStepOutput(latents=latents + dt * model_output, state=None)


def create_diffusion_algorithm(backbone: str, scheduler=None, **kwargs) -> DiffusionAlgorithm:
    """
    Factory: pick the right diffusion algorithm based on backbone name.

    Args:
        backbone: Model ID (e.g. "THUDM/CogVideoX-2b", "Wan-AI/Wan2.1-T2V-14B")
        scheduler: Scheduler instance (required for DDPM-based backbones)
    """
    backbone_lower = backbone.lower()

    if "cogvideo" in backbone_lower:
        assert scheduler is not None, "CogVideoX requires a scheduler instance"
        return DDPMVPrediction(scheduler)
    elif "wan" in backbone_lower:
        return FlowMatching(**kwargs)
    else:
        # Default to flow matching for unknown backbones
        return FlowMatching(**kwargs)
