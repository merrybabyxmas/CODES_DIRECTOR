"""
DIRECTOR Context Encoding Modules.

Implements the Unified Visual Context Space:
  - LocalContextEncoder: VAE-based encoding of previous shot's last frame -> N_local tokens
  - GlobalContextEncoder: CLIP Vision encoding of character references -> N_global tokens per character
  - ContextProjection: Linear projections to align heterogeneous embeddings to D-dim
  - UnifiedContextBuilder: Concatenation with positional info

Tensor shape conventions (commented inline):
  B = batch size
  N_l = local token count (e.g., 256)
  N_g = global token count per character (e.g., 64)
  K = number of characters
  D = context dimension (1920 for CogVideoX-2b)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor


@dataclass
class ContextConfig:
    """Configuration for context encoding modules."""
    local_token_count: int = 256
    num_local_frames: int = 2       # number of previous frames (t-1, t-2) for two-shot context
    global_token_count: int = 64
    max_characters: int = 4
    context_dim: int = 1920
    clip_vision_dim: int = 1024
    vae_latent_channels: int = 16
    clip_model: str = "openai/clip-vit-large-patch14"
    vae_spatial_scale: int = 8
    patch_size: int = 2


class LocalContextEncoder(nn.Module):
    """
    Encodes previous shot frames into local context tokens using the CogVideoX VAE.

    Supports two-shot local context (t-1, t-2) for cinematic continuity
    (e.g., shot-reverse-shot patterns). Each frame produces `target_token_count`
    tokens; total output is `num_frames * target_token_count` tokens.

    Input: list of (B, 3, H, W) RGB frames in [0, 1]
    Output: (B, num_frames * N_local, D) local context tokens
    """

    def __init__(
        self,
        vae: nn.Module,
        target_token_count: int = 256,
        num_local_frames: int = 2,
        context_dim: int = 1920,
        vae_latent_channels: int = 16,
        vae_spatial_scale: int = 8,
        vae_scaling_factor: float = 1.15258426,
    ):
        super().__init__()
        self.vae = vae
        self.target_token_count = target_token_count
        self.num_local_frames = num_local_frames
        self.context_dim = context_dim
        self.vae_latent_channels = vae_latent_channels
        self.vae_spatial_scale = vae_spatial_scale
        self.vae_scaling_factor = vae_scaling_factor

        # Projection from flattened VAE latent to context_dim
        # VAE output per spatial position: vae_latent_channels (16)
        self.projection = ContextProjection(
            input_dim=vae_latent_channels,
            output_dim=context_dim,
            num_layers=2,
        )

        # Per-frame positional embeddings (spatial)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, target_token_count, context_dim) * 0.02
        )

        # Temporal embeddings to distinguish t-1 vs t-2
        self.temporal_embedding = nn.Embedding(num_local_frames, context_dim)

    @torch.no_grad()
    def encode_frame_to_latent(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Encode a single RGB frame through the CogVideoX VAE encoder.

        Args:
            frame: (B, 3, H, W) in [0, 1]

        Returns:
            latent: (B, C_latent, H_lat, W_lat) VAE latent
        """
        # CogVideoX VAE expects [-1, 1] input
        frame = frame * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        # CogVideoX VAE expects (B, C, T, H, W) with T frames
        # For a single frame, T=1
        frame_5d = frame.unsqueeze(2)  # (B, 3, 1, H, W)

        # Encode through VAE
        posterior = self.vae.encode(frame_5d)
        if hasattr(posterior, 'latent_dist'):
            latent = posterior.latent_dist.mode()  # (B, C_lat, T_lat, H_lat, W_lat)
        else:
            latent = posterior.sample() if hasattr(posterior, 'sample') else posterior

        latent = latent * self.vae_scaling_factor

        # Squeeze temporal dimension (single frame -> T_lat=1)
        if latent.dim() == 5:
            latent = latent.squeeze(2)  # (B, C_lat, H_lat, W_lat)

        return latent

    def _encode_single_frame(self, frame: Optional[torch.Tensor] = None,
                             precomputed_latent: Optional[torch.Tensor] = None,
                             temporal_idx: int = 0) -> torch.Tensor:
        """Encode a single frame to local tokens with spatial + temporal embeddings."""
        if precomputed_latent is not None:
            latent = precomputed_latent
        else:
            latent = self.encode_frame_to_latent(frame)

        B, C, H_lat, W_lat = latent.shape
        num_spatial = H_lat * W_lat

        tokens = latent.flatten(2).transpose(1, 2)  # (B, H_lat*W_lat, C_lat)

        # Adaptive pooling if spatial count != target_token_count
        if num_spatial != self.target_token_count:
            tokens_2d = latent
            target_h = int(math.sqrt(self.target_token_count * H_lat / W_lat))
            target_w = self.target_token_count // target_h
            while target_h * target_w != self.target_token_count:
                target_h += 1
                target_w = self.target_token_count // target_h
                if target_h > self.target_token_count:
                    target_h = self.target_token_count
                    target_w = 1
                    break
            tokens_2d = F.adaptive_avg_pool2d(tokens_2d, (target_h, target_w))
            tokens = tokens_2d.flatten(2).transpose(1, 2)

        local_tokens = self.projection(tokens)  # (B, N_local, D)

        # Add spatial positional embeddings
        local_tokens = local_tokens + self.pos_embedding[:, :local_tokens.size(1)]

        # Add temporal embedding to distinguish t-1 from t-2
        temp_emb = self.temporal_embedding(
            torch.tensor(temporal_idx, device=local_tokens.device)
        )  # (D,)
        local_tokens = local_tokens + temp_emb.unsqueeze(0).unsqueeze(0)

        return local_tokens

    def forward(
        self,
        frames: Optional[List[torch.Tensor]] = None,
        precomputed_latents: Optional[List[torch.Tensor]] = None,
        # Legacy single-frame interface
        frame: Optional[torch.Tensor] = None,
        precomputed_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode one or more previous frames into local context tokens.

        Args:
            frames: list of (B, 3, H, W) RGB frames [t-1, t-2, ...], newest first
            precomputed_latents: list of (B, C_lat, H_lat, W_lat) pre-encoded latents
            frame: (B, 3, H, W) single frame (legacy, treated as [frame])
            precomputed_latent: single pre-encoded latent (legacy)

        Returns:
            local_tokens: (B, num_frames * N_local, D) concatenated local context tokens
        """
        # Normalize to list interface
        if frames is None and frame is not None:
            frames = [frame]
        if precomputed_latents is None and precomputed_latent is not None:
            precomputed_latents = [precomputed_latent]

        all_tokens = []
        num_inputs = len(frames) if frames is not None else (
            len(precomputed_latents) if precomputed_latents is not None else 0
        )

        for i in range(num_inputs):
            f = frames[i] if frames is not None else None
            pl = precomputed_latents[i] if precomputed_latents is not None else None
            all_tokens.append(self._encode_single_frame(f, pl, temporal_idx=i))

        return torch.cat(all_tokens, dim=1)  # (B, num_frames * N_local, D)


class GlobalContextEncoder(nn.Module):
    """
    Encodes character reference images via CLIP Vision Encoder.

    Each character image -> CLIP patch tokens -> project to D-dim.

    Input: List of (B, 3, 224, 224) character reference images
    Output: (B, K * N_global, D) global context tokens
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        global_token_count: int = 64,
        context_dim: int = 1920,
        clip_vision_dim: int = 1024,
        max_characters: int = 4,
        freeze_clip: bool = True,
    ):
        super().__init__()
        self.global_token_count = global_token_count
        self.context_dim = context_dim
        self.max_characters = max_characters
        self.clip_vision_dim = clip_vision_dim

        # Load CLIP vision model
        self.clip_vision = CLIPVisionModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPImageProcessor.from_pretrained(clip_model_name)

        if freeze_clip:
            for param in self.clip_vision.parameters():
                param.requires_grad = False
            self.clip_vision.eval()

        # Projection from CLIP dim to context_dim
        self.projection = ContextProjection(
            input_dim=clip_vision_dim,
            output_dim=context_dim,
            num_layers=2,
        )

        # Token compression: CLIP outputs 257 tokens (1 CLS + 256 patches for ViT-L/14)
        # We compress to global_token_count via learned attention pooling
        self.token_compressor = AttentionPooling(
            input_dim=clip_vision_dim,
            num_queries=global_token_count,
            num_heads=8,
        )

        # Per-character type embeddings (to distinguish character A from B)
        self.character_type_embedding = nn.Embedding(
            max_characters, context_dim
        )

        # Positional embeddings for global tokens
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_characters * global_token_count, context_dim) * 0.02
        )

    @torch.no_grad()
    def encode_clip(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images through frozen CLIP vision encoder.

        Args:
            images: (N, 3, 224, 224) preprocessed images

        Returns:
            features: (N, num_patches + 1, clip_dim) CLIP features
        """
        outputs = self.clip_vision(pixel_values=images, output_hidden_states=True)
        # Use last hidden state which includes all patch tokens
        features = outputs.last_hidden_state  # (N, 257, 1024) for ViT-L/14
        return features

    def forward(
        self,
        character_images: List[torch.Tensor],
        character_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            character_images: list of K tensors, each (B, 3, 224, 224), or
                              single tensor (B, K, 3, 224, 224)
            character_masks: (B, K) binary mask indicating valid characters

        Returns:
            global_tokens: (B, K * N_global, D) concatenated global tokens
            token_mask: (B, K * N_global) validity mask
        """
        if isinstance(character_images, torch.Tensor) and character_images.dim() == 5:
            # (B, K, 3, 224, 224) -> list of (B, 3, 224, 224)
            B, K = character_images.shape[:2]
            char_list = [character_images[:, k] for k in range(K)]
        else:
            char_list = character_images
            B = char_list[0].shape[0]
            K = len(char_list)

        device = char_list[0].device
        dtype = char_list[0].dtype

        all_global_tokens = []
        all_masks = []

        for k in range(K):
            char_img = char_list[k]  # (B, 3, 224, 224)

            # Encode through CLIP
            clip_features = self.encode_clip(char_img)  # (B, 257, clip_dim)

            # Remove CLS token, keep patch tokens
            patch_features = clip_features[:, 1:]  # (B, 256, clip_dim)

            # Compress to global_token_count via attention pooling
            compressed = self.token_compressor(patch_features)  # (B, N_global, clip_dim)

            # Project to context_dim
            projected = self.projection(compressed)  # (B, N_global, D)

            # Add character type embedding
            char_type_emb = self.character_type_embedding(
                torch.full((B,), k, device=device, dtype=torch.long)
            )  # (B, D)
            projected = projected + char_type_emb.unsqueeze(1)  # (B, N_global, D)

            all_global_tokens.append(projected)

            # Build mask
            if character_masks is not None:
                # (B,) -> (B, N_global)
                mask_k = character_masks[:, k].unsqueeze(1).expand(-1, self.global_token_count)
            else:
                mask_k = torch.ones(B, self.global_token_count, device=device, dtype=dtype)
            all_masks.append(mask_k)

        # Pad remaining character slots if K < max_characters
        for k in range(K, self.max_characters):
            pad_tokens = torch.zeros(
                B, self.global_token_count, self.context_dim,
                device=device, dtype=dtype
            )
            all_global_tokens.append(pad_tokens)
            all_masks.append(
                torch.zeros(B, self.global_token_count, device=device, dtype=dtype)
            )

        # Concatenate all character tokens
        global_tokens = torch.cat(all_global_tokens, dim=1)  # (B, K_max * N_global, D)
        token_mask = torch.cat(all_masks, dim=1)  # (B, K_max * N_global)

        # Add positional embeddings
        global_tokens = global_tokens + self.pos_embedding[:, :global_tokens.size(1)]

        return global_tokens, token_mask


class ContextProjection(nn.Module):
    """
    Multi-layer projection with LayerNorm and GELU.

    Maps from input_dim to output_dim with optional intermediate layers.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        dims = [input_dim]
        # Intermediate dims: linear interpolation
        for i in range(1, num_layers):
            mid = input_dim + (output_dim - input_dim) * i // num_layers
            dims.append(mid)
        dims.append(output_dim)

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        # Final LayerNorm
        layers.append(nn.LayerNorm(output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., input_dim)
        Returns:
            y: (..., output_dim)
        """
        return self.net(x)


class AttentionPooling(nn.Module):
    """
    Learnable attention pooling to compress a variable number of tokens
    into a fixed number of query tokens.

    Uses cross-attention: learnable queries attend to input tokens.
    """

    def __init__(
        self,
        input_dim: int,
        num_queries: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.input_dim = input_dim

        # Learnable query tokens
        self.queries = nn.Parameter(
            torch.randn(1, num_queries, input_dim) * (input_dim ** -0.5)
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N_in, D_in) input tokens

        Returns:
            out: (B, num_queries, D_in) compressed tokens
        """
        B = x.shape[0]
        queries = self.queries.expand(B, -1, -1)  # (B, num_queries, D_in)

        # Cross-attention: queries attend to input tokens
        out, _ = self.cross_attn(
            query=queries,
            key=x,
            value=x,
        )  # (B, num_queries, D_in)

        out = self.norm(out + queries)  # residual + norm
        return out


class UnifiedContextBuilder(nn.Module):
    """
    Concatenates local and global context tokens into a unified context tensor
    with proper masking and type embeddings.

    Output: Unified Context = [Local Tokens] + [Global Tokens (char A)] + [Global Tokens (char B)] + ...
    """

    def __init__(
        self,
        context_dim: int = 1920,
        local_token_count: int = 256,
        global_token_count: int = 64,
        max_characters: int = 4,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.local_token_count = local_token_count
        self.global_token_count = global_token_count
        self.max_characters = max_characters

        total_tokens = local_token_count + global_token_count * max_characters

        # Type embeddings: 0 = local, 1 = global
        self.type_embedding = nn.Embedding(2, context_dim)

        # Layer norm for final unified context
        self.output_norm = nn.LayerNorm(context_dim)

    def forward(
        self,
        local_tokens: torch.Tensor,
        global_tokens: torch.Tensor,
        local_mask: Optional[torch.Tensor] = None,
        global_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            local_tokens: (B, N_local, D)
            global_tokens: (B, N_global_total, D) where N_global_total = K * N_g
            local_mask: (B, N_local) or None
            global_mask: (B, N_global_total) or None

        Returns:
            unified_context: (B, N_local + N_global_total, D)
            unified_mask: (B, N_local + N_global_total)
        """
        B = local_tokens.shape[0]
        device = local_tokens.device
        dtype = local_tokens.dtype

        # Add type embeddings
        local_type = self.type_embedding(
            torch.zeros(B, local_tokens.size(1), device=device, dtype=torch.long)
        )  # (B, N_local, D)
        global_type = self.type_embedding(
            torch.ones(B, global_tokens.size(1), device=device, dtype=torch.long)
        )  # (B, N_global_total, D)

        local_tokens = local_tokens + local_type
        global_tokens = global_tokens + global_type

        # Concatenate
        unified_context = torch.cat(
            [local_tokens, global_tokens], dim=1
        )  # (B, N_local + N_global_total, D)

        # Build unified mask
        if local_mask is None:
            local_mask = torch.ones(B, local_tokens.size(1), device=device, dtype=dtype)
        if global_mask is None:
            global_mask = torch.ones(B, global_tokens.size(1), device=device, dtype=dtype)
        unified_mask = torch.cat([local_mask, global_mask], dim=1)

        # Normalize
        unified_context = self.output_norm(unified_context)

        return unified_context, unified_mask
