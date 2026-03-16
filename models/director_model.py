"""
DIRECTOR Model: Core architecture integrating Unified Visual Context into CogVideoX.

Architecture (Post-Block Adapter):
  - Backbone CogVideoX transformer blocks are kept COMPLETELY UNTOUCHED.
  - After each block, a lightweight ContextAdapter cross-attention module is applied:
      Q = joint hidden states (text + video),  K/V = unified context tokens
  - The adapter output is added to the block output via tanh-gated residual.
  - This design is fully compatible with gradient checkpointing.

Mathematical formulation:
  h_out = h_block + tanh(gate) * Adapter(h_block, ctx)
  Adapter(h, ctx) = OutProj( Softmax( Q_norm(Q) @ K_norm(K)^T / sqrt(d) ) @ V )
  where Q = W_q(h),  K = W_k(ctx),  V = W_v(ctx)
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import CogVideoXPipeline, CogVideoXTransformer3DModel
from diffusers.models.transformers.cogvideox_transformer_3d import Transformer2DModelOutput

from .context_encoder import (
    ContextConfig,
    GlobalContextEncoder,
    LocalContextEncoder,
    UnifiedContextBuilder,
)


@dataclass
class DirectorConfig:
    """Full DIRECTOR model configuration."""
    backbone: str = "THUDM/CogVideoX-2b"
    inner_dim: int = 1920
    text_embed_dim: int = 4096
    num_heads: int = 30
    head_dim: int = 64
    num_layers: int = 30
    max_text_seq_length: int = 226

    # Context
    context: ContextConfig = field(default_factory=ContextConfig)

    # Multi-context dropout
    drop_global_prob: float = 0.10
    drop_local_prob: float = 0.10
    keep_both_prob: float = 0.80

    # Attention injection
    inject_layers: Union[str, List[int]] = "all"
    context_gate_init: float = 0.0


# ===========================================================================
# Post-Block Context Adapter (replaces custom attention processor)
# ===========================================================================

class ContextAdapter(nn.Module):
    """
    Lightweight cross-attention adapter applied AFTER each CogVideoX block.

    Q = linear(joint_hidden)  -->  (B, S, D)
    K = linear(context)       -->  (B, N_ctx, D)
    V = linear(context)       -->  (B, N_ctx, D)

    Output is tanh-gated and added as residual.
    Fully compatible with gradient checkpointing since no backbone modification.
    """

    def __init__(
        self,
        inner_dim: int = 1920,
        num_heads: int = 30,
        head_dim: int = 64,
        context_dim: int = 1920,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Cross-attention projections
        self.to_q = nn.Linear(inner_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # QK normalization for stable training
        self.q_norm = nn.LayerNorm(head_dim, elementwise_affine=True)
        self.k_norm = nn.LayerNorm(head_dim, elementwise_affine=True)

        # Output projection
        self.to_out = nn.Linear(inner_dim, inner_dim, bias=False)

        # Tanh gate (init → 0 so adapter starts as identity)
        self.gate = nn.Parameter(torch.tensor(gate_init))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.to_q.weight, gain=0.01)
        nn.init.xavier_uniform_(self.to_k.weight, gain=0.01)
        nn.init.xavier_uniform_(self.to_v.weight, gain=0.01)
        nn.init.xavier_uniform_(self.to_out.weight, gain=0.01)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        unified_context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states:          (B, T_video, D)  video tokens from block output
            encoder_hidden_states:  (B, T_text, D)   text tokens from block output
            unified_context:        (B, N_ctx, D)    unified context [local + global]
            context_mask:           (B, N_ctx)        validity mask

        Returns:
            hidden_states, encoder_hidden_states  (with adapter residual added)
        """
        text_len = encoder_hidden_states.size(1)
        # Concatenate text + video for joint Q
        joint = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # (B, S, D)

        B, S, _ = joint.shape
        N_ctx = unified_context.size(1)

        q = self.to_q(joint)                      # (B, S, D)
        k = self.to_k(unified_context)             # (B, N_ctx, D)
        v = self.to_v(unified_context)             # (B, N_ctx, D)

        # Reshape to multi-head: (B, heads, seq, head_dim)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N_ctx, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N_ctx, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Attention mask
        attn_mask = None
        if context_mask is not None:
            attn_mask = context_mask.unsqueeze(1).unsqueeze(2)          # (B, 1, 1, N_ctx)
            attn_mask = attn_mask.to(dtype=q.dtype)
            attn_mask = (1.0 - attn_mask) * torch.finfo(q.dtype).min

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
        )  # (B, heads, S, head_dim)

        out = out.transpose(1, 2).reshape(B, S, self.inner_dim)        # (B, S, D)
        out = self.to_out(out)

        # Split back
        ctx_text, ctx_video = out.split([text_len, S - text_len], dim=1)

        gate_val = torch.tanh(self.gate)
        hidden_states = hidden_states + gate_val * ctx_video
        encoder_hidden_states = encoder_hidden_states + gate_val * ctx_text

        return hidden_states, encoder_hidden_states


# ===========================================================================
# Multi-Context Dropout
# ===========================================================================

class MultiContextDropout(nn.Module):
    """
    Per-batch-element dropout of local / global context tokens.

    During training:
      - 10% drop global  (forces local-only path → motion learning)
      - 10% drop local   (forces global-only path → identity learning)
      - 80% keep both    (attention competition training)
    """

    def __init__(self, drop_global_prob: float = 0.10, drop_local_prob: float = 0.10):
        super().__init__()
        assert drop_global_prob + drop_local_prob <= 1.0
        self.drop_global_prob = drop_global_prob
        self.drop_local_prob = drop_local_prob

    def forward(
        self,
        local_tokens: torch.Tensor,
        global_tokens: torch.Tensor,
        local_mask: torch.Tensor,
        global_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training:
            return local_tokens, global_tokens, local_mask, global_mask

        B = local_tokens.shape[0]
        rand = torch.rand(B, device=local_tokens.device)

        drop_global = (rand < self.drop_global_prob).float()
        drop_local = (
            (rand >= self.drop_global_prob)
            & (rand < self.drop_global_prob + self.drop_local_prob)
        ).float()

        global_keep = (1.0 - drop_global).unsqueeze(1)
        global_tokens = global_tokens * global_keep.unsqueeze(2)
        global_mask = global_mask * global_keep

        local_keep = (1.0 - drop_local).unsqueeze(1)
        local_tokens = local_tokens * local_keep.unsqueeze(2)
        local_mask = local_mask * local_keep

        return local_tokens, global_tokens, local_mask, global_mask


# ===========================================================================
# DIRECTOR Transformer (post-block adapter design)
# ===========================================================================

class DirectorTransformer(nn.Module):
    """
    CogVideoX backbone + post-block ContextAdapters.

    The backbone is kept completely frozen and unmodified.
    After each transformer block, a ContextAdapter injects unified context
    via cross-attention with tanh-gated residual.

    Gradient checkpointing on the backbone is fully supported.
    """

    def __init__(self, config: DirectorConfig):
        super().__init__()
        self.config = config

        # Load pretrained backbone
        self.backbone = CogVideoXTransformer3DModel.from_pretrained(
            config.backbone,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Context encoders
        self.local_encoder = None  # set via set_vae()
        self.global_encoder = GlobalContextEncoder(
            clip_model_name=config.context.clip_model,
            global_token_count=config.context.global_token_count,
            context_dim=config.context.context_dim,
            clip_vision_dim=config.context.clip_vision_dim,
            max_characters=config.context.max_characters,
        )

        self.context_builder = UnifiedContextBuilder(
            context_dim=config.context.context_dim,
            local_token_count=config.context.local_token_count,
            global_token_count=config.context.global_token_count,
            max_characters=config.context.max_characters,
        )

        self.context_dropout = MultiContextDropout(
            drop_global_prob=config.drop_global_prob,
            drop_local_prob=config.drop_local_prob,
        )

        # Create post-block adapters
        self._create_adapters()

    def _create_adapters(self):
        """Create a ContextAdapter for each injected layer."""
        inject = self.config.inject_layers
        num_blocks = len(self.backbone.transformer_blocks)

        if inject == "all":
            self._inject_indices = list(range(num_blocks))
        else:
            self._inject_indices = list(inject)

        self.adapters = nn.ModuleDict()
        for idx in self._inject_indices:
            self.adapters[str(idx)] = ContextAdapter(
                inner_dim=self.config.inner_dim,
                num_heads=self.config.num_heads,
                head_dim=self.config.head_dim,
                context_dim=self.config.context.context_dim,
                gate_init=self.config.context_gate_init,
            )

    def set_vae(self, vae: nn.Module):
        """Set the VAE for local context encoding."""
        self.local_encoder = LocalContextEncoder(
            vae=vae,
            target_token_count=self.config.context.local_token_count,
            context_dim=self.config.context.context_dim,
            vae_latent_channels=self.config.context.vae_latent_channels,
            vae_spatial_scale=self.config.context.vae_spatial_scale,
        )

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Return only DIRECTOR-added parameters (backbone stays frozen)."""
        trainable = []
        if self.local_encoder is not None:
            trainable.extend(self.local_encoder.parameters())
        trainable.extend(self.global_encoder.parameters())
        trainable.extend(self.context_builder.parameters())
        trainable.extend(self.adapters.parameters())
        return trainable

    # ----- context encoding (unchanged) -----

    def encode_context(
        self,
        prev_frame: Optional[torch.Tensor] = None,
        character_images: Optional[List[torch.Tensor]] = None,
        character_masks: Optional[torch.Tensor] = None,
        precomputed_local_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode local + global context into unified context tokens.

        Returns:
            unified_context: (B, N_total, D)
            unified_mask:    (B, N_total)
        """
        # Determine B and device from any available tensor
        ref = None
        if prev_frame is not None:
            ref = prev_frame
        elif character_images is not None and len(character_images) > 0:
            ref = character_images[0]
        elif precomputed_local_latent is not None:
            ref = precomputed_local_latent
        elif character_masks is not None:
            ref = character_masks

        if ref is None:
            # Fallback: null context for batch=1
            B, device = 1, next(self.parameters()).device
        else:
            B, device = ref.shape[0], ref.device
        dtype = torch.bfloat16

        # Local context
        if prev_frame is not None or precomputed_local_latent is not None:
            local_tokens = self.local_encoder(
                frame=prev_frame, precomputed_latent=precomputed_local_latent,
            )
            local_mask = torch.ones(B, local_tokens.size(1), device=device, dtype=dtype)
        else:
            N_l = self.config.context.local_token_count
            local_tokens = torch.zeros(B, N_l, self.config.context.context_dim, device=device, dtype=dtype)
            local_mask = torch.zeros(B, N_l, device=device, dtype=dtype)

        # Global context
        if character_images is not None and len(character_images) > 0:
            global_tokens, global_mask = self.global_encoder(
                character_images=character_images, character_masks=character_masks,
            )
        else:
            N_g = self.config.context.global_token_count * self.config.context.max_characters
            global_tokens = torch.zeros(B, N_g, self.config.context.context_dim, device=device, dtype=dtype)
            global_mask = torch.zeros(B, N_g, device=device, dtype=dtype)

        # Multi-context dropout
        local_tokens, global_tokens, local_mask, global_mask = self.context_dropout(
            local_tokens, global_tokens, local_mask, global_mask,
        )

        return self.context_builder(local_tokens, global_tokens, local_mask, global_mask)

    # ----- custom forward: backbone blocks + post-block adapters -----

    def _adapter_block(
        self,
        block_idx: int,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        unified_context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply adapter for a given block index (if it exists)."""
        key = str(block_idx)
        if key in self.adapters:
            hidden_states, encoder_hidden_states = self.adapters[key](
                hidden_states, encoder_hidden_states, unified_context, context_mask,
            )
        return hidden_states, encoder_hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        unified_context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        Custom forward that replicates CogVideoX's forward but adds
        post-block adapter calls.
        """
        backbone = self.backbone
        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        t_emb = backbone.time_proj(timestep)
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = backbone.time_embedding(t_emb, timestep_cond)

        if hasattr(backbone, 'ofs_embedding') and backbone.ofs_embedding is not None:
            pass  # CogVideoX-2b doesn't use OFS

        # 2. Patch embedding
        hidden_states = backbone.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = backbone.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks + adapters
        for i, block in enumerate(backbone.transformer_blocks):
            if torch.is_grad_enabled() and backbone.gradient_checkpointing:
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    None,  # attention_kwargs
                    use_reentrant=False,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )

            # Post-block adapter (outside gradient checkpoint scope of the block)
            if unified_context is not None:
                hidden_states, encoder_hidden_states = self._adapter_block(
                    i, hidden_states, encoder_hidden_states,
                    unified_context, context_mask,
                )

        # 4. Final norm + projection
        hidden_states = backbone.norm_final(hidden_states)
        hidden_states = backbone.norm_out(hidden_states, temb=emb)
        hidden_states = backbone.proj_out(hidden_states)

        # 5. Unpatchify
        p = backbone.config.patch_size
        p_t = backbone.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


# ===========================================================================
# DIRECTOR Pipeline
# ===========================================================================

class DirectorPipeline:
    """
    Full DIRECTOR pipeline wrapping CogVideoXPipeline.

    Provides methods for:
      - Training: encode context + run forward for loss computation
      - Inference: multi-shot autoregressive generation with Multi-CFG
    """

    def __init__(
        self,
        config: DirectorConfig,
        device: torch.device = torch.device("cuda"),
    ):
        self.config = config
        self.device = device

        # Load base CogVideoX pipeline
        self.base_pipeline = CogVideoXPipeline.from_pretrained(
            config.backbone,
            torch_dtype=torch.bfloat16,
        )

        # Create DIRECTOR transformer
        self.director_transformer = DirectorTransformer(config)
        self.director_transformer.set_vae(self.base_pipeline.vae)

        # Move to device
        self.base_pipeline.vae.to(device)
        self.base_pipeline.text_encoder.to(device)
        self.director_transformer.to(device)

        # Expose components
        self.vae = self.base_pipeline.vae
        self.text_encoder = self.base_pipeline.text_encoder
        self.tokenizer = self.base_pipeline.tokenizer
        self.scheduler = self.base_pipeline.scheduler

    @torch.no_grad()
    def encode_text(self, prompt: str, max_length: int = 226) -> torch.Tensor:
        """Encode text prompt via T5.  Returns (1, S, 4096)."""
        inputs = self.tokenizer(
            prompt, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        ).to(self.device)
        return self.text_encoder(**inputs).last_hidden_state

    @torch.no_grad()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video (B, T, 3, H, W) [0,1] → latents (B, T_lat, C, H_lat, W_lat)."""
        B, T, C, H, W = video.shape
        video = video.permute(0, 2, 1, 3, 4)  # (B, 3, T, H, W)
        posterior = self.vae.encode(video)
        latent = posterior.latent_dist.mode() if hasattr(posterior, 'latent_dist') else posterior
        latent = latent * self.vae.config.scaling_factor
        if latent.dim() == 5:
            latent = latent.permute(0, 2, 1, 3, 4)
        return latent

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latents → video (B, T, 3, H, W) [0,1]."""
        latent = latent.permute(0, 2, 1, 3, 4) / self.vae.config.scaling_factor
        video = self.vae.decode(latent).sample
        return video.permute(0, 2, 1, 3, 4).clamp(0, 1)

    def compute_flow_matching_loss(
        self,
        x_1: torch.Tensor,
        text_embeds: torch.Tensor,
        unified_context: torch.Tensor,
        context_mask: torch.Tensor,
        sigma_min: float = 0.001,
        time_sampling: str = "logit_normal",
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Flow matching loss:
          L = E ||v_theta(X_t, t, ...) - (X_1 - X_0)||^2
          X_t = t*X_1 + (1-t)*X_0,  X_0 ~ N(0,I)
        """
        B = x_1.shape[0]
        device, dtype = x_1.device, x_1.dtype

        x_0 = torch.randn_like(x_1)

        # Sample timestep
        if time_sampling == "logit_normal":
            u = torch.randn(B, device=device, dtype=dtype) * logit_normal_std + logit_normal_mean
            t = torch.sigmoid(u)
        else:
            t = torch.rand(B, device=device, dtype=dtype)

        t = t.clamp(sigma_min, 1.0 - sigma_min)
        t_exp = t.view(B, 1, 1, 1, 1)

        x_t = t_exp * x_1 + (1.0 - t_exp) * x_0
        v_target = x_1 - x_0

        timestep = (t * 1000.0).long()

        v_pred = self.director_transformer(
            hidden_states=x_t,
            encoder_hidden_states=text_embeds,
            timestep=timestep,
            unified_context=unified_context,
            context_mask=context_mask,
            return_dict=False,
        )[0]

        loss = F.mse_loss(v_pred, v_target, reduction="mean")

        with torch.no_grad():
            pred_norm = v_pred.flatten(1).norm(dim=1).mean()
            target_norm = v_target.flatten(1).norm(dim=1).mean()

        return {
            "loss": loss,
            "pred_norm": pred_norm,
            "target_norm": target_norm,
            "timestep_mean": t.mean(),
        }

    # ----- Inference -----

    @torch.no_grad()
    def generate_shot(
        self,
        prompt: str,
        prev_frame: Optional[torch.Tensor] = None,
        character_images: Optional[List[torch.Tensor]] = None,
        character_masks: Optional[torch.Tensor] = None,
        omega_text: float = 6.0,
        omega_local: float = 2.0,
        omega_global: float = 3.0,
        num_steps: int = 50,
        height: int = 480,
        width: int = 720,
        num_frames: int = 49,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Single shot generation with Multi-CFG:
          v_out = v_null + w_t*(v_text - v_null) + w_l*(v_local - v_null) + w_g*(v_global - v_null)
        """
        self.director_transformer.eval()
        self.director_transformer.to(torch.bfloat16)

        text_embeds = self.encode_text(prompt)
        null_text_embeds = self.encode_text("")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ctx_full, mask_full = self.director_transformer.encode_context(
                prev_frame=prev_frame, character_images=character_images,
                character_masks=character_masks,
            )
            ctx_local, mask_local = self.director_transformer.encode_context(
                prev_frame=prev_frame, character_images=None,
            )
            ctx_global, mask_global = self.director_transformer.encode_context(
                prev_frame=None, character_images=character_images,
                character_masks=character_masks,
            )
            ctx_null, mask_null = self.director_transformer.encode_context(
                prev_frame=None, character_images=None,
            )

        latent_h, latent_w = height // 8, width // 8
        latent_t = (num_frames - 1) // 4 + 1
        latent_c = self.vae.config.latent_channels

        x = torch.randn(
            1, latent_t, latent_c, latent_h, latent_w,
            device=self.device, dtype=torch.bfloat16, generator=generator,
        )

        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_val = step * dt
            t_tensor = torch.tensor([t_val * 1000.0], device=self.device, dtype=torch.long)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                fwd = lambda te, ctx, msk: self.director_transformer(
                    hidden_states=x, encoder_hidden_states=te, timestep=t_tensor,
                    unified_context=ctx, context_mask=msk, return_dict=False,
                )[0]

                v_null  = fwd(null_text_embeds, ctx_null, mask_null)
                v_text  = fwd(text_embeds, ctx_null, mask_null)
                v_local = fwd(text_embeds, ctx_local, mask_local)
                v_glob  = fwd(text_embeds, ctx_global, mask_global)

            v_out = (v_null
                     + omega_text  * (v_text  - v_null)
                     + omega_local * (v_local - v_null)
                     + omega_global* (v_glob  - v_null))

            x = x + dt * v_out

        return self.decode_latent(x)

    @torch.no_grad()
    def generate_multi_shot(
        self,
        prompts: List[str],
        character_images: Optional[List[torch.Tensor]] = None,
        character_masks: Optional[torch.Tensor] = None,
        omega_text: float = 6.0,
        omega_local: float = 2.0,
        omega_global: float = 3.0,
        num_steps: int = 50,
        height: int = 480,
        width: int = 720,
        num_frames: int = 49,
        seed: int = 42,
    ) -> List[torch.Tensor]:
        """Autoregressive multi-shot generation with O(1) memory per shot."""
        all_shots = []
        prev_frame = None

        for shot_idx, prompt in enumerate(prompts):
            gen = torch.Generator(device=self.device)
            gen.manual_seed(seed + shot_idx)

            video = self.generate_shot(
                prompt=prompt, prev_frame=prev_frame,
                character_images=character_images, character_masks=character_masks,
                omega_text=omega_text, omega_local=omega_local, omega_global=omega_global,
                num_steps=num_steps, height=height, width=width, num_frames=num_frames,
                generator=gen,
            )
            all_shots.append(video)
            prev_frame = video[:, -1]
            torch.cuda.empty_cache()

        return all_shots
