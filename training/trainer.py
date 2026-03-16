"""
DIRECTOR Training Loop.

Implements flow matching training with:
  - Multi-context dropout
  - Mixed precision (BF16) via torch.amp
  - Gradient checkpointing
  - TensorBoard logging
  - Checkpoint management (best + last + periodic)
  - Proper seeding for full reproducibility
  - Optional DeepSpeed/Accelerate integration

Flow Matching Objective:
  L(theta) = E_{t, X_0, X_1, C} ||v_theta(X_t, t, m_text*E_text, m_local*Z_local, m_global*Z_global) - (X_1 - X_0)||^2
  where X_t = t*X_1 + (1-t)*X_0, X_0 ~ N(0,I)
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import yaml

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class DirectorTrainer:
    """
    Training loop for the DIRECTOR model.

    Handles:
      - Flow matching loss computation with multi-context dropout
      - Mixed precision training (BF16)
      - Gradient accumulation and clipping
      - Learning rate scheduling with warmup
      - TensorBoard logging
      - Checkpoint save/load
    """

    def __init__(
        self,
        director_pipeline,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
    ):
        # Load config
        if config is None and config_path is not None:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        self.config = config or {}

        self.pipeline = director_pipeline
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training config
        train_cfg = self.config.get("training", {})
        self.num_epochs = train_cfg.get("num_epochs", 50)
        self.max_steps = train_cfg.get("max_steps", 100000)
        self.grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 4)
        self.max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        self.mixed_precision = train_cfg.get("mixed_precision", "bf16")
        self.gradient_checkpointing = train_cfg.get("gradient_checkpointing", True)

        # Flow matching config
        fm_cfg = train_cfg.get("flow_matching", {})
        self.sigma_min = fm_cfg.get("sigma_min", 0.001)
        self.time_sampling = fm_cfg.get("time_sampling", "logit_normal")
        self.logit_normal_mean = fm_cfg.get("logit_normal_mean", 0.0)
        self.logit_normal_std = fm_cfg.get("logit_normal_std", 1.0)

        # Logging config
        log_cfg = train_cfg.get("logging", {})
        self.log_every = log_cfg.get("log_every_steps", 50)
        self.log_grad_norm = log_cfg.get("log_grad_norm", True)
        tb_dir = log_cfg.get("tensorboard_dir", "logs/tensorboard")
        self.writer = SummaryWriter(log_dir=tb_dir)

        # Checkpoint config
        ckpt_cfg = train_cfg.get("checkpoint", {})
        self.save_every = ckpt_cfg.get("save_every_steps", 5000)
        self.save_dir = Path(ckpt_cfg.get("save_dir", "checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = ckpt_cfg.get("keep_last_n", 3)
        self.save_best = ckpt_cfg.get("save_best", True)
        self.best_metric = ckpt_cfg.get("best_metric", "loss")
        self.best_metric_val = float("inf")

        # Set up device
        self.device = director_pipeline.device

        # Enable gradient checkpointing
        if self.gradient_checkpointing:
            self.pipeline.director_transformer.backbone.enable_gradient_checkpointing()

        # Collect trainable parameters
        self.trainable_params = self.pipeline.director_transformer.get_trainable_parameters()
        total_params = sum(p.numel() for p in self.trainable_params)
        logger.info(f"Trainable parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

        # Optimizer
        opt_cfg = train_cfg.get("optimizer", {})
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=opt_cfg.get("lr", 1e-5),
            weight_decay=opt_cfg.get("weight_decay", 0.01),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            eps=opt_cfg.get("eps", 1e-8),
        )

        # Scheduler
        sched_cfg = train_cfg.get("scheduler", {})
        warmup_steps = sched_cfg.get("warmup_steps", 500)
        min_lr_ratio = sched_cfg.get("min_lr_ratio", 0.1)
        total_steps = min(self.max_steps, self.num_epochs * len(train_loader))

        self.scheduler = self._create_scheduler(
            sched_cfg.get("type", "cosine"),
            warmup_steps,
            total_steps,
            min_lr_ratio,
        )

        # Mixed precision
        self.use_amp = self.mixed_precision in ("bf16", "fp16")
        self.amp_dtype = torch.bfloat16 if self.mixed_precision == "bf16" else torch.float16
        # BF16 does not need GradScaler; FP16 does
        self.scaler = GradScaler(enabled=(self.mixed_precision == "fp16"))

        # State tracking
        self.global_step = 0
        self.epoch = 0
        self.running_loss = 0.0
        self.saved_checkpoints = []

    def _create_scheduler(
        self, scheduler_type: str, warmup_steps: int, total_steps: int, min_lr_ratio: float
    ):
        """Create learning rate scheduler with linear warmup."""

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            if scheduler_type == "cosine":
                return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
                    1.0 + math.cos(math.pi * progress)
                )
            elif scheduler_type == "linear":
                return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)
            else:
                return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(self):
        """Main training loop."""
        logger.info("Starting DIRECTOR training...")
        logger.info(f"  Epochs: {self.num_epochs}")
        logger.info(f"  Max steps: {self.max_steps}")
        logger.info(f"  Gradient accumulation: {self.grad_accum_steps}")
        logger.info(f"  Mixed precision: {self.mixed_precision}")
        logger.info(f"  Batch size per step: {self.train_loader.batch_size}")
        logger.info(f"  Effective batch size: {self.train_loader.batch_size * self.grad_accum_steps}")

        self.pipeline.director_transformer.train()

        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch
            epoch_loss = self._train_epoch()

            logger.info(f"Epoch {epoch}: avg_loss={epoch_loss:.6f}")
            self.writer.add_scalar("epoch/loss", epoch_loss, epoch)

            # Validation
            if self.val_loader is not None:
                val_loss = self._validate()
                logger.info(f"Epoch {epoch}: val_loss={val_loss:.6f}")
                self.writer.add_scalar("epoch/val_loss", val_loss, epoch)

                # Save best model
                if self.save_best and val_loss < self.best_metric_val:
                    self.best_metric_val = val_loss
                    self._save_checkpoint("best")
                    logger.info(f"New best model: val_loss={val_loss:.6f}")

            if self.global_step >= self.max_steps:
                break

        # Save final checkpoint
        self._save_checkpoint("last")
        self.writer.close()
        logger.info("Training complete.")

    def _train_epoch(self) -> float:
        """Train for one epoch. Returns average loss."""
        self.pipeline.director_transformer.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            if self.global_step >= self.max_steps:
                break

            loss_dict = self._train_step(batch)
            loss_val = loss_dict["loss"].item()
            total_loss += loss_val
            num_batches += 1
            self.running_loss += loss_val

            # Logging
            if self.global_step % self.log_every == 0 and self.global_step > 0:
                avg_loss = self.running_loss / self.log_every
                lr = self.scheduler.get_last_lr()[0]

                self.writer.add_scalar("train/loss", avg_loss, self.global_step)
                self.writer.add_scalar("train/lr", lr, self.global_step)
                self.writer.add_scalar(
                    "train/pred_norm", loss_dict.get("pred_norm", 0), self.global_step
                )
                self.writer.add_scalar(
                    "train/target_norm", loss_dict.get("target_norm", 0), self.global_step
                )
                self.writer.add_scalar(
                    "train/timestep_mean", loss_dict.get("timestep_mean", 0), self.global_step
                )

                if self.log_grad_norm:
                    grad_norm = self._compute_grad_norm()
                    self.writer.add_scalar("train/grad_norm", grad_norm, self.global_step)

                # Log gate values from DIRECTOR adapters
                gate_values = self._get_gate_values()
                if gate_values:
                    gate_mean = np.mean(gate_values)
                    gate_max = np.max(gate_values)
                    gate_min = np.min(gate_values)
                    gate_std = np.std(gate_values)
                    self.writer.add_scalar("train/gate_mean", gate_mean, self.global_step)
                    self.writer.add_scalar("train/gate_max", gate_max, self.global_step)
                    self.writer.add_scalar("train/gate_min", gate_min, self.global_step)
                    self.writer.add_scalar("train/gate_std", gate_std, self.global_step)
                    # Per-layer gate values (every 5th layer to avoid clutter)
                    for i, gv in enumerate(gate_values):
                        if i % 5 == 0 or i == len(gate_values) - 1:
                            self.writer.add_scalar(f"gates/layer_{i}", gv, self.global_step)

                gate_str = f", gate={np.mean(gate_values):.6f}" if gate_values else ""
                gn_str = f"{grad_norm:.2e}" if self.log_grad_norm else "N/A"
                logger.info(
                    f"Step {self.global_step}: loss={avg_loss:.6f}, lr={lr:.2e}, "
                    f"grad_norm={gn_str}{gate_str}"
                )
                self.running_loss = 0.0

            # Periodic checkpoint
            if self.global_step % self.save_every == 0 and self.global_step > 0:
                self._save_checkpoint(f"step_{self.global_step}")

        return total_loss / max(1, num_batches)

    def _train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Execute one training step with gradient accumulation.

        Args:
            batch: dict from DirectorDataCollator

        Returns:
            loss_dict with 'loss' and auxiliary metrics
        """
        # Move batch to device
        target_video = batch["target_video"].to(self.device, dtype=torch.float32)
        prev_frame = batch["prev_frame"].to(self.device, dtype=torch.float32)
        prev_prev_frame = batch["prev_prev_frame"].to(self.device, dtype=torch.float32)
        has_prev_prev = batch["has_prev_prev"]  # (B,) bool
        anchor_rgb = batch["anchor_rgb"].to(self.device, dtype=torch.float32)  # (B, K, 3, 224, 224)
        char_mask = batch["character_mask"].to(self.device, dtype=torch.float32)  # (B, K)
        captions = batch["captions"]

        B, K = anchor_rgb.shape[:2]

        # Build prev_frames list: always [t-1, t-2] with per-sample validity mask
        prev_frames = [prev_frame, prev_prev_frame]
        # local_frame_valid: (num_frames, B) — t-1 always valid, t-2 per-sample
        local_frame_valid = torch.stack([
            torch.ones(B, device=self.device),                           # t-1: always valid
            has_prev_prev.float().to(self.device),                       # t-2: per-sample
        ], dim=0)  # (2, B)

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            # 1. Encode target video through VAE to get latents X_1
            with torch.no_grad():
                x_1 = self.pipeline.encode_video(target_video)  # (B, T_lat, C, H_lat, W_lat)

            # 2. Encode text
            with torch.no_grad():
                text_embeds_list = []
                for caption in captions:
                    te = self.pipeline.encode_text(caption)  # (1, S, D_text)
                    text_embeds_list.append(te)
                text_embeds = torch.cat(text_embeds_list, dim=0)  # (B, S, D_text)

            # 3. Encode context (local + global) with multi-context dropout
            # Decompose anchor_rgb (B, K, 3, 224, 224) -> list of K tensors [(B, 3, 224, 224)]
            char_list = [anchor_rgb[:, k] for k in range(K)]

            unified_context, context_mask = self.pipeline.director_transformer.encode_context(
                prev_frames=prev_frames,
                character_images=char_list,
                character_masks=char_mask,
                local_frame_valid=local_frame_valid,
            )  # (B, N_ctx, D), (B, N_ctx)

            # 4. Compute flow matching loss
            loss_dict = self.pipeline.compute_flow_matching_loss(
                x_1=x_1,
                text_embeds=text_embeds,
                unified_context=unified_context,
                context_mask=context_mask,
                sigma_min=self.sigma_min,
                time_sampling=self.time_sampling,
                logit_normal_mean=self.logit_normal_mean,
                logit_normal_std=self.logit_normal_std,
            )

            loss = loss_dict["loss"] / self.grad_accum_steps

        # Backward pass
        self.scaler.scale(loss).backward()

        # Gradient accumulation step
        self.global_step += 1
        if self.global_step % self.grad_accum_steps == 0:
            # Unscale for grad clipping
            self.scaler.unscale_(self.optimizer)

            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.trainable_params, self.max_grad_norm
                )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

        return loss_dict

    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation loop. Returns average validation loss."""
        self.pipeline.director_transformer.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            target_video = batch["target_video"].to(self.device, dtype=torch.float32)
            prev_frame = batch["prev_frame"].to(self.device, dtype=torch.float32)
            prev_prev_frame = batch["prev_prev_frame"].to(self.device, dtype=torch.float32)
            has_prev_prev = batch["has_prev_prev"]
            anchor_rgb = batch["anchor_rgb"].to(self.device, dtype=torch.float32)  # (B, K, 3, 224, 224)
            char_mask = batch["character_mask"].to(self.device, dtype=torch.float32)  # (B, K)
            captions = batch["captions"]
            B, K = anchor_rgb.shape[:2]

            prev_frames = [prev_frame, prev_prev_frame]
            local_frame_valid = torch.stack([
                torch.ones(B, device=self.device),
                has_prev_prev.float().to(self.device),
            ], dim=0)

            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                x_1 = self.pipeline.encode_video(target_video)
                text_embeds_list = [self.pipeline.encode_text(c) for c in captions]
                text_embeds = torch.cat(text_embeds_list, dim=0)

                char_list = [anchor_rgb[:, k] for k in range(K)]
                unified_context, context_mask = self.pipeline.director_transformer.encode_context(
                    prev_frames=prev_frames,
                    character_images=char_list,
                    character_masks=char_mask,
                    local_frame_valid=local_frame_valid,
                )

                loss_dict = self.pipeline.compute_flow_matching_loss(
                    x_1=x_1,
                    text_embeds=text_embeds,
                    unified_context=unified_context,
                    context_mask=context_mask,
                )

            total_loss += loss_dict["loss"].item()
            num_batches += 1

        self.pipeline.director_transformer.train()
        return total_loss / max(1, num_batches)

    def _compute_grad_norm(self) -> float:
        """Compute the total gradient norm across all trainable parameters."""
        total_norm = 0.0
        for p in self.trainable_params:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def _get_gate_values(self) -> List[float]:
        """Extract tanh(gate) values from all DIRECTOR context adapters."""
        gate_vals = []
        adapters = self.pipeline.director_transformer.adapters
        for key, adapter in adapters.items():
            gate_vals.append(torch.tanh(adapter.gate).item())
        return gate_vals

    def _save_checkpoint(self, name: str):
        """Save a training checkpoint."""
        ckpt_path = self.save_dir / f"checkpoint_{name}.pt"

        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric_val": self.best_metric_val,
            "adapters": self.pipeline.director_transformer.adapters.state_dict(),
            "context_builder": self.pipeline.director_transformer.context_builder.state_dict(),
            "global_encoder": {
                k: v
                for k, v in self.pipeline.director_transformer.global_encoder.state_dict().items()
                if "clip_vision" not in k  # Don't save frozen CLIP weights
            },
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

        if self.pipeline.director_transformer.local_encoder is not None:
            state["local_encoder"] = {
                k: v
                for k, v in self.pipeline.director_transformer.local_encoder.state_dict().items()
                if "vae" not in k  # Don't save frozen VAE weights
            }

        torch.save(state, ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")

        # Track checkpoints for cleanup
        if name not in ("best", "last"):
            self.saved_checkpoints.append(ckpt_path)
            while len(self.saved_checkpoints) > self.keep_last_n:
                old = self.saved_checkpoints.pop(0)
                if old.exists():
                    old.unlink()
                    logger.info(f"Removed old checkpoint: {old}")

    def load_checkpoint(self, path: str):
        """Load a training checkpoint."""
        state = torch.load(path, map_location=self.device)

        self.global_step = state["global_step"]
        self.epoch = state["epoch"]
        self.best_metric_val = state.get("best_metric_val", float("inf"))

        self.pipeline.director_transformer.adapters.load_state_dict(
            state["adapters"]
        )
        self.pipeline.director_transformer.context_builder.load_state_dict(
            state["context_builder"]
        )

        # Load global encoder (partial, excluding frozen CLIP)
        if "global_encoder" in state:
            missing, unexpected = self.pipeline.director_transformer.global_encoder.load_state_dict(
                state["global_encoder"], strict=False
            )

        if "local_encoder" in state and self.pipeline.director_transformer.local_encoder is not None:
            missing, unexpected = self.pipeline.director_transformer.local_encoder.load_state_dict(
                state["local_encoder"], strict=False
            )

        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])

        logger.info(f"Loaded checkpoint from {path} (step={self.global_step}, epoch={self.epoch})")


def main():
    """Entry point for standalone training."""
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from models.director_model import DirectorConfig, DirectorPipeline
    from data.dataset import create_dataloader

    parser = argparse.ArgumentParser(description="DIRECTOR Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(config.get("seed", 42))

    # Set device
    device = torch.device(f"cuda:{config.get('cuda_device', 0)}")

    # Create model
    model_cfg = config.get("model", {})
    from models.context_encoder import ContextConfig
    ctx_cfg = model_cfg.get("context", {})
    dropout_cfg = model_cfg.get("dropout", {})

    director_config = DirectorConfig(
        backbone=model_cfg.get("backbone", "THUDM/CogVideoX-2b"),
        inner_dim=model_cfg.get("inner_dim", 1920),
        text_embed_dim=model_cfg.get("text_embed_dim", 4096),
        num_heads=model_cfg.get("num_heads", 30),
        head_dim=model_cfg.get("head_dim", 64),
        num_layers=model_cfg.get("num_layers", 30),
        context=ContextConfig(
            local_token_count=ctx_cfg.get("local_token_count", 256),
            num_local_frames=ctx_cfg.get("num_local_frames", 2),
            global_token_count=ctx_cfg.get("global_token_count", 64),
            max_characters=ctx_cfg.get("max_characters", 4),
            context_dim=ctx_cfg.get("context_dim", 1920),
            clip_vision_dim=ctx_cfg.get("clip_vision_dim", 1024),
            clip_model=ctx_cfg.get("clip_model", "openai/clip-vit-large-patch14"),
        ),
        drop_global_prob=dropout_cfg.get("drop_global_prob", 0.10),
        drop_local_prob=dropout_cfg.get("drop_local_prob", 0.10),
        inject_layers=model_cfg.get("attention", {}).get("inject_layers", "all"),
        context_gate_init=model_cfg.get("attention", {}).get("context_gate_init", 0.0),
    )

    logger.info("Initializing DIRECTOR pipeline...")
    pipeline = DirectorPipeline(config=director_config, device=device)

    # Create dataloaders
    ds_cfg = config.get("dataset", {})
    dl_cfg = ds_cfg.get("dataloader", {})
    vid_cfg = ds_cfg.get("video", {})

    dataset_dir = ds_cfg.get("dataset_dir", ds_cfg.get("triplet_dir", "data/processed_dataset"))
    train_cfg = config.get("training", {})
    # Use training-specific resolution (may differ from native video resolution)
    train_h = train_cfg.get("train_height", vid_cfg.get("height", 480))
    train_w = train_cfg.get("train_width", vid_cfg.get("width", 720))
    train_f = train_cfg.get("train_frames", vid_cfg.get("num_frames", 49))

    train_loader = create_dataloader(
        dataset_dir=dataset_dir,
        batch_size=dl_cfg.get("batch_size", 1),
        num_workers=dl_cfg.get("num_workers", 4),
        split="train",
        target_height=train_h,
        target_width=train_w,
        target_frames=train_f,
        tokenizer=pipeline.tokenizer,
        seed=config.get("seed", 42),
    )

    val_loader = create_dataloader(
        dataset_dir=dataset_dir,
        batch_size=dl_cfg.get("batch_size", 1),
        num_workers=dl_cfg.get("num_workers", 4),
        split="val",
        target_height=train_h,
        target_width=train_w,
        target_frames=train_f,
        tokenizer=pipeline.tokenizer,
        seed=config.get("seed", 42),
    )

    # Create trainer
    trainer = DirectorTrainer(
        director_pipeline=pipeline,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
