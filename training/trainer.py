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

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

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
        local_rank: int = -1,
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
        # Only main process writes TensorBoard
        self.writer = None  # set after DDP state is determined

        # Checkpoint config
        ckpt_cfg = train_cfg.get("checkpoint", {})
        self.save_every = ckpt_cfg.get("save_every_steps", 5000)
        self.save_dir = Path(ckpt_cfg.get("save_dir", "checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = ckpt_cfg.get("keep_last_n", 3)
        self.save_best = ckpt_cfg.get("save_best", True)
        self.best_metric = ckpt_cfg.get("best_metric", "loss")
        self.best_metric_val = float("inf")

        # Sample generation config
        sample_cfg = train_cfg.get("sample_generation", {})
        self.sample_every_steps = sample_cfg.get("every_steps", 10000)
        self.sample_num_steps = sample_cfg.get("num_steps", 20)
        self.sample_num_frames_to_log = sample_cfg.get("num_frames_to_log", 8)
        self.sample_dir = Path(sample_cfg.get("save_dir", "samples"))
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.ablation_every_steps = sample_cfg.get("ablation_every_steps", 10000)
        self.multishot_every_steps = sample_cfg.get("multishot_every_steps", 10000)
        self.multishot_num_shots = sample_cfg.get("multishot_num_shots", 2)
        self._fixed_sample = None  # cached validation sample for consistent comparison
        self._last_sample_step = 0  # track last sample generation step

        # T5/CLIP CPU offloading config
        self.offload_text_encoder = train_cfg.get("offload_text_encoder", False)

        # Text embedding cache: pre-encode all captions once, then free T5 entirely
        self.cache_text_embeddings = train_cfg.get("cache_text_embeddings", False)
        self._text_embed_cache: Dict[str, torch.Tensor] = {}  # caption_str → (1, S, D) CPU tensor

        # Set up device
        self.device = director_pipeline.device

        # Enable gradient checkpointing
        if self.gradient_checkpointing:
            self.pipeline.director_transformer.backbone.enable_gradient_checkpointing()

        # Collect trainable parameters with separate groups
        opt_cfg = train_cfg.get("optimizer", {})
        base_lr = opt_cfg.get("lr", 1e-5)
        adapter_lr = opt_cfg.get("adapter_lr", base_lr)
        gate_lr = opt_cfg.get("gate_lr", adapter_lr)

        param_groups_dict = self.pipeline.director_transformer.get_trainable_param_groups()
        backbone_lr = opt_cfg.get("backbone_lr", base_lr * 0.1)
        lora_lr = opt_cfg.get("lora_lr", backbone_lr)
        self.trainable_params = (
            param_groups_dict["encoder"]
            + param_groups_dict["adapter"]
            + param_groups_dict["gate"]
            + param_groups_dict.get("backbone", [])
            + param_groups_dict.get("lora", [])
        )
        total_params = sum(p.numel() for p in self.trainable_params)
        enc_count = sum(p.numel() for p in param_groups_dict["encoder"])
        adp_count = sum(p.numel() for p in param_groups_dict["adapter"])
        gate_count = sum(p.numel() for p in param_groups_dict["gate"])
        backbone_count = sum(p.numel() for p in param_groups_dict.get("backbone", []))
        lora_count = sum(p.numel() for p in param_groups_dict.get("lora", []))
        logger.info(
            f"Trainable parameters: {total_params:,} ({total_params / 1e6:.2f}M) "
            f"[encoder={enc_count:,}, adapter={adp_count:,}, gate={gate_count:,}, "
            f"backbone={backbone_count:,}, lora={lora_count:,}]"
        )
        lr_info = f"LR: encoder={base_lr:.1e}, adapter={adapter_lr:.1e}, gate={gate_lr:.1e}"
        if backbone_count > 0:
            lr_info += f", backbone={backbone_lr:.1e}"
        if lora_count > 0:
            lr_info += f", lora={lora_lr:.1e}"
        logger.info(lr_info)

        # Optimizer with separate param groups
        param_groups = []
        if param_groups_dict["encoder"]:
            param_groups.append({"params": param_groups_dict["encoder"], "lr": base_lr})
        if param_groups_dict["adapter"]:
            param_groups.append({"params": param_groups_dict["adapter"], "lr": adapter_lr})
        if param_groups_dict["gate"]:
            param_groups.append({"params": param_groups_dict["gate"], "lr": gate_lr, "weight_decay": 0.0})
            # Gate gradient amplification (skip if gate is frozen)
            gate_grad_scale = opt_cfg.get("gate_grad_scale", 1.0)
            if gate_grad_scale != 1.0:
                for gp in param_groups_dict["gate"]:
                    if gp.requires_grad:
                        gp.register_hook(lambda grad, s=gate_grad_scale: grad * s)
                logger.info(f"Gate gradient amplification: {gate_grad_scale}x")
        if param_groups_dict.get("backbone"):
            param_groups.append({"params": param_groups_dict["backbone"], "lr": backbone_lr})
        if param_groups_dict.get("lora"):
            param_groups.append({"params": param_groups_dict["lora"], "lr": lora_lr})

        opt_type = opt_cfg.get("type", "adamw")
        opt_kwargs = dict(
            lr=base_lr,
            weight_decay=opt_cfg.get("weight_decay", 0.01),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            eps=opt_cfg.get("eps", 1e-8),
        )
        if opt_type == "adamw8bit":
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(param_groups, **opt_kwargs)
            logger.info("Using 8-bit AdamW (bitsandbytes)")
        else:
            self.optimizer = torch.optim.AdamW(param_groups, **opt_kwargs)

        # Store initial LR per param group (used to override stale LR from checkpoints)
        self._initial_param_groups_lr = [pg["lr"] for pg in param_groups]

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

        # DDP setup
        self.local_rank = local_rank
        self.is_distributed = local_rank >= 0 and dist.is_initialized()
        self.is_main_process = not self.is_distributed or local_rank == 0

        if self.is_distributed:
            # Manual gradient sync instead of DDP wrapping.
            # DDP wrapping the full transformer adds ~1GB overhead for gradient buckets,
            # pushing past GPU memory limits. Instead, we manually all_reduce gradients
            # for trainable parameters after backward.
            logger.info(f"DDP enabled on rank {local_rank}, world_size={dist.get_world_size()} (manual grad sync)")

        # Create TensorBoard writer only on main process
        if self.is_main_process:
            self.writer = SummaryWriter(log_dir=tb_dir)

    @property
    def _transformer(self):
        """Get the unwrapped DirectorTransformer (handles DDP wrapping)."""
        t = self.pipeline.director_transformer
        return t.module if hasattr(t, "module") else t

    def _precache_text_embeddings(self):
        """Pre-encode all unique captions with T5 and cache on CPU.

        After caching, T5 text encoder is deleted entirely to free ~9.4GB VRAM/RAM.
        This is much more efficient than per-step CPU offloading because:
        1. No T5 GPU↔CPU transfers during training
        2. No T5 forward pass overhead per step
        3. T5 memory is freed permanently (not just offloaded)
        """
        import json

        cache_path = self.save_dir / "text_embed_cache.pt"

        # Try loading existing cache
        if cache_path.exists():
            logger.info(f"Loading text embedding cache from {cache_path}")
            self._text_embed_cache = torch.load(cache_path, map_location="cpu")
            logger.info(f"  Loaded {len(self._text_embed_cache)} cached embeddings")
        else:
            # Collect all unique captions directly from caption.json files (no video loading)
            logger.info("Pre-caching text embeddings for all dataset captions...")
            dataset = self.train_loader.dataset
            unique_captions = set()
            for seq_dir in dataset.seq_dirs:
                caption_path = seq_dir / "caption.json"
                if caption_path.exists():
                    with open(caption_path) as f:
                        cap_data = json.load(f)
                    full_raw = cap_data.get("full", "")
                    if isinstance(full_raw, str) and full_raw:
                        unique_captions.add(full_raw)
                    else:
                        # Reconstruct from identity + motion (same logic as dataset)
                        identity_raw = cap_data.get("identity", "")
                        motion_raw = cap_data.get("motion", "")
                        if isinstance(identity_raw, dict):
                            identity_text = ", ".join(f"{v}" for v in identity_raw.values() if v and v != "neutral")
                        else:
                            identity_text = str(identity_raw)
                        if isinstance(motion_raw, dict):
                            motion_text = ", ".join(f"{v}" for v in motion_raw.values() if v and v != "neutral")
                        else:
                            motion_text = str(motion_raw)
                        unique_captions.add(f"{identity_text}. {motion_text}".strip(". "))
            # Also add null caption for CFG
            unique_captions.add("")

            # Add 5-shot evaluation prompts so they're cached too
            _eval_prompts = [
                "The character walks forward through the scene as the camera steadily dollies in, "
                "revealing details of the environment. The lighting is natural and the composition "
                "focuses on the character's presence in the space.",
                "The same character continues through the scene as the camera pans right while "
                "moving forward, tracking their movement through the environment. The character "
                "maintains a natural walking pace.",
                "Cut to a new scene: a different character appears in a completely different "
                "environment. The camera captures them from a medium angle as they stand or "
                "move in their new surroundings, establishing a clear scene change.",
                "Both characters are now visible together in the same scene. The camera holds "
                "a wide static shot showing them interacting in a shared space. Each character "
                "maintains their distinct appearance and positioning.",
                "The scene continues with only the first character visible. The camera pans left "
                "while moving forward, following the character as they walk alone through the "
                "environment. The second character is no longer present.",
            ]
            for p in _eval_prompts:
                unique_captions.add(p)

            logger.info(f"  Found {len(unique_captions)} unique captions")

            # Encode all captions
            self.pipeline.text_encoder.to(self.device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                for i, caption in enumerate(sorted(unique_captions)):
                    te = self.pipeline.encode_text(caption)  # (1, S, D)
                    self._text_embed_cache[caption] = te.cpu()
                    if (i + 1) % 100 == 0:
                        logger.info(f"  Encoded {i+1}/{len(unique_captions)} captions")

            # Save cache to disk for restarts
            torch.save(self._text_embed_cache, cache_path)
            logger.info(f"  Saved cache to {cache_path} ({len(self._text_embed_cache)} entries)")

        # Free T5 entirely — no longer needed
        if self.pipeline.text_encoder is not None:
            logger.info("  Freeing T5 text encoder from memory...")
            self.pipeline.text_encoder.cpu()
            del self.pipeline.text_encoder
            self.pipeline.text_encoder = None
            if hasattr(self.pipeline, 'tokenizer') and self.pipeline.tokenizer is not None:
                del self.pipeline.tokenizer
                self.pipeline.tokenizer = None
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            logger.info("  T5 text encoder freed. Text embeddings served from cache.")

    def _get_text_embeds(self, captions, device=None) -> torch.Tensor:
        """Look up text embeddings from cache. Falls back to live encoding if not cached."""
        if device is None:
            device = self.device

        if self._text_embed_cache:
            embeds = []
            for c in (captions if isinstance(captions, (list, tuple)) else [captions]):
                if c in self._text_embed_cache:
                    embeds.append(self._text_embed_cache[c].to(device))
                else:
                    raise RuntimeError(
                        f"Text embedding cache miss: '{c[:80]}...'. "
                        f"Delete {self.save_dir / 'text_embed_cache.pt'} and restart to rebuild cache."
                    )
            return torch.cat(embeds, dim=0)
        else:
            # No cache — encode live
            embeds = [self.pipeline.encode_text(c) for c in
                      (captions if isinstance(captions, (list, tuple)) else [captions])]
            return torch.cat(embeds, dim=0)

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

        # Pre-cache all text embeddings and free T5 if enabled
        if self.cache_text_embeddings:
            self._precache_text_embeddings()

        self._transformer.train()

        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch
            # Set epoch on DistributedSampler for proper shuffling
            if self.is_distributed and hasattr(self, '_train_sampler') and self._train_sampler is not None:
                self._train_sampler.set_epoch(epoch)
            epoch_loss = self._train_epoch()

            if self.is_main_process:
                logger.info(f"Epoch {epoch}: avg_loss={epoch_loss:.6f}")
                self.writer.add_scalar("epoch/loss", epoch_loss, epoch)

            # Validation
            if self.val_loader is not None:
                val_loss = self._validate()
                if self.is_main_process:
                    logger.info(f"Epoch {epoch}: val_loss={val_loss:.6f}")
                    self.writer.add_scalar("epoch/val_loss", val_loss, epoch)

                    # Save best model
                    if self.save_best and val_loss < self.best_metric_val:
                        self.best_metric_val = val_loss
                        self._save_checkpoint("best")
                        logger.info(f"New best model: val_loss={val_loss:.6f}")

            if self.global_step >= self.max_steps:
                break

        # Save final checkpoint (main process only)
        if self.is_main_process:
            self._save_checkpoint("last")
            if self.writer is not None:
                self.writer.close()
        if self.is_distributed:
            dist.barrier()
        logger.info("Training complete.")

    def _train_epoch(self) -> float:
        """Train for one epoch. Returns average loss."""
        self._transformer.train()
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

            # Logging (main process only)
            if self.global_step % self.log_every == 0 and self.global_step > 0 and self.is_main_process:
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
                    grad_norm = loss_dict.get("grad_norm", 0.0)
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

                # Log gate gradient norms
                gate_grad_norms = self._get_gate_grad_norms()
                if gate_grad_norms:
                    self.writer.add_scalar("train/gate_grad_mean", np.mean(gate_grad_norms), self.global_step)

                # Per-module monitoring: grad norms and weight norms
                module_stats = loss_dict.get("module_stats", self._get_module_stats())
                for mod_name, stats in module_stats.items():
                    self.writer.add_scalar(f"modules/{mod_name}/grad_norm", stats["grad_norm"], self.global_step)
                    self.writer.add_scalar(f"modules/{mod_name}/weight_norm", stats["weight_norm"], self.global_step)
                    self.writer.add_scalar(f"modules/{mod_name}/weight_delta", stats["weight_delta"], self.global_step)

                gate_str = f", gate={np.mean(gate_values):.6f}" if gate_values else ""
                if gate_grad_norms:
                    gate_str += f" (grad={np.mean(gate_grad_norms):.2e})"
                gn_str = f"{grad_norm:.2e}" if self.log_grad_norm else "N/A"

                # Compact module monitoring string
                mod_str = ""
                if module_stats:
                    parts = [f"{k}={v['grad_norm']:.2e}" for k, v in module_stats.items()]
                    mod_str = f" | gnorms: {', '.join(parts)}"

                logger.info(
                    f"Step {self.global_step}: loss={avg_loss:.6f}, lr={lr:.2e}, "
                    f"grad_norm={gn_str}{gate_str}{mod_str}"
                )
                self.running_loss = 0.0

                # Save best based on train loss if no val loader
                if self.save_best and self.val_loader is None and avg_loss < self.best_metric_val:
                    self.best_metric_val = avg_loss
                    self._save_checkpoint("best")
                    logger.info(f"New best model (train): loss={avg_loss:.6f}")

            # Periodic checkpoint (main process only)
            if self.global_step % self.save_every == 0 and self.global_step > 0 and self.is_main_process:
                self._save_checkpoint(f"step_{self.global_step}")

            # Periodic sample generation (main process only)
            if (self.global_step % self.sample_every_steps == 0
                    and self.global_step > 0
                    and self.global_step != self._last_sample_step
                    and self.is_main_process):
                self._last_sample_step = self.global_step
                # Debug: write trigger to file (bypasses log buffering)
                with open("samples/sample_debug.log", "a") as dbg:
                    dbg.write(f"[{self.global_step}] Sample trigger fired\n")
                    dbg.flush()
                try:
                    self._generate_samples(self.global_step)
                    with open("samples/sample_debug.log", "a") as dbg:
                        dbg.write(f"[{self.global_step}] Sample generation succeeded\n")
                        dbg.flush()
                except Exception as e:
                    import traceback
                    err_msg = f"Sample generation failed (step {self.global_step}): {e}\n{traceback.format_exc()}"
                    logger.warning(err_msg)
                    with open("samples/sample_debug.log", "a") as dbg:
                        dbg.write(f"[{self.global_step}] {err_msg}\n")
                        dbg.flush()
                    # Ensure optimizer state is back on GPU even if sample gen fails
                    self._reload_optimizer_to_gpu()
                    self._transformer.train()
                    torch.cuda.empty_cache()

            # Periodic 5-shot multi-transition evaluation (main process only)
            if (self.global_step % self.multishot_every_steps == 0
                    and self.global_step > 0
                    and self.is_main_process):
                try:
                    self._generate_5shot_evaluation(self.global_step)
                except Exception as e:
                    import traceback
                    logger.warning(f"5-shot evaluation failed (step {self.global_step}): {e}\n{traceback.format_exc()}")
                    self._reload_optimizer_to_gpu()
                    self._transformer.train()
                    torch.cuda.empty_cache()

                # GT comparison: generate all training samples and compare with GT
                try:
                    self._generate_gt_comparison(self.global_step)
                except Exception as e:
                    import traceback
                    logger.warning(f"GT comparison failed (step {self.global_step}): {e}\n{traceback.format_exc()}")
                    self._reload_optimizer_to_gpu()
                    self._transformer.train()
                    torch.cuda.empty_cache()

            # Synchronize DDP processes after checkpoint/sample gen
            if self.is_distributed:
                dist.barrier()

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

            # 2. Encode text (cached if cache_text_embeddings=true, else T5 with optional offloading)
            with torch.no_grad():
                if self._text_embed_cache:
                    text_embeds = self._get_text_embeds(captions)
                else:
                    if self.offload_text_encoder:
                        self.pipeline.text_encoder.to(self.device)
                    text_embeds_list = []
                    for caption in captions:
                        te = self.pipeline.encode_text(caption)  # (1, S, D_text)
                        text_embeds_list.append(te)
                    text_embeds = torch.cat(text_embeds_list, dim=0)  # (B, S, D_text)
                    if self.offload_text_encoder:
                        self.pipeline.text_encoder.to("cpu")
                        torch.cuda.empty_cache()

            # 3. Encode context (local + global) with multi-context dropout
            # Reload CLIP to GPU if it was offloaded in a previous step
            if self.offload_text_encoder:
                self._transformer.global_encoder.clip_model.to(self.device)
            # Decompose anchor_rgb (B, K, 3, 224, 224) -> list of K tensors [(B, 3, 224, 224)]
            char_list = [anchor_rgb[:, k] for k in range(K)]

            unified_context, context_mask = self._transformer.encode_context(
                prev_frames=prev_frames,
                character_images=char_list,
                character_masks=char_mask,
                local_frame_valid=local_frame_valid,
            )  # (B, N_ctx, D), (B, N_ctx)

            # Offload CLIP to CPU after context encoding to free VRAM during backward
            if self.offload_text_encoder:
                self._transformer.global_encoder.clip_model.to("cpu")
                torch.cuda.empty_cache()

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

            # Manual gradient all-reduce for DDP (average across ranks)
            if self.is_distributed:
                world_size = dist.get_world_size()
                for p in self.trainable_params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad.div_(world_size)

            # Compute grad norm BEFORE clipping (for logging)
            if self.log_grad_norm:
                loss_dict["grad_norm"] = self._compute_grad_norm()

            # Capture per-module grad norms BEFORE clipping (for monitoring)
            if self.global_step % self.log_every == 0:
                loss_dict["module_stats"] = self._get_module_stats()

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
        self._transformer.eval()
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
                text_embeds = self._get_text_embeds(captions)

                char_list = [anchor_rgb[:, k] for k in range(K)]
                unified_context, context_mask = self._transformer.encode_context(
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

        self._transformer.train()
        return total_loss / max(1, num_batches)

    def _offload_optimizer_to_cpu(self):
        """Temporarily move Adam optimizer state to CPU to free GPU memory."""
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        torch.cuda.empty_cache()

    def _reload_optimizer_to_gpu(self):
        """Move Adam optimizer state back to GPU."""
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

    @torch.no_grad()
    def _generate_samples(self, step: int):
        """
        Generate reconstruction-based samples for visual monitoring.

        Approach: encode GT video → add noise at t=0.25,0.75 → single-step denoise
        → decode → compare with GT. This directly shows model's denoising ability
        without relying on full ODE integration.

        Also runs a full generation (Euler ODE) for end-to-end quality check.

        Saves:
          - Reconstruction grids to TensorBoard
          - Full generation MP4 to samples/
        """
        self._transformer.eval()

        # Free ~5GB by moving optimizer state to CPU
        logger.info(f"Sample gen (step {step}): offloading optimizer to CPU...")
        self._offload_optimizer_to_cpu()

        # Cache a fixed sample — pick brightest from first N batches for better visual assessment
        if self._fixed_sample is None:
            # Use val_loader if it has data, otherwise fall back to train_loader
            loader = None
            if self.val_loader is not None and len(self.val_loader) > 0:
                loader = self.val_loader
            if loader is None:
                loader = self.train_loader
            best_sample = None
            best_brightness = -1.0
            best_id = "?"
            for i, batch in enumerate(loader):
                if i >= 20:
                    break
                brightness = batch["target_video"].mean().item()
                seq_id = batch.get("seq_ids", ["?"])[0]
                if brightness > best_brightness:
                    best_brightness = brightness
                    best_sample = batch
                    best_id = seq_id
            self._fixed_sample = best_sample
            logger.info(f"Cached fixed sample: seq_id={best_id} (brightness={best_brightness:.4f}, scanned {min(i+1, 20)} batches)")

        batch = self._fixed_sample
        target_video = batch["target_video"].to(self.device, dtype=torch.float32)
        prev_frame = batch["prev_frame"].to(self.device, dtype=torch.float32)
        prev_prev_frame = batch["prev_prev_frame"].to(self.device, dtype=torch.float32)
        has_prev_prev = batch["has_prev_prev"]
        anchor_rgb = batch["anchor_rgb"].to(self.device, dtype=torch.float32)
        char_mask = batch["character_mask"].to(self.device, dtype=torch.float32)
        caption = batch["captions"][0]
        B, K = anchor_rgb.shape[:2]

        prev_frames = [prev_frame]
        if has_prev_prev[0]:
            prev_frames.append(prev_prev_frame)
        char_list = [anchor_rgb[:, k] for k in range(K)]
        local_frame_valid = torch.stack([
            torch.ones(B, device=self.device),
            has_prev_prev.float().to(self.device),
        ], dim=0)

        start_t = time.time()

        # === Part 1: Reconstruction samples (single-step denoising) ===
        # Memory-efficient: decode one at a time, move to CPU immediately
        # VAE stays in bf16 (casting to float32 doubles memory → OOM)
        # GT decode under bf16 may be dark — reconstruction panels are the real signal
        recon_frames_cpu = []  # list of (3, H, W) CPU tensors
        gt_mid_cpu = None

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            # Encode GT to latent
            x_1 = self.pipeline.encode_video(target_video)

            # Encode text + context
            text_embeds = self._get_text_embeds(caption)
            unified_context, context_mask = self._transformer.encode_context(
                prev_frames=prev_frames,
                character_images=char_list,
                character_masks=char_mask,
                local_frame_valid=local_frame_valid,
            )

            # Use raw target_video middle frame as GT reference (skip VAE decode to save memory)
            T_vid = target_video.shape[1]
            gt_mid_cpu = target_video[0, T_vid // 2].cpu().float().clamp(0, 1)  # (3, H, W)
            del target_video
            torch.cuda.empty_cache()

            # Single-step denoising at two noise levels
            # Sample at two noise levels for reconstruction comparison
            sample_timesteps = [250, 750]  # mild noise, heavy noise
            noise = torch.randn_like(x_1)
            diffusion = self.pipeline.diffusion

            # Pre-check: cache x_t at heavy noise for ablation if needed
            run_ablation = (step % self.ablation_every_steps == 0)
            x_t_750_cpu = None
            text_embeds_cpu = None

            # Phase 1: All transformer forward passes (keep transformer on GPU)
            x0_hat_latents_cpu = []  # list of latent tensors on CPU
            for ts_val in sample_timesteps:
                timestep = torch.tensor([ts_val], device=self.device, dtype=torch.long)

                x_t = diffusion.add_noise(x_1, noise, timestep)

                # Cache x_t at heavy noise for ablation
                if run_ablation and ts_val == 750:
                    x_t_750_cpu = x_t.cpu()
                    text_embeds_cpu = text_embeds.cpu()

                v_pred = self.pipeline.director_transformer(
                    hidden_states=x_t,
                    encoder_hidden_states=text_embeds,
                    timestep=timestep,
                    unified_context=unified_context,
                    context_mask=context_mask,
                    return_dict=False,
                )[0]

                # Recover clean sample from model prediction
                x_0_hat = diffusion.recover_clean(x_t, v_pred, timestep)
                x0_hat_latents_cpu.append(x_0_hat.cpu())
                del v_pred, x_t, x_0_hat
                torch.cuda.empty_cache()

            del x_1, noise, unified_context, context_mask, text_embeds

        # Phase 2: Offload transformer + text encoder to CPU, decode latents with VAE
        self._transformer.cpu()
        if self.pipeline.text_encoder is not None:
            self.pipeline.text_encoder.cpu()
        torch.cuda.empty_cache()
        self.pipeline.vae.to(self.device)

        for lat_cpu in x0_hat_latents_cpu:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                recon_video = self.pipeline.decode_latent(lat_cpu.to(self.device))
                recon_mid = recon_video[0].cpu().float().clamp(0, 1)
                mid_idx = recon_mid.shape[0] // 2
                recon_frames_cpu.append(recon_mid[mid_idx])  # (3, H, W)
                del recon_video, recon_mid
                torch.cuda.empty_cache()
        del x0_hat_latents_cpu

        # Build comparison grid on CPU: [GT_raw, t250_recon, t750_recon]
        # Resize recon frames to match GT raw frame size
        gt_h, gt_w = gt_mid_cpu.shape[1], gt_mid_cpu.shape[2]
        resized_recons = []
        for rf in recon_frames_cpu:
            rf_resized = F.interpolate(rf.unsqueeze(0), size=(gt_h, gt_w), mode="bilinear", align_corners=False)[0]
            resized_recons.append(rf_resized.clamp(0, 1))
        all_frames = [gt_mid_cpu] + resized_recons
        recon_grid = make_grid(all_frames, nrow=len(all_frames), padding=4, normalize=False)
        self.writer.add_image("samples/recon_gt_t250_t750", recon_grid, step)

        # Log GT raw frame as ground truth reference
        self.writer.add_image("samples/ground_truth", gt_mid_cpu, step)

        # Log context reference (prev frame + anchors)
        ref_images = [prev_frame[0].cpu().float()]
        for k in range(min(K, 4)):
            if char_mask[0, k] > 0:
                ref_images.append(anchor_rgb[0, k].cpu().float())
        if len(ref_images) > 1:
            ref_h = ref_images[0].shape[1]
            ref_w = ref_images[0].shape[2]
            resized = []
            for img in ref_images:
                img = F.interpolate(img.unsqueeze(0), size=(ref_h, ref_w), mode="bilinear", align_corners=False)[0]
                resized.append(img.clamp(0, 1))
            ref_grid = make_grid(resized, nrow=len(resized), padding=2, normalize=False)
            self.writer.add_image("samples/context_ref", ref_grid, step)

        # === Part 2: Multi-CFG Ablation (every ablation_every_steps) ===
        # Single-step denoising at t=0.75 with 4 context configurations:
        #   1) text_only:  no local, no global
        #   2) local_only: local context, no global
        #   3) global_only: no local, global context
        #   4) full:       local + global (DIRECTOR)
        # Uses x_t and text_embeds cached from reconstruction (no re-encoding needed).
        if run_ablation and x_t_750_cpu is not None:
            logger.info(f"Running Multi-CFG ablation at step {step}...")
            ablation_start = time.time()

            # Phase 1: Bring transformer back to GPU for forward passes
            self._transformer.to(self.device)
            torch.cuda.empty_cache()

            abl_latents_cpu = []  # list of x_0_hat latents on CPU
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                # Move cached tensors back to GPU
                x_t_abl = x_t_750_cpu.to(self.device)
                text_embeds_abl = text_embeds_cpu.to(self.device)
                timestep_abl = torch.tensor([750], device=self.device, dtype=torch.long)
                del x_t_750_cpu, text_embeds_cpu

                # Encode 4 context variants
                ctx_configs = {
                    "text_only": {"prev_frames": None, "character_images": None, "character_masks": None},
                    "local_only": {"prev_frames": prev_frames, "character_images": None, "character_masks": None},
                    "global_only": {"prev_frames": None, "character_images": char_list, "character_masks": char_mask},
                    "full": {"prev_frames": prev_frames, "character_images": char_list, "character_masks": char_mask},
                }

                for cfg_name, cfg_kwargs in ctx_configs.items():
                    lfv = None
                    if cfg_kwargs["prev_frames"] is not None:
                        lfv = local_frame_valid

                    ctx_abl, mask_abl = self._transformer.encode_context(
                        prev_frames=cfg_kwargs["prev_frames"],
                        character_images=cfg_kwargs["character_images"],
                        character_masks=cfg_kwargs["character_masks"],
                        local_frame_valid=lfv,
                    )

                    v_pred_abl = self.pipeline.director_transformer(
                        hidden_states=x_t_abl,
                        encoder_hidden_states=text_embeds_abl,
                        timestep=timestep_abl,
                        unified_context=ctx_abl,
                        context_mask=mask_abl,
                        return_dict=False,
                    )[0]

                    x_0_hat_abl = diffusion.recover_clean(x_t_abl, v_pred_abl, timestep_abl)
                    abl_latents_cpu.append(x_0_hat_abl.cpu())
                    del v_pred_abl, ctx_abl, mask_abl, x_0_hat_abl
                    torch.cuda.empty_cache()

                del x_t_abl, text_embeds_abl
                torch.cuda.empty_cache()

            # Phase 2: Offload transformer + text encoder, decode with VAE
            self._transformer.cpu()
            if self.pipeline.text_encoder is not None:
                self.pipeline.text_encoder.cpu()
            torch.cuda.empty_cache()
            self.pipeline.vae.to(self.device)

            ablation_frames = []
            for lat_cpu in abl_latents_cpu:
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    recon_abl = self.pipeline.decode_latent(lat_cpu.to(self.device))
                    mid_idx_abl = recon_abl.shape[1] // 2
                    ablation_frames.append(recon_abl[0, mid_idx_abl].cpu().float().clamp(0, 1))
                    del recon_abl
                    torch.cuda.empty_cache()
            del abl_latents_cpu

            # Build 4-column grid: [text_only, local_only, global_only, full]
            abl_h, abl_w = ablation_frames[0].shape[1], ablation_frames[0].shape[2]
            resized_abl = []
            for af in ablation_frames:
                af_r = F.interpolate(af.unsqueeze(0), size=(abl_h, abl_w), mode="bilinear", align_corners=False)[0]
                resized_abl.append(af_r.clamp(0, 1))
            ablation_grid = make_grid(resized_abl, nrow=4, padding=4, normalize=False)
            self.writer.add_image("samples/ablation_text_local_global_full", ablation_grid, step)

            # Also log individual ablation frames
            cfg_names = ["text_only", "local_only", "global_only", "full"]
            for i, name in enumerate(cfg_names):
                self.writer.add_image(f"samples/ablation_{name}", resized_abl[i], step)

            del ablation_frames, resized_abl
            torch.cuda.empty_cache()

            ablation_elapsed = time.time() - ablation_start
            logger.info(f"Multi-CFG ablation complete in {ablation_elapsed:.1f}s")

        # === Part 3: Per-layer gate values ===
        gate_vals = self._get_gate_values()
        for i, gv in enumerate(gate_vals):
            self.writer.add_scalar(f"gates/layer_{i:02d}", gv, step)

        del recon_frames_cpu, gt_mid_cpu
        torch.cuda.empty_cache()

        # Move transformer + text encoder back to GPU for training
        self._transformer.to(self.device)
        if self.pipeline.text_encoder is not None:
            self.pipeline.text_encoder.to(self.device)
        torch.cuda.empty_cache()

        elapsed = time.time() - start_t
        logger.info(f"Sample generation complete in {elapsed:.1f}s")

        self.writer.add_text("samples/caption", caption, step)
        self.writer.flush()

        self._transformer.train()
        torch.cuda.empty_cache()

        # Reload optimizer state back to GPU
        logger.info("Reloading optimizer state to GPU...")
        self._reload_optimizer_to_gpu()

    def _generate_gt_comparison(self, step: int):
        """Generate Multi-CFG samples for ALL training samples and compare with GT.

        For each sample in the dataset:
          1. Full Multi-CFG generation (same anchor, prev_frame, caption as training)
          2. Decode both GT and generated latents
          3. Save side-by-side video: [GT | Generated] for each sample
          4. Save combined grid to TensorBoard

        This is the definitive overfit verification: if the model memorized
        the data, generated videos should match GT closely.
        """
        from torchvision.utils import make_grid
        import torchvision.transforms.functional as TF

        logger.info(f"GT comparison (step {step}): generating for all {len(self.train_loader.dataset)} samples...")
        start_t = time.time()

        self._transformer.eval()
        self._offload_optimizer_to_cpu()

        train_cfg = self.config["training"]
        height = train_cfg.get("train_height", 320)
        width = train_cfg.get("train_width", 512)
        num_frames = train_cfg.get("train_frames", 13)

        latent_h, latent_w = height // 8, width // 8
        latent_t = (num_frames - 1) // 4 + 1
        latent_c = self.pipeline.vae.config.latent_channels

        inf_guidance = self.config.get("inference", {}).get("guidance", {})
        omega_text = inf_guidance.get("omega_text", 6.0)
        omega_local = inf_guidance.get("omega_local", 2.0)
        omega_global = inf_guidance.get("omega_global", 3.0)
        num_steps = self.sample_num_steps

        diffusion = self.pipeline.diffusion
        transformer = self._transformer
        max_chars = self.config.get("model", {}).get("context", {}).get("max_characters", 4)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            null_text_embeds = self._get_text_embeds("")

        gt_gen_pairs = []  # list of (seq_id, role, gt_mid_frame, gen_mid_frame)

        for batch_idx, batch in enumerate(self.train_loader):
            seq_id = batch.get("seq_ids", [f"sample_{batch_idx}"])[0]

            target_video = batch["target_video"].to(self.device, dtype=torch.float32)
            prev_frame = batch["prev_frame"].to(self.device, dtype=torch.float32)
            prev_prev_frame = batch["prev_prev_frame"].to(self.device, dtype=torch.float32)
            has_prev_prev = batch["has_prev_prev"]
            anchor_rgb = batch["anchor_rgb"].to(self.device, dtype=torch.float32)
            char_mask = batch["character_mask"].to(self.device, dtype=torch.float32)
            caption = batch["captions"][0]
            B, K = anchor_rgb.shape[:2]

            prev_frames = [prev_frame]
            if has_prev_prev[0]:
                prev_frames.append(prev_prev_frame)
            char_list = [anchor_rgb[:, k] for k in range(K)]
            local_frame_valid = torch.stack([
                torch.ones(B, device=self.device),
                has_prev_prev.float().to(self.device),
            ], dim=0)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                # Encode GT
                gt_latents = self.pipeline.encode_video(target_video)

                # Get text + context embeddings
                text_embeds = self._get_text_embeds(caption)
                ctx_full, mask_full = transformer.encode_context(
                    prev_frames=prev_frames,
                    character_images=char_list,
                    character_masks=char_mask,
                    local_frame_valid=local_frame_valid,
                )
                ctx_local, mask_local = transformer.encode_context(
                    prev_frames=prev_frames,
                    character_images=None,
                )

                # Full Multi-CFG ODE generation
                gen = torch.Generator(device=self.device).manual_seed(42)
                x = torch.randn(
                    1, latent_t, latent_c, latent_h, latent_w,
                    device=self.device, dtype=torch.bfloat16, generator=gen,
                )
                timesteps = diffusion.prepare_inference(num_steps, self.device)
                state = None
                for i, t in enumerate(timesteps):
                    t_tensor = t.expand(1)
                    fwd = lambda te, ctx, msk: transformer(
                        hidden_states=x, encoder_hidden_states=te, timestep=t_tensor,
                        unified_context=ctx, context_mask=msk, return_dict=False,
                    )[0]

                    v_null  = fwd(null_text_embeds, None, None)
                    v_text  = fwd(text_embeds, None, None)
                    v_local = fwd(text_embeds, ctx_local, mask_local)
                    v_full  = fwd(text_embeds, ctx_full, mask_full)

                    v_out = (v_null
                             + omega_text   * (v_text  - v_null)
                             + omega_local  * (v_local - v_text)
                             + omega_global * (v_full  - v_local))
                    v_out = v_out.float()
                    step_out = diffusion.inference_step(v_out, x.float(), t, i, timesteps, state=state)
                    x = step_out.latents.to(torch.bfloat16)
                    state = step_out.state

                gen_latents = x.cpu()

            logger.info(f"  [{seq_id}] Generated ({caption[:50]}...)")

            # Store for decoding after all generation is done
            gt_gen_pairs.append((seq_id, gt_latents.cpu(), gen_latents))

            del target_video, prev_frame, prev_prev_frame, anchor_rgb, char_mask
            del text_embeds, ctx_full, mask_full, ctx_local, mask_local, x
            torch.cuda.empty_cache()

        # Phase 2: Decode all with VAE (transformer offloaded)
        transformer.cpu()
        torch.cuda.empty_cache()
        self.pipeline.vae.to(self.device)

        comparison_frames = []
        sample_dir = Path(self.config["training"]["sample_generation"]["save_dir"])
        sample_dir.mkdir(parents=True, exist_ok=True)

        for seq_id, gt_lat, gen_lat in gt_gen_pairs:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                gt_video = self.pipeline.decode_latent(gt_lat.to(self.device))   # (1, T, 3, H, W)
                gen_video = self.pipeline.decode_latent(gen_lat.to(self.device))  # (1, T, 3, H, W)

            gt_vid = gt_video[0].cpu().float().clamp(0, 1)    # (T, 3, H, W)
            gen_vid = gen_video[0].cpu().float().clamp(0, 1)   # (T, 3, H, W)

            # Save side-by-side video: [GT | Generated]
            T_vid = min(gt_vid.shape[0], gen_vid.shape[0])
            sbs_frames = []
            for t_idx in range(T_vid):
                gt_f = (gt_vid[t_idx].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                gen_f = (gen_vid[t_idx].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                # Add labels
                gt_labeled = gt_f.copy()
                gen_labeled = gen_f.copy()
                cv2.putText(gt_labeled, "GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(gen_labeled, "GEN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                sbs = np.concatenate([gt_labeled, gen_labeled], axis=1)
                sbs_frames.append(sbs)

            # Write side-by-side video
            sbs_path = sample_dir / f"gt_vs_gen_{seq_id}_step{step:06d}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            h_sbs, w_sbs = sbs_frames[0].shape[:2]
            writer = cv2.VideoWriter(str(sbs_path), fourcc, 8, (w_sbs, h_sbs))
            for frame in sbs_frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            logger.info(f"  Saved: {sbs_path}")

            # Store mid frames for grid
            mid = T_vid // 2
            gt_mid = gt_vid[mid]
            gen_mid = gen_vid[mid]
            comparison_frames.append(gt_mid)
            comparison_frames.append(gen_mid)

            del gt_video, gen_video, gt_vid, gen_vid
            torch.cuda.empty_cache()

        # TensorBoard: grid of all [GT, Gen] pairs
        if comparison_frames and self.writer is not None:
            grid = make_grid(comparison_frames, nrow=2, padding=4, normalize=False)
            self.writer.add_image("gt_comparison/all_samples", grid, step)
            self.writer.flush()

        # Cleanup: move everything back
        self.pipeline.vae.cpu()
        torch.cuda.empty_cache()
        transformer.to(self.device)
        transformer.train()
        torch.cuda.empty_cache()

        elapsed = time.time() - start_t
        logger.info(f"GT comparison complete in {elapsed:.1f}s ({len(gt_gen_pairs)} samples)")

        logger.info("Reloading optimizer state to GPU...")
        self._reload_optimizer_to_gpu()

    def _generate_multishot_ablation(self, step: int):
        """
        Generate a multi-shot autoregressive sequence to verify shot-to-shot consistency.

        Generates N shots where each shot's last frame becomes the next shot's prev_frame.
        Uses simple CFG (not full multi-cfg) for speed: v_out = v_null + scale * (v_cond - v_null).
        Logs to TensorBoard:
          - Grid of key frames (first/mid/last) from each shot
          - Shot transition pairs (last frame of shot i, first frame of shot i+1)
          - Saves concatenated MP4 video
        """
        num_shots = self.multishot_num_shots
        num_steps = self.sample_num_steps  # reuse existing config (e.g. 20)
        logger.info(f"Multi-shot ablation (step {step}): generating {num_shots} shots, {num_steps} ODE steps...")
        start_t = time.time()

        self._transformer.eval()
        self._offload_optimizer_to_cpu()

        # Use the same fixed sample as single-shot ablation
        if self._fixed_sample is None:
            loader = self.val_loader if self.val_loader is not None else self.train_loader
            best_sample, best_brightness = None, -1.0
            for i, batch in enumerate(loader):
                if i >= 20:
                    break
                brightness = batch["target_video"].mean().item()
                if brightness > best_brightness:
                    best_brightness = brightness
                    best_sample = batch
            self._fixed_sample = best_sample

        batch = self._fixed_sample
        anchor_rgb = batch["anchor_rgb"].to(self.device, dtype=torch.float32)
        char_mask = batch["character_mask"].to(self.device, dtype=torch.float32)
        prev_frame_init = batch["prev_frame"].to(self.device, dtype=torch.float32)
        caption = batch["captions"][0]
        B, K = anchor_rgb.shape[:2]
        char_list = [anchor_rgb[:, k] for k in range(K)]

        # Use training resolution for memory efficiency
        train_cfg = self.config["training"]
        height = train_cfg.get("train_height", 320)
        width = train_cfg.get("train_width", 512)
        num_frames = train_cfg.get("train_frames", 13)

        # Multi-CFG guidance scales from config
        inf_guidance = self.config.get("inference", {}).get("guidance", {})
        omega_text = inf_guidance.get("omega_text", 6.0)
        omega_local = inf_guidance.get("omega_local", 2.0)
        omega_global = inf_guidance.get("omega_global", 3.0)

        latent_h, latent_w = height // 8, width // 8
        latent_t = (num_frames - 1) // 4 + 1
        latent_c = self.pipeline.vae.config.latent_channels

        diffusion = self.pipeline.diffusion
        transformer = self._transformer

        # Pre-encode text (stays fixed across shots)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            text_embeds = self._get_text_embeds(caption)
            null_text_embeds = self._get_text_embeds("")

        # Generate shots autoregressively
        shot_latents_cpu = []  # list of latent tensors on CPU
        prev_frames_ar = [prev_frame_init]  # start with dataset's prev_frame

        for shot_idx in range(num_shots):
            logger.info(f"  Shot {shot_idx+1}/{num_shots}...")

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                # Encode context variants for Multi-CFG:
                # Full context (local + global)
                ctx_full, mask_full = transformer.encode_context(
                    prev_frames=prev_frames_ar,
                    character_images=char_list,
                    character_masks=char_mask,
                )
                # Local-only context (no global anchors)
                ctx_local, mask_local = transformer.encode_context(
                    prev_frames=prev_frames_ar,
                    character_images=None,
                )
                # Null context: None to skip adapters (avoids NaN from all-zero mask)

                # Initial noise
                gen = torch.Generator(device=self.device)
                gen.manual_seed(42 + shot_idx)
                x = torch.randn(
                    1, latent_t, latent_c, latent_h, latent_w,
                    device=self.device, dtype=torch.bfloat16, generator=gen,
                )

                # ODE solve with Multi-CFG
                timesteps = diffusion.prepare_inference(num_steps, self.device)
                state = None
                for i, t in enumerate(timesteps):
                    t_tensor = t.expand(1)

                    fwd = lambda te, ctx, msk: transformer(
                        hidden_states=x, encoder_hidden_states=te, timestep=t_tensor,
                        unified_context=ctx, context_mask=msk, return_dict=False,
                    )[0]

                    # 4-pass Multi-CFG
                    v_null  = fwd(null_text_embeds, None, None)         # unconditional
                    v_text  = fwd(text_embeds, None, None)              # text-only
                    v_local = fwd(text_embeds, ctx_local, mask_local)   # text + local
                    v_full  = fwd(text_embeds, ctx_full, mask_full)     # text + local + global

                    v_out = (v_null
                             + omega_text   * (v_text  - v_null)
                             + omega_local  * (v_local - v_text)
                             + omega_global * (v_full  - v_local))
                    # Cast to float32 for numerical precision in scheduler step
                    v_out = v_out.float()
                    step_out = diffusion.inference_step(v_out, x.float(), t, i, timesteps, state=state)
                    x = step_out.latents.to(torch.bfloat16)
                    state = step_out.state

                shot_latents_cpu.append(x.cpu())
                del x, ctx_full, mask_full, ctx_local, mask_local
                torch.cuda.empty_cache()

            # Decode last frame for next shot's prev_frame (if not last shot)
            if shot_idx < num_shots - 1:
                transformer.cpu()
                torch.cuda.empty_cache()
                self.pipeline.vae.to(self.device)

                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    video_tmp = self.pipeline.decode_latent(shot_latents_cpu[-1].to(self.device))
                last_frame = video_tmp[:, -1]  # (1, 3, H, W) on GPU
                del video_tmp
                torch.cuda.empty_cache()

                # Bring transformer back for next shot
                self.pipeline.vae.cpu()
                torch.cuda.empty_cache()
                transformer.to(self.device)

                prev_frames_ar = [last_frame]

        del text_embeds, null_text_embeds
        torch.cuda.empty_cache()

        # Decode all shot latents to video
        logger.info("  Decoding all shots...")
        transformer.cpu()
        if self.pipeline.text_encoder is not None:
            self.pipeline.text_encoder.cpu()
        torch.cuda.empty_cache()
        self.pipeline.vae.to(self.device)

        shot_videos = []  # list of (T, 3, H, W) CPU tensors
        for lat_cpu in shot_latents_cpu:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                video = self.pipeline.decode_latent(lat_cpu.to(self.device))
            shot_videos.append(video[0].cpu().float().clamp(0, 1))
            del video
            torch.cuda.empty_cache()
        del shot_latents_cpu

        # === Log to TensorBoard ===

        # 1. Key frames grid: [shot1_first, shot1_mid, shot1_last, shot2_first, shot2_mid, shot2_last, ...]
        key_frames = []
        for sv in shot_videos:
            T = sv.shape[0]
            key_frames.extend([sv[0], sv[T // 2], sv[-1]])
        kf_grid = make_grid(key_frames, nrow=3, padding=4, normalize=False)
        self.writer.add_image("multishot/key_frames", kf_grid, step)

        # 2. Shot transitions: [shot_i_last | shot_{i+1}_first] pairs
        if num_shots >= 2:
            transition_frames = []
            for i in range(num_shots - 1):
                transition_frames.append(shot_videos[i][-1])    # last frame of shot i
                transition_frames.append(shot_videos[i + 1][0]) # first frame of shot i+1
            tr_grid = make_grid(transition_frames, nrow=2, padding=4, normalize=False)
            self.writer.add_image("multishot/transitions", tr_grid, step)

        # 3. Save concatenated MP4
        full_video = torch.cat(shot_videos, dim=0)  # (T_total, 3, H, W)
        mp4_path = self.sample_dir / f"multishot_step{step:06d}.mp4"
        video_np = (full_video.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()
        T_out, H_out, W_out, _ = video_np.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_cv = cv2.VideoWriter(str(mp4_path), fourcc, 8, (W_out, H_out))
        for t_idx in range(T_out):
            writer_cv.write(cv2.cvtColor(video_np[t_idx], cv2.COLOR_RGB2BGR))
        writer_cv.release()
        logger.info(f"  Saved multi-shot video: {mp4_path} ({T_out} frames)")

        # 4. Add video to tensorboard (subsample to 16 frames max for TB)
        max_tb_frames = 16
        stride = max(1, T_out // max_tb_frames)
        tb_video = full_video[::stride].unsqueeze(0)  # (1, T_sub, 3, H, W)
        self.writer.add_video("multishot/video", tb_video, step, fps=4)

        del shot_videos, full_video, key_frames, video_np
        torch.cuda.empty_cache()

        # Move transformer + text encoder back to GPU
        transformer.to(self.device)
        if self.pipeline.text_encoder is not None:
            self.pipeline.text_encoder.to(self.device)
        torch.cuda.empty_cache()

        self.writer.flush()
        elapsed = time.time() - start_t
        logger.info(f"Multi-shot ablation complete in {elapsed:.1f}s ({num_shots} shots)")

        self._transformer.train()
        torch.cuda.empty_cache()
        self._reload_optimizer_to_gpu()

    def _generate_5shot_evaluation(self, step: int):
        """
        Generate 3×5 multi-domain multi-transition evaluation.

        3 Anchor Domains:
          D1: T2I Synthetic (high-fidelity generated characters)
          D2: In-the-wild / Heterogeneous (cartoon + noisy real photo)
          D3: Self-extracted (SAM2-extracted from video dataset)

        5 Shots per domain (same scenario):
          Shot 1: Zoom-in, Character A  (Identity Alignment)
          Shot 2: Zoom-in, Character A  (Spatial Inertia / Continuity)
          Shot 3: Cross-cut to Char B   (Context Blocking / Swap)
          Shot 4: Char A + B together   (Feature Composition / Multi-entity)
          Shot 5: Char A only, B gone   (Selective Erasure)

        Outputs per domain:
          1. Full-length 5-shot concatenated video
          2. 4-subplot transition comparison video with subtitles
        """
        from PIL import Image as _Image
        from torchvision import transforms as _transforms

        num_shots = 5
        num_steps = self.sample_num_steps
        logger.info(f"3x5 evaluation (step {step}): 3 domains × {num_shots} shots, {num_steps} ODE steps...")
        start_t = time.time()

        self._transformer.eval()
        self._offload_optimizer_to_cpu()

        # --- Anchor loading utilities ---
        clip_size = 224
        clip_normalize = _transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

        def load_anchor_from_path(path: str):
            """Load RGBA anchor as CLIP-normalized (3, 224, 224) tensor."""
            p = Path(path)
            if not p.exists():
                return None
            img = _Image.open(str(p)).convert("RGBA")
            arr = np.array(img)
            t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            t = _transforms.Resize((clip_size, clip_size), antialias=True)(t)
            rgb = t[:3]
            alpha = t[3:4]
            composited = rgb * alpha + torch.ones_like(rgb) * (1 - alpha)
            return clip_normalize(composited)

        # --- Load anchors for all 3 domains ---
        eval_anchor_dir = Path("evaluation/anchors")
        dataset_dir = Path(self.config.get("dataset", {}).get(
            "dataset_dir", self.config.get("dataset", {}).get("triplet_dir", "data/director_10k/output")))

        domains = [
            {
                "name": "t2i",
                "label": "D1:T2I-Synthetic",
                "char_a": str(eval_anchor_dir / "t2i" / "char_a.png"),
                "char_b": str(eval_anchor_dir / "t2i" / "char_b.png"),
            },
            {
                "name": "wild",
                "label": "D2:In-the-Wild",
                "char_a": str(eval_anchor_dir / "wild" / "char_a.png"),
                "char_b": str(eval_anchor_dir / "wild" / "char_b.png"),
            },
            {
                "name": "extracted",
                "label": "D3:Self-Extracted",
                "char_a": str(eval_anchor_dir / "extracted" / "char_a.png"),
                "char_b": str(eval_anchor_dir / "extracted" / "char_b.png"),
            },
        ]

        # Filter to available domains only
        valid_domains = []
        for dom in domains:
            a = load_anchor_from_path(dom["char_a"])
            b = load_anchor_from_path(dom["char_b"])
            if a is not None and b is not None:
                dom["anchor_a"] = a
                dom["anchor_b"] = b
                valid_domains.append(dom)
            else:
                logger.warning(f"Skipping domain {dom['name']}: anchors not found")

        if not valid_domains:
            # Fallback: use dataset anchors directly
            logger.warning("No eval anchors found, falling back to dataset anchors")
            a = load_anchor_from_path(str(dataset_dir / "seq_00047" / "global_anchor_0.png"))
            b = load_anchor_from_path(str(dataset_dir / "seq_00002" / "global_anchor_0.png"))
            if a is None or b is None:
                logger.error("Cannot find any valid anchors for evaluation!")
                self._transformer.train()
                self._reload_optimizer_to_gpu()
                return
            valid_domains = [{"name": "extracted", "label": "D3:Self-Extracted",
                              "anchor_a": a, "anchor_b": b}]

        logger.info(f"  Running evaluation for {len(valid_domains)} domains: {[d['name'] for d in valid_domains]}")

        # Load initial prev_frame (from dataset)
        prev_frame_path = dataset_dir / "seq_00047" / "prev_shot_last_frame.jpg"
        if not prev_frame_path.exists():
            # Find any valid prev frame
            for seq in sorted(dataset_dir.iterdir()):
                p = seq / "prev_shot_last_frame.jpg"
                if p.exists():
                    prev_frame_path = p
                    break
        prev_img = cv2.imread(str(prev_frame_path))
        if prev_img is None:
            logger.error(f"Cannot load prev frame from {prev_frame_path}")
            self._transformer.train()
            self._reload_optimizer_to_gpu()
            return
        prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB)

        # Training resolution
        train_cfg = self.config["training"]
        height = train_cfg.get("train_height", 320)
        width = train_cfg.get("train_width", 512)
        num_frames = train_cfg.get("train_frames", 13)
        max_chars = self.config.get("dataset", {}).get("max_characters", 4)

        video_resize = _transforms.Resize((height, width), antialias=True)
        prev_tensor = torch.from_numpy(prev_img).permute(2, 0, 1).float() / 255.0
        prev_frame_base = video_resize(prev_tensor.unsqueeze(0)).squeeze(0)  # (3, H, W)

        # Multi-CFG guidance scales
        inf_guidance = self.config.get("inference", {}).get("guidance", {})
        omega_text = inf_guidance.get("omega_text", 6.0)
        omega_local = inf_guidance.get("omega_local", 2.0)
        omega_global = inf_guidance.get("omega_global", 3.0)

        latent_h, latent_w = height // 8, width // 8
        latent_t = (num_frames - 1) // 4 + 1
        latent_c = self.pipeline.vae.config.latent_channels

        diffusion = self.pipeline.diffusion
        transformer = self._transformer

        # Pre-encode text (shared across all domains)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            null_text_embeds = self._get_text_embeds("")

        # Transition labels for subplot videos
        transition_labels = [
            "T1: Continuity (Fwd->Pan-R)\nSame char A, motion change.\nSuccess: A consistent",
            "T2: Cross-Cut (A->B)\nCharacter swap.\nSuccess: B appears, not A",
            "T3: Multi-Entity (B->A+B)\nA joins B's scene.\nSuccess: Both A+B visible",
            "T4: Erasure (A+B->A)\nB removed, pan-left.\nSuccess: Only A remains",
        ]

        # === Generate 5 shots for each domain ===
        for dom in valid_domains:
            domain_name = dom["name"]
            domain_label = dom["label"]
            anchor_a = dom["anchor_a"]
            anchor_b = dom["anchor_b"]

            logger.info(f"  --- Domain: {domain_label} ---")

            shot_configs = [
                # S1: Character A established in scene, camera dollies forward
                {"caption": "The character walks forward through the scene as the camera steadily dollies in, "
                            "revealing details of the environment. The lighting is natural and the composition "
                            "focuses on the character's presence in the space.",
                 "anchors": [anchor_a], "subtitle": f"[{domain_label}] S1: Dolly-In | A"},
                # S2: Same character A, camera pans right — test continuity across motion change
                {"caption": "The same character continues through the scene as the camera pans right while "
                            "moving forward, tracking their movement through the environment. The character "
                            "maintains a natural walking pace.",
                 "anchors": [anchor_a], "subtitle": f"[{domain_label}] S2: Pan-R | A (cont.)"},
                # S3: Cross-cut to character B — test identity swap
                {"caption": "Cut to a new scene: a different character appears in a completely different "
                            "environment. The camera captures them from a medium angle as they stand or "
                            "move in their new surroundings, establishing a clear scene change.",
                 "anchors": [anchor_b], "subtitle": f"[{domain_label}] S3: Cross-Cut | B"},
                # S4: Both A and B together — test multi-entity composition
                {"caption": "Both characters are now visible together in the same scene. The camera holds "
                            "a wide static shot showing them interacting in a shared space. Each character "
                            "maintains their distinct appearance and positioning.",
                 "anchors": [anchor_a, anchor_b], "subtitle": f"[{domain_label}] S4: Static | A+B"},
                # S5: Only A remains, B gone — test selective erasure
                {"caption": "The scene continues with only the first character visible. The camera pans left "
                            "while moving forward, following the character as they walk alone through the "
                            "environment. The second character is no longer present.",
                 "anchors": [anchor_a], "subtitle": f"[{domain_label}] S5: Pan-L | A only"},
            ]

            prev_frame_init = prev_frame_base.unsqueeze(0).to(self.device)
            shot_latents_cpu = []
            prev_frames_ar = [prev_frame_init]

            for shot_idx in range(num_shots):
                sc = shot_configs[shot_idx]
                logger.info(f"    {sc['subtitle']}...")

                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    text_embeds = self._get_text_embeds(sc["caption"])

                    anchor_rgb = torch.zeros(1, max_chars, 3, clip_size, clip_size, device=self.device)
                    char_mask = torch.zeros(1, max_chars, device=self.device)
                    for k, anc in enumerate(sc["anchors"]):
                        if k < max_chars:
                            anchor_rgb[0, k] = anc.to(self.device)
                            char_mask[0, k] = 1.0
                    char_list = [anchor_rgb[:, k] for k in range(max_chars)]

                    ctx_full, mask_full = transformer.encode_context(
                        prev_frames=prev_frames_ar,
                        character_images=char_list,
                        character_masks=char_mask,
                    )
                    ctx_local, mask_local = transformer.encode_context(
                        prev_frames=prev_frames_ar,
                        character_images=None,
                    )

                    gen = torch.Generator(device=self.device)
                    gen.manual_seed(42 + shot_idx)
                    x = torch.randn(
                        1, latent_t, latent_c, latent_h, latent_w,
                        device=self.device, dtype=torch.bfloat16, generator=gen,
                    )

                    timesteps = diffusion.prepare_inference(num_steps, self.device)
                    state = None
                    for i, t in enumerate(timesteps):
                        t_tensor = t.expand(1)
                        fwd = lambda te, ctx, msk: transformer(
                            hidden_states=x, encoder_hidden_states=te, timestep=t_tensor,
                            unified_context=ctx, context_mask=msk, return_dict=False,
                        )[0]

                        v_null  = fwd(null_text_embeds, None, None)
                        v_text  = fwd(text_embeds, None, None)
                        v_local = fwd(text_embeds, ctx_local, mask_local)
                        v_full  = fwd(text_embeds, ctx_full, mask_full)

                        v_out = (v_null
                                 + omega_text   * (v_text  - v_null)
                                 + omega_local  * (v_local - v_text)
                                 + omega_global * (v_full  - v_local))
                        v_out = v_out.float()
                        step_out = diffusion.inference_step(v_out, x.float(), t, i, timesteps, state=state)
                        x = step_out.latents.to(torch.bfloat16)
                        state = step_out.state

                    shot_latents_cpu.append(x.cpu())
                    del x, ctx_full, mask_full, ctx_local, mask_local, text_embeds
                    torch.cuda.empty_cache()

                # Decode last frame for autoregressive chaining
                if shot_idx < num_shots - 1:
                    transformer.cpu()
                    torch.cuda.empty_cache()
                    self.pipeline.vae.to(self.device)

                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                        video_tmp = self.pipeline.decode_latent(shot_latents_cpu[-1].to(self.device))
                    last_frame = video_tmp[:, -1]
                    del video_tmp
                    torch.cuda.empty_cache()

                    self.pipeline.vae.cpu()
                    torch.cuda.empty_cache()
                    transformer.to(self.device)
                    prev_frames_ar = [last_frame]

            # === Decode all shots for this domain ===
            logger.info(f"    Decoding all 5 shots for {domain_label}...")
            transformer.cpu()
            if self.pipeline.text_encoder is not None:
                self.pipeline.text_encoder.cpu()
            torch.cuda.empty_cache()
            self.pipeline.vae.to(self.device)

            shot_videos = []
            for lat_cpu in shot_latents_cpu:
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    video = self.pipeline.decode_latent(lat_cpu.to(self.device))
                shot_videos.append(video[0].cpu().float().clamp(0, 1))
                del video
                torch.cuda.empty_cache()
            del shot_latents_cpu

            # Move transformer back for next domain (or final cleanup)
            self.pipeline.vae.cpu()
            torch.cuda.empty_cache()
            transformer.to(self.device)

            # === Write outputs for this domain ===
            self._write_5shot_videos(
                shot_videos, shot_configs, transition_labels,
                step, domain_name, domain_label,
                height, width,
            )

            del shot_videos
            torch.cuda.empty_cache()

        del null_text_embeds
        torch.cuda.empty_cache()

        # Move everything back to GPU
        transformer.to(self.device)
        if self.pipeline.text_encoder is not None:
            self.pipeline.text_encoder.to(self.device)
        torch.cuda.empty_cache()

        self.writer.flush()
        elapsed = time.time() - start_t
        logger.info(f"3x5 evaluation complete in {elapsed:.1f}s ({len(valid_domains)} domains)")

        self._transformer.train()
        torch.cuda.empty_cache()
        self._reload_optimizer_to_gpu()

    def _write_5shot_videos(
        self,
        shot_videos: list,
        shot_configs: list,
        transition_labels: list,
        step: int,
        domain_name: str,
        domain_label: str,
        height: int,
        width: int,
    ):
        """Write output videos for one domain of the 5-shot evaluation."""
        H_out, W_out = height, width
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Output 1: Full-length concatenated video with shot labels
        full_video = torch.cat(shot_videos, dim=0)
        mp4_full = self.sample_dir / f"5shot_{domain_name}_full_step{step:06d}.mp4"
        video_np = (full_video.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()
        T_out = video_np.shape[0]

        writer_cv = cv2.VideoWriter(str(mp4_full), fourcc, 8, (W_out, H_out))
        frame_idx = 0
        for shot_i, sv in enumerate(shot_videos):
            label = shot_configs[shot_i]["subtitle"]
            for _ in range(sv.shape[0]):
                frame_bgr = cv2.cvtColor(video_np[frame_idx], cv2.COLOR_RGB2BGR)
                cv2.putText(frame_bgr, label, (10, H_out - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame_bgr, label, (10, H_out - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                writer_cv.write(frame_bgr)
                frame_idx += 1
        writer_cv.release()
        logger.info(f"    Saved: {mp4_full} ({T_out} frames)")

        # Output 2: 4-subplot transition comparison
        mp4_comp = self.sample_dir / f"5shot_{domain_name}_transitions_step{step:06d}.mp4"
        n_ctx = 3
        canvas_w = W_out * 4
        canvas_h = H_out + 60

        transition_clips = []
        for t_idx in range(4):
            before = shot_videos[t_idx][-n_ctx:]
            after = shot_videos[t_idx + 1][:n_ctx]
            transition_clips.append(torch.cat([before, after], dim=0))
        max_frames = max(c.shape[0] for c in transition_clips)

        writer_cv2 = cv2.VideoWriter(str(mp4_comp), fourcc, 4, (canvas_w, canvas_h))
        for f_idx in range(max_frames):
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            for t_idx in range(4):
                clip = transition_clips[t_idx]
                actual_f = min(f_idx, clip.shape[0] - 1)
                frame = (clip[actual_f].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if f_idx == n_ctx:
                    cv2.line(frame_bgr, (0, 0), (0, H_out), (0, 0, 255), 3)

                x_off = t_idx * W_out
                canvas[:H_out, x_off:x_off + W_out] = frame_bgr

                for li, line in enumerate(transition_labels[t_idx].split("\n")):
                    cv2.putText(canvas, line, (x_off + 5, H_out + 15 + li * 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

                pos = f"Shot {t_idx+1} (-{n_ctx-f_idx})" if f_idx < n_ctx else f"Shot {t_idx+2} (+{f_idx-n_ctx+1})"
                cv2.putText(canvas, pos, (x_off + 5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
            writer_cv2.write(canvas)
        writer_cv2.release()
        logger.info(f"    Saved: {mp4_comp} ({max_frames} frames, {canvas_w}x{canvas_h})")

        # TensorBoard logging
        key_frames = []
        for sv in shot_videos:
            T = sv.shape[0]
            key_frames.extend([sv[0], sv[T // 2], sv[-1]])
        kf_grid = make_grid(key_frames, nrow=3, padding=4, normalize=False)
        self.writer.add_image(f"5shot_{domain_name}/key_frames", kf_grid, step)

        tr_frames = []
        for i in range(4):
            tr_frames.append(shot_videos[i][-1])
            tr_frames.append(shot_videos[i + 1][0])
        tr_grid = make_grid(tr_frames, nrow=2, padding=4, normalize=False)
        self.writer.add_image(f"5shot_{domain_name}/transitions", tr_grid, step)

    def _compute_grad_norm(self) -> float:
        """Compute the total gradient norm across all trainable parameters."""
        total_norm = 0.0
        for p in self.trainable_params:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def _get_gate_values(self) -> List[float]:
        """Extract gate values from all DIRECTOR context adapters."""
        gate_vals = []
        adapters = self._transformer.adapters
        for key, adapter in adapters.items():
            gate_vals.append(adapter.gate.item())
        return gate_vals

    def _get_gate_grad_norms(self) -> List[float]:
        """Extract gate gradient norms (after hooks) from all context adapters."""
        norms = []
        adapters = self._transformer.adapters
        for key, adapter in adapters.items():
            if adapter.gate.grad is not None:
                norms.append(adapter.gate.grad.abs().item())
        return norms

    def _get_module_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute per-module grad norms and weight norms for monitoring.

        Tracks: encoder (local+global proj), adapter (cross-attn), gate, lora.
        """
        if not hasattr(self, '_prev_weight_norms'):
            self._prev_weight_norms = {}

        param_groups_dict = self.pipeline.director_transformer.get_trainable_param_groups()
        stats = {}
        for group_name in ["encoder", "adapter", "gate", "lora"]:
            params = param_groups_dict.get(group_name, [])
            if not params:
                continue
            # Grad norm
            grad_sq = 0.0
            for p in params:
                if p.grad is not None:
                    grad_sq += p.grad.data.float().norm().item() ** 2
            grad_norm = grad_sq ** 0.5

            # Weight norm
            weight_sq = sum(p.data.float().norm().item() ** 2 for p in params)
            weight_norm = weight_sq ** 0.5

            # Weight delta (change since last check)
            prev = self._prev_weight_norms.get(group_name, weight_norm)
            weight_delta = abs(weight_norm - prev)
            self._prev_weight_norms[group_name] = weight_norm

            stats[group_name] = {
                "grad_norm": grad_norm,
                "weight_norm": weight_norm,
                "weight_delta": weight_delta,
            }
        return stats

    def _unwrap_module(self, module):
        """Unwrap DDP module to get the underlying module."""
        return module.module if hasattr(module, "module") else module

    def _save_checkpoint(self, name: str):
        """Save a training checkpoint."""
        ckpt_path = self.save_dir / f"checkpoint_{name}.pt"

        # Use unwrapped transformer for saving (so checkpoints are DDP-agnostic)
        t = self._transformer
        adapters = t.adapters
        context_builder = t.context_builder
        global_encoder = t.global_encoder
        local_encoder = t.local_encoder

        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric_val": self.best_metric_val,
            "adapters": adapters.state_dict(),
            "context_builder": context_builder.state_dict(),
            "global_encoder": {
                k: v
                for k, v in global_encoder.state_dict().items()
                if "clip_vision" not in k  # Don't save frozen CLIP weights
            },
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

        if local_encoder is not None:
            state["local_encoder"] = {
                k: v
                for k, v in local_encoder.state_dict().items()
                if "vae" not in k  # Don't save frozen VAE weights
            }

        # Save LoRA or backbone trainable params
        if t.lora_enabled:
            # Save only LoRA adapter weights
            from peft import get_peft_model_state_dict
            lora_state = get_peft_model_state_dict(t.backbone)
            if lora_state:
                state["lora"] = lora_state
        else:
            # Save backbone trainable params (when partially/fully unfrozen)
            backbone_trainable = {
                k: v for k, v in t.backbone.state_dict().items()
                if any(p.requires_grad and p.data_ptr() == v.data_ptr()
                       for p in t.backbone.parameters())
            }
            if not backbone_trainable:
                # Fallback: check by matching named_parameters with requires_grad
                trainable_names = {n for n, p in t.backbone.named_parameters() if p.requires_grad}
                backbone_trainable = {
                    k: v for k, v in t.backbone.state_dict().items()
                    if k in trainable_names
                }
            if backbone_trainable:
                state["backbone"] = backbone_trainable

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

    def load_checkpoint(self, path: str, weights_only: bool = False):
        """Load a training checkpoint (DDP-agnostic: checkpoints are always unwrapped).

        Args:
            weights_only: If True, only load model weights (skip optimizer/scheduler/step).
                         Useful when changing LR config mid-training.
        """
        state = torch.load(path, map_location=self.device)

        if not weights_only:
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
            self.best_metric_val = state.get("best_metric_val", float("inf"))

        # Use unwrapped transformer for loading (checkpoints saved without DDP prefix)
        t = self._transformer
        t.adapters.load_state_dict(state["adapters"])
        t.context_builder.load_state_dict(state["context_builder"])

        # Load global encoder (partial, excluding frozen CLIP)
        if "global_encoder" in state:
            missing, unexpected = t.global_encoder.load_state_dict(
                state["global_encoder"], strict=False
            )

        if "local_encoder" in state and t.local_encoder is not None:
            missing, unexpected = t.local_encoder.load_state_dict(
                state["local_encoder"], strict=False
            )

        # Load LoRA or backbone trainable params
        if "lora" in state and t.lora_enabled:
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(t.backbone, state["lora"])
            logger.info(f"Loaded LoRA params: {len(state['lora'])} keys")
        elif "backbone" in state:
            missing, unexpected = t.backbone.load_state_dict(
                state["backbone"], strict=False
            )
            logger.info(f"Loaded backbone params: {len(state['backbone'])} keys")

        if weights_only:
            logger.info(f"Loaded model weights only from {path} (skipping optimizer/scheduler/step)")
            return

        # Load optimizer state — skip if param groups changed (e.g., added separate LR groups)
        old_n_groups = len(state["optimizer"]["param_groups"])
        new_n_groups = len(self.optimizer.param_groups)
        if old_n_groups == new_n_groups:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
                logger.info("Optimizer state loaded successfully")
                # Override LR from current config (checkpoint may have stale LR values)
                for pg_old, pg_new in zip(self.optimizer.param_groups,
                                          self._initial_param_groups_lr):
                    if pg_old["lr"] != pg_new:
                        logger.info(f"  Overriding LR: {pg_old['lr']:.1e} -> {pg_new:.1e}")
                        pg_old["lr"] = pg_new
            except (ValueError, KeyError) as e:
                logger.warning(f"Optimizer state load failed, starting fresh: {e}")
        else:
            logger.warning(
                f"Optimizer param groups changed ({old_n_groups} -> {new_n_groups}), "
                f"starting optimizer & scheduler fresh with new LR config"
            )

        # Load scheduler only if optimizer was loaded (same param group count)
        if old_n_groups == new_n_groups:
            try:
                self.scheduler.load_state_dict(state["scheduler"])
            except Exception as e:
                logger.warning(f"Scheduler state incompatible, starting fresh: {e}")
        # If param groups changed, scheduler stays fresh (already initialized for new groups)

        if "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])

        logger.info(f"Loaded checkpoint from {path} (step={self.global_step}, epoch={self.epoch})")


def _build_pipeline_and_config(config):
    """Build DirectorConfig and DirectorPipeline from config dict."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from models.director_model import DirectorConfig, DirectorPipeline
    from models.context_encoder import ContextConfig

    model_cfg = config.get("model", {})
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
        adapter_init_gain=model_cfg.get("attention", {}).get("adapter_init_gain", 0.01),
        freeze_backbone=model_cfg.get("freeze_backbone", True),
        unfreeze_backbone_last_n=model_cfg.get("unfreeze_backbone_last_n", 0),
        freeze_gate=model_cfg.get("attention", {}).get("freeze_gate", False),
        # LoRA config
        lora_enabled=model_cfg.get("lora", {}).get("enabled", False),
        lora_rank=model_cfg.get("lora", {}).get("rank", 16),
        lora_alpha=model_cfg.get("lora", {}).get("alpha", 16),
        lora_target_modules=model_cfg.get("lora", {}).get(
            "target_modules", ["attn1.to_q", "attn1.to_v"]
        ),
        lora_dropout=model_cfg.get("lora", {}).get("dropout", 0.0),
        lora_layers=model_cfg.get("lora", {}).get("layers", None),
    )

    return director_config, DirectorPipeline


def _create_dataloaders(config, pipeline, distributed=False):
    """Create train and val dataloaders."""
    from data.dataset import create_dataloader, DirectorDataset, DirectorDataCollator

    ds_cfg = config.get("dataset", {})
    dl_cfg = ds_cfg.get("dataloader", {})
    vid_cfg = ds_cfg.get("video", {})
    train_cfg = config.get("training", {})

    dataset_dir = ds_cfg.get("dataset_dir", ds_cfg.get("triplet_dir", "data/processed_dataset"))
    train_h = train_cfg.get("train_height", vid_cfg.get("height", 480))
    train_w = train_cfg.get("train_width", vid_cfg.get("width", 720))
    train_f = train_cfg.get("train_frames", vid_cfg.get("num_frames", 49))
    seed = config.get("seed", 42)
    batch_size = dl_cfg.get("batch_size", 1)
    num_workers = dl_cfg.get("num_workers", 4)

    if distributed:
        # Reduce workers per rank to save memory (each worker has per-process overhead)
        num_workers = min(num_workers, 2)
        # For DDP: use DistributedSampler instead of create_dataloader's built-in shuffle
        train_dataset = DirectorDataset(
            dataset_dir=dataset_dir,
            target_height=train_h,
            target_width=train_w,
            target_frames=train_f,
            augment=True,
            split="train",
            seed=seed,
        )
        val_dataset = DirectorDataset(
            dataset_dir=dataset_dir,
            target_height=train_h,
            target_width=train_w,
            target_frames=train_f,
            augment=False,
            split="val",
            seed=seed,
        )

        collator = DirectorDataCollator(
            tokenizer=pipeline.tokenizer,
            max_text_length=226,
        )

        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=seed)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
        )

        return train_loader, val_loader, train_sampler
    else:
        train_loader = create_dataloader(
            dataset_dir=dataset_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            split="train",
            target_height=train_h,
            target_width=train_w,
            target_frames=train_f,
            tokenizer=pipeline.tokenizer,
            seed=seed,
        )
        val_loader = create_dataloader(
            dataset_dir=dataset_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            split="val",
            target_height=train_h,
            target_width=train_w,
            target_frames=train_f,
            tokenizer=pipeline.tokenizer,
            seed=seed,
        )
        # If val set is empty (e.g. 1-sample overfit), set to None
        if len(val_loader.dataset) == 0:
            val_loader = None
        return train_loader, val_loader, None


def main():
    """Entry point for standalone training (single GPU)."""
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    parser = argparse.ArgumentParser(description="DIRECTOR Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--weights-only", action="store_true", help="Load only model weights, fresh optimizer/scheduler")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(config.get("seed", 42))

    # Set device
    device = torch.device(f"cuda:{config.get('cuda_device', 0)}")

    # Create model
    director_config, PipelineClass = _build_pipeline_and_config(config)
    logger.info("Initializing DIRECTOR pipeline...")
    pipeline = PipelineClass(config=director_config, device=device)

    # Create dataloaders
    train_loader, val_loader, _ = _create_dataloaders(config, pipeline, distributed=False)

    # Create trainer
    trainer = DirectorTrainer(
        director_pipeline=pipeline,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume, weights_only=args.weights_only)

    # Train
    trainer.train()


def main_ddp():
    """Entry point for DDP multi-GPU training via torchrun."""
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    parser = argparse.ArgumentParser(description="DIRECTOR DDP Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--weights-only", action="store_true", help="Load only model weights, fresh optimizer/scheduler")
    args = parser.parse_args()

    # DDP environment variables set by torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize process group
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set seed (offset by rank for data diversity, but model init is same)
    set_seed(config.get("seed", 42) + local_rank)

    # Create model on this rank's GPU
    director_config, PipelineClass = _build_pipeline_and_config(config)
    if local_rank == 0:
        logger.info(f"Initializing DIRECTOR pipeline (world_size={world_size})...")
    pipeline = PipelineClass(config=director_config, device=device)

    # Create dataloaders with DistributedSampler
    train_loader, val_loader, train_sampler = _create_dataloaders(
        config, pipeline, distributed=True
    )

    # Create trainer with DDP
    trainer = DirectorTrainer(
        director_pipeline=pipeline,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        local_rank=local_rank,
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume, weights_only=args.weights_only)

    # Store sampler for epoch-based re-seeding
    trainer._train_sampler = train_sampler

    # Train
    trainer.train()

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    # Check if launched via torchrun (DDP)
    if "LOCAL_RANK" in os.environ:
        main_ddp()
    else:
        main()
