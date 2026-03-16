"""
DIRECTOR Inference Pipeline.

Supports:
  1. Pre-production: generate/cache character assets
  2. Autoregressive multi-shot generation with O(1) memory per shot
  3. Multi-CFG with controllable omega_text, omega_local, omega_global
  4. Euler ODE solver for flow matching

Usage:
  python -m inference.generate \
    --config configs/default.yaml \
    --checkpoint checkpoints/best.pt \
    --prompts prompts.json \
    --output_dir outputs/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


class CharacterCache:
    """
    Pre-production character asset cache.

    Stores character reference images and their CLIP-encoded global tokens
    for reuse across shots. Provides O(1) access to character identity.
    """

    def __init__(self, max_characters: int = 4, device: torch.device = torch.device("cuda")):
        self.max_characters = max_characters
        self.device = device

        self.references: Dict[str, torch.Tensor] = {}     # name -> (3, 224, 224)
        self.encoded: Dict[str, torch.Tensor] = {}         # name -> (N_global, D)
        self.order: List[str] = []                          # insertion order

    def add_character(
        self,
        name: str,
        image: torch.Tensor,
        encoded_tokens: Optional[torch.Tensor] = None,
    ):
        """
        Add a character reference to the cache.

        Args:
            name: character identifier
            image: (3, 224, 224) preprocessed reference image
            encoded_tokens: (N_global, D) pre-encoded global tokens (optional)
        """
        if len(self.references) >= self.max_characters and name not in self.references:
            logger.warning(
                f"Cache full ({self.max_characters}). Evicting oldest: {self.order[0]}"
            )
            oldest = self.order.pop(0)
            del self.references[oldest]
            if oldest in self.encoded:
                del self.encoded[oldest]

        self.references[name] = image.to(self.device)
        if encoded_tokens is not None:
            self.encoded[name] = encoded_tokens.to(self.device)
        if name not in self.order:
            self.order.append(name)

    def get_images(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Get all character images as batched tensors.

        Returns:
            char_images: list of (1, 3, 224, 224) tensors
            char_mask: (1, K) validity mask
        """
        images = []
        mask = []
        for i in range(self.max_characters):
            if i < len(self.order):
                name = self.order[i]
                images.append(self.references[name].unsqueeze(0))  # (1, 3, 224, 224)
                mask.append(1.0)
            else:
                images.append(torch.zeros(1, 3, 224, 224, device=self.device))
                mask.append(0.0)

        char_mask = torch.tensor([mask], device=self.device, dtype=torch.float32)
        return images, char_mask

    def clear(self):
        """Clear all cached characters."""
        self.references.clear()
        self.encoded.clear()
        self.order.clear()


class DirectorInferencePipeline:
    """
    Full DIRECTOR inference pipeline.

    Handles:
      - Character asset loading and caching
      - Autoregressive shot-by-shot generation
      - Multi-CFG guidance
      - Video concatenation and export
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: Optional[str] = None,
        device: torch.device = torch.device("cuda"),
    ):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device = device
        inf_cfg = self.config.get("inference", {})
        self.guidance_cfg = inf_cfg.get("guidance", {})
        self.solver_cfg = inf_cfg.get("solver", {})
        self.gen_cfg = inf_cfg.get("generation", {})
        self.ar_cfg = inf_cfg.get("autoregressive", {})
        self.output_cfg = inf_cfg.get("output", {})

        # Initialize model
        self._init_model(checkpoint_path)

        # Character cache
        self.character_cache = CharacterCache(
            max_characters=self.ar_cfg.get("character_cache_size", 4),
            device=device,
        )

    def _init_model(self, checkpoint_path: Optional[str]):
        """Initialize the DIRECTOR model from config and optional checkpoint."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

        from models.director_model import DirectorConfig, DirectorPipeline
        from models.context_encoder import ContextConfig

        model_cfg = self.config.get("model", {})
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

        self.pipeline = DirectorPipeline(config=director_config, device=self.device)

        # Load checkpoint if available
        if checkpoint_path and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            dt = self.pipeline.director_transformer

            # Load adapters (post-block architecture)
            if "adapters" in state:
                dt.adapters.load_state_dict(state["adapters"])
            elif "director_processors" in state:
                # Legacy checkpoint format
                dt.adapters.load_state_dict(state["director_processors"])

            dt.context_builder.load_state_dict(state["context_builder"])

            if "global_encoder" in state:
                dt.global_encoder.load_state_dict(
                    state["global_encoder"], strict=False
                )
            if "local_encoder" in state and dt.local_encoder is not None:
                dt.local_encoder.load_state_dict(
                    state["local_encoder"], strict=False
                )
            logger.info(f"Loaded checkpoint: {checkpoint_path}")

        self.pipeline.director_transformer.eval()

    def load_character(self, name: str, image_path: str):
        """
        Load a character reference image into the cache.

        Args:
            name: character identifier
            image_path: path to character reference image (RGB)
        """
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224), Image.LANCZOS)
        tensor = transforms.ToTensor()(img)  # (3, 224, 224)
        self.character_cache.add_character(name, tensor)
        logger.info(f"Loaded character '{name}' from {image_path}")

    def generate_single_shot(
        self,
        prompt: str,
        prev_frame: Optional[torch.Tensor] = None,
        omega_text: Optional[float] = None,
        omega_local: Optional[float] = None,
        omega_global: Optional[float] = None,
        num_steps: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        seed: int = 42,
    ) -> torch.Tensor:
        """
        Generate a single video shot.

        Args:
            prompt: text description
            prev_frame: (1, 3, H, W) previous shot's last frame, or None for first shot
            omega_*: guidance scales (use config defaults if None)
            num_steps: ODE solver steps
            height, width, num_frames: video dimensions
            seed: random seed

        Returns:
            video: (1, T, 3, H, W) generated video in [0, 1]
        """
        omega_text = omega_text or self.guidance_cfg.get("omega_text", 6.0)
        omega_local = omega_local or self.guidance_cfg.get("omega_local", 2.0)
        omega_global = omega_global or self.guidance_cfg.get("omega_global", 3.0)
        num_steps = num_steps or self.solver_cfg.get("num_steps", 50)
        height = height or self.gen_cfg.get("height", 480)
        width = width or self.gen_cfg.get("width", 720)
        num_frames = num_frames or self.gen_cfg.get("num_frames", 49)

        # Get character references from cache
        char_images, char_mask = self.character_cache.get_images()

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        video = self.pipeline.generate_shot(
            prompt=prompt,
            prev_frame=prev_frame,
            character_images=char_images if any(m > 0 for m in char_mask[0]) else None,
            character_masks=char_mask,
            omega_text=omega_text,
            omega_local=omega_local,
            omega_global=omega_global,
            num_steps=num_steps,
            height=height,
            width=width,
            num_frames=num_frames,
            generator=generator,
        )

        return video

    def generate_multi_shot_story(
        self,
        shot_prompts: List[str],
        character_refs: Optional[Dict[str, str]] = None,
        omega_text: Optional[float] = None,
        omega_local: Optional[float] = None,
        omega_global: Optional[float] = None,
        num_steps: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        seed: int = 42,
        save_dir: Optional[str] = None,
    ) -> List[torch.Tensor]:
        """
        Generate a multi-shot video story autoregressively.

        Each shot is conditioned on:
          - Text prompt (what happens in this shot)
          - Previous shot's last frame (local context for continuity)
          - Character references (global context for identity)

        Args:
            shot_prompts: list of text prompts, one per shot
            character_refs: dict {name: image_path} for character references
            omega_*, num_steps, height, width, num_frames: generation params
            seed: base random seed
            save_dir: optional directory to save individual shots

        Returns:
            List of video tensors (1, T, 3, H, W) per shot
        """
        # Load character references
        self.character_cache.clear()
        if character_refs:
            for name, path in character_refs.items():
                self.load_character(name, path)

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

        all_shots = []
        prev_frame = None
        total_time = 0.0

        for shot_idx, prompt in enumerate(shot_prompts):
            logger.info(f"Generating shot {shot_idx + 1}/{len(shot_prompts)}: '{prompt[:50]}...'")

            start_time = time.time()

            video = self.generate_single_shot(
                prompt=prompt,
                prev_frame=prev_frame,
                omega_text=omega_text,
                omega_local=omega_local,
                omega_global=omega_global,
                num_steps=num_steps,
                height=height,
                width=width,
                num_frames=num_frames,
                seed=seed + shot_idx,
            )  # (1, T, 3, H, W)

            elapsed = time.time() - start_time
            total_time += elapsed
            logger.info(f"Shot {shot_idx + 1} generated in {elapsed:.1f}s")

            all_shots.append(video)

            # Extract last frame for next shot (O(1) memory)
            prev_frame = video[:, -1].clone()  # (1, 3, H, W)

            # Save individual shot
            if save_dir:
                self._save_video(
                    video[0],  # (T, 3, H, W)
                    save_path / f"shot_{shot_idx:03d}.mp4",
                    fps=self.gen_cfg.get("fps", 8),
                )

            # Free GPU memory
            torch.cuda.empty_cache()

        logger.info(
            f"Generated {len(shot_prompts)} shots in {total_time:.1f}s "
            f"({total_time / len(shot_prompts):.1f}s/shot)"
        )

        # Save concatenated video
        if save_dir:
            self._save_concatenated(
                all_shots,
                save_path / "full_story.mp4",
                fps=self.gen_cfg.get("fps", 8),
            )

        return all_shots

    def _save_video(self, video: torch.Tensor, path: Path, fps: int = 8):
        """
        Save a video tensor to file.

        Args:
            video: (T, 3, H, W) float tensor in [0, 1]
            path: output file path
            fps: frames per second
        """
        video_np = (video.cpu().float().clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()
        T, H, W, C = video_np.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        quality = self.output_cfg.get("video_quality", 23)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (W, H))

        for t in range(T):
            frame_bgr = cv2.cvtColor(video_np[t], cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        writer.release()
        logger.info(f"Saved video: {path} ({T} frames, {W}x{H}, {fps}fps)")

    def _save_concatenated(
        self, shots: List[torch.Tensor], path: Path, fps: int = 8
    ):
        """Concatenate all shots into a single video file."""
        all_frames = []
        for shot in shots:
            # shot: (1, T, 3, H, W)
            frames = shot[0].cpu().float().clamp(0, 1)  # (T, 3, H, W)
            all_frames.append(frames)

        concat = torch.cat(all_frames, dim=0)  # (T_total, 3, H, W)
        self._save_video(concat, path, fps)


def main():
    parser = argparse.ArgumentParser(description="DIRECTOR Video Generation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--prompts", type=str, required=True, help="JSON file with shot prompts")
    parser.add_argument("--characters", type=str, default=None, help="JSON file with character refs")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--omega_text", type=float, default=None)
    parser.add_argument("--omega_local", type=float, default=None)
    parser.add_argument("--omega_global", type=float, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load prompts
    with open(args.prompts) as f:
        prompts_data = json.load(f)

    if isinstance(prompts_data, list):
        shot_prompts = prompts_data
    else:
        shot_prompts = prompts_data.get("shots", prompts_data.get("prompts", []))

    # Load character references
    character_refs = None
    if args.characters:
        with open(args.characters) as f:
            character_refs = json.load(f)

    # Initialize pipeline
    device_id = yaml.safe_load(open(args.config)).get("cuda_device", 0)
    device = torch.device(f"cuda:{device_id}")

    pipeline = DirectorInferencePipeline(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # Generate
    shots = pipeline.generate_multi_shot_story(
        shot_prompts=shot_prompts,
        character_refs=character_refs,
        omega_text=args.omega_text,
        omega_local=args.omega_local,
        omega_global=args.omega_global,
        num_steps=args.num_steps,
        seed=args.seed,
        save_dir=args.output_dir,
    )

    logger.info(f"Generation complete. Output saved to {args.output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
