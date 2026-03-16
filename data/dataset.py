"""
DIRECTOR PyTorch Dataset.

Loads triplet data produced by dataset_pipeline.py (v2 flat format) and
returns tensors suitable for DIRECTOR training.

Expected on-disk layout:
  data/processed_dataset/
  ├── seq_00001/
  │   ├── global_anchor.png         # RGBA character with transparent bg
  │   ├── prev_shot_last_frame.jpg  # Last frame of S_{t-1}
  │   ├── target_shot.mp4           # S_t video clip (8fps, max 49 frames)
  │   └── caption.json              # {"identity": "...", "motion": "...", "full": "..."}
  ├── seq_00002/ ...
  └── metadata.jsonl

Each __getitem__ returns:
  target_video  : (T, 3, H, W)   float32 [0, 1]  – the target shot
  prev_frame    : (3, H, W)      float32 [0, 1]  – last frame of previous shot
  anchor_image  : (4, H_a, W_a)  float32 [0, 1]  – RGBA global anchor (resized)
  anchor_rgb    : (3, 224, 224)   float32 [0, 1]  – anchor RGB for CLIP encoder
  caption       : str                              – full caption text
  identity_text : str                              – identity-only caption
  motion_text   : str                              – motion-only caption
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class DirectorDataset(Dataset):
    """
    PyTorch Dataset for DIRECTOR training triplets (v2 flat layout).

    Reads seq_XXXXX/ directories, each containing:
      global_anchor.png, prev_shot_last_frame.jpg, target_shot.mp4, caption.json
    """

    def __init__(
        self,
        dataset_dir: str,
        target_height: int = 480,
        target_width: int = 720,
        target_frames: int = 49,
        anchor_size: int = 512,
        clip_image_size: int = 224,
        augment: bool = True,
        split: str = "train",
        split_ratio: float = 0.9,
        seed: int = 42,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.target_height = target_height
        self.target_width = target_width
        self.target_frames = target_frames
        self.anchor_size = anchor_size
        self.clip_image_size = clip_image_size
        self.augment = augment and split == "train"

        # Discover all seq_XXXXX directories
        self.seq_dirs = self._discover_sequences()

        # Train/val split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(self.seq_dirs))
        split_point = int(len(indices) * split_ratio)
        if split == "train":
            self.indices = indices[:split_point]
        else:
            self.indices = indices[split_point:]

        logger.info(
            f"DirectorDataset [{split}]: {len(self.indices)} samples "
            f"from {len(self.seq_dirs)} total sequences"
        )

        # Transforms
        self.video_resize = transforms.Resize(
            (target_height, target_width), antialias=True
        )
        self.anchor_resize = transforms.Resize(
            (anchor_size, anchor_size), antialias=True
        )
        self.clip_resize = transforms.Resize(
            (clip_image_size, clip_image_size), antialias=True
        )
        self.clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

    def _discover_sequences(self) -> List[Path]:
        """Find all valid seq_XXXXX directories."""
        seq_dirs = []
        for entry in sorted(self.dataset_dir.iterdir()):
            if entry.is_dir() and entry.name.startswith("seq_"):
                # Check required files exist
                required = ["global_anchor.png", "prev_shot_last_frame.jpg",
                            "target_shot.mp4", "caption.json"]
                if all((entry / f).exists() for f in required):
                    seq_dirs.append(entry)
                else:
                    missing = [f for f in required if not (entry / f).exists()]
                    logger.warning(f"Skipping {entry.name}: missing {missing}")
        return seq_dirs

    def _load_video_from_mp4(self, mp4_path: str) -> torch.Tensor:
        """
        Load frames from target_shot.mp4.

        Returns:
            (T, 3, H, W) float32 in [0, 1], padded/subsampled to target_frames.
        """
        cap = cv2.VideoCapture(mp4_path)
        frames = []
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # (H, W, 3) -> (3, H, W) float [0, 1]
            t = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
            t = self.video_resize(t.unsqueeze(0)).squeeze(0)  # (3, H, W)
            frames.append(t)
        cap.release()

        if len(frames) == 0:
            return torch.zeros(self.target_frames, 3, self.target_height, self.target_width)

        # Subsample or pad to target_frames
        if len(frames) >= self.target_frames:
            step = len(frames) / self.target_frames
            selected = [frames[int(i * step)] for i in range(self.target_frames)]
        else:
            selected = list(frames)
            while len(selected) < self.target_frames:
                selected.append(frames[-1])  # repeat last frame

        return torch.stack(selected)  # (T, 3, H, W)

    def _load_image(self, path: str) -> torch.Tensor:
        """Load a single RGB image as (3, H, W) float [0, 1]."""
        img = cv2.imread(path)
        if img is None:
            return torch.zeros(3, self.target_height, self.target_width)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return t

    def _load_anchor(self, path: str) -> torch.Tensor:
        """
        Load RGBA global anchor as (4, H, W) float [0, 1].
        """
        img = Image.open(path).convert("RGBA")
        arr = np.array(img)  # (H, W, 4)
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0  # (4, H, W)
        t = self.anchor_resize(t)  # (4, anchor_size, anchor_size)
        return t

    def _anchor_to_clip_rgb(self, anchor_rgba: torch.Tensor) -> torch.Tensor:
        """
        Convert RGBA anchor to RGB on white background, resize for CLIP.

        Args:
            anchor_rgba: (4, H, W) float [0, 1]

        Returns:
            (3, 224, 224) float, CLIP-normalised.
        """
        rgb = anchor_rgba[:3]    # (3, H, W)
        alpha = anchor_rgba[3:4]  # (1, H, W)
        # Composite on white background
        white_bg = torch.ones_like(rgb)
        composited = rgb * alpha + white_bg * (1 - alpha)  # (3, H, W)
        composited = self.clip_resize(composited)  # (3, 224, 224)
        composited = self.clip_normalize(composited)  # CLIP normalization
        return composited

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq_dir = self.seq_dirs[self.indices[idx]]

        # Load target video
        target_video = self._load_video_from_mp4(
            str(seq_dir / "target_shot.mp4")
        )  # (T, 3, H, W)

        # Load previous shot's last frame (t-1)
        prev_frame = self._load_image(
            str(seq_dir / "prev_shot_last_frame.jpg")
        )  # (3, H, W)
        prev_frame = self.video_resize(prev_frame.unsqueeze(0)).squeeze(0)

        # Load t-2 frame if available
        prev_prev_path = seq_dir / "prev_prev_shot_last_frame.jpg"
        if prev_prev_path.exists():
            prev_prev_frame = self._load_image(str(prev_prev_path))
            prev_prev_frame = self.video_resize(prev_prev_frame.unsqueeze(0)).squeeze(0)
            has_prev_prev = True
        else:
            prev_prev_frame = torch.zeros_like(prev_frame)
            has_prev_prev = False

        # Load global anchor
        anchor_rgba = self._load_anchor(
            str(seq_dir / "global_anchor.png")
        )  # (4, anchor_size, anchor_size)

        # CLIP-ready RGB anchor
        anchor_rgb = self._anchor_to_clip_rgb(anchor_rgba)  # (3, 224, 224)

        # Load caption
        caption_path = seq_dir / "caption.json"
        with open(caption_path, "r") as f:
            caption_data = json.load(f)

        identity_raw = caption_data.get("identity", "")
        motion_raw = caption_data.get("motion", "")

        # VLM may output dicts — flatten to descriptive strings
        if isinstance(identity_raw, dict):
            identity_text = ", ".join(f"{v}" for v in identity_raw.values() if v and v != "neutral")
        else:
            identity_text = str(identity_raw)

        if isinstance(motion_raw, dict):
            motion_text = ", ".join(f"{v}" for v in motion_raw.values() if v and v != "neutral")
        else:
            motion_text = str(motion_raw)

        full_raw = caption_data.get("full", "")
        if isinstance(full_raw, str) and full_raw:
            full_text = full_raw
        else:
            full_text = f"{identity_text}. {motion_text}".strip(". ")

        # Augmentation: synchronised horizontal flip
        if self.augment and torch.rand(1).item() < 0.5:
            target_video = torch.flip(target_video, dims=[-1])
            prev_frame = torch.flip(prev_frame, dims=[-1])
            prev_prev_frame = torch.flip(prev_prev_frame, dims=[-1])
            anchor_rgba = torch.flip(anchor_rgba, dims=[-1])
            anchor_rgb = torch.flip(anchor_rgb, dims=[-1])

        return {
            "target_video": target_video,        # (T, 3, H, W)
            "prev_frame": prev_frame,            # (3, H, W) - t-1
            "prev_prev_frame": prev_prev_frame,  # (3, H, W) - t-2
            "has_prev_prev": has_prev_prev,       # bool
            "anchor_image": anchor_rgba,         # (4, anchor_size, anchor_size)
            "anchor_rgb": anchor_rgb,            # (3, 224, 224)
            "caption": full_text,                # str
            "identity_text": identity_text,      # str
            "motion_text": motion_text,          # str
            "seq_id": seq_dir.name,              # str
        }


class DirectorDataCollator:
    """
    Custom collator for DIRECTOR dataset.

    Stacks tensors and collects captions as lists.
    Optionally tokenises text if a tokenizer is provided.
    """

    def __init__(self, tokenizer=None, max_text_length: int = 226):
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = {
            "target_video": torch.stack([s["target_video"] for s in batch]),     # (B, T, 3, H, W)
            "prev_frame": torch.stack([s["prev_frame"] for s in batch]),         # (B, 3, H, W)
            "prev_prev_frame": torch.stack([s["prev_prev_frame"] for s in batch]),  # (B, 3, H, W)
            "has_prev_prev": torch.tensor([s["has_prev_prev"] for s in batch]),  # (B,)
            "anchor_image": torch.stack([s["anchor_image"] for s in batch]),     # (B, 4, Ha, Wa)
            "anchor_rgb": torch.stack([s["anchor_rgb"] for s in batch]),         # (B, 3, 224, 224)
            "captions": [s["caption"] for s in batch],
            "identity_texts": [s["identity_text"] for s in batch],
            "motion_texts": [s["motion_text"] for s in batch],
            "seq_ids": [s["seq_id"] for s in batch],
        }

        if self.tokenizer is not None:
            text_inputs = self.tokenizer(
                result["captions"],
                max_length=self.max_text_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            result["input_ids"] = text_inputs.input_ids
            result["attention_mask"] = text_inputs.attention_mask

        return result


def create_dataloader(
    dataset_dir: str,
    batch_size: int = 1,
    num_workers: int = 4,
    split: str = "train",
    target_height: int = 480,
    target_width: int = 720,
    target_frames: int = 49,
    anchor_size: int = 512,
    clip_image_size: int = 224,
    tokenizer=None,
    seed: int = 42,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for DIRECTOR training/validation.

    Args:
        dataset_dir: path to processed_dataset/ root (containing seq_XXXXX dirs)
        batch_size: batch size
        num_workers: data loading workers
        split: 'train' or 'val'
        target_height, target_width: video frame dimensions
        target_frames: number of video frames per sample
        anchor_size: global anchor image size
        clip_image_size: CLIP input size for anchor RGB
        tokenizer: optional text tokenizer
        seed: random seed
        pin_memory: pin GPU memory

    Returns:
        DataLoader instance
    """
    dataset = DirectorDataset(
        dataset_dir=dataset_dir,
        target_height=target_height,
        target_width=target_width,
        target_frames=target_frames,
        anchor_size=anchor_size,
        clip_image_size=clip_image_size,
        augment=(split == "train"),
        split=split,
        seed=seed,
    )

    collator = DirectorDataCollator(
        tokenizer=tokenizer,
        max_text_length=226,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=(split == "train"),
        generator=torch.Generator().manual_seed(seed),
    )

    return loader
