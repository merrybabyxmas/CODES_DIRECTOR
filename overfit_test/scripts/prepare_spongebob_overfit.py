"""
Prepare SpongeBob single-batch overfitting dataset.

Extracts 5 shots from the SpongeBob video for architecture verification:
  Shot 1: SpongeBob solo (0:22) - baseline
  Shot 2: Synthetic zoom-in of Shot 1 - continuity test
  Shot 3: Patrick solo (1:00) - cross-cut test
  Shot 4: SpongeBob & Patrick (2:32) - multi-character test
  Shot 5: SpongeBob solo (3:07) - filtering test

Creates:
  data/overfit_dataset/seq_0000{1-5}/ with proper DIRECTOR format
"""

import cv2
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Project root
ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = ROOT / "data/raw_videos/spongebob_test/spongebob_full.mp4"
OUTPUT_DIR = ROOT / "data/overfit_dataset"

# Timestamps (seconds) and durations
SHOTS = {
    "shot1_spongebob_solo": {"start": 22.0, "duration": 2.5},
    "shot3_patrick_solo":   {"start": 60.0, "duration": 2.5},
    "shot4_both":           {"start": 152.0, "duration": 2.5},
    "shot5_spongebob_close":{"start": 187.0, "duration": 2.5},
}

TARGET_FPS = 8
TARGET_FRAMES = 49
TARGET_H, TARGET_W = 320, 512
CLIP_SIZE = 224
ANCHOR_SIZE = 512


def extract_frames(video_path, start_sec, duration_sec, target_fps=8):
    """Extract frames from video at given timestamp."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_sec * fps)
    total_extract = int(duration_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(total_extract):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    # Subsample to target_fps
    if len(frames) == 0:
        return []
    step = fps / target_fps
    selected = [frames[min(int(i * step), len(frames) - 1)] for i in range(int(len(frames) / step))]
    return selected


def resize_frame(frame, h, w):
    """Resize frame to target resolution."""
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LANCZOS4)


def pad_or_subsample(frames, target_count):
    """Pad or subsample frames to exact target count."""
    if len(frames) >= target_count:
        step = len(frames) / target_count
        return [frames[int(i * step)] for i in range(target_count)]
    else:
        result = list(frames)
        while len(result) < target_count:
            result.append(frames[-1])
        return result


def create_zoom_in(frames, zoom_start=1.0, zoom_end=2.0):
    """Create synthetic zoom-in effect from existing frames."""
    zoomed = []
    n = len(frames)
    for i, frame in enumerate(frames):
        t = i / max(1, n - 1)
        zoom = zoom_start + (zoom_end - zoom_start) * t
        h, w = frame.shape[:2]
        # Crop center region
        crop_h = int(h / zoom)
        crop_w = int(w / zoom)
        y1 = (h - crop_h) // 2
        x1 = (w - crop_w) // 2
        cropped = frame[y1:y1+crop_h, x1:x1+crop_w]
        # Resize back to original size
        zoomed.append(cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4))
    return zoomed


def frames_to_video(frames, output_path, fps=8):
    """Save frames as mp4."""
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def extract_character_anchor(frame, bbox_ratio=(0.2, 0.1, 0.8, 0.9)):
    """
    Extract character region from frame as RGBA anchor.
    bbox_ratio: (x1_ratio, y1_ratio, x2_ratio, y2_ratio) of frame.
    Simple center crop for cartoon characters.
    """
    h, w = frame.shape[:2]
    x1 = int(w * bbox_ratio[0])
    y1 = int(h * bbox_ratio[1])
    x2 = int(w * bbox_ratio[2])
    y2 = int(h * bbox_ratio[3])

    crop = frame[y1:y2, x1:x2]
    crop_resized = cv2.resize(crop, (ANCHOR_SIZE, ANCHOR_SIZE), interpolation=cv2.INTER_LANCZOS4)

    # Create RGBA with full alpha (cartoon character on solid bg)
    rgba = np.zeros((ANCHOR_SIZE, ANCHOR_SIZE, 4), dtype=np.uint8)
    rgba[:, :, :3] = crop_resized
    rgba[:, :, 3] = 255  # Full alpha
    return rgba


def save_anchor(rgba_array, path):
    """Save RGBA numpy array as PNG."""
    img = Image.fromarray(rgba_array, 'RGBA')
    img.save(str(path))


def create_seq_dir(seq_id, target_frames, prev_frame, prev_prev_frame,
                   anchor_rgbas, caption_data):
    """Create a seq_XXXXX directory with proper DIRECTOR format."""
    seq_dir = OUTPUT_DIR / f"seq_{seq_id:05d}"
    seq_dir.mkdir(parents=True, exist_ok=True)

    # Save target video
    frames_resized = [resize_frame(f, TARGET_H, TARGET_W) for f in target_frames]
    frames_padded = pad_or_subsample(frames_resized, TARGET_FRAMES)
    frames_to_video(frames_padded, seq_dir / "target_shot.mp4", fps=TARGET_FPS)

    # Save prev_shot_last_frame (t-1)
    if prev_frame is not None:
        prev_resized = resize_frame(prev_frame, TARGET_H, TARGET_W)
        cv2.imwrite(str(seq_dir / "prev_shot_last_frame.jpg"),
                     cv2.cvtColor(prev_resized, cv2.COLOR_RGB2BGR))
    else:
        # Zero frame (no previous shot)
        zero = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
        cv2.imwrite(str(seq_dir / "prev_shot_last_frame.jpg"), zero)

    # Save prev_prev_shot_last_frame (t-2)
    if prev_prev_frame is not None:
        pp_resized = resize_frame(prev_prev_frame, TARGET_H, TARGET_W)
        cv2.imwrite(str(seq_dir / "prev_prev_shot_last_frame.jpg"),
                     cv2.cvtColor(pp_resized, cv2.COLOR_RGB2BGR))

    # Save global anchors
    for k, rgba in enumerate(anchor_rgbas):
        save_anchor(rgba, seq_dir / f"global_anchor_{k}.png")

    # Save caption
    with open(seq_dir / "caption.json", "w") as f:
        json.dump(caption_data, f, indent=2)

    print(f"  Created {seq_dir.name}: {len(frames_padded)} frames, "
          f"{len(anchor_rgbas)} characters")
    return seq_dir


def main():
    print("=" * 60)
    print("SpongeBob Single-Batch Overfitting Dataset Preparation")
    print("=" * 60)

    # ── Step 1: Extract raw shots ──
    print("\n[1/4] Extracting raw shots from video...")
    raw_shots = {}
    for name, info in SHOTS.items():
        frames = extract_frames(VIDEO_PATH, info["start"], info["duration"], TARGET_FPS)
        print(f"  {name}: {len(frames)} frames @ {info['start']}s")
        raw_shots[name] = frames

    # ── Step 2: Create synthetic zoom-in (Shot 2) ──
    print("\n[2/4] Creating synthetic zoom-in for Shot 2...")
    shot1_frames = raw_shots["shot1_spongebob_solo"]
    shot2_frames = create_zoom_in(shot1_frames, zoom_start=1.0, zoom_end=2.0)
    print(f"  shot2_zoom_in: {len(shot2_frames)} frames (synthetic)")

    # ── Step 3: Extract character anchors ──
    print("\n[3/4] Extracting character anchors...")
    # SpongeBob anchor from shot 1 (center character)
    spongebob_anchor = extract_character_anchor(
        shot1_frames[len(shot1_frames) // 2],
        bbox_ratio=(0.25, 0.1, 0.75, 0.9)
    )
    # Patrick anchor from shot 3 (center character)
    patrick_frames = raw_shots["shot3_patrick_solo"]
    patrick_anchor = extract_character_anchor(
        patrick_frames[len(patrick_frames) // 2],
        bbox_ratio=(0.25, 0.1, 0.75, 0.9)
    )
    print(f"  SpongeBob anchor: {ANCHOR_SIZE}x{ANCHOR_SIZE} RGBA")
    print(f"  Patrick anchor: {ANCHOR_SIZE}x{ANCHOR_SIZE} RGBA")

    # ── Step 4: Package into DIRECTOR format ──
    print("\n[4/4] Packaging into DIRECTOR training format...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # seq_00001: S1 (SpongeBob solo) + SpongeBob anchor + No local context
    create_seq_dir(
        seq_id=1,
        target_frames=shot1_frames,
        prev_frame=None,
        prev_prev_frame=None,
        anchor_rgbas=[spongebob_anchor],
        caption_data={
            "identity": {"gender": "male", "clothing": "white shirt with red tie",
                         "appearance": "yellow square sponge character"},
            "motion": {"camera": "static", "person": "standing",
                       "environment": "fast food restaurant kitchen"},
            "full": "A yellow square sponge character wearing a white shirt with red tie, "
                    "standing in a fast food restaurant kitchen"
        }
    )

    # seq_00002: S2 (Zoom-in) + SpongeBob anchor + S1 last frame
    create_seq_dir(
        seq_id=2,
        target_frames=shot2_frames,
        prev_frame=shot1_frames[-1],
        prev_prev_frame=None,
        anchor_rgbas=[spongebob_anchor],
        caption_data={
            "identity": {"gender": "male", "clothing": "white shirt with red tie",
                         "appearance": "yellow square sponge character"},
            "motion": {"camera": "zoom-in", "person": "standing",
                       "environment": "fast food restaurant kitchen"},
            "full": "Camera zooms in on a yellow square sponge character wearing a white shirt "
                    "with red tie in a fast food restaurant kitchen"
        }
    )

    # seq_00003: S3 (Patrick solo) + Patrick anchor + S2 last frame
    create_seq_dir(
        seq_id=3,
        target_frames=patrick_frames,
        prev_frame=shot2_frames[-1],
        prev_prev_frame=shot1_frames[-1],
        anchor_rgbas=[patrick_anchor],
        caption_data={
            "identity": {"gender": "male", "clothing": "green and purple shorts",
                         "appearance": "pink starfish character"},
            "motion": {"camera": "static", "person": "sitting",
                       "environment": "fast food restaurant dining area"},
            "full": "A pink starfish character wearing green and purple shorts, "
                    "sitting in a fast food restaurant dining area"
        }
    )

    # seq_00004: S4 (Both) + SpongeBob & Patrick anchors + S3 last frame
    both_frames = raw_shots["shot4_both"]
    create_seq_dir(
        seq_id=4,
        target_frames=both_frames,
        prev_frame=patrick_frames[-1],
        prev_prev_frame=shot2_frames[-1],
        anchor_rgbas=[spongebob_anchor, patrick_anchor],
        caption_data={
            "identity": {"gender": "male", "clothing": "white shirt and green shorts",
                         "appearance": "yellow sponge and pink starfish together"},
            "motion": {"camera": "static", "person": "standing together",
                       "environment": "fast food restaurant kitchen"},
            "full": "A yellow square sponge character and a pink starfish character "
                    "standing together in a fast food restaurant kitchen"
        }
    )

    # seq_00005: S5 (SpongeBob close-up) + SpongeBob anchor + S4 last frame
    close_frames = raw_shots["shot5_spongebob_close"]
    create_seq_dir(
        seq_id=5,
        target_frames=close_frames,
        prev_frame=both_frames[-1],
        prev_prev_frame=patrick_frames[-1],
        anchor_rgbas=[spongebob_anchor],
        caption_data={
            "identity": {"gender": "male", "clothing": "white shirt with red tie",
                         "appearance": "yellow square sponge character close-up"},
            "motion": {"camera": "close-up", "person": "talking",
                       "environment": "fast food restaurant"},
            "full": "Close-up of a yellow square sponge character wearing a white shirt "
                    "with red tie, talking in a fast food restaurant"
        }
    )

    # Save metadata
    metadata = {
        "description": "SpongeBob single-batch overfitting test dataset",
        "num_sequences": 5,
        "target_resolution": f"{TARGET_W}x{TARGET_H}",
        "target_frames": TARGET_FRAMES,
        "target_fps": TARGET_FPS,
        "shots": {
            "seq_00001": "SpongeBob solo (baseline)",
            "seq_00002": "Zoom-in on SpongeBob (continuity)",
            "seq_00003": "Patrick solo (cross-cut)",
            "seq_00004": "SpongeBob & Patrick (multi-character)",
            "seq_00005": "SpongeBob close-up (filtering)",
        }
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Dataset created at: {OUTPUT_DIR}")
    print(f"Sequences: 5")
    print(f"Ready for single-batch overfitting test!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
