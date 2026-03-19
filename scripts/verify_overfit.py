"""
Verify single-batch overfitting by generating video with the EXACT same inputs
used during training and comparing to the original target.

Generates all 5 shots with their exact training conditions.
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.director_model import DirectorConfig, DirectorPipeline
from models.context_encoder import ContextConfig


def load_image(path, height, width):
    """Load image as (1, 3, H, W) tensor in [0, 1]."""
    img = cv2.imread(str(path))
    if img is None:
        return torch.zeros(1, 3, height, width)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)  # (1, 3, H, W)


def load_anchor(path, clip_size=224):
    """Load RGBA anchor → CLIP-normalized (1, 3, 224, 224)."""
    img = Image.open(str(path)).convert("RGBA")
    arr = np.array(img)
    rgba = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0  # (4, H, W)

    rgb = rgba[:3]
    alpha = rgba[3:4]
    white_bg = torch.ones_like(rgb)
    composited = rgb * alpha + white_bg * (1 - alpha)

    resize = transforms.Resize((clip_size, clip_size), antialias=True)
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    composited = normalize(resize(composited))
    return composited.unsqueeze(0)  # (1, 3, 224, 224)


def extract_first_frame_from_mp4(path, height, width):
    """Extract first frame from target_shot.mp4 for visual comparison."""
    cap = cv2.VideoCapture(str(path))
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height))
        return frame
    return np.zeros((height, width, 3), dtype=np.uint8)


def save_comparison(generated_video, target_path, output_path, height, width):
    """Save side-by-side comparison: target first frame vs generated first frame."""
    target_frame = extract_first_frame_from_mp4(target_path, height, width)

    # Generated first frame
    gen_frame = generated_video[0, 0].cpu().float().clamp(0, 1)  # (3, H, W)
    gen_frame = (gen_frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Generated middle frame
    mid_idx = generated_video.shape[1] // 2
    gen_mid = generated_video[0, mid_idx].cpu().float().clamp(0, 1)
    gen_mid = (gen_mid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Side-by-side: target | gen_first | gen_mid
    comparison = np.concatenate([target_frame, gen_frame, gen_mid], axis=1)
    comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), comparison_bgr)
    print(f"  Saved comparison: {output_path}")


def save_video(video_tensor, path, fps=8):
    """Save (1, T, 3, H, W) video tensor as mp4."""
    video = video_tensor[0].cpu().float().clamp(0, 1)  # (T, 3, H, W)
    video_np = (video.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    T, H, W, C = video_np.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, fps, (W, H))
    for t in range(T):
        writer.write(cv2.cvtColor(video_np[t], cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints_overfit/checkpoint_best.pt")
    parser.add_argument("--dataset_dir", default="data/overfit_dataset")
    parser.add_argument("--output_dir", default="samples/overfit_verify")
    parser.add_argument("--num_steps", type=int, default=30)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=49)
    args = parser.parse_args()

    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)

    print("=" * 60)
    print("OVERFIT VERIFICATION: Generating with exact training inputs")
    print("=" * 60)

    # Build model
    print("\n[1/3] Loading model...")
    config = DirectorConfig(
        backbone="THUDM/CogVideoX-2b",
        context=ContextConfig(
            num_local_frames=2,
            local_token_count=256,
            global_token_count=64,
            max_characters=4,
            context_dim=1920,
            clip_vision_dim=1024,
        ),
        drop_global_prob=0.0,
        drop_local_prob=0.0,
    )
    pipeline = DirectorPipeline(config=config, device=device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    dt = pipeline.director_transformer
    dt.adapters.load_state_dict(ckpt["adapters"])
    dt.context_builder.load_state_dict(ckpt["context_builder"])
    if "global_encoder" in ckpt:
        dt.global_encoder.load_state_dict(ckpt["global_encoder"], strict=False)
    if "local_encoder" in ckpt and dt.local_encoder is not None:
        dt.local_encoder.load_state_dict(ckpt["local_encoder"], strict=False)
    if "backbone" in ckpt:
        dt.backbone.load_state_dict(ckpt["backbone"], strict=False)
        print(f"  Loaded backbone params: {len(ckpt['backbone'])} keys")
    loaded_step = ckpt.get("global_step", ckpt.get("step", "?"))
    print(f"  Loaded checkpoint: {args.checkpoint} (step {loaded_step})")

    # Define 5 shots with exact training inputs
    shots = [
        {
            "name": "shot1_spongebob_solo",
            "seq": "seq_00001",
            "caption": "A yellow square sponge character wearing a white shirt with red tie, standing in a fast food restaurant kitchen",
            "prev_frames": [],  # No local context
            "anchors": ["global_anchor_0.png"],
        },
        {
            "name": "shot2_zoom_in",
            "seq": "seq_00002",
            "caption": "Camera zooms in on a yellow square sponge character wearing a white shirt with red tie in a fast food restaurant kitchen",
            "prev_frames": ["seq_00001"],  # t-1 from shot 1
            "anchors": ["global_anchor_0.png"],
        },
        {
            "name": "shot3_patrick_solo",
            "seq": "seq_00003",
            "caption": "A pink starfish character wearing green and purple shorts, sitting in a fast food restaurant dining area",
            "prev_frames": ["seq_00002", "seq_00001"],  # t-1, t-2
            "anchors": ["global_anchor_0.png"],
        },
        {
            "name": "shot4_both",
            "seq": "seq_00004",
            "caption": "A yellow square sponge character and a pink starfish character standing together in a fast food restaurant kitchen",
            "prev_frames": ["seq_00003", "seq_00002"],
            "anchors": ["global_anchor_0.png", "global_anchor_1.png"],
        },
        {
            "name": "shot5_spongebob_close",
            "seq": "seq_00005",
            "caption": "Close-up of a yellow square sponge character wearing a white shirt with red tie, talking in a fast food restaurant",
            "prev_frames": ["seq_00004", "seq_00003"],
            "anchors": ["global_anchor_0.png"],
        },
    ]

    # Generate each shot
    print("\n[2/3] Generating shots...")
    for i, shot in enumerate(shots):
        print(f"\n  Shot {i+1}/5: {shot['name']}")
        seq_dir = dataset_dir / shot["seq"]

        # Load prev_frames (local context)
        prev_frames = []
        for prev_seq in shot["prev_frames"]:
            prev_dir = dataset_dir / prev_seq
            pf = load_image(
                prev_dir / "prev_shot_last_frame.jpg" if prev_seq == shot["prev_frames"][0]
                else prev_dir / "prev_shot_last_frame.jpg",
                args.height, args.width,
            )
            prev_frames.append(pf.to(device))

        # Actually, for proper AR chain, use the LAST frame of prev seq's target video
        # But for training verification, use the exact prev_shot_last_frame.jpg stored
        prev_frames_loaded = []
        if shot["prev_frames"]:
            # t-1: prev_shot_last_frame.jpg of current seq
            pf = load_image(seq_dir / "prev_shot_last_frame.jpg", args.height, args.width)
            prev_frames_loaded.append(pf.to(device))

            # t-2: prev_prev_shot_last_frame.jpg of current seq (if exists)
            pp_path = seq_dir / "prev_prev_shot_last_frame.jpg"
            if pp_path.exists():
                pp = load_image(pp_path, args.height, args.width)
                prev_frames_loaded.append(pp.to(device))

        # Load anchors (global context)
        char_images = []
        for anchor_name in shot["anchors"]:
            anchor = load_anchor(seq_dir / anchor_name)
            char_images.append(anchor.to(device))

        # Pad to max_characters
        max_chars = 4
        char_mask_vals = [1.0] * len(char_images)
        while len(char_images) < max_chars:
            char_images.append(torch.zeros(1, 3, 224, 224, device=device))
            char_mask_vals.append(0.0)
        char_mask = torch.tensor([char_mask_vals], device=device)

        # Generate
        gen = torch.Generator(device=device)
        gen.manual_seed(42 + i)

        with torch.no_grad():
            video = pipeline.generate_shot(
                prompt=shot["caption"],
                prev_frames=prev_frames_loaded if prev_frames_loaded else None,
                character_images=char_images,
                character_masks=char_mask,
                omega_text=6.0,
                omega_local=2.0,
                omega_global=3.0,
                num_steps=args.num_steps,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                generator=gen,
            )

        # Save generated video
        save_video(video, output_dir / f"{shot['name']}_generated.mp4")

        # Save comparison image (target vs generated)
        target_mp4 = seq_dir / "target_shot.mp4"
        save_comparison(video, target_mp4,
                        output_dir / f"{shot['name']}_comparison.jpg",
                        args.height, args.width)

        torch.cuda.empty_cache()

    # Save metadata
    print("\n[3/3] Saving metadata...")
    metadata = {
        "checkpoint": args.checkpoint,
        "loaded_step": loaded_step,
        "num_steps": args.num_steps,
        "resolution": f"{args.width}x{args.height}",
        "num_frames": args.num_frames,
        "guidance": {"omega_text": 6.0, "omega_local": 2.0, "omega_global": 3.0},
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Verification complete! Results in: {output_dir}/")
    print(f"Check *_comparison.jpg files: left=target, center=gen_first, right=gen_mid")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
