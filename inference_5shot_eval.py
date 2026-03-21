"""
DIRECTOR: 5-Shot Systematic Evaluation Pipeline.

Shot 1 [Baseline]:     Subject A in Setting 1, no local context
Shot 2 [Continuity]:   Same setting, camera move, uses Shot 1's last frame
Shot 3 [Transition]:   Subject B in Setting 2, uses Shot 2's last frame (must forget)
Shot 4 [Multi-ID]:     Both A & B in Setting 1, uses Shot 3's last frame
Shot 5 [Filtering]:    Only Subject A close-up, uses Shot 4's last frame (must filter B out)

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n paper_env python inference_5shot_eval.py \
        --config configs/full_training_lora.yaml \
        --checkpoint checkpoints_full_lora/checkpoint_best.pt
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.utils import make_grid

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ── 5-shot definitions ──────────────────────────────────────────────────────
SHOT_DEFS = [
    {
        "name": "Shot 1: BASELINE",
        "subtitle": "[Baseline] Identity & Environment — Subject A in modern office",
        "caption": "A man in a gray suit sits in a modern office, holding a white cup, looking forward. Static camera, warm interior lighting.",
        "anchor_seqs": [1038],       # Subject A only (anchor_1 = the man)
        "anchor_indices": [1],       # Use anchor index 1 from seq_01038
        "use_local": False,          # No local context
        "use_prev_prev": False,
    },
    {
        "name": "Shot 2: CONTINUITY",
        "subtitle": "[Continuity] Spatial consistency — Camera zoom-in on Subject A",
        "caption": "A man in a gray suit holds a cup in a modern office. The camera slowly zooms in on his face. He takes a sip from the cup.",
        "anchor_seqs": [1038],       # Subject A
        "anchor_indices": [1],
        "use_local": True,           # Uses Shot 1's last frame
        "use_prev_prev": False,
    },
    {
        "name": "Shot 3: TRANSITION",
        "subtitle": "[Transition] Context switch — Subject B in bright meadow (forget office)",
        "caption": "A young woman with blonde hair in a white dress stands in a bright green meadow. The camera is static. She smiles and looks around at the scenery.",
        "anchor_seqs": [337],        # Subject B only
        "anchor_indices": [0],
        "use_local": True,           # Uses Shot 2's last frame (must forget office)
        "use_prev_prev": False,
    },
    {
        "name": "Shot 4: MULTI-ID",
        "subtitle": "[Multi-ID] Both subjects in Setting 1 — A & B together in office",
        "caption": "A man in a gray suit and a young blonde woman in a white dress sit together in a modern office. Static camera. They are having a conversation.",
        "anchor_seqs": [1038, 337],  # Both Subject A & B
        "anchor_indices": [1, 0],
        "use_local": True,           # Uses Shot 3's last frame
        "use_prev_prev": True,       # Also uses Shot 2's last frame (t-2 retrieval)
    },
    {
        "name": "Shot 5: FILTERING",
        "subtitle": "[Filtering] Selective isolation — Close-up of Subject A only",
        "caption": "A close-up of a man in a gray suit in a modern office. He looks directly at the camera with a serious expression. The woman is no longer visible.",
        "anchor_seqs": [1038],       # Only Subject A (filter out B)
        "anchor_indices": [1],
        "use_local": True,           # Uses Shot 4's last frame (has both A & B)
        "use_prev_prev": False,
    },
]


def load_anchor_from_dataset(ds, seq_id, anchor_idx):
    """Load a specific anchor from a dataset sequence, properly CLIP-preprocessed."""
    # Find the seq_dir
    seq_dir = None
    for sd in ds.seq_dirs:
        if sd.name == f"seq_{seq_id:05d}":
            seq_dir = sd
            break
    if seq_dir is None:
        raise ValueError(f"seq_{seq_id:05d} not found")

    anchor_path = seq_dir / f"global_anchor_{anchor_idx}.png"
    if not anchor_path.exists():
        raise FileNotFoundError(f"Anchor not found: {anchor_path}")

    # Use dataset's preprocessing (CLIP resize + normalize)
    anchor_rgba = ds._load_anchor(str(anchor_path))
    anchor_rgb = ds._anchor_to_clip_rgb(anchor_rgba)  # (3, 224, 224) CLIP-normalized
    return anchor_rgb


def add_subtitle(frame_np, text, font_size=18):
    """Add subtitle text at the bottom of a frame (H, W, 3) numpy array."""
    img = Image.fromarray(frame_np)
    draw = ImageDraw.Draw(img)

    # Try to use a nice font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()

    H, W = frame_np.shape[:2]

    # Wrap text if too long
    max_chars = W // (font_size // 2)
    lines = []
    words = text.split()
    current_line = ""
    for word in words:
        test = current_line + " " + word if current_line else word
        if len(test) <= max_chars:
            current_line = test
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    # Draw semi-transparent background
    line_height = font_size + 4
    total_height = len(lines) * line_height + 8
    y_start = H - total_height - 4

    # Draw black rectangle background
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [(0, y_start), (W, H)],
        fill=(0, 0, 0, 180)
    )
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw text
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        x = (W - tw) // 2
        y = y_start + 4 + i * line_height
        # White text with black outline
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                draw.text((x + dx, y + dy), line, fill=(0, 0, 0), font=font)
        draw.text((x, y), line, fill=(255, 255, 255), font=font)

    return np.array(img)


def main():
    parser = argparse.ArgumentParser(description="DIRECTOR 5-Shot Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--guidance-scale", type=float, default=None,
                        help="Legacy single-CFG scale (ignored when Multi-CFG omegas are set)")
    parser.add_argument("--omega-text", type=float, default=None)
    parser.add_argument("--omega-local", type=float, default=None)
    parser.add_argument("--omega-global", type=float, default=None)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="samples/5shot_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda:0")
    train_h, train_w = args.height, args.width
    train_frames = config["training"].get("train_frames", 13)

    # Multi-CFG guidance scales: CLI args > config > defaults
    inf_guidance = config.get("inference", {}).get("guidance", {})
    omega_text = args.omega_text if args.omega_text is not None else inf_guidance.get("omega_text", 6.0)
    omega_local = args.omega_local if args.omega_local is not None else inf_guidance.get("omega_local", 2.0)
    omega_global = args.omega_global if args.omega_global is not None else inf_guidance.get("omega_global", 3.0)
    print(f"Multi-CFG guidance: omega_text={omega_text}, omega_local={omega_local}, omega_global={omega_global}")

    # Build model
    from training.trainer import _build_pipeline_and_config
    director_config, PipelineClass = _build_pipeline_and_config(config)

    print("Initializing DIRECTOR pipeline...")
    pipeline = PipelineClass(config=director_config, device=device)
    transformer = pipeline.director_transformer

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device)
    transformer.adapters.load_state_dict(state["adapters"])
    transformer.context_builder.load_state_dict(state["context_builder"])
    if "global_encoder" in state:
        transformer.global_encoder.load_state_dict(state["global_encoder"], strict=False)
    if "local_encoder" in state and transformer.local_encoder is not None:
        transformer.local_encoder.load_state_dict(state["local_encoder"], strict=False)
    if "lora" in state and transformer.lora_enabled:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(transformer.backbone, state["lora"])
        print(f"Loaded LoRA params: {len(state['lora'])} keys")
    elif "backbone" in state:
        transformer.backbone.load_state_dict(state["backbone"], strict=False)
    step = state.get("global_step", "?")
    print(f"Checkpoint loaded (step={step})")

    # Enable tiled VAE decoding for high-resolution (480x720) without OOM
    pipeline.vae.enable_tiling()
    pipeline.vae.enable_slicing()
    print("VAE tiling & slicing enabled for high-res decoding")

    # Load dataset for anchor preprocessing
    from data.dataset import DirectorDataset
    dataset_dir = config["dataset"]["dataset_dir"]
    ds = DirectorDataset(
        dataset_dir=dataset_dir,
        target_height=train_h,
        target_width=train_w,
        target_frames=train_frames,
        augment=False, split="train", split_ratio=1.0, seed=42,
    )

    # Setup
    latent_h, latent_w = train_h // 8, train_w // 8
    latent_t = (train_frames - 1) // 4 + 1
    latent_c = pipeline.vae.config.latent_channels
    diffusion = pipeline.diffusion
    amp_dtype = torch.bfloat16
    max_characters = 4

    transformer.eval()

    # Pre-encode null text
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        null_text_embeds = pipeline.encode_text("")

    # Generate 5 shots
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shot_videos = []       # list of (T, 3, H, W) CPU tensors
    shot_last_frames = []  # list of (1, 3, H, W) GPU tensors - last frame of each shot
    start_total = time.time()

    for shot_idx, shot_def in enumerate(SHOT_DEFS):
        print(f"\n{'='*60}")
        print(f"  {shot_def['name']}")
        print(f"  Caption: {shot_def['caption'][:80]}...")
        print(f"{'='*60}")
        shot_start = time.time()

        # --- Prepare anchors ---
        anchor_list = []
        for seq_id, anc_idx in zip(shot_def["anchor_seqs"], shot_def["anchor_indices"]):
            anc = load_anchor_from_dataset(ds, seq_id, anc_idx)
            anchor_list.append(anc)
        # Pad to max_characters
        while len(anchor_list) < max_characters:
            anchor_list.append(torch.zeros(3, 224, 224))

        anchor_rgb = torch.stack(anchor_list).unsqueeze(0).to(device, dtype=torch.float32)  # (1, K, 3, 224, 224)
        char_mask = torch.zeros(1, max_characters, device=device, dtype=torch.float32)
        char_mask[0, :len(shot_def["anchor_seqs"])] = 1.0
        char_list = [anchor_rgb[:, k] for k in range(max_characters)]

        # --- Prepare local context (prev frames) ---
        prev_frames_ar = []
        if shot_def["use_local"] and len(shot_last_frames) > 0:
            prev_frames_ar.append(shot_last_frames[-1])  # t-1
            if shot_def["use_prev_prev"] and len(shot_last_frames) >= 2:
                prev_frames_ar.append(shot_last_frames[-2])  # t-2

        # --- Encode text ---
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            text_embeds = pipeline.encode_text(shot_def["caption"])

        # --- Encode context variants for Multi-CFG ---
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            # Full context (local + global)
            if len(prev_frames_ar) > 0:
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
            else:
                # No local context - only global anchors
                ctx_full, mask_full = transformer.encode_context(
                    prev_frames=None,
                    character_images=char_list,
                    character_masks=char_mask,
                )
                # No local, no global = null
                ctx_local, mask_local = None, None

            # Initial noise
            gen = torch.Generator(device=device)
            gen.manual_seed(args.seed + shot_idx)
            x = torch.randn(
                1, latent_t, latent_c, latent_h, latent_w,
                device=device, dtype=torch.bfloat16, generator=gen,
            )

            # ODE solve with Multi-CFG
            timesteps = diffusion.prepare_inference(args.num_steps, device)
            state_ode = None
            for i, t in enumerate(timesteps):
                t_tensor = t.expand(1)

                fwd = lambda te, ctx, msk: transformer(
                    hidden_states=x, encoder_hidden_states=te, timestep=t_tensor,
                    unified_context=ctx, context_mask=msk, return_dict=False,
                )[0]

                # 4-pass Multi-CFG
                v_null  = fwd(null_text_embeds, None, None)             # unconditional
                v_text  = fwd(text_embeds, None, None)                  # text-only
                v_local = fwd(text_embeds, ctx_local, mask_local)       # text + local
                v_full  = fwd(text_embeds, ctx_full, mask_full)         # text + local + global

                v_out = (v_null
                         + omega_text   * (v_text  - v_null)
                         + omega_local  * (v_local - v_text)
                         + omega_global * (v_full  - v_local))
                v_out = v_out.float()
                step_out = diffusion.inference_step(v_out, x.float(), t, i, timesteps, state=state_ode)
                x = step_out.latents.to(torch.bfloat16)
                state_ode = step_out.state

            latent_cpu = x.cpu()
            del x, ctx_full, mask_full, ctx_local, mask_local, text_embeds
            torch.cuda.empty_cache()

        print(f"  ODE done in {time.time() - shot_start:.1f}s")

        # Decode video — offload everything except VAE
        transformer.cpu()
        pipeline.text_encoder.cpu()
        torch.cuda.empty_cache()
        pipeline.vae.to(device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            video = pipeline.decode_latent(latent_cpu.to(device))  # (1, T, 3, H, W)

        shot_video = video[0].cpu().float().clamp(0, 1)  # (T, 3, H, W)
        last_frame = video[:, -1].clone()  # (1, 3, H, W) on GPU

        del video, latent_cpu
        torch.cuda.empty_cache()

        pipeline.vae.cpu()
        torch.cuda.empty_cache()
        transformer.to(device)
        pipeline.text_encoder.to(device)

        shot_videos.append(shot_video)
        shot_last_frames.append(last_frame)
        print(f"  Shot {shot_idx+1} complete: {shot_video.shape[0]} frames")

    del null_text_embeds
    torch.cuda.empty_cache()

    # ── Save outputs ────────────────────────────────────────────────────────

    print("\nSaving outputs...")

    # 1. Individual shot MP4s with subtitles
    for i, (sv, shot_def) in enumerate(zip(shot_videos, SHOT_DEFS)):
        mp4_path = output_dir / f"shot{i+1}_{shot_def['name'].split(':')[0].strip().lower().replace(' ', '')}.mp4"
        video_np = (sv.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()
        T_out, H_out, W_out, _ = video_np.shape

        # Add subtitles
        subtitle = shot_def["subtitle"]
        subtitled_frames = []
        for t_idx in range(T_out):
            frame = add_subtitle(video_np[t_idx], subtitle)
            subtitled_frames.append(frame)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_cv = cv2.VideoWriter(str(mp4_path), fourcc, 8, (W_out, H_out))
        for frame in subtitled_frames:
            writer_cv.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer_cv.release()
        print(f"  Saved {mp4_path.name} ({T_out} frames)")

    # 2. Full concatenated MP4 with subtitles
    mp4_path = output_dir / "5shot_full.mp4"
    all_frames = []
    for sv, shot_def in zip(shot_videos, SHOT_DEFS):
        video_np = (sv.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()
        subtitle = shot_def["subtitle"]
        for t_idx in range(video_np.shape[0]):
            frame = add_subtitle(video_np[t_idx], subtitle)
            all_frames.append(frame)

    H_out, W_out = all_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_cv = cv2.VideoWriter(str(mp4_path), fourcc, 8, (W_out, H_out))
    for frame in all_frames:
        writer_cv.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer_cv.release()
    print(f"  Saved 5shot_full.mp4 ({len(all_frames)} frames)")

    # 3. Key frames grid (first/mid/last per shot)
    key_frames = []
    for sv in shot_videos:
        T = sv.shape[0]
        key_frames.extend([sv[0], sv[T // 2], sv[-1]])
    kf_grid = make_grid(key_frames, nrow=3, padding=4, normalize=False)
    kf_img = (kf_grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(kf_img).save(output_dir / "keyframes_5shot.jpg")
    print("  Saved keyframes_5shot.jpg")

    # 4. Transition pairs grid
    transition_frames = []
    for i in range(len(shot_videos) - 1):
        transition_frames.append(shot_videos[i][-1])
        transition_frames.append(shot_videos[i + 1][0])
    tr_grid = make_grid(transition_frames, nrow=2, padding=4, normalize=False)
    tr_img = (tr_grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(tr_img).save(output_dir / "transitions_5shot.jpg")
    print("  Saved transitions_5shot.jpg")

    # 5. Individual frames
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    frame_idx = 0
    for i, (sv, shot_def) in enumerate(zip(shot_videos, SHOT_DEFS)):
        video_np = (sv.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()
        subtitle = shot_def["subtitle"]
        for t in range(video_np.shape[0]):
            frame = add_subtitle(video_np[t], subtitle)
            Image.fromarray(frame).save(frames_dir / f"frame_{frame_idx:04d}_shot{i+1}.jpg")
            frame_idx += 1

    elapsed = time.time() - start_total
    print(f"\nDone! Total time: {elapsed:.1f}s")
    print(f"Output directory: {output_dir}")
    print(f"\nShot summary:")
    for i, sd in enumerate(SHOT_DEFS):
        print(f"  {sd['name']}: {sd['subtitle']}")


if __name__ == "__main__":
    main()
