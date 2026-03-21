"""
DIRECTOR: Multi-shot inference script.
Generates autoregressive multi-shot videos from a checkpoint and specific sequences.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n paper_env python inference_multishot.py \
        --config configs/full_training_lora.yaml \
        --checkpoint checkpoints_full_lora/checkpoint_best.pt \
        --seq-ids 336 337 338 \
        --num-shots 3 \
        --guidance-scale 6.0 \
        --num-steps 50 \
        --output-dir samples/inference_multishot
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.utils import make_grid

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description="DIRECTOR Multi-shot Inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seq-ids", type=int, nargs="+", required=True,
                        help="Sequence IDs to use as starting point (first one used for init)")
    parser.add_argument("--num-shots", type=int, default=3)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="samples/inference_multishot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda:0")
    train_h = args.height
    train_w = args.width
    train_frames = config["training"].get("train_frames", 13)

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
    print(f"Checkpoint loaded (step={state.get('global_step', '?')})")

    # Load sequence data using DirectorDataset (handles CLIP preprocessing correctly)
    from data.dataset import DirectorDataset
    dataset_dir = config["dataset"]["dataset_dir"]

    # Create a dataset just to use its loading/preprocessing logic
    ds = DirectorDataset(
        dataset_dir=dataset_dir,
        target_height=train_h,
        target_width=train_w,
        target_frames=train_frames,
        augment=False,
        split="train",
        split_ratio=1.0,  # Include all sequences
        seed=42,
    )

    # Find the internal index for the requested seq_id
    first_seq_id = args.seq_ids[0]
    seq_idx = None
    for i, sd in enumerate(ds.seq_dirs):
        if sd.name == f"seq_{first_seq_id:05d}":
            # Find position in ds.indices
            for j, idx in enumerate(ds.indices):
                if idx == i:
                    seq_idx = j
                    break
            if seq_idx is None:
                # Force direct access
                seq_idx = i
                ds.indices = np.array([i])
                seq_idx = 0
            break

    if seq_idx is None:
        raise ValueError(f"Sequence seq_{first_seq_id:05d} not found in dataset")

    batch = ds[seq_idx]
    # Add batch dimension
    prev_frame = batch["prev_frame"].unsqueeze(0).to(device, dtype=torch.float32)
    prev_prev_frame = batch["prev_prev_frame"].unsqueeze(0).to(device, dtype=torch.float32)
    has_prev_prev = batch["has_prev_prev"]
    anchor_rgb = batch["anchor_rgb"].unsqueeze(0).to(device, dtype=torch.float32)
    char_mask = batch["character_mask"].unsqueeze(0).to(device, dtype=torch.float32)
    caption = batch["captions"]

    B, K = anchor_rgb.shape[:2]
    char_list = [anchor_rgb[:, k] for k in range(K)]

    print(f"Starting sequence: seq_{first_seq_id:05d}")
    print(f"Caption: {caption[:100]}...")
    print(f"Resolution: {train_h}x{train_w}, frames: {train_frames}")
    print(f"Generating {args.num_shots} shots, {args.num_steps} ODE steps, guidance={args.guidance_scale}")

    # Setup
    latent_h, latent_w = train_h // 8, train_w // 8
    latent_t = (train_frames - 1) // 4 + 1
    latent_c = pipeline.vae.config.latent_channels
    diffusion = pipeline.diffusion
    amp_dtype = torch.bfloat16

    transformer.eval()

    # Encode text
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        text_embeds = pipeline.encode_text(caption)
        null_text_embeds = pipeline.encode_text("")

    # Initialize prev_frames from first sequence
    prev_frames_ar = [prev_frame]
    if has_prev_prev:
        prev_frames_ar.append(prev_prev_frame)

    # Generate shots autoregressively
    shot_latents_cpu = []
    start_t = time.time()

    for shot_idx in range(args.num_shots):
        print(f"\n  Shot {shot_idx+1}/{args.num_shots}...")
        shot_start = time.time()

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            # Encode context
            ctx_cond, mask_cond = transformer.encode_context(
                prev_frames=prev_frames_ar,
                character_images=char_list,
                character_masks=char_mask,
            )

            # Initial noise
            gen = torch.Generator(device=device)
            gen.manual_seed(args.seed + shot_idx)
            x = torch.randn(
                1, latent_t, latent_c, latent_h, latent_w,
                device=device, dtype=torch.bfloat16, generator=gen,
            )

            # ODE solve with simple CFG
            timesteps = diffusion.prepare_inference(args.num_steps, device)
            state_ode = None
            for i, t in enumerate(timesteps):
                t_tensor = t.expand(1)
                v_null = transformer(
                    hidden_states=x, encoder_hidden_states=null_text_embeds, timestep=t_tensor,
                    unified_context=None, context_mask=None, return_dict=False,
                )[0]
                v_cond = transformer(
                    hidden_states=x, encoder_hidden_states=text_embeds, timestep=t_tensor,
                    unified_context=ctx_cond, context_mask=mask_cond, return_dict=False,
                )[0]
                v_out = v_null + args.guidance_scale * (v_cond - v_null)
                v_out = v_out.float()
                step_out = diffusion.inference_step(v_out, x.float(), t, i, timesteps, state=state_ode)
                x = step_out.latents.to(torch.bfloat16)
                state_ode = step_out.state

            shot_latents_cpu.append(x.cpu())
            del x, ctx_cond, mask_cond
            torch.cuda.empty_cache()

        print(f"    ODE done in {time.time() - shot_start:.1f}s")

        # Decode last frame for next shot's prev_frame
        if shot_idx < args.num_shots - 1:
            transformer.cpu()
            torch.cuda.empty_cache()
            pipeline.vae.to(device)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
                video_tmp = pipeline.decode_latent(shot_latents_cpu[-1].to(device))
            last_frame = video_tmp[:, -1]  # (1, 3, H, W)
            del video_tmp
            torch.cuda.empty_cache()

            pipeline.vae.cpu()
            torch.cuda.empty_cache()
            transformer.to(device)

            prev_frames_ar = [last_frame]

    del text_embeds, null_text_embeds
    torch.cuda.empty_cache()

    # Decode all shots to video
    print("\nDecoding all shots to video...")
    transformer.cpu()
    pipeline.text_encoder.cpu()
    torch.cuda.empty_cache()
    pipeline.vae.to(device)

    shot_videos = []
    for i, lat_cpu in enumerate(shot_latents_cpu):
        print(f"  Decoding shot {i+1}/{args.num_shots}...")
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            video = pipeline.decode_latent(lat_cpu.to(device))
        shot_videos.append(video[0].cpu().float().clamp(0, 1))
        del video
        torch.cuda.empty_cache()

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_tag = "_".join(f"{s:05d}" for s in args.seq_ids[:3])

    # 1. Save key frames grid
    key_frames = []
    for sv in shot_videos:
        T = sv.shape[0]
        key_frames.extend([sv[0], sv[T // 2], sv[-1]])
    kf_grid = make_grid(key_frames, nrow=3, padding=4, normalize=False)
    kf_img = (kf_grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(kf_img).save(output_dir / f"keyframes_{seq_tag}.jpg")
    print(f"  Saved key frames: keyframes_{seq_tag}.jpg")

    # 2. Save transition frames
    if args.num_shots >= 2:
        transition_frames = []
        for i in range(args.num_shots - 1):
            transition_frames.append(shot_videos[i][-1])
            transition_frames.append(shot_videos[i + 1][0])
        tr_grid = make_grid(transition_frames, nrow=2, padding=4, normalize=False)
        tr_img = (tr_grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(tr_img).save(output_dir / f"transitions_{seq_tag}.jpg")
        print(f"  Saved transitions: transitions_{seq_tag}.jpg")

    # 3. Save individual shot MP4s
    for i, sv in enumerate(shot_videos):
        mp4_path = output_dir / f"shot{i+1}_{seq_tag}.mp4"
        video_np = (sv.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()
        T_out, H_out, W_out, _ = video_np.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_cv = cv2.VideoWriter(str(mp4_path), fourcc, 8, (W_out, H_out))
        for t_idx in range(T_out):
            writer_cv.write(cv2.cvtColor(video_np[t_idx], cv2.COLOR_RGB2BGR))
        writer_cv.release()
        print(f"  Saved shot {i+1}: {mp4_path} ({T_out} frames)")

    # 4. Save concatenated MP4
    full_video = torch.cat(shot_videos, dim=0)
    mp4_path = output_dir / f"multishot_{seq_tag}.mp4"
    video_np = (full_video.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()
    T_out, H_out, W_out, _ = video_np.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_cv = cv2.VideoWriter(str(mp4_path), fourcc, 8, (W_out, H_out))
    for t_idx in range(T_out):
        writer_cv.write(cv2.cvtColor(video_np[t_idx], cv2.COLOR_RGB2BGR))
    writer_cv.release()
    print(f"  Saved full multi-shot: {mp4_path} ({T_out} frames)")

    # 5. Save individual frames
    frames_dir = output_dir / f"frames_{seq_tag}"
    frames_dir.mkdir(exist_ok=True)
    for t_idx in range(T_out):
        frame = video_np[t_idx]
        Image.fromarray(frame).save(frames_dir / f"frame_{t_idx:04d}.jpg")

    elapsed = time.time() - start_t
    print(f"\nDone! Total time: {elapsed:.1f}s")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
