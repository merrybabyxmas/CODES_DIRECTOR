"""
Multi-CFG ablation with full ODE solving + autoregressive multi-shot video generation.

For each context configuration (text_only, local_only, global_only, full, multi_cfg),
generates a long-sequence video by autoregressively chaining multiple shots.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n paper_env python scripts/run_ablation_video.py \
        --num_shots 3 --num_steps 30 --checkpoint checkpoints/checkpoint_best.pt
"""
import argparse
import os
import sys
import cv2
import json
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from pathlib import Path
from typing import List, Optional, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))

import yaml
from data.dataset import DirectorDataset, DirectorDataCollator
from torch.utils.data import DataLoader
from trainer import _build_pipeline_and_config


def save_video(video: torch.Tensor, path: Path, fps: int = 8):
    """Save (T, 3, H, W) float [0,1] tensor as MP4."""
    video_np = (video.cpu().float().clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()
    T, H, W, C = video_np.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (W, H))
    for t in range(T):
        frame_bgr = cv2.cvtColor(video_np[t], cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    writer.release()
    print(f"  Saved: {path} ({T} frames, {W}x{H}, {fps}fps)")


def save_side_by_side_video(
    videos: Dict[str, torch.Tensor], path: Path, fps: int = 8, labels: bool = True
):
    """
    Save multiple videos side-by-side as a single comparison MP4.
    videos: {name: (T, 3, H, W) tensor}
    All videos must have the same T.
    """
    names = list(videos.keys())
    tensors = [videos[n] for n in names]

    # Resize all to same height
    T = tensors[0].shape[0]
    H = min(t.shape[2] for t in tensors)
    W = min(t.shape[3] for t in tensors)

    resized = []
    for t in tensors:
        r = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
        resized.append(r.clamp(0, 1))

    # Concatenate horizontally: (T, 3, H, W*N)
    combined = torch.cat(resized, dim=3)

    # Add labels on top if requested
    if labels:
        video_np = (combined.cpu().float().clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()
        for t_idx in range(video_np.shape[0]):
            frame = video_np[t_idx].copy()
            for i, name in enumerate(names):
                x_pos = i * W + 5
                cv2.putText(frame, name, (x_pos, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, name, (x_pos, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, name, (x_pos, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            video_np[t_idx] = frame
        # Write with cv2
        T_out, H_out, W_out, _ = video_np.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (W_out, H_out))
        for t_idx in range(T_out):
            writer.write(cv2.cvtColor(video_np[t_idx], cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"  Saved comparison: {path} ({T_out} frames, {W_out}x{H_out}, {fps}fps)")
    else:
        save_video(combined, path, fps)


def generate_shot_single_cfg(
    pipeline, transformer, prompt: str,
    prev_frames: Optional[List[torch.Tensor]],
    character_images: Optional[List[torch.Tensor]],
    character_masks: Optional[torch.Tensor],
    num_steps: int = 30,
    height: int = 480, width: int = 720, num_frames: int = 49,
    seed: int = 42,
) -> torch.Tensor:
    """
    Generate a single shot using ONLY the provided context (no Multi-CFG decomposition).
    Uses simple classifier-free guidance: v_out = v_null + scale * (v_cond - v_null).
    Returns: (1, T, 3, H, W) tensor on CPU.
    """
    device = pipeline.device
    transformer.eval()

    with torch.no_grad():
        text_embeds = pipeline.encode_text(prompt)
        null_text_embeds = pipeline.encode_text("")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ctx_cond, mask_cond = transformer.encode_context(
                prev_frames=prev_frames,
                character_images=character_images,
                character_masks=character_masks,
            )
            # Null context: pass None to skip adapters (avoids NaN from all-zero mask)
            ctx_null, mask_null = None, None

        latent_h, latent_w = height // 8, width // 8
        latent_t = (num_frames - 1) // 4 + 1
        latent_c = pipeline.vae.config.latent_channels

        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        x = torch.randn(
            1, latent_t, latent_c, latent_h, latent_w,
            device=device, dtype=torch.bfloat16, generator=gen,
        )

        guidance_scale = 6.0

        # Backbone-agnostic inference loop
        timesteps = pipeline.diffusion.prepare_inference(num_steps, device)

        state = None
        for i, t in enumerate(timesteps):
            t_tensor = t.expand(1)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                v_null = transformer(
                    hidden_states=x, encoder_hidden_states=null_text_embeds, timestep=t_tensor,
                    unified_context=None, context_mask=None, return_dict=False,
                )[0]
                v_cond = transformer(
                    hidden_states=x, encoder_hidden_states=text_embeds, timestep=t_tensor,
                    unified_context=ctx_cond, context_mask=mask_cond, return_dict=False,
                )[0]

            v_out = v_null + guidance_scale * (v_cond - v_null)

            # Cast to float32 for numerical precision in scheduler step
            v_out = v_out.float()
            step_out = pipeline.diffusion.inference_step(
                v_out, x.float(), t, i, timesteps, state=state,
            )
            x = step_out.latents.to(torch.bfloat16)
            state = step_out.state

        del text_embeds, null_text_embeds, ctx_cond, mask_cond
        torch.cuda.empty_cache()
        return x  # Return latent, decode later


def generate_shot_multi_cfg(
    pipeline, transformer, prompt: str,
    prev_frames: Optional[List[torch.Tensor]],
    character_images: Optional[List[torch.Tensor]],
    character_masks: Optional[torch.Tensor],
    omega_text: float = 6.0, omega_local: float = 2.0, omega_global: float = 3.0,
    num_steps: int = 30,
    height: int = 480, width: int = 720, num_frames: int = 49,
    seed: int = 42,
) -> torch.Tensor:
    """
    Generate a single shot with full Multi-CFG guidance (DIRECTOR inference).
    Returns latent on GPU.
    """
    device = pipeline.device
    transformer.eval()

    with torch.no_grad():
        text_embeds = pipeline.encode_text(prompt)
        null_text_embeds = pipeline.encode_text("")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ctx_local, mask_local = transformer.encode_context(
                prev_frames=prev_frames, character_images=None,
            )
            ctx_global, mask_global = transformer.encode_context(
                prev_frames=None, character_images=character_images,
                character_masks=character_masks,
            )
            # Null context: pass None to skip adapters (avoids NaN from all-zero mask)

        latent_h, latent_w = height // 8, width // 8
        latent_t = (num_frames - 1) // 4 + 1
        latent_c = pipeline.vae.config.latent_channels

        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        x = torch.randn(
            1, latent_t, latent_c, latent_h, latent_w,
            device=device, dtype=torch.bfloat16, generator=gen,
        )

        # Backbone-agnostic inference loop
        timesteps = pipeline.diffusion.prepare_inference(num_steps, device)

        state = None
        for i, t in enumerate(timesteps):
            t_tensor = t.expand(1)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                fwd = lambda te, ctx, msk: transformer(
                    hidden_states=x, encoder_hidden_states=te, timestep=t_tensor,
                    unified_context=ctx, context_mask=msk, return_dict=False,
                )[0]

                v_null = fwd(null_text_embeds, None, None)
                v_text = fwd(text_embeds, None, None)
                v_local = fwd(text_embeds, ctx_local, mask_local)
                v_glob = fwd(text_embeds, ctx_global, mask_global)

            v_out = (v_null
                     + omega_text   * (v_text  - v_null)
                     + omega_local  * (v_local - v_text)
                     + omega_global * (v_glob  - v_text))

            # Cast to float32 for numerical precision in scheduler step
            v_out = v_out.float()
            step_out = pipeline.diffusion.inference_step(
                v_out, x.float(), t, i, timesteps, state=state,
            )
            x = step_out.latents.to(torch.bfloat16)
            state = step_out.state

        del text_embeds, null_text_embeds
        del ctx_local, mask_local, ctx_global, mask_global
        torch.cuda.empty_cache()
        return x


def decode_and_offload(pipeline, latent: torch.Tensor, device) -> torch.Tensor:
    """Decode latent to video on CPU. Assumes VAE is on device."""
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        latent_gpu = latent.to(device) if latent.device != device else latent
        video = pipeline.decode_latent(latent_gpu)  # (1, T, 3, H, W)
    result = video[0].cpu().float().clamp(0, 1)  # (T, 3, H, W)
    del video, latent_gpu
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="Multi-CFG Ablation: Long Sequence Video Generation")
    parser.add_argument("--num_shots", type=int, default=3, help="Number of autoregressive shots")
    parser.add_argument("--num_steps", type=int, default=30, help="ODE solver steps (30=fast, 50=quality)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_best.pt")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda:0")

    # Load config & pipeline
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    director_config, PipelineClass = _build_pipeline_and_config(cfg)

    print("Loading pipeline...")
    pipeline = PipelineClass(director_config, device=device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    transformer = pipeline.director_transformer
    transformer.adapters.load_state_dict(state["adapters"])
    transformer.context_builder.load_state_dict(state["context_builder"])
    transformer.global_encoder.load_state_dict(state["global_encoder"], strict=False)
    if "local_encoder" in state and transformer.local_encoder is not None:
        transformer.local_encoder.load_state_dict(state["local_encoder"], strict=False)
    loaded_step = state.get("global_step", "?")
    print(f"Loaded step {loaded_step}")
    del state
    torch.cuda.empty_cache()
    transformer.eval()

    # Load dataset - pick sample with most characters for interesting ablation
    print("Loading dataset...")
    dataset = DirectorDataset(
        dataset_dir=cfg["dataset"]["dataset_dir"],
        target_frames=cfg["dataset"]["video"]["num_frames"],
        target_height=args.height,
        target_width=args.width,
        augment=False,
        split="val",
    )
    collator = DirectorDataCollator()
    loader = DataLoader(dataset, batch_size=1, collate_fn=collator, shuffle=False)

    # Pick sample with most characters (for meaningful global ablation)
    best_sample = None
    best_score = -1.0
    for i, batch in enumerate(loader):
        if i >= 30:
            break
        n_chars = batch["character_mask"].sum().item()
        brightness = batch["target_video"].mean().item()
        score = n_chars * 10 + brightness  # Prefer more characters
        if score > best_score:
            best_score = score
            best_sample = batch

    batch = best_sample
    seq_id = batch["seq_ids"][0]
    caption = batch["captions"][0]
    anchor_rgb = batch["anchor_rgb"].to(device, dtype=torch.float32)  # (1, K, 3, 224, 224)
    char_mask = batch["character_mask"].to(device, dtype=torch.float32)  # (1, K)
    prev_frame_init = batch["prev_frame"].to(device, dtype=torch.float32)  # (1, 3, H, W)
    K = anchor_rgb.shape[1]
    char_list = [anchor_rgb[:, k] for k in range(K)]
    n_active = int(char_mask[0].sum().item())

    print(f"Selected: {seq_id} ({n_active} characters)")
    print(f"Caption: {caption[:100]}...")
    print(f"Generating {args.num_shots} shots × 5 configs = {args.num_shots * 5} total generations")
    print(f"ODE steps: {args.num_steps}, Resolution: {args.width}x{args.height}, Frames/shot: {args.num_frames}")

    # Use same caption for all shots (in practice each shot would have its own)
    # Slightly vary captions for autoregressive diversity
    prompts = [caption] * args.num_shots

    # 5 context configurations
    cfg_names = ["text_only", "local_only", "global_only", "full", "multi_cfg"]
    all_shot_latents = {name: [] for name in cfg_names}  # Store latents on CPU

    # For each config, generate shots autoregressively
    for cfg_name in cfg_names:
        print(f"\n{'='*60}")
        print(f"Config: {cfg_name} — Generating {args.num_shots} shots")
        print(f"{'='*60}")

        prev_frames_ar: List[torch.Tensor] = [prev_frame_init]  # Start with dataset's prev_frame

        for shot_idx in range(args.num_shots):
            shot_seed = args.seed + shot_idx
            print(f"  Shot {shot_idx+1}/{args.num_shots} (seed={shot_seed})...", end=" ", flush=True)

            if cfg_name == "text_only":
                latent = generate_shot_single_cfg(
                    pipeline, transformer, prompts[shot_idx],
                    prev_frames=None, character_images=None, character_masks=None,
                    num_steps=args.num_steps, height=args.height, width=args.width,
                    num_frames=args.num_frames, seed=shot_seed,
                )
            elif cfg_name == "local_only":
                latent = generate_shot_single_cfg(
                    pipeline, transformer, prompts[shot_idx],
                    prev_frames=prev_frames_ar, character_images=None, character_masks=None,
                    num_steps=args.num_steps, height=args.height, width=args.width,
                    num_frames=args.num_frames, seed=shot_seed,
                )
            elif cfg_name == "global_only":
                latent = generate_shot_single_cfg(
                    pipeline, transformer, prompts[shot_idx],
                    prev_frames=None, character_images=char_list, character_masks=char_mask,
                    num_steps=args.num_steps, height=args.height, width=args.width,
                    num_frames=args.num_frames, seed=shot_seed,
                )
            elif cfg_name == "full":
                latent = generate_shot_single_cfg(
                    pipeline, transformer, prompts[shot_idx],
                    prev_frames=prev_frames_ar, character_images=char_list, character_masks=char_mask,
                    num_steps=args.num_steps, height=args.height, width=args.width,
                    num_frames=args.num_frames, seed=shot_seed,
                )
            elif cfg_name == "multi_cfg":
                latent = generate_shot_multi_cfg(
                    pipeline, transformer, prompts[shot_idx],
                    prev_frames=prev_frames_ar, character_images=char_list, character_masks=char_mask,
                    omega_text=cfg["inference"]["guidance"]["omega_text"],
                    omega_local=cfg["inference"]["guidance"]["omega_local"],
                    omega_global=cfg["inference"]["guidance"]["omega_global"],
                    num_steps=args.num_steps, height=args.height, width=args.width,
                    num_frames=args.num_frames, seed=shot_seed,
                )

            # Cache latent on CPU
            all_shot_latents[cfg_name].append(latent.cpu())
            del latent
            torch.cuda.empty_cache()

            # For autoregressive: decode last frame for next shot's local context
            if shot_idx < args.num_shots - 1 and cfg_name in ["local_only", "full", "multi_cfg"]:
                print("(decoding last frame for AR)...", end=" ", flush=True)
                # Offload transformer to CPU, decode, then bring transformer back
                transformer.cpu()
                torch.cuda.empty_cache()
                pipeline.vae.to(device)

                last_latent = all_shot_latents[cfg_name][-1].to(device)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    video_tmp = pipeline.decode_latent(last_latent)
                last_frame = video_tmp[:, -1].cpu()  # (1, 3, H, W)
                del video_tmp, last_latent
                torch.cuda.empty_cache()

                # Bring transformer back for next shot
                pipeline.vae.cpu()
                torch.cuda.empty_cache()
                transformer.to(device)

                prev_frames_ar.insert(0, last_frame.to(device))
                prev_frames_ar = prev_frames_ar[:2]  # Keep max 2 frames (t-1, t-2)

            print("done")

    # =========================================================================
    # Decode all latents → videos
    # Offload transformer to CPU to free memory for VAE
    # =========================================================================
    print("\nOffloading transformer to CPU for VAE decode...")
    transformer.cpu()
    torch.cuda.empty_cache()
    pipeline.vae.to(device)

    out_dir = Path(f"samples/ablation_video_step{loaded_step}")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_videos = {}  # {cfg_name: (T_total, 3, H, W)}

    for cfg_name in cfg_names:
        print(f"\nDecoding {cfg_name}...")
        shot_videos = []
        for shot_idx, latent_cpu in enumerate(all_shot_latents[cfg_name]):
            print(f"  Shot {shot_idx+1}...", end=" ", flush=True)
            video = decode_and_offload(pipeline, latent_cpu, device)
            shot_videos.append(video)
            print("done")

        # Concatenate all shots: (T_total, 3, H, W)
        full_video = torch.cat(shot_videos, dim=0)
        all_videos[cfg_name] = full_video

        # Save individual config video
        save_video(full_video, out_dir / f"{cfg_name}.mp4", fps=args.fps)

        # Save first frame of each shot as reference grid
        shot_first_frames = [v[0] for v in shot_videos]
        if len(shot_first_frames) > 1:
            grid = make_grid(shot_first_frames, nrow=len(shot_first_frames), padding=4)
            save_image(grid, out_dir / f"{cfg_name}_shot_firsts.png")

        del shot_videos
        torch.cuda.empty_cache()

    del all_shot_latents

    # Save side-by-side comparison video
    # Truncate to shortest length
    min_T = min(v.shape[0] for v in all_videos.values())
    trimmed = {k: v[:min_T] for k, v in all_videos.items()}
    save_side_by_side_video(trimmed, out_dir / "comparison_all.mp4", fps=args.fps)

    # Save middle-frame comparison grid (from first shot)
    T_shot = args.num_frames
    mid_idx = T_shot // 2
    mid_frames = []
    for cfg_name in cfg_names:
        if mid_idx < all_videos[cfg_name].shape[0]:
            mid_frames.append(all_videos[cfg_name][mid_idx])
    if mid_frames:
        grid = make_grid(mid_frames, nrow=len(mid_frames), padding=4)
        save_image(grid, out_dir / "comparison_midframe.png")

    # Save context reference image
    ref_images = [prev_frame_init[0].cpu().float().clamp(0, 1)]
    for k in range(K):
        if char_mask[0, k] > 0:
            ref_images.append(anchor_rgb[0, k].cpu().float().clamp(0, 1))
    if len(ref_images) > 1:
        ref_h, ref_w = ref_images[0].shape[1], ref_images[0].shape[2]
        ref_resized = [
            F.interpolate(img.unsqueeze(0), size=(ref_h, ref_w),
                         mode="bilinear", align_corners=False)[0].clamp(0, 1)
            for img in ref_images
        ]
        ref_grid = make_grid(ref_resized, nrow=len(ref_resized), padding=2)
        save_image(ref_grid, out_dir / "context_ref.png")

    # Save metadata
    meta = {
        "seq_id": seq_id,
        "caption": caption,
        "num_shots": args.num_shots,
        "num_steps": args.num_steps,
        "checkpoint": args.checkpoint,
        "loaded_step": loaded_step,
        "resolution": f"{args.width}x{args.height}",
        "num_frames_per_shot": args.num_frames,
        "seed": args.seed,
        "n_active_characters": n_active,
        "guidance": {
            "omega_text": cfg["inference"]["guidance"]["omega_text"],
            "omega_local": cfg["inference"]["guidance"]["omega_local"],
            "omega_global": cfg["inference"]["guidance"]["omega_global"],
        },
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"All results saved to {out_dir}/")
    print(f"{'='*60}")
    print(f"Videos ({args.num_shots} shots each):")
    for name in cfg_names:
        print(f"  {name}.mp4")
    print(f"  comparison_all.mp4 (side-by-side)")
    print(f"Images:")
    print(f"  comparison_midframe.png, context_ref.png")
    print(f"  *_shot_firsts.png (per-config)")


if __name__ == "__main__":
    main()
