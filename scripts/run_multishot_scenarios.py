"""
Comprehensive multi-shot scenario sampling for DIRECTOR.

Scenarios:
  1. Zoom-in: Same character, camera progressively zooms in across shots
  2. Entity addition: Start with 1 character, add more in subsequent shots
  3. Baseline: Standard autoregressive multi-shot (same prompt)

For each scenario, generates with "full" and "multi_cfg" configs.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n paper_env python scripts/run_multishot_scenarios.py \
        --checkpoint checkpoints/checkpoint_step_52000.pt \
        --num_steps 20 --height 320 --width 512 --num_frames 49 --seed 42
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
from typing import List, Optional, Dict
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))

import yaml
from data.dataset import DirectorDataset, DirectorDataCollator
from torch.utils.data import DataLoader
from trainer import _build_pipeline_and_config


def save_video(video: torch.Tensor, path: Path, fps: int = 8):
    video_np = (video.cpu().float().clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()
    T, H, W, C = video_np.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (W, H))
    for t in range(T):
        writer.write(cv2.cvtColor(video_np[t], cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  Saved: {path} ({T} frames, {W}x{H})")


def save_side_by_side_video(videos: Dict[str, torch.Tensor], path: Path, fps: int = 8):
    names = list(videos.keys())
    tensors = [videos[n] for n in names]
    T = min(t.shape[0] for t in tensors)
    H = min(t.shape[2] for t in tensors)
    W = min(t.shape[3] for t in tensors)

    resized = []
    for t in tensors:
        r = F.interpolate(t[:T], size=(H, W), mode="bilinear", align_corners=False).clamp(0, 1)
        resized.append(r)

    combined = torch.cat(resized, dim=3)
    video_np = (combined.cpu().float() * 255).byte().permute(0, 2, 3, 1).numpy()
    T_out, H_out, W_out, _ = video_np.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (W_out, H_out))
    for t_idx in range(T_out):
        frame = video_np[t_idx].copy()
        for i, name in enumerate(names):
            x_pos = i * W + 5
            cv2.putText(frame, name, (x_pos, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  Saved comparison: {path}")


def generate_shot_full(
    pipeline, transformer, prompt, prev_frames, char_images, char_mask,
    num_steps, height, width, num_frames, seed, device,
):
    """Generate with full context (local + global), simple CFG."""
    transformer.eval()
    with torch.no_grad():
        text_embeds = pipeline.encode_text(prompt)
        null_text_embeds = pipeline.encode_text("")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ctx_cond, mask_cond = transformer.encode_context(
                prev_frames=prev_frames,
                character_images=char_images,
                character_masks=char_mask,
            )
            ctx_null, mask_null = transformer.encode_context(
                prev_frames=None, character_images=None,
            )

        latent_h, latent_w = height // 8, width // 8
        latent_t = (num_frames - 1) // 4 + 1
        latent_c = pipeline.vae.config.latent_channels

        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        x = torch.randn(1, latent_t, latent_c, latent_h, latent_w,
                        device=device, dtype=torch.bfloat16, generator=gen)

        timesteps = pipeline.diffusion.prepare_inference(num_steps, device)
        state = None
        for i, t in enumerate(timesteps):
            t_tensor = t.expand(1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                v_null = transformer(
                    hidden_states=x, encoder_hidden_states=null_text_embeds,
                    timestep=t_tensor, unified_context=ctx_null,
                    context_mask=mask_null, return_dict=False,
                )[0]
                v_cond = transformer(
                    hidden_states=x, encoder_hidden_states=text_embeds,
                    timestep=t_tensor, unified_context=ctx_cond,
                    context_mask=mask_cond, return_dict=False,
                )[0]
            v_out = v_null + 6.0 * (v_cond - v_null)
            step_out = pipeline.diffusion.inference_step(v_out, x, t, i, timesteps, state=state)
            x = step_out.latents
            state = step_out.state

        del text_embeds, null_text_embeds, ctx_cond, mask_cond, ctx_null, mask_null
        torch.cuda.empty_cache()
        return x


def generate_shot_multi_cfg(
    pipeline, transformer, prompt, prev_frames, char_images, char_mask,
    omega_text, omega_local, omega_global,
    num_steps, height, width, num_frames, seed, device,
):
    """Generate with Multi-CFG guidance."""
    transformer.eval()
    with torch.no_grad():
        text_embeds = pipeline.encode_text(prompt)
        null_text_embeds = pipeline.encode_text("")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ctx_local, mask_local = transformer.encode_context(
                prev_frames=prev_frames, character_images=None,
            )
            ctx_global, mask_global = transformer.encode_context(
                prev_frames=None, character_images=char_images,
                character_masks=char_mask,
            )
            ctx_null, mask_null = transformer.encode_context(
                prev_frames=None, character_images=None,
            )

        latent_h, latent_w = height // 8, width // 8
        latent_t = (num_frames - 1) // 4 + 1
        latent_c = pipeline.vae.config.latent_channels

        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        x = torch.randn(1, latent_t, latent_c, latent_h, latent_w,
                        device=device, dtype=torch.bfloat16, generator=gen)

        timesteps = pipeline.diffusion.prepare_inference(num_steps, device)
        state = None
        for i, t in enumerate(timesteps):
            t_tensor = t.expand(1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                fwd = lambda te, ctx, msk: transformer(
                    hidden_states=x, encoder_hidden_states=te,
                    timestep=t_tensor, unified_context=ctx,
                    context_mask=msk, return_dict=False,
                )[0]
                v_null = fwd(null_text_embeds, ctx_null, mask_null)
                v_text = fwd(text_embeds, ctx_null, mask_null)
                v_local = fwd(text_embeds, ctx_local, mask_local)
                v_glob = fwd(text_embeds, ctx_global, mask_global)

            v_out = (v_null
                     + omega_text * (v_text - v_null)
                     + omega_local * (v_local - v_text)
                     + omega_global * (v_glob - v_text))

            step_out = pipeline.diffusion.inference_step(v_out, x, t, i, timesteps, state=state)
            x = step_out.latents
            state = step_out.state

        del text_embeds, null_text_embeds
        del ctx_local, mask_local, ctx_global, mask_global, ctx_null, mask_null
        torch.cuda.empty_cache()
        return x


def decode_latent(pipeline, latent, device):
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        latent_gpu = latent.to(device)
        video = pipeline.decode_latent(latent_gpu)
    result = video[0].cpu().float().clamp(0, 1)
    del video, latent_gpu
    torch.cuda.empty_cache()
    return result


def ar_decode_last_frame(pipeline, transformer, latent_cpu, device):
    """Offload transformer, decode last frame, bring transformer back."""
    transformer.cpu()
    torch.cuda.empty_cache()
    pipeline.vae.to(device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        video_tmp = pipeline.decode_latent(latent_cpu.to(device))
    last_frame = video_tmp[:, -1].cpu()
    del video_tmp
    torch.cuda.empty_cache()

    pipeline.vae.cpu()
    torch.cuda.empty_cache()
    transformer.to(device)
    return last_frame


def load_sample(dataset, loader, seq_idx=None, max_chars=False):
    """Load a dataset sample. If max_chars, find one with most characters."""
    if max_chars:
        best_sample, best_score = None, -1
        for i, batch in enumerate(loader):
            if i >= 50:
                break
            n_chars = batch["character_mask"].sum().item()
            if n_chars > best_score:
                best_score = n_chars
                best_sample = batch
        return best_sample
    else:
        for i, batch in enumerate(loader):
            if i == seq_idx:
                return batch
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_step_52000.pt")
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
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

    # Load dataset
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

    # Guidance scales
    omega_text = cfg["inference"]["guidance"]["omega_text"]
    omega_local = cfg["inference"]["guidance"]["omega_local"]
    omega_global = cfg["inference"]["guidance"]["omega_global"]

    out_dir = Path(f"samples/multishot_step{loaded_step}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Find good samples: one with zoom-in caption, one with multiple characters
    # =========================================================================
    print("Scanning dataset for suitable samples...")
    zoom_sample = None
    multi_char_sample = None
    baseline_sample = None
    all_samples = []

    for i, batch in enumerate(loader):
        if i >= 80:
            break
        all_samples.append(batch)
        caption = batch["captions"][0]
        n_chars = int(batch["character_mask"][0].sum().item())

        if zoom_sample is None and "zoom" in caption.lower():
            zoom_sample = batch
            print(f"  Zoom sample: idx={i}, chars={n_chars}, caption={caption[:80]}...")
        if multi_char_sample is None and n_chars >= 3:
            multi_char_sample = batch
            print(f"  Multi-char sample: idx={i}, chars={n_chars}, caption={caption[:80]}...")
        if baseline_sample is None and n_chars >= 2 and "static" in caption.lower():
            baseline_sample = batch
            print(f"  Baseline sample: idx={i}, chars={n_chars}, caption={caption[:80]}...")

    # Fallbacks
    if zoom_sample is None:
        zoom_sample = all_samples[0]
        print(f"  Zoom fallback: idx=0")
    if multi_char_sample is None:
        multi_char_sample = all_samples[4] if len(all_samples) > 4 else all_samples[0]
        print(f"  Multi-char fallback")
    if baseline_sample is None:
        baseline_sample = all_samples[9] if len(all_samples) > 9 else all_samples[0]
        print(f"  Baseline fallback")

    gen_kwargs = dict(
        num_steps=args.num_steps, height=args.height, width=args.width,
        num_frames=args.num_frames, device=device,
    )

    scenarios = {}

    # =========================================================================
    # SCENARIO 1: ZOOM-IN (3 shots, camera progressively closer)
    # =========================================================================
    print(f"\n{'='*70}")
    print("SCENARIO 1: ZOOM-IN (3 shots, progressive camera zoom)")
    print(f"{'='*70}")

    batch = zoom_sample
    caption_data = batch["captions"][0]
    anchor_rgb = batch["anchor_rgb"].to(device, dtype=torch.float32)
    char_mask = batch["character_mask"].to(device, dtype=torch.float32)
    prev_frame = batch["prev_frame"].to(device, dtype=torch.float32)
    K = anchor_rgb.shape[1]
    char_list = [anchor_rgb[:, k] for k in range(K)]

    # Construct zoom-in prompts with progressively closer camera
    base_identity = caption_data.split("}")[0] + "}"
    zoom_prompts = [
        f"{base_identity} {{'camera': 'wide shot', 'person': 'standing'}}",
        f"{base_identity} {{'camera': 'medium shot', 'person': 'standing'}}",
        f"{base_identity} {{'camera': 'close-up', 'person': 'standing'}}",
    ]
    print(f"  Prompts:")
    for p in zoom_prompts:
        print(f"    {p[:90]}...")

    for cfg_name, gen_fn in [("full", generate_shot_full), ("multi_cfg", generate_shot_multi_cfg)]:
        print(f"\n  Config: {cfg_name}")
        shot_latents = []
        prev_frames_ar = [prev_frame]

        for shot_idx in range(3):
            seed = args.seed + shot_idx
            print(f"    Shot {shot_idx+1}/3 (seed={seed})...", end=" ", flush=True)

            if cfg_name == "full":
                latent = gen_fn(
                    pipeline, transformer, zoom_prompts[shot_idx],
                    prev_frames_ar, char_list, char_mask,
                    seed=seed, **gen_kwargs,
                )
            else:
                latent = gen_fn(
                    pipeline, transformer, zoom_prompts[shot_idx],
                    prev_frames_ar, char_list, char_mask,
                    omega_text, omega_local, omega_global,
                    seed=seed, **gen_kwargs,
                )

            shot_latents.append(latent.cpu())
            del latent
            torch.cuda.empty_cache()

            # AR: decode last frame for next shot
            if shot_idx < 2:
                print("(AR decode)...", end=" ", flush=True)
                last_frame = ar_decode_last_frame(
                    pipeline, transformer, shot_latents[-1], device
                )
                prev_frames_ar = [last_frame.to(device)]
            print("done")

        scenarios[f"zoom_{cfg_name}"] = shot_latents

    # =========================================================================
    # SCENARIO 2: ENTITY ADDITION (3 shots, add characters progressively)
    # =========================================================================
    print(f"\n{'='*70}")
    print("SCENARIO 2: ENTITY ADDITION (3 shots, adding characters)")
    print(f"{'='*70}")

    batch = multi_char_sample
    caption_data = batch["captions"][0]
    anchor_rgb = batch["anchor_rgb"].to(device, dtype=torch.float32)
    char_mask_full = batch["character_mask"].to(device, dtype=torch.float32)
    prev_frame = batch["prev_frame"].to(device, dtype=torch.float32)
    K = anchor_rgb.shape[1]
    n_active = int(char_mask_full[0].sum().item())
    char_list_all = [anchor_rgb[:, k] for k in range(K)]

    # Progressive character masks: shot1=1char, shot2=2chars, shot3=all chars
    char_counts = [1, min(2, n_active), n_active]
    entity_prompts = [
        f"{caption_data.split('}')[0]}}}" + " {'camera': 'static', 'person': 'standing alone'}",
        f"{caption_data.split('}')[0]}}}" + " {'camera': 'static', 'person': 'joined by another person'}",
        caption_data,  # Original caption with all characters
    ]
    print(f"  Characters available: {n_active}")
    print(f"  Progressive: {char_counts}")

    for cfg_name, gen_fn in [("full", generate_shot_full), ("multi_cfg", generate_shot_multi_cfg)]:
        print(f"\n  Config: {cfg_name}")
        shot_latents = []
        prev_frames_ar = [prev_frame]

        for shot_idx in range(3):
            seed = args.seed + shot_idx
            n_chars = char_counts[shot_idx]

            # Build progressive character mask
            mask_progressive = torch.zeros_like(char_mask_full)
            mask_progressive[0, :n_chars] = char_mask_full[0, :n_chars]

            print(f"    Shot {shot_idx+1}/3 ({n_chars} chars, seed={seed})...", end=" ", flush=True)

            if cfg_name == "full":
                latent = gen_fn(
                    pipeline, transformer, entity_prompts[shot_idx],
                    prev_frames_ar, char_list_all, mask_progressive,
                    seed=seed, **gen_kwargs,
                )
            else:
                latent = gen_fn(
                    pipeline, transformer, entity_prompts[shot_idx],
                    prev_frames_ar, char_list_all, mask_progressive,
                    omega_text, omega_local, omega_global,
                    seed=seed, **gen_kwargs,
                )

            shot_latents.append(latent.cpu())
            del latent
            torch.cuda.empty_cache()

            if shot_idx < 2:
                print("(AR decode)...", end=" ", flush=True)
                last_frame = ar_decode_last_frame(
                    pipeline, transformer, shot_latents[-1], device
                )
                prev_frames_ar = [last_frame.to(device)]
            print("done")

        scenarios[f"entity_{cfg_name}"] = shot_latents

    # =========================================================================
    # SCENARIO 3: BASELINE (3 shots, same prompt, standard AR)
    # =========================================================================
    print(f"\n{'='*70}")
    print("SCENARIO 3: BASELINE (3 shots, same prompt, standard AR)")
    print(f"{'='*70}")

    batch = baseline_sample
    caption_data = batch["captions"][0]
    anchor_rgb = batch["anchor_rgb"].to(device, dtype=torch.float32)
    char_mask = batch["character_mask"].to(device, dtype=torch.float32)
    prev_frame = batch["prev_frame"].to(device, dtype=torch.float32)
    K = anchor_rgb.shape[1]
    char_list = [anchor_rgb[:, k] for k in range(K)]

    print(f"  Caption: {caption_data[:100]}...")

    for cfg_name, gen_fn in [("full", generate_shot_full), ("multi_cfg", generate_shot_multi_cfg)]:
        print(f"\n  Config: {cfg_name}")
        shot_latents = []
        prev_frames_ar = [prev_frame]

        for shot_idx in range(3):
            seed = args.seed + shot_idx
            print(f"    Shot {shot_idx+1}/3 (seed={seed})...", end=" ", flush=True)

            if cfg_name == "full":
                latent = gen_fn(
                    pipeline, transformer, caption_data,
                    prev_frames_ar, char_list, char_mask,
                    seed=seed, **gen_kwargs,
                )
            else:
                latent = gen_fn(
                    pipeline, transformer, caption_data,
                    prev_frames_ar, char_list, char_mask,
                    omega_text, omega_local, omega_global,
                    seed=seed, **gen_kwargs,
                )

            shot_latents.append(latent.cpu())
            del latent
            torch.cuda.empty_cache()

            if shot_idx < 2:
                print("(AR decode)...", end=" ", flush=True)
                last_frame = ar_decode_last_frame(
                    pipeline, transformer, shot_latents[-1], device
                )
                prev_frames_ar = [last_frame.to(device)]
            print("done")

        scenarios[f"baseline_{cfg_name}"] = shot_latents

    # =========================================================================
    # DECODE ALL & SAVE
    # =========================================================================
    print(f"\n{'='*70}")
    print("Decoding all latents...")
    print(f"{'='*70}")

    transformer.cpu()
    torch.cuda.empty_cache()
    pipeline.vae.to(device)

    all_videos = {}

    for scenario_name, shot_latents in scenarios.items():
        print(f"\n  Decoding {scenario_name}...")
        shot_videos = []
        for shot_idx, lat in enumerate(shot_latents):
            print(f"    Shot {shot_idx+1}...", end=" ", flush=True)
            video = decode_latent(pipeline, lat, device)
            shot_videos.append(video)
            print("done")

        full_video = torch.cat(shot_videos, dim=0)
        all_videos[scenario_name] = full_video

        # Save concatenated video
        save_video(full_video, out_dir / f"{scenario_name}.mp4", fps=args.fps)

        # Save per-shot first frames
        shot_firsts = [v[0] for v in shot_videos]
        grid = make_grid(shot_firsts, nrow=len(shot_firsts), padding=4)
        save_image(grid, out_dir / f"{scenario_name}_shot_firsts.png")

        # Save per-shot mid frames
        mid = args.num_frames // 2
        shot_mids = [v[min(mid, v.shape[0]-1)] for v in shot_videos]
        grid = make_grid(shot_mids, nrow=len(shot_mids), padding=4)
        save_image(grid, out_dir / f"{scenario_name}_shot_mids.png")

        del shot_videos

    # Save comparisons: full vs multi_cfg for each scenario
    for scenario in ["zoom", "entity", "baseline"]:
        full_key = f"{scenario}_full"
        mcfg_key = f"{scenario}_multi_cfg"
        if full_key in all_videos and mcfg_key in all_videos:
            save_side_by_side_video(
                {"full": all_videos[full_key], "multi_cfg": all_videos[mcfg_key]},
                out_dir / f"{scenario}_comparison.mp4", fps=args.fps,
            )

    # Save metadata
    meta = {
        "checkpoint": args.checkpoint,
        "loaded_step": loaded_step,
        "resolution": f"{args.width}x{args.height}",
        "num_frames_per_shot": args.num_frames,
        "num_shots": 3,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "guidance": {
            "omega_text": omega_text,
            "omega_local": omega_local,
            "omega_global": omega_global,
        },
        "scenarios": {
            "zoom": {
                "sample": zoom_sample["seq_ids"][0],
                "caption": zoom_sample["captions"][0][:200],
            },
            "entity": {
                "sample": multi_char_sample["seq_ids"][0],
                "caption": multi_char_sample["captions"][0][:200],
                "n_characters": int(multi_char_sample["character_mask"][0].sum().item()),
            },
            "baseline": {
                "sample": baseline_sample["seq_ids"][0],
                "caption": baseline_sample["captions"][0][:200],
            },
        },
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"All results saved to {out_dir}/")
    print(f"{'='*70}")
    for f_name in sorted(os.listdir(out_dir)):
        print(f"  {f_name}")


if __name__ == "__main__":
    main()
