"""
Quick standalone ablation test: 4 context configurations at t=0.75
Run on a single GPU without disturbing training.
"""
import os
import sys
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from data.dataset import DirectorDataset, DirectorDataCollator
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))
from trainer import _build_pipeline_and_config


def main():
    device = torch.device("cuda:0")  # Will be mapped by CUDA_VISIBLE_DEVICES

    # Load config
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    director_config, PipelineClass = _build_pipeline_and_config(cfg)

    # Load pipeline on target device
    print("Loading pipeline...")
    pipeline = PipelineClass(director_config, device=device)

    # Load checkpoint (modules saved individually — frozen weights already loaded by pipeline)
    ckpt_path = "checkpoints/checkpoint_best.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    transformer = pipeline.director_transformer
    transformer.adapters.load_state_dict(state["adapters"])
    transformer.context_builder.load_state_dict(state["context_builder"])
    transformer.global_encoder.load_state_dict(state["global_encoder"], strict=False)
    if "local_encoder" in state and transformer.local_encoder is not None:
        transformer.local_encoder.load_state_dict(state["local_encoder"], strict=False)
    print(f"Loaded step {state.get('global_step', '?')}")
    del state
    torch.cuda.empty_cache()

    transformer.eval()

    # Load dataset and pick brightest sample
    print("Loading dataset...")
    dataset = DirectorDataset(
        dataset_dir=cfg["dataset"]["dataset_dir"],
        target_frames=cfg["dataset"]["video"]["num_frames"],
        target_height=cfg["training"]["train_height"],
        target_width=cfg["training"]["train_width"],
        augment=False,
        split="val",
    )
    collator = DirectorDataCollator()
    loader = DataLoader(dataset, batch_size=1, collate_fn=collator, shuffle=False)

    best_sample = None
    best_brightness = -1.0
    for i, batch in enumerate(loader):
        if i >= 20:
            break
        brightness = batch["target_video"].mean().item()
        if brightness > best_brightness:
            best_brightness = brightness
            best_sample = batch
            best_id = batch.get("seq_ids", ["?"])[0]

    print(f"Selected sample: {best_id} (brightness={best_brightness:.4f})")

    # Prepare inputs
    batch = best_sample
    target_video = batch["target_video"].to(device, dtype=torch.float32)
    prev_frame = batch["prev_frame"].to(device, dtype=torch.float32)
    prev_prev_frame = batch["prev_prev_frame"].to(device, dtype=torch.float32)
    has_prev_prev = batch["has_prev_prev"]
    anchor_rgb = batch["anchor_rgb"].to(device, dtype=torch.float32)
    char_mask = batch["character_mask"].to(device, dtype=torch.float32)
    caption = batch["captions"][0]
    B, K = anchor_rgb.shape[:2]

    prev_frames = [prev_frame]
    if has_prev_prev[0]:
        prev_frames.append(prev_prev_frame)
    char_list = [anchor_rgb[:, k] for k in range(K)]
    local_frame_valid = torch.stack([
        torch.ones(B, device=device),
        has_prev_prev.float().to(device),
    ], dim=0)

    # Get GT middle frame
    T_vid = target_video.shape[1]
    gt_mid = target_video[0, T_vid // 2].cpu().float().clamp(0, 1)

    print(f"Caption: {caption[:80]}...")
    print("Running ablation...")

    transformer = pipeline.director_transformer

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Encode GT to latent
        x_clean = pipeline.encode_video(target_video)
        del target_video
        torch.cuda.empty_cache()

        # Create noisy input at timestep 750 (heavy noise)
        noise = torch.randn_like(x_clean)
        timestep = torch.tensor([750], device=device, dtype=torch.long)
        diffusion = pipeline.diffusion

        x_t = diffusion.add_noise(x_clean, noise, timestep)

        del x_clean, noise
        torch.cuda.empty_cache()

        # Encode text
        text_embeds = pipeline.encode_text(caption)

        # 4 context configurations
        ctx_configs = [
            ("text_only",   {"prev_frames": None, "character_images": None, "character_masks": None}),
            ("local_only",  {"prev_frames": prev_frames, "character_images": None, "character_masks": None}),
            ("global_only", {"prev_frames": None, "character_images": char_list, "character_masks": char_mask}),
            ("full",        {"prev_frames": prev_frames, "character_images": char_list, "character_masks": char_mask}),
        ]

        # Run all 4 forward passes, recover x_0 from v-prediction
        x0_hats_cpu = []
        for cfg_name, cfg_kwargs in ctx_configs:
            print(f"  {cfg_name} forward...", end=" ", flush=True)
            lfv = local_frame_valid if cfg_kwargs["prev_frames"] is not None else None

            ctx, mask = transformer.encode_context(
                prev_frames=cfg_kwargs["prev_frames"],
                character_images=cfg_kwargs["character_images"],
                character_masks=cfg_kwargs["character_masks"],
                local_frame_valid=lfv,
            )

            v_pred = transformer(
                hidden_states=x_t,
                encoder_hidden_states=text_embeds,
                timestep=timestep,
                unified_context=ctx,
                context_mask=mask,
                return_dict=False,
            )[0]

            # Recover clean sample via diffusion wrapper
            x_0_hat = diffusion.recover_clean(x_t, v_pred, timestep)
            x0_hats_cpu.append(x_0_hat.cpu())
            del v_pred, ctx, mask, x_0_hat
            torch.cuda.empty_cache()
            print("done")

        del x_t, text_embeds
        torch.cuda.empty_cache()

    # Offload transformer to CPU to free ~25GB for VAE decode
    print("  Offloading transformer to CPU for VAE decode...")
    transformer.cpu()
    torch.cuda.empty_cache()

    # Ensure VAE is on GPU for decode
    pipeline.vae.to(device)

    ablation_frames = []
    for i, (cfg_name, _) in enumerate(ctx_configs):
        print(f"  {cfg_name} decode...", end=" ", flush=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x_0_hat_gpu = x0_hats_cpu[i].to(device)
            recon = pipeline.decode_latent(x_0_hat_gpu)
            mid_idx = recon.shape[1] // 2
            frame = recon[0, mid_idx].cpu().float().clamp(0, 1)
            ablation_frames.append(frame)
            del recon, x_0_hat_gpu
            torch.cuda.empty_cache()
            print("done")
    del x0_hats_cpu

    # Save results
    out_dir = Path("samples/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save individual frames
    for i, (name, _) in enumerate(ctx_configs):
        save_image(ablation_frames[i], out_dir / f"{name}.png")

    # Save GT
    save_image(gt_mid, out_dir / "ground_truth.png")

    # Save context reference
    ref_images = [prev_frame[0].cpu().float().clamp(0, 1)]
    for k in range(min(K, 4)):
        if char_mask[0, k] > 0:
            ref_images.append(anchor_rgb[0, k].cpu().float().clamp(0, 1))

    # Save 5-column comparison: [GT, text_only, local_only, global_only, full]
    all_frames = [gt_mid] + ablation_frames
    # Resize all to same size
    h, w = all_frames[0].shape[1], all_frames[0].shape[2]
    resized = []
    for f in all_frames:
        f_r = F.interpolate(f.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)[0]
        resized.append(f_r.clamp(0, 1))

    grid = make_grid(resized, nrow=5, padding=4, normalize=False)
    save_image(grid, out_dir / "comparison_gt_text_local_global_full.png")

    # Save context ref grid
    if len(ref_images) > 1:
        ref_h, ref_w = ref_images[0].shape[1], ref_images[0].shape[2]
        ref_resized = [F.interpolate(img.unsqueeze(0), size=(ref_h, ref_w), mode="bilinear", align_corners=False)[0].clamp(0, 1) for img in ref_images]
        ref_grid = make_grid(ref_resized, nrow=len(ref_resized), padding=2, normalize=False)
        save_image(ref_grid, out_dir / "context_ref.png")

    # Print per-layer gate values
    print("\nPer-layer gate values (tanh(gate)):")
    adapters = transformer.adapters
    for key in sorted(adapters.keys(), key=int):
        gv = torch.tanh(adapters[key].gate).item()
        bar = "+" * int(abs(gv) * 100) if gv > 0 else "-" * int(abs(gv) * 100)
        print(f"  Layer {int(key):2d}: {gv:+.4f}  {'|' + bar}")

    print(f"\nSaved to {out_dir}/")
    print("Files: comparison_gt_text_local_global_full.png, text_only.png, local_only.png, global_only.png, full.png, ground_truth.png, context_ref.png")


if __name__ == "__main__":
    main()
