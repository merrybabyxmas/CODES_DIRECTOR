"""
1-sample overfit test for DIRECTOR.

Verifies that the diffusion loss (DDPM v-prediction for CogVideoX, flow matching
for WanVideo, etc.) is correctly implemented by overfitting adapters on a single
training example.

Success criteria:
  - Loss drops to < 0.01 by step 500
  - Single-step reconstruction at t=750 produces a recognizable image
  - Full inference (from pure noise) produces a recognizable video

Usage:
  CUDA_VISIBLE_DEVICES=1 conda run -n paper_env python scripts/run_overfit_test.py
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from torchvision.utils import save_image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from data.dataset import DirectorDataset, DirectorDataCollator
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))
from trainer import _build_pipeline_and_config


def main():
    device = torch.device("cuda:0")
    out_dir = Path("samples/overfit_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/overfit_test.yaml"
    print(f"Loading config: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    director_config, PipelineClass = _build_pipeline_and_config(cfg)
    print("Loading pipeline...")
    pipeline = PipelineClass(director_config, device=device)

    transformer = pipeline.director_transformer
    transformer.train()

    # --- Load single sample ---
    print("Loading dataset (single sample)...")
    train_cfg = cfg["training"]
    dataset = DirectorDataset(
        dataset_dir=cfg["dataset"]["dataset_dir"],
        target_frames=cfg["dataset"]["video"]["num_frames"],
        target_height=train_cfg["train_height"],
        target_width=train_cfg["train_width"],
        augment=False,
        split="train",
    )
    collator = DirectorDataCollator()
    loader = DataLoader(dataset, batch_size=1, collate_fn=collator, shuffle=False)

    # Pick first sample
    batch = next(iter(loader))
    seq_id = batch.get("seq_ids", ["?"])[0]
    caption = batch["captions"][0]
    print(f"Sample: {seq_id}")
    print(f"Caption: {caption[:100]}...")

    # Move to device once and keep
    target_video = batch["target_video"].to(device, dtype=torch.float32)
    prev_frame = batch["prev_frame"].to(device, dtype=torch.float32)
    prev_prev_frame = batch["prev_prev_frame"].to(device, dtype=torch.float32)
    has_prev_prev = batch["has_prev_prev"]
    anchor_rgb = batch["anchor_rgb"].to(device, dtype=torch.float32)
    char_mask = batch["character_mask"].to(device, dtype=torch.float32)
    B, K = anchor_rgb.shape[:2]

    # Save GT
    T_vid = target_video.shape[1]
    gt_mid = target_video[0, T_vid // 2].cpu().float().clamp(0, 1)
    save_image(gt_mid, out_dir / "ground_truth.png")

    # Pre-encode
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        x_1 = pipeline.encode_video(target_video)
        text_embeds = pipeline.encode_text(caption)

        prev_frames = [prev_frame]
        if has_prev_prev[0]:
            prev_frames.append(prev_prev_frame)
        char_list = [anchor_rgb[:, k] for k in range(K)]
        local_frame_valid = torch.stack([
            torch.ones(B, device=device),
            has_prev_prev.float().to(device),
        ], dim=0)

    del target_video, prev_frame, prev_prev_frame, anchor_rgb
    torch.cuda.empty_cache()

    # Enable gradient checkpointing to fit in memory
    if train_cfg.get("gradient_checkpointing", True):
        transformer.backbone.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled")

    # --- Setup optimizer (no dropout, high LR) ---
    opt_cfg = train_cfg["optimizer"]
    lr = opt_cfg.get("lr", 5e-4)
    params = list(transformer.get_trainable_parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")

    # FM config
    fm_cfg = train_cfg.get("flow_matching", {})
    sigma_min = fm_cfg.get("sigma_min", 0.001)
    time_sampling = fm_cfg.get("time_sampling", "logit_normal")
    logit_normal_mean = fm_cfg.get("logit_normal_mean", 0.0)
    logit_normal_std = fm_cfg.get("logit_normal_std", 1.0)

    max_steps = train_cfg.get("max_steps", 500)
    log_every = train_cfg["logging"].get("log_every_steps", 10)
    sample_every = train_cfg.get("sample_generation", {}).get("sample_every_steps", 100)

    print(f"\n{'='*60}")
    print(f"1-SAMPLE OVERFIT TEST")
    print(f"LR={lr}, max_steps={max_steps}")
    print(f"Diffusion algorithm: {type(pipeline.diffusion).__name__}")
    print(f"{'='*60}\n")

    # --- Training loop ---
    losses = []
    start_time = time.time()

    for step in range(1, max_steps + 1):
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Encode context (with no dropout)
            unified_context, context_mask = transformer.encode_context(
                prev_frames=prev_frames,
                character_images=char_list,
                character_masks=char_mask,
                local_frame_valid=local_frame_valid,
            )

            loss_dict = pipeline.compute_flow_matching_loss(
                x_1=x_1,
                text_embeds=text_embeds,
                unified_context=unified_context,
                context_mask=context_mask,
                sigma_min=sigma_min,
                time_sampling=time_sampling,
                logit_normal_mean=logit_normal_mean,
                logit_normal_std=logit_normal_std,
            )
            loss = loss_dict["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        losses.append(loss_val)

        if step % log_every == 0 or step == 1:
            elapsed = time.time() - start_time
            pred_norm = loss_dict["pred_norm"].item()
            target_norm = loss_dict["target_norm"].item()
            ts_mean = loss_dict["timestep_mean"].item()

            # Gate stats
            gate_vals = []
            for key in transformer.adapters.keys():
                gv = torch.tanh(transformer.adapters[key].gate).item()
                gate_vals.append(gv)
            gate_mean = sum(gate_vals) / len(gate_vals)
            gate_max = max(gate_vals)

            print(f"Step {step:4d} | loss={loss_val:.6f} | "
                  f"pred={pred_norm:.2f} tgt={target_norm:.2f} | "
                  f"ts={ts_mean:.0f} | gate={gate_mean:+.4f} max={gate_max:+.4f} | "
                  f"{elapsed:.0f}s")

        # Sample reconstruction
        if step % sample_every == 0 or step == max_steps:
            print(f"\n--- Sampling at step {step} ---")
            transformer.eval()

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Single-step reconstruction at t=750
                noise = torch.randn_like(x_1)
                timestep = torch.tensor([750], device=device, dtype=torch.long)

                x_t = pipeline.diffusion.add_noise(x_1, noise, timestep)

                ctx, mask = transformer.encode_context(
                    prev_frames=prev_frames,
                    character_images=char_list,
                    character_masks=char_mask,
                    local_frame_valid=local_frame_valid,
                )

                v_pred = transformer(
                    hidden_states=x_t,
                    encoder_hidden_states=text_embeds,
                    timestep=timestep,
                    unified_context=ctx,
                    context_mask=mask,
                    return_dict=False,
                )[0]

                x_0_hat = pipeline.diffusion.recover_clean(x_t, v_pred, timestep)
                x_0_hat_cpu = x_0_hat.cpu()
                del noise, x_t, v_pred, ctx, mask, x_0_hat
                torch.cuda.empty_cache()

            # Offload transformer to CPU for VAE decode
            transformer.cpu()
            torch.cuda.empty_cache()
            pipeline.vae.to(device)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                recon = pipeline.decode_latent(x_0_hat_cpu.to(device))
                mid_idx = recon.shape[1] // 2
                recon_frame = recon[0, mid_idx].cpu().float().clamp(0, 1)
                save_image(recon_frame, out_dir / f"recon_step{step:04d}.png")
                del recon, x_0_hat_cpu
                torch.cuda.empty_cache()

            # Move transformer back to GPU
            transformer.to(device)
            print(f"Saved reconstruction to {out_dir}/recon_step{step:04d}.png")
            transformer.train()

    # --- Final summary ---
    print(f"\n{'='*60}")
    print(f"OVERFIT TEST COMPLETE")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Min loss:   {min(losses):.6f}")
    if losses[-1] < 0.01:
        print("PASS: Loss < 0.01")
    elif losses[-1] < 0.05:
        print("MARGINAL: Loss < 0.05 (may need more steps)")
    else:
        print("FAIL: Loss too high — check diffusion algorithm implementation")
    print(f"{'='*60}")

    # Save checkpoint for inference test
    ckpt = {
        "adapters": transformer.adapters.state_dict(),
        "context_builder": transformer.context_builder.state_dict(),
        "global_encoder": transformer.global_encoder.state_dict(),
        "global_step": max_steps,
    }
    if transformer.local_encoder is not None:
        ckpt["local_encoder"] = transformer.local_encoder.state_dict()
    torch.save(ckpt, out_dir / "overfit_checkpoint.pt")
    print(f"Saved checkpoint to {out_dir}/overfit_checkpoint.pt")


if __name__ == "__main__":
    main()
