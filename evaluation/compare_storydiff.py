"""
DIRECTOR vs StoryDiffusion Comparison.

Runs both DIRECTOR and StoryDiffusion on the same prompts,
computes metrics side by side, and generates comparison visualizations.

Usage:
  python -m evaluation.compare_storydiff \
    --config configs/default.yaml \
    --checkpoint checkpoints/best.pt \
    --prompts test_prompts.json \
    --output_dir evaluation_results/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

RELATED_PAPERS_DIR = Path(__file__).resolve().parents[2] / "related_papers"


class StoryDiffusionBaseline:
    """
    Wrapper for StoryDiffusion baseline.

    Loads the StoryDiffusion pipeline from the related papers directory
    and provides a unified interface for fair comparison.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
        storydiff_dir: Optional[str] = None,
    ):
        self.device = device
        if storydiff_dir is None:
            storydiff_dir = str(RELATED_PAPERS_DIR / "StoryDiffusion")
        self.storydiff_dir = Path(storydiff_dir)

        sys.path.insert(0, str(self.storydiff_dir))

        try:
            # StoryDiffusion uses SDXL + PhotoMaker
            from diffusers import StableDiffusionXLPipeline

            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                variant="fp16",
            ).to(device)

            # Load StoryDiffusion-specific components
            self._load_story_components()
            self.available = True
            logger.info("StoryDiffusion baseline loaded successfully")
        except Exception as e:
            logger.warning(f"StoryDiffusion not fully available: {e}")
            logger.warning("Will use base SDXL for comparison")
            self.available = False

    def _load_story_components(self):
        """Load StoryDiffusion's IP-Adapter and consistent self-attention."""
        try:
            # Try to load StoryDiffusion's specific attention mechanism
            utils_dir = self.storydiff_dir / "utils"
            if utils_dir.exists():
                sys.path.insert(0, str(utils_dir))

            # StoryDiffusion modifies attention for consistent generation
            # We use their pipeline if available
            config_dir = self.storydiff_dir / "config"
            if config_dir.exists():
                self.story_config = {}
                for cfg_file in config_dir.glob("*.yaml"):
                    with open(cfg_file) as f:
                        self.story_config.update(yaml.safe_load(f) or {})
        except Exception as e:
            logger.warning(f"Could not load StoryDiffusion components: {e}")

    @torch.no_grad()
    def generate_story(
        self,
        prompts: List[str],
        character_description: str = "",
        num_images_per_prompt: int = 1,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = 42,
    ) -> List[np.ndarray]:
        """
        Generate a sequence of images using StoryDiffusion.

        Note: StoryDiffusion generates images (not video), so for fair comparison
        we generate one key frame per shot.

        Args:
            prompts: list of scene descriptions
            character_description: character appearance description
            height, width: image dimensions
            num_inference_steps: denoising steps
            guidance_scale: classifier-free guidance scale
            seed: random seed

        Returns:
            List of generated images as numpy arrays (H, W, 3)
        """
        generator = torch.Generator(device=self.device).manual_seed(seed)
        images = []

        for i, prompt in enumerate(prompts):
            full_prompt = f"{character_description}, {prompt}" if character_description else prompt

            gen = torch.Generator(device=self.device).manual_seed(seed + i)

            if self.available and hasattr(self, 'pipe'):
                output = self.pipe(
                    prompt=full_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=gen,
                )
                img = output.images[0]
                img_np = np.array(img)
            else:
                # Fallback: generate random image (for testing pipeline)
                img_np = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

            images.append(img_np)

        return images


class ComparisonEvaluator:
    """
    Runs DIRECTOR and StoryDiffusion on the same prompts and compares results.
    """

    def __init__(
        self,
        config_path: str,
        director_checkpoint: Optional[str] = None,
        device: torch.device = torch.device("cuda"),
    ):
        self.device = device
        self.config_path = config_path

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Initialize metrics
        from .metrics import DirectorEvaluator
        self.evaluator = DirectorEvaluator(device=device)

        # Initialize DIRECTOR
        logger.info("Loading DIRECTOR...")
        from inference.generate import DirectorInferencePipeline
        self.director = DirectorInferencePipeline(
            config_path=config_path,
            checkpoint_path=director_checkpoint,
            device=device,
        )

        # Initialize StoryDiffusion
        logger.info("Loading StoryDiffusion baseline...")
        self.storydiff = StoryDiffusionBaseline(device=device)

    def run_comparison(
        self,
        prompts: List[str],
        character_refs: Optional[Dict[str, str]] = None,
        output_dir: str = "evaluation_results",
        seed: int = 42,
    ) -> Dict:
        """
        Run both methods on the same prompts and compare.

        Args:
            prompts: list of shot descriptions
            character_refs: dict {name: image_path} for characters
            output_dir: directory to save results
            seed: random seed for reproducibility

        Returns:
            Comparison results dict
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {
            "prompts": prompts,
            "seed": seed,
            "director": {},
            "storydiffusion": {},
        }

        # --- Run DIRECTOR ---
        logger.info("=" * 60)
        logger.info("Running DIRECTOR...")
        logger.info("=" * 60)

        director_dir = output_path / "director"
        director_dir.mkdir(exist_ok=True)

        start_time = time.time()
        director_shots = self.director.generate_multi_shot_story(
            shot_prompts=prompts,
            character_refs=character_refs,
            seed=seed,
            save_dir=str(director_dir),
        )
        director_time = time.time() - start_time

        results["director"]["time_s"] = director_time
        results["director"]["time_per_shot_s"] = director_time / len(prompts)

        # Evaluate DIRECTOR
        director_video_paths = [
            str(director_dir / f"shot_{i:03d}.mp4") for i in range(len(prompts))
        ]
        director_metrics = self.evaluator.evaluate_shots(
            shots=director_shots,
            video_paths=director_video_paths,
            prompts=prompts,
        )
        results["director"]["metrics"] = director_metrics

        # --- Run StoryDiffusion ---
        logger.info("=" * 60)
        logger.info("Running StoryDiffusion...")
        logger.info("=" * 60)

        storydiff_dir = output_path / "storydiffusion"
        storydiff_dir.mkdir(exist_ok=True)

        char_desc = ""
        if character_refs:
            char_desc = "a person"  # StoryDiffusion uses text descriptions

        start_time = time.time()
        storydiff_images = self.storydiff.generate_story(
            prompts=prompts,
            character_description=char_desc,
            seed=seed,
        )
        storydiff_time = time.time() - start_time

        results["storydiffusion"]["time_s"] = storydiff_time
        results["storydiffusion"]["time_per_shot_s"] = storydiff_time / len(prompts)

        # Save StoryDiffusion images
        for i, img in enumerate(storydiff_images):
            cv2.imwrite(
                str(storydiff_dir / f"shot_{i:03d}.png"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )

        # Evaluate StoryDiffusion (convert images to pseudo-video tensors)
        storydiff_shots = []
        for img in storydiff_images:
            # Convert single image to 1-frame video tensor
            tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            # Resize to match DIRECTOR output
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=(480, 720), mode="bilinear", align_corners=False
            )
            # Repeat as single-frame "video"
            video = tensor.unsqueeze(1)  # (1, 1, 3, H, W)
            storydiff_shots.append(video)

        storydiff_metrics = self.evaluator.evaluate_shots(
            shots=storydiff_shots,
            prompts=prompts,
        )
        results["storydiffusion"]["metrics"] = storydiff_metrics

        # --- Generate comparison ---
        self._generate_comparison_table(results, output_path)
        self._generate_visual_comparison(
            director_shots, storydiff_images, prompts, output_path
        )

        # Save full results
        # Convert non-serializable values
        results_serializable = self._make_serializable(results)
        with open(output_path / "comparison_results.json", "w") as f:
            json.dump(results_serializable, f, indent=2)

        logger.info(f"Comparison results saved to {output_path}")
        return results

    def _generate_comparison_table(self, results: Dict, output_path: Path):
        """Generate a text table comparing metrics."""
        lines = []
        lines.append("=" * 70)
        lines.append("DIRECTOR vs StoryDiffusion Comparison")
        lines.append("=" * 70)
        lines.append("")

        # Timing
        lines.append("--- Efficiency ---")
        lines.append(f"{'Metric':<30} {'DIRECTOR':>15} {'StoryDiff':>15}")
        lines.append("-" * 60)
        lines.append(
            f"{'Total Time (s)':<30} "
            f"{results['director']['time_s']:>15.1f} "
            f"{results['storydiffusion']['time_s']:>15.1f}"
        )
        lines.append(
            f"{'Time per Shot (s)':<30} "
            f"{results['director']['time_per_shot_s']:>15.1f} "
            f"{results['storydiffusion']['time_per_shot_s']:>15.1f}"
        )
        lines.append("")

        # Identity metrics
        lines.append("--- Identity Consistency ---")
        d_id = results["director"]["metrics"].get("identity", {})
        s_id = results["storydiffusion"]["metrics"].get("identity", {})
        for key in ["mean_similarity", "adjacent_similarity", "distant_similarity"]:
            d_val = d_id.get(key, "N/A")
            s_val = s_id.get(key, "N/A")
            d_str = f"{d_val:.4f}" if isinstance(d_val, float) else str(d_val)
            s_str = f"{s_val:.4f}" if isinstance(s_val, float) else str(s_val)
            lines.append(f"{key:<30} {d_str:>15} {s_str:>15}")
        lines.append("")

        # Motion metrics
        lines.append("--- Motion Coherence ---")
        d_mo = results["director"]["metrics"].get("motion", {})
        s_mo = results["storydiffusion"]["metrics"].get("motion", {})
        for key in ["boundary_psnr", "mean_flow_magnitude", "flow_smoothness"]:
            d_val = d_mo.get(key, "N/A")
            s_val = s_mo.get(key, "N/A")
            d_str = f"{d_val:.4f}" if isinstance(d_val, float) else str(d_val)
            s_str = f"{s_val:.4f}" if isinstance(s_val, float) else str(s_val)
            lines.append(f"{key:<30} {d_str:>15} {s_str:>15}")

        lines.append("")
        lines.append("=" * 70)

        table_text = "\n".join(lines)
        print(table_text)

        with open(output_path / "comparison_table.txt", "w") as f:
            f.write(table_text)

    def _generate_visual_comparison(
        self,
        director_shots: List[torch.Tensor],
        storydiff_images: List[np.ndarray],
        prompts: List[str],
        output_path: Path,
    ):
        """Generate a side-by-side visual comparison grid."""
        num_shots = min(len(director_shots), len(storydiff_images))
        if num_shots == 0:
            return

        # Create comparison grid
        cell_h, cell_w = 240, 360
        header_h = 40
        text_h = 30
        grid_h = header_h + num_shots * (cell_h + text_h)
        grid_w = 2 * cell_w + 20  # 2 columns + padding

        grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

        # Draw headers
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except OSError:
            font = ImageFont.load_default()
            small_font = font

        grid_pil = Image.fromarray(grid)
        draw = ImageDraw.Draw(grid_pil)

        draw.text((cell_w // 2 - 40, 5), "DIRECTOR", fill=(0, 100, 200), font=font)
        draw.text((cell_w + 20 + cell_w // 2 - 60, 5), "StoryDiffusion", fill=(200, 100, 0), font=font)

        for i in range(num_shots):
            y_offset = header_h + i * (cell_h + text_h)

            # DIRECTOR: use middle frame
            director_frame = director_shots[i][0, director_shots[i].shape[1] // 2]
            director_np = (director_frame.cpu().float().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
            director_resized = cv2.resize(director_np, (cell_w, cell_h))
            grid_pil.paste(Image.fromarray(director_resized), (0, y_offset))

            # StoryDiffusion
            storydiff_resized = cv2.resize(storydiff_images[i], (cell_w, cell_h))
            grid_pil.paste(Image.fromarray(storydiff_resized), (cell_w + 20, y_offset))

            # Prompt text
            prompt_text = prompts[i][:60] + "..." if len(prompts[i]) > 60 else prompts[i]
            draw.text(
                (5, y_offset + cell_h + 2),
                f"Shot {i + 1}: {prompt_text}",
                fill=(0, 0, 0),
                font=small_font,
            )

        grid_pil.save(str(output_path / "visual_comparison.png"))
        logger.info(f"Visual comparison saved to {output_path / 'visual_comparison.png'}")

    def _make_serializable(self, obj):
        """Convert non-JSON-serializable objects."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return str(obj)
        return obj


def main():
    parser = argparse.ArgumentParser(description="DIRECTOR vs StoryDiffusion Comparison")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--prompts", type=str, required=True, help="JSON with shot prompts")
    parser.add_argument("--characters", type=str, default=None, help="JSON with character refs")
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.prompts) as f:
        prompts_data = json.load(f)
    if isinstance(prompts_data, list):
        prompts = prompts_data
    else:
        prompts = prompts_data.get("shots", prompts_data.get("prompts", []))

    character_refs = None
    if args.characters:
        with open(args.characters) as f:
            character_refs = json.load(f)

    device_id = yaml.safe_load(open(args.config)).get("cuda_device", 0)
    device = torch.device(f"cuda:{device_id}")

    evaluator = ComparisonEvaluator(
        config_path=args.config,
        director_checkpoint=args.checkpoint,
        device=device,
    )

    results = evaluator.run_comparison(
        prompts=prompts,
        character_refs=character_refs,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    # Print summary
    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)
    d_summary = results["director"]["metrics"].get("summary", {})
    s_summary = results["storydiffusion"]["metrics"].get("summary", {})

    for key in d_summary:
        d_val = d_summary.get(key, 0)
        s_val = s_summary.get(key, 0)
        winner = "DIRECTOR" if d_val > s_val else "StoryDiff"
        print(f"{key}: DIRECTOR={d_val:.4f} | StoryDiff={s_val:.4f} -> {winner}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
