"""
DIRECTOR Evaluation Metrics.

Implements:
  1. IdentityConsistencyMetric: DINOv2 cosine similarity between shots
  2. MotionCoherenceMetric: Optical flow smoothness at shot boundaries
  3. EfficiencyMetric: GPU memory + time per shot
  4. VBench integration for comprehensive video quality

All metrics operate on lists of video tensors (one per shot).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


class IdentityConsistencyMetric:
    """
    Measures identity consistency across shots using DINOv2.

    For each pair of shots, extracts DINOv2 features from character regions
    and computes cosine similarity. Higher = better identity preservation.

    Metrics:
      - mean_similarity: average cosine sim across all shot pairs
      - adjacent_similarity: cosine sim between consecutive shots
      - distant_similarity: cosine sim between non-adjacent shots (harder)
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: torch.device = torch.device("cuda"),
    ):
        self.device = device

        from transformers import AutoModel, AutoImageProcessor

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def extract_frame_features(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Extract DINOv2 CLS features from a frame.

        Args:
            frame: (3, H, W) float tensor in [0, 1]

        Returns:
            features: (D,) normalized feature vector
        """
        # Convert to PIL for processor
        frame_np = (frame.cpu().float().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        pil_img = Image.fromarray(frame_np)

        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        features = outputs.last_hidden_state[:, 0]  # CLS token, (1, D)
        features = F.normalize(features, p=2, dim=-1)
        return features.squeeze(0)  # (D,)

    def compute(
        self,
        shots: List[torch.Tensor],
        pairs: str = "all",
        frames_per_shot: int = 3,
    ) -> Dict[str, float]:
        """
        Compute identity consistency metrics across shots.

        Args:
            shots: list of (1, T, 3, H, W) video tensors
            pairs: 'all', 'adjacent', or 'distant'
            frames_per_shot: number of frames to sample per shot

        Returns:
            dict with similarity metrics
        """
        num_shots = len(shots)
        if num_shots < 2:
            return {"mean_similarity": 1.0, "adjacent_similarity": 1.0, "distant_similarity": 1.0}

        # Extract features from sampled frames per shot
        shot_features = []
        for shot in shots:
            video = shot[0]  # (T, 3, H, W)
            T = video.shape[0]
            indices = torch.linspace(0, T - 1, frames_per_shot).long()
            feats = []
            for idx in indices:
                f = self.extract_frame_features(video[idx])  # (D,)
                feats.append(f)
            # Average features across sampled frames
            shot_feat = torch.stack(feats).mean(dim=0)  # (D,)
            shot_feat = F.normalize(shot_feat, p=2, dim=-1)
            shot_features.append(shot_feat)

        features = torch.stack(shot_features)  # (N, D)

        # Compute pairwise similarities
        sim_matrix = features @ features.T  # (N, N)

        # Adjacent pairs
        adjacent_sims = []
        for i in range(num_shots - 1):
            adjacent_sims.append(sim_matrix[i, i + 1].item())

        # Distant pairs (gap > 1)
        distant_sims = []
        for i in range(num_shots):
            for j in range(i + 2, num_shots):
                distant_sims.append(sim_matrix[i, j].item())

        # All pairs
        all_sims = []
        for i in range(num_shots):
            for j in range(i + 1, num_shots):
                all_sims.append(sim_matrix[i, j].item())

        results = {
            "mean_similarity": np.mean(all_sims) if all_sims else 0.0,
            "adjacent_similarity": np.mean(adjacent_sims) if adjacent_sims else 0.0,
            "distant_similarity": np.mean(distant_sims) if distant_sims else 0.0,
            "min_similarity": min(all_sims) if all_sims else 0.0,
            "num_pairs": len(all_sims),
        }

        return results


class MotionCoherenceMetric:
    """
    Measures motion coherence at shot boundaries using optical flow.

    Computes optical flow between the last frame of shot t-1 and the first
    frame of shot t. Smoother flow = better temporal continuity.

    Metrics:
      - mean_flow_magnitude: average optical flow magnitude at boundaries
      - flow_smoothness: variance of flow vectors (lower = smoother)
      - boundary_psnr: PSNR between last/first frames of adjacent shots
    """

    def __init__(
        self,
        flow_method: str = "farneback",
        boundary_window: int = 5,
    ):
        self.flow_method = flow_method
        self.boundary_window = boundary_window

    def _compute_optical_flow(
        self, frame1: np.ndarray, frame2: np.ndarray
    ) -> np.ndarray:
        """
        Compute optical flow between two frames.

        Args:
            frame1, frame2: (H, W, 3) uint8 numpy arrays

        Returns:
            flow: (H, W, 2) float32 optical flow
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

        if self.flow_method == "farneback":
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2,
                flow=None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
        elif self.flow_method == "raft":
            # RAFT requires torchvision
            try:
                from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
                weights = Raft_Large_Weights.DEFAULT
                model = raft_large(weights=weights).eval()

                transform = weights.transforms()

                f1 = torch.from_numpy(frame1).permute(2, 0, 1).unsqueeze(0).float()
                f2 = torch.from_numpy(frame2).permute(2, 0, 1).unsqueeze(0).float()
                f1, f2 = transform(f1, f2)

                with torch.no_grad():
                    flow_pred = model(f1, f2)
                flow = flow_pred[-1].squeeze(0).permute(1, 2, 0).numpy()
            except ImportError:
                logger.warning("RAFT not available, falling back to Farneback")
                return self._compute_optical_flow_farneback(frame1, frame2)
        else:
            raise ValueError(f"Unknown flow method: {self.flow_method}")

        return flow

    def _compute_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute PSNR between two images."""
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        if mse == 0:
            return float("inf")
        return 10 * np.log10(255.0 ** 2 / mse)

    def compute(
        self,
        shots: List[torch.Tensor],
    ) -> Dict[str, float]:
        """
        Compute motion coherence metrics at shot boundaries.

        Args:
            shots: list of (1, T, 3, H, W) video tensors

        Returns:
            dict with motion coherence metrics
        """
        num_shots = len(shots)
        if num_shots < 2:
            return {
                "mean_flow_magnitude": 0.0,
                "flow_smoothness": 0.0,
                "boundary_psnr": float("inf"),
            }

        flow_magnitudes = []
        flow_variances = []
        boundary_psnrs = []

        for i in range(num_shots - 1):
            # Last frame of current shot
            last_frame = shots[i][0, -1]  # (3, H, W)
            # First frame of next shot
            first_frame = shots[i + 1][0, 0]  # (3, H, W)

            # Convert to numpy uint8
            last_np = (last_frame.cpu().float().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
            first_np = (first_frame.cpu().float().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()

            # Compute optical flow
            flow = self._compute_optical_flow(last_np, first_np)  # (H, W, 2)
            magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

            flow_magnitudes.append(np.mean(magnitude))
            flow_variances.append(np.var(magnitude))

            # PSNR at boundary
            psnr = self._compute_psnr(last_np, first_np)
            boundary_psnrs.append(psnr)

        # Also compute intra-shot flow for reference
        intra_flow_mags = []
        for shot in shots:
            video = shot[0]  # (T, 3, H, W)
            T = video.shape[0]
            if T < 2:
                continue
            # Flow between consecutive frames within the shot
            mid = T // 2
            f1 = (video[mid].cpu().float().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
            f2 = (video[mid + 1].cpu().float().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
            flow = self._compute_optical_flow(f1, f2)
            mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
            intra_flow_mags.append(np.mean(mag))

        results = {
            "mean_flow_magnitude": float(np.mean(flow_magnitudes)),
            "flow_smoothness": float(np.mean(flow_variances)),
            "boundary_psnr": float(np.mean(boundary_psnrs)),
            "intra_shot_flow": float(np.mean(intra_flow_mags)) if intra_flow_mags else 0.0,
            "flow_ratio": (
                float(np.mean(flow_magnitudes) / max(np.mean(intra_flow_mags), 1e-8))
                if intra_flow_mags else 0.0
            ),
        }

        return results


class EfficiencyMetric:
    """
    Measures computational efficiency.

    Metrics:
      - peak_gpu_memory_mb: peak GPU memory during generation
      - time_per_shot_s: wall-clock time per shot
      - total_time_s: total generation time
      - memory_scaling: memory usage vs. number of shots (should be O(1))
    """

    def __init__(self, device: torch.device = torch.device("cuda")):
        self.device = device

    def measure_generation(
        self,
        generate_fn,
        num_shots: int = 5,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Measure efficiency of the generation function.

        Args:
            generate_fn: callable that generates shots
            num_shots: number of shots to generate
            **kwargs: additional arguments for generate_fn

        Returns:
            dict with efficiency metrics
        """
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.empty_cache()

        # Baseline memory
        base_memory = torch.cuda.memory_allocated(self.device) / 1024 ** 2

        start_time = time.time()
        shot_times = []
        shot_memories = []

        for i in range(num_shots):
            shot_start = time.time()
            torch.cuda.reset_peak_memory_stats(self.device)

            generate_fn(shot_index=i, **kwargs)

            shot_time = time.time() - shot_start
            shot_peak = torch.cuda.max_memory_allocated(self.device) / 1024 ** 2

            shot_times.append(shot_time)
            shot_memories.append(shot_peak)

            torch.cuda.empty_cache()

        total_time = time.time() - start_time
        peak_memory = max(shot_memories)

        # Check memory scaling: O(1) means memory shouldn't grow with shots
        if len(shot_memories) > 2:
            memory_growth = (shot_memories[-1] - shot_memories[0]) / max(shot_memories[0], 1)
        else:
            memory_growth = 0.0

        results = {
            "peak_gpu_memory_mb": peak_memory,
            "base_memory_mb": base_memory,
            "time_per_shot_s": float(np.mean(shot_times)),
            "total_time_s": total_time,
            "time_std_s": float(np.std(shot_times)),
            "memory_scaling_ratio": memory_growth,
            "is_constant_memory": abs(memory_growth) < 0.05,
        }

        return results


class VBenchEvaluator:
    """
    VBench integration for comprehensive video quality evaluation.

    Wraps the VBench evaluation framework to assess:
      - Subject consistency
      - Temporal flickering
      - Motion smoothness
      - Aesthetic quality
      - Imaging quality
    """

    def __init__(
        self,
        vbench_dir: Optional[str] = None,
        device: torch.device = torch.device("cuda"),
        dimensions: Optional[List[str]] = None,
    ):
        self.device = device
        self.dimensions = dimensions or [
            "subject_consistency",
            "temporal_flickering",
            "motion_smoothness",
            "aesthetic_quality",
            "imaging_quality",
        ]

        # Set VBench path
        if vbench_dir is None:
            vbench_dir = str(
                Path(__file__).resolve().parents[2] / "related_papers" / "VBench"
            )
        self.vbench_dir = Path(vbench_dir)

        try:
            import sys
            sys.path.insert(0, str(self.vbench_dir))
            from vbench import VBench
            self.vbench = VBench(device=str(device), full_json_dir=str(self.vbench_dir / "vbench"))
            self.available = True
        except (ImportError, Exception) as e:
            logger.warning(f"VBench not available: {e}. Falling back to basic metrics.")
            self.available = False

    def evaluate(
        self,
        video_paths: List[str],
        prompts: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate videos using VBench dimensions.

        Args:
            video_paths: list of paths to generated video files
            prompts: optional text prompts used for generation

        Returns:
            dict mapping dimension names to scores
        """
        if not self.available:
            return self._basic_evaluation(video_paths)

        results = {}
        for dimension in self.dimensions:
            try:
                score = self.vbench.evaluate(
                    videos_path=video_paths,
                    name=dimension,
                    prompt_list=prompts,
                    dimension_list=[dimension],
                )
                if isinstance(score, dict):
                    results[dimension] = score.get(dimension, 0.0)
                else:
                    results[dimension] = float(score)
            except Exception as e:
                logger.warning(f"VBench dimension '{dimension}' failed: {e}")
                results[dimension] = -1.0

        return results

    def _basic_evaluation(self, video_paths: List[str]) -> Dict[str, float]:
        """Fallback evaluation when VBench is not available."""
        results = {}

        for path in video_paths:
            cap = cv2.VideoCapture(path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            if len(frames) < 2:
                continue

            # Temporal flickering: frame-to-frame pixel difference
            diffs = []
            for i in range(1, len(frames)):
                diff = np.mean(np.abs(frames[i].astype(float) - frames[i - 1].astype(float)))
                diffs.append(diff)
            results.setdefault("temporal_flickering_raw", []).append(np.mean(diffs))

            # Motion smoothness: variance of optical flow magnitudes
            gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
            flow_mags = []
            for i in range(1, min(len(gray_frames), 10)):
                flow = cv2.calcOpticalFlowFarneback(
                    gray_frames[i - 1], gray_frames[i],
                    None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
                flow_mags.append(np.mean(mag))
            results.setdefault("motion_smoothness_raw", []).append(np.std(flow_mags))

        # Aggregate
        final = {}
        for key, values in results.items():
            final[key.replace("_raw", "")] = float(np.mean(values))

        return final


class DirectorEvaluator:
    """
    Combined evaluator running all DIRECTOR metrics.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
        vbench_dir: Optional[str] = None,
    ):
        self.device = device
        self.identity_metric = IdentityConsistencyMetric(device=device)
        self.motion_metric = MotionCoherenceMetric()
        self.efficiency_metric = EfficiencyMetric(device=device)
        self.vbench_evaluator = VBenchEvaluator(vbench_dir=vbench_dir, device=device)

    def evaluate_shots(
        self,
        shots: List[torch.Tensor],
        video_paths: Optional[List[str]] = None,
        prompts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run all evaluation metrics on generated shots.

        Args:
            shots: list of (1, T, 3, H, W) video tensors
            video_paths: optional saved video paths for VBench
            prompts: optional prompts used for generation

        Returns:
            dict with all metric results
        """
        results = {}

        # Identity consistency
        logger.info("Computing identity consistency...")
        identity_results = self.identity_metric.compute(shots)
        results["identity"] = identity_results

        # Motion coherence
        logger.info("Computing motion coherence...")
        motion_results = self.motion_metric.compute(shots)
        results["motion"] = motion_results

        # VBench (if video paths available)
        if video_paths:
            logger.info("Running VBench evaluation...")
            vbench_results = self.vbench_evaluator.evaluate(video_paths, prompts)
            results["vbench"] = vbench_results

        # Summary
        results["summary"] = {
            "identity_consistency": identity_results.get("mean_similarity", 0.0),
            "motion_coherence_psnr": motion_results.get("boundary_psnr", 0.0),
            "flow_smoothness": motion_results.get("flow_smoothness", 0.0),
        }

        return results
