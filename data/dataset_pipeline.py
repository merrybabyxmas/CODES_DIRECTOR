"""
DIRECTOR Dataset Construction Pipeline (Production Spec v2).

Full pipeline for constructing training triplets from raw video:
  Step 1: Shot boundary detection (TransNetV2) + identity filtering (YOLOv8 + DINOv2)
  Step 2: Global anchor extraction (best frame selection + SAM2 segmentation)
  Step 3: Decoupled captioning (VLM or template fallback)
  Step 4: Triplet packaging into flat seq_XXXXX/ structure

Output structure:
  data/processed_dataset/
  ├── seq_00001/
  │   ├── global_anchor.png              # RGBA, character with transparent bg
  │   ├── prev_shot_last_frame.jpg        # Last frame of S_{t-1}
  │   ├── prev_prev_shot_last_frame.jpg   # Last frame of S_{t-2} (if available)
  │   ├── target_shot.mp4                 # S_t video clip (8fps, max 49 frames)
  │   └── caption.json                    # {"identity": "...", "motion": "...", "full": "..."}
  ├── seq_00002/
  │   └── ...
  └── metadata.jsonl                # One JSON per line

Usage:
  CUDA_VISIBLE_DEVICES=3 python -m data.dataset_pipeline \\
      --video_dir data/raw_videos --output_dir data/processed_dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Monkey-patch transformers numpy version check (numpy 2.x vs transformers <4.45)
# Must happen BEFORE importing transformers.
# ---------------------------------------------------------------------------
try:
    import transformers.utils.versions as _tv
    _orig_require = _tv.require_version
    def _patched_require(requirement, hint=None):
        if "numpy" in requirement:
            return  # skip numpy version gate
        return _orig_require(requirement, hint)
    _tv.require_version = _patched_require
    _tv.require_version_core = _patched_require
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("director.pipeline")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RELATED_PAPERS_DIR = Path(__file__).resolve().parents[2] / "related_papers"
TRANSNET_DIR = RELATED_PAPERS_DIR / "TransNetV2" / "inference-pytorch"
TRANSNET_TF_WEIGHTS_DIR = RELATED_PAPERS_DIR / "TransNetV2" / "inference" / "transnetv2-weights"
SAM2_ROOT = RELATED_PAPERS_DIR / "segment-anything-2"
SAM2_CHECKPOINT = SAM2_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

sys.path.insert(0, str(TRANSNET_DIR))
sys.path.insert(0, str(SAM2_ROOT))


# ===========================================================================
# Configuration
# ===========================================================================
@dataclass
class PipelineConfig:
    """All tuneable parameters for the dataset pipeline."""
    # TransNetV2
    shot_threshold: float = 0.5
    min_shot_length_frames: int = 24

    # YOLOv8 person detection
    yolo_model: str = "yolov8n.pt"
    yolo_person_class: int = 0
    yolo_conf_threshold: float = 0.35
    num_representative_frames: int = 5  # frames per shot for person detection

    # DINOv2 identity matching
    dino_model: str = "facebook/dinov2-base"
    identity_similarity_threshold: float = 0.75

    # Best frame selection
    face_aspect_ratio_target: float = 0.7  # ideal w/h for frontal face

    # SAM2
    sam2_config: str = SAM2_CONFIG
    sam2_checkpoint: str = str(SAM2_CHECKPOINT)

    # Captioning
    vlm_model: str = "Qwen/Qwen2-VL-2B-Instruct"  # try first
    vlm_fallback: str = "template"

    # Video output
    target_fps: int = 8
    max_frames: int = 49  # CogVideoX native

    # General
    device: str = "cuda"
    seed: int = 42


# ===========================================================================
# Step 1: Shot Boundary Detection (PySceneDetect - robust, no weights needed)
# ===========================================================================
class ShotBoundaryDetector:
    """PySceneDetect-based shot boundary detection (ContentDetector).

    Uses adaptive content-based detection which analyses pixel changes
    between consecutive frames. More robust than TransNetV2 when pre-trained
    weights are unavailable and produces high-quality results on cinematic
    content (trailers, movies).
    """

    def __init__(self, config: PipelineConfig):
        self.threshold = config.shot_threshold
        self.min_shot_length = config.min_shot_length_frames

    def detect(self, video_path: str) -> List[Tuple[int, int]]:
        """
        Detect shot boundaries using PySceneDetect ContentDetector.

        Returns:
            List of (start_frame, end_frame) per shot.
        """
        from scenedetect import open_video, SceneManager, ContentDetector

        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(
                threshold=27.0,  # default content threshold (works well for movies)
                min_scene_len=self.min_shot_length,
            )
        )
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()

        shots: List[Tuple[int, int]] = []
        for scene in scene_list:
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames() - 1  # exclusive -> inclusive
            if end_frame - start_frame >= self.min_shot_length:
                shots.append((start_frame, end_frame))

        total_frames = video.duration.get_frames()
        fps = video.frame_rate
        logger.info(
            f"  SBD: {Path(video_path).name}, frames={total_frames}, "
            f"fps={fps:.1f}, detected {len(shots)} shots"
        )
        return shots


# ===========================================================================
# Step 1 (cont.): Person Detection (YOLOv8) + Identity Filtering (DINOv2)
# ===========================================================================
class PersonDetector:
    """YOLOv8-based person bounding-box detector."""

    def __init__(self, config: PipelineConfig):
        from ultralytics import YOLO
        self.model = YOLO(config.yolo_model)
        self.person_class = config.yolo_person_class
        self.conf_threshold = config.yolo_conf_threshold

    def detect_persons(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Detect person bounding boxes in a single frame.

        Args:
            frame: (H, W, 3) uint8 BGR or RGB numpy array.

        Returns:
            List of [x1, y1, x2, y2] arrays (absolute pixel coords).
        """
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        boxes: List[np.ndarray] = []
        for r in results:
            for box_data in r.boxes:
                cls_id = int(box_data.cls.item())
                if cls_id == self.person_class:
                    xyxy = box_data.xyxy[0].cpu().numpy()  # (4,)
                    boxes.append(xyxy)
        return boxes


class IdentityMatcher:
    """DINOv2-based identity matching between person crops."""

    def __init__(self, config: PipelineConfig):
        self.threshold = config.identity_similarity_threshold
        self.device = torch.device(config.device)

        from transformers import AutoModel, AutoImageProcessor
        self.processor = AutoImageProcessor.from_pretrained(config.dino_model)
        self.model = AutoModel.from_pretrained(config.dino_model).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_features(self, crops: List[np.ndarray]) -> torch.Tensor:
        """
        Extract DINOv2 CLS features from person crops.

        Args:
            crops: list of (H, W, 3) uint8 RGB arrays.

        Returns:
            (N, D) L2-normalised feature tensor.
        """
        if len(crops) == 0:
            return torch.zeros(0, 768, device=self.device)
        pil_images = [Image.fromarray(c) for c in crops]
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        feats = outputs.last_hidden_state[:, 0]  # CLS token, (N, D)
        return F.normalize(feats, p=2, dim=-1)

    def match_consecutive(
        self,
        crops_prev: List[np.ndarray],
        crops_curr: List[np.ndarray],
    ) -> Tuple[bool, float, Optional[Tuple[int, int]]]:
        """
        Check if any person in crops_prev matches any person in crops_curr.

        Returns:
            (is_match, best_similarity, (prev_idx, curr_idx) or None)
        """
        if len(crops_prev) == 0 or len(crops_curr) == 0:
            return False, 0.0, None

        feats_prev = self.extract_features(crops_prev)  # (Np, D)
        feats_curr = self.extract_features(crops_curr)  # (Nc, D)
        sim_matrix = feats_prev @ feats_curr.T  # (Np, Nc)
        best_val = sim_matrix.max().item()
        if best_val > self.threshold:
            idx = sim_matrix.argmax().item()
            pi = idx // sim_matrix.shape[1]
            ci = idx % sim_matrix.shape[1]
            return True, best_val, (pi, ci)
        return False, best_val, None


# ===========================================================================
# Step 2: Global Anchor Extraction (Best Frame + SAM2)
# ===========================================================================
class GlobalAnchorExtractor:
    """Select the best frame and segment the character with SAM2."""

    def __init__(self, config: PipelineConfig):
        self.device = torch.device(config.device)
        self.face_ar_target = config.face_aspect_ratio_target

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_model = build_sam2(
            config.sam2_config,
            config.sam2_checkpoint,
            device=config.device,
        )
        self.predictor = SAM2ImagePredictor(sam2_model)
        logger.info("SAM2 image predictor initialised")

    def select_best_frame(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        person_detector: PersonDetector,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Search all frames in [start_frame, end_frame] for the best person frame.

        Scoring criteria (weighted sum):
          - Person bbox area (larger = better)          weight 0.4
          - Frontality heuristic (bbox aspect ratio)    weight 0.3
          - Frame sharpness (Laplacian variance)        weight 0.3

        Returns:
            (best_frame_rgb, best_bbox_xyxy, best_frame_idx)
        """
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(video_path)

        # Sample every 3rd frame to balance speed vs coverage
        stride = max(1, (end_frame - start_frame) // 50)
        candidate_indices = list(range(start_frame, min(end_frame + 1, len(vr)), stride))
        if not candidate_indices:
            candidate_indices = [start_frame]

        best_score = -1.0
        best_frame: Optional[np.ndarray] = None
        best_bbox: Optional[np.ndarray] = None
        best_idx = candidate_indices[0]

        # Process in mini-batches for memory efficiency
        batch_size = 16
        for batch_start in range(0, len(candidate_indices), batch_size):
            batch_indices = candidate_indices[batch_start:batch_start + batch_size]
            frames = vr.get_batch(batch_indices).numpy().astype(np.uint8)  # (B, H, W, 3)

            for local_i, frame_rgb in enumerate(frames):
                global_idx = batch_indices[local_i]
                boxes = person_detector.detect_persons(frame_rgb)
                if len(boxes) == 0:
                    continue

                frame_h, frame_w = frame_rgb.shape[:2]
                total_area = float(frame_h * frame_w)

                # Sharpness: Laplacian variance (computed once per frame)
                gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

                for box in boxes:
                    x1, y1, x2, y2 = box
                    bw = x2 - x1
                    bh = y2 - y1
                    area = bw * bh

                    # Area score: fraction of image
                    area_score = area / total_area

                    # Frontality heuristic: aspect ratio closeness to target
                    aspect_ratio = bw / max(bh, 1.0)
                    frontality_score = 1.0 - min(abs(aspect_ratio - self.face_ar_target), 1.0)

                    # Sharpness score: normalise heuristically (log scale)
                    sharp_score = min(np.log1p(sharpness) / 10.0, 1.0)

                    combined = 0.4 * area_score + 0.3 * frontality_score + 0.3 * sharp_score

                    if combined > best_score:
                        best_score = combined
                        best_frame = frame_rgb.copy()
                        best_bbox = np.array([x1, y1, x2, y2])
                        best_idx = global_idx

        if best_frame is None:
            # Fallback: use the middle frame with a centre crop bbox
            mid_idx = (start_frame + end_frame) // 2
            mid_idx = min(mid_idx, len(vr) - 1)
            best_frame = vr[mid_idx].numpy().astype(np.uint8)
            h, w = best_frame.shape[:2]
            best_bbox = np.array([w * 0.2, h * 0.1, w * 0.8, h * 0.9])
            best_idx = mid_idx

        return best_frame, best_bbox, best_idx

    def segment_character(self, frame_rgb: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
        """
        Use SAM2 with bbox prompt to segment the person.

        Args:
            frame_rgb: (H, W, 3) uint8 RGB.
            bbox_xyxy: (4,) array [x1, y1, x2, y2].

        Returns:
            RGBA image (H, W, 4) uint8 with transparent background.
        """
        self.predictor.set_image(frame_rgb)
        masks, scores, _ = self.predictor.predict(
            box=bbox_xyxy,
            multimask_output=False,
        )
        # masks: (1, H, W) bool
        mask = masks[0]  # (H, W)

        # Create RGBA
        alpha = (mask.astype(np.uint8) * 255)[:, :, None]  # (H, W, 1)
        rgba = np.concatenate([frame_rgb, alpha], axis=2)  # (H, W, 4)

        # Crop to bounding box of mask (with small padding)
        ys, xs = np.where(mask)
        if len(ys) > 0:
            pad = 10
            y_min = max(ys.min() - pad, 0)
            y_max = min(ys.max() + pad, frame_rgb.shape[0])
            x_min = max(xs.min() - pad, 0)
            x_max = min(xs.max() + pad, frame_rgb.shape[1])
            rgba = rgba[y_min:y_max, x_min:x_max]

        return rgba


# ===========================================================================
# Step 3: Decoupled Captioning
# ===========================================================================
CAPTION_SYSTEM_PROMPT = """You are a professional cinematographer. Describe the following video frame.
You MUST output ONLY a valid JSON object with exactly two keys:
1. "identity": Describe the main person's physical appearance, clothing, and gender. Do not describe actions.
2. "motion": Describe the camera movement (e.g., zoom-in, panning), the person's action, and the interaction with the environment."""


class DecoupledCaptioner:
    """VLM-based or template-based decoupled captioning."""

    def __init__(self, config: PipelineConfig):
        self.device = torch.device(config.device)
        self.vlm_available = False
        self.vlm_type: Optional[str] = None

        # Try Qwen2-VL
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            self.vlm_processor = AutoProcessor.from_pretrained(
                config.vlm_model, trust_remote_code=True
            )
            self.vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
                config.vlm_model,
                torch_dtype=torch.float16,
                device_map={"": self.device},
                trust_remote_code=True,
            )
            self.vlm_model.eval()
            self.vlm_available = True
            self.vlm_type = "qwen2vl"
            logger.info(f"Loaded VLM: {config.vlm_model}")
        except Exception as e:
            logger.warning(f"Qwen2-VL not available ({e}). Falling back to template captioner.")

    @torch.no_grad()
    def caption_frame(self, frame_rgb: np.ndarray) -> Dict[str, str]:
        """
        Generate decoupled caption for a single frame.

        Returns:
            {"identity": "...", "motion": "...", "full": "..."}
        """
        if self.vlm_available and self.vlm_type == "qwen2vl":
            return self._caption_qwen2vl(frame_rgb)
        return self._template_caption(frame_rgb)

    def _caption_qwen2vl(self, frame_rgb: np.ndarray) -> Dict[str, str]:
        """Caption using Qwen2-VL."""
        pil_img = Image.fromarray(frame_rgb)

        messages = [
            {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": "Describe this frame."},
            ]},
        ]

        text_prompt = self.vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.vlm_processor(
            text=[text_prompt], images=[pil_img], return_tensors="pt", padding=True
        ).to(self.device)

        output_ids = self.vlm_model.generate(**inputs, max_new_tokens=300, do_sample=False)
        # Decode only the generated portion
        gen_ids = output_ids[:, inputs.input_ids.shape[1]:]
        text = self.vlm_processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

        return self._parse_vlm_output(text)

    @staticmethod
    def _parse_vlm_output(text: str) -> Dict[str, str]:
        """Parse VLM output into identity/motion dict."""
        # Try JSON parsing
        try:
            # Find JSON object in text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                identity = data.get("identity", "")
                motion = data.get("motion", "")
                return {
                    "identity": identity,
                    "motion": motion,
                    "full": f"{identity} {motion}".strip(),
                }
        except json.JSONDecodeError:
            pass

        # Fallback: treat entire text as full caption
        return {"identity": text, "motion": "", "full": text}

    @staticmethod
    def _template_caption(frame_rgb: np.ndarray) -> Dict[str, str]:
        """
        Template-based fallback captioner.
        Uses basic heuristics from the frame to construct a reasonable caption.
        """
        h, w = frame_rgb.shape[:2]
        aspect = w / max(h, 1)

        # Estimate brightness
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        mean_brightness = gray.mean()

        # Lighting description
        if mean_brightness < 80:
            lighting = "dimly lit"
        elif mean_brightness > 180:
            lighting = "brightly lit"
        else:
            lighting = "moderately lit"

        # Shot type heuristic from aspect ratio
        if aspect > 2.0:
            shot_type = "wide cinematic shot"
        elif aspect > 1.5:
            shot_type = "medium shot"
        else:
            shot_type = "close-up shot"

        # Simple motion estimation via frame gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_density = (np.sqrt(sobelx**2 + sobely**2) > 30).mean()

        if edge_density > 0.3:
            scene_desc = "a detailed, textured environment"
        else:
            scene_desc = "a scene with smooth backgrounds"

        identity = f"A person in a {lighting} {shot_type}."
        motion = f"Static camera framing {scene_desc}."

        return {
            "identity": identity,
            "motion": motion,
            "full": f"{identity} {motion}",
        }


# ===========================================================================
# Step 4: Video Writer Utility
# ===========================================================================
def write_target_shot_mp4(
    video_path: str,
    start_frame: int,
    end_frame: int,
    output_path: str,
    target_fps: int = 8,
    max_frames: int = 49,
) -> int:
    """
    Extract a shot from video, subsample to target_fps, write as MP4.

    Returns:
        Number of frames written.
    """
    import decord
    decord.bridge.set_bridge("torch")

    vr = decord.VideoReader(video_path)
    src_fps = vr.get_avg_fps()
    total = len(vr)

    # Compute frame indices at target_fps
    stride = max(1, int(round(src_fps / target_fps)))
    indices = list(range(start_frame, min(end_frame + 1, total), stride))

    # Limit to max_frames
    if len(indices) > max_frames:
        step = len(indices) / max_frames
        indices = [indices[int(i * step)] for i in range(max_frames)]

    if len(indices) == 0:
        indices = [start_frame]

    frames = vr.get_batch(indices).numpy().astype(np.uint8)  # (T, H, W, 3) RGB
    h, w = frames.shape[1], frames.shape[2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, target_fps, (w, h))

    for frame_rgb in frames:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    return len(frames)


# ===========================================================================
# Main Pipeline
# ===========================================================================
class DatasetPipeline:
    """
    End-to-end dataset pipeline producing flat seq_XXXXX/ triplets.

    Processes one video at a time, clears GPU cache between steps.
    Resumable: skips already-processed sequences.
    """

    def __init__(self, config: PipelineConfig, video_dir: str, output_dir: str):
        self.config = config
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(config.device)

        # Global sequence counter (read from existing metadata.jsonl)
        self.metadata_path = self.output_dir / "metadata.jsonl"
        self.seq_counter = self._count_existing_sequences()
        self._processed_videos = self._load_processed_videos()

        # Lazy-initialised components (loaded on first use, freed between videos)
        self._shot_detector: Optional[ShotBoundaryDetector] = None
        self._person_detector: Optional[PersonDetector] = None
        self._identity_matcher: Optional[IdentityMatcher] = None
        self._anchor_extractor: Optional[GlobalAnchorExtractor] = None
        self._captioner: Optional[DecoupledCaptioner] = None

    # -- Lazy component loading -------------------------------------------

    def _get_shot_detector(self) -> ShotBoundaryDetector:
        if self._shot_detector is None:
            self._shot_detector = ShotBoundaryDetector(self.config)
        return self._shot_detector

    def _get_person_detector(self) -> PersonDetector:
        if self._person_detector is None:
            self._person_detector = PersonDetector(self.config)
        return self._person_detector

    def _get_identity_matcher(self) -> IdentityMatcher:
        if self._identity_matcher is None:
            self._identity_matcher = IdentityMatcher(self.config)
        return self._identity_matcher

    def _get_anchor_extractor(self) -> GlobalAnchorExtractor:
        if self._anchor_extractor is None:
            self._anchor_extractor = GlobalAnchorExtractor(self.config)
        return self._anchor_extractor

    def _get_captioner(self) -> DecoupledCaptioner:
        if self._captioner is None:
            self._captioner = DecoupledCaptioner(self.config)
        return self._captioner

    def _unload_component(self, attr: str) -> None:
        """Unload a component and free GPU memory."""
        obj = getattr(self, attr, None)
        if obj is not None:
            del obj
            setattr(self, attr, None)
            torch.cuda.empty_cache()

    # -- Resumability helpers ---------------------------------------------

    def _count_existing_sequences(self) -> int:
        if not self.metadata_path.exists():
            return 0
        count = 0
        with open(self.metadata_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    count += 1
        return count

    def _load_processed_videos(self) -> set:
        processed = set()
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            processed.add(entry.get("video_source", ""))
                        except json.JSONDecodeError:
                            pass
        return processed

    def _next_seq_id(self) -> str:
        self.seq_counter += 1
        return f"seq_{self.seq_counter:05d}"

    def _append_metadata(self, entry: Dict[str, Any]) -> None:
        with open(self.metadata_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # -- Core pipeline ----------------------------------------------------

    def process_video(self, video_path: str) -> int:
        """
        Process a single video end-to-end.

        Returns:
            Number of valid sequences extracted.
        """
        video_name = Path(video_path).stem
        if video_path in self._processed_videos:
            logger.info(f"Skipping already-processed video: {video_name}")
            return 0

        import decord
        decord.bridge.set_bridge("torch")

        logger.info(f"{'='*60}")
        logger.info(f"Processing: {video_name}")
        logger.info(f"{'='*60}")

        vr = decord.VideoReader(video_path)
        src_fps = vr.get_avg_fps()
        total_frames = len(vr)

        # -- Step 1a: Shot boundary detection --------------------------------
        logger.info("Step 1a: Shot boundary detection ...")
        shot_detector = self._get_shot_detector()
        shots = shot_detector.detect(video_path)
        if len(shots) < 2:
            logger.warning(f"  Only {len(shots)} shot(s) found – need >= 2 for pairs. Skipping.")
            return 0

        # Free TransNetV2 from GPU
        self._unload_component("_shot_detector")

        # -- Step 1b: Person detection per shot ------------------------------
        logger.info("Step 1b: Person detection (YOLOv8) ...")
        person_detector = self._get_person_detector()

        # For each shot, detect persons on representative frames
        shot_person_crops: Dict[int, List[np.ndarray]] = {}  # shot_idx -> list of RGB crops
        shot_person_boxes: Dict[int, List[np.ndarray]] = {}  # shot_idx -> list of xyxy boxes

        for shot_idx, (s_start, s_end) in enumerate(shots):
            # Sample representative frames
            n_repr = min(self.config.num_representative_frames, s_end - s_start + 1)
            repr_indices = np.linspace(s_start, s_end, n_repr, dtype=int).tolist()
            repr_indices = [min(idx, total_frames - 1) for idx in repr_indices]

            repr_frames = vr.get_batch(repr_indices).numpy().astype(np.uint8)  # (N, H, W, 3)

            # Collect all person detections for this shot
            all_crops: List[np.ndarray] = []
            all_boxes: List[np.ndarray] = []
            for frame_rgb in repr_frames:
                boxes = person_detector.detect_persons(frame_rgb)
                for box in boxes:
                    x1, y1, x2, y2 = box.astype(int)
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(frame_rgb.shape[1], x2); y2 = min(frame_rgb.shape[0], y2)
                    if x2 - x1 < 10 or y2 - y1 < 10:
                        continue
                    crop = frame_rgb[y1:y2, x1:x2].copy()
                    all_crops.append(crop)
                    all_boxes.append(np.array([x1, y1, x2, y2]))

            shot_person_crops[shot_idx] = all_crops
            shot_person_boxes[shot_idx] = all_boxes

        logger.info(f"  Person crops per shot: {[len(shot_person_crops.get(i, [])) for i in range(len(shots))]}")

        # -- Step 1c: Identity filtering (consecutive pairs) ----------------
        logger.info("Step 1c: Identity filtering (DINOv2) ...")
        identity_matcher = self._get_identity_matcher()

        valid_pairs: List[Dict[str, Any]] = []  # list of {prev_idx, curr_idx, similarity, matched_person_boxes}

        for i in range(1, len(shots)):
            prev_crops = shot_person_crops.get(i - 1, [])
            curr_crops = shot_person_crops.get(i, [])

            is_match, sim, match_indices = identity_matcher.match_consecutive(prev_crops, curr_crops)
            if is_match and match_indices is not None:
                # Find the best bbox in current shot for anchor extraction
                curr_box = shot_person_boxes[i][match_indices[1]] if match_indices[1] < len(shot_person_boxes.get(i, [])) else None
                valid_pairs.append({
                    "prev_shot_idx": i - 1,
                    "curr_shot_idx": i,
                    "similarity": sim,
                    "curr_person_box": curr_box,
                })

        logger.info(f"  Valid consecutive pairs (sim>{self.config.identity_similarity_threshold}): {len(valid_pairs)}")

        if len(valid_pairs) == 0:
            logger.warning("  No valid identity-matched pairs found. Skipping video.")
            self._unload_component("_identity_matcher")
            return 0

        # Free DINOv2
        self._unload_component("_identity_matcher")

        # -- Step 2: Global anchor extraction --------------------------------
        logger.info("Step 2: Global anchor extraction (SAM2) ...")
        anchor_extractor = self._get_anchor_extractor()

        pair_anchors: List[Optional[np.ndarray]] = []
        for pair in valid_pairs:
            ci = pair["curr_shot_idx"]
            s_start, s_end = shots[ci]

            # Also include previous shot frames for best-frame search
            pi = pair["prev_shot_idx"]
            ps_start, ps_end = shots[pi]
            search_start = ps_start
            search_end = s_end

            best_frame, best_bbox, best_fidx = anchor_extractor.select_best_frame(
                video_path, search_start, search_end, person_detector
            )

            # Segment with SAM2
            try:
                rgba = anchor_extractor.segment_character(best_frame, best_bbox)
                pair_anchors.append(rgba)
            except Exception as e:
                logger.warning(f"  SAM2 segmentation failed for pair prev={pi} curr={ci}: {e}")
                pair_anchors.append(None)

        # Free SAM2
        self._unload_component("_anchor_extractor")

        # -- Step 3: Decoupled captioning ------------------------------------
        logger.info("Step 3: Decoupled captioning ...")
        captioner = self._get_captioner()

        pair_captions: List[Dict[str, str]] = []
        for pair in valid_pairs:
            ci = pair["curr_shot_idx"]
            s_start, s_end = shots[ci]
            mid_idx = (s_start + s_end) // 2
            mid_idx = min(mid_idx, total_frames - 1)
            mid_frame = vr[mid_idx].numpy().astype(np.uint8)  # (H, W, 3)
            caption = captioner.caption_frame(mid_frame)
            pair_captions.append(caption)

        # Free VLM
        self._unload_component("_captioner")

        # -- Step 4: Triplet packaging ---------------------------------------
        logger.info("Step 4: Triplet packaging ...")
        num_sequences = 0

        for pair_idx, pair in enumerate(valid_pairs):
            anchor_rgba = pair_anchors[pair_idx]
            caption = pair_captions[pair_idx]
            pi = pair["prev_shot_idx"]
            ci = pair["curr_shot_idx"]
            ps_start, ps_end = shots[pi]
            cs_start, cs_end = shots[ci]

            # Skip if anchor extraction failed
            if anchor_rgba is None:
                logger.warning(f"  Skipping pair {pair_idx}: no anchor")
                continue

            seq_id = self._next_seq_id()
            seq_dir = self.output_dir / seq_id
            seq_dir.mkdir(parents=True, exist_ok=True)

            # (a) Save global_anchor.png
            anchor_img = Image.fromarray(anchor_rgba, mode="RGBA")
            anchor_img.save(str(seq_dir / "global_anchor.png"))

            # (b) Save prev_shot_last_frame.jpg (t-1)
            last_frame_idx = min(ps_end, total_frames - 1)
            last_frame = vr[last_frame_idx].numpy().astype(np.uint8)  # (H, W, 3) RGB
            cv2.imwrite(
                str(seq_dir / "prev_shot_last_frame.jpg"),
                cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR),
            )

            # (b2) Save prev_prev_shot_last_frame.jpg (t-2) if available
            if pi > 0:
                pp_start, pp_end = shots[pi - 1]
                pp_last_idx = min(pp_end, total_frames - 1)
                pp_last_frame = vr[pp_last_idx].numpy().astype(np.uint8)
                cv2.imwrite(
                    str(seq_dir / "prev_prev_shot_last_frame.jpg"),
                    cv2.cvtColor(pp_last_frame, cv2.COLOR_RGB2BGR),
                )

            # (c) Save target_shot.mp4
            n_written = write_target_shot_mp4(
                video_path,
                cs_start,
                cs_end,
                str(seq_dir / "target_shot.mp4"),
                target_fps=self.config.target_fps,
                max_frames=self.config.max_frames,
            )

            # (d) Save caption.json
            with open(seq_dir / "caption.json", "w") as f:
                json.dump(caption, f, indent=2)

            # (e) Append to metadata.jsonl
            meta_entry = {
                "seq_id": seq_id,
                "video_source": video_path,
                "video_name": video_name,
                "prev_shot": {"start": ps_start, "end": ps_end, "idx": pi},
                "curr_shot": {"start": cs_start, "end": cs_end, "idx": ci},
                "identity_similarity": round(pair["similarity"], 4),
                "target_shot_frames": n_written,
                "target_fps": self.config.target_fps,
                "src_fps": round(src_fps, 2),
            }
            self._append_metadata(meta_entry)
            num_sequences += 1

        self._processed_videos.add(video_path)
        logger.info(f"  Finished {video_name}: {num_sequences} sequences extracted")
        torch.cuda.empty_cache()
        return num_sequences

    def process_all(self) -> int:
        """Process all videos in video_dir. Returns total sequences."""
        video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
        video_files = sorted([
            f for f in self.video_dir.iterdir()
            if f.suffix.lower() in video_extensions
        ])
        logger.info(f"Found {len(video_files)} video files in {self.video_dir}")

        total_seqs = 0
        for vi, vf in enumerate(video_files):
            logger.info(f"\n[{vi+1}/{len(video_files)}] {vf.name}")
            try:
                n = self.process_video(str(vf))
                total_seqs += n
            except Exception as e:
                logger.error(f"Failed to process {vf.name}: {e}", exc_info=True)

        logger.info(f"\nPipeline complete. Total sequences: {total_seqs}")
        return total_seqs


# ===========================================================================
# CLI entry point
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="DIRECTOR Dataset Construction Pipeline v2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing raw video files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for processed_dataset/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--shot_threshold", type=float, default=0.5)
    parser.add_argument("--min_shot_length", type=int, default=24)
    parser.add_argument("--identity_threshold", type=float, default=0.75)
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt")
    parser.add_argument("--yolo_conf", type=float, default=0.35)
    parser.add_argument("--dino_model", type=str, default="facebook/dinov2-base")
    parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--target_fps", type=int, default=8)
    parser.add_argument("--max_frames", type=int, default=49)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    config = PipelineConfig(
        shot_threshold=args.shot_threshold,
        min_shot_length_frames=args.min_shot_length,
        yolo_model=args.yolo_model,
        yolo_conf_threshold=args.yolo_conf,
        dino_model=args.dino_model,
        identity_similarity_threshold=args.identity_threshold,
        vlm_model=args.vlm_model,
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        device=args.device,
        seed=args.seed,
    )

    pipeline = DatasetPipeline(config, args.video_dir, args.output_dir)
    pipeline.process_all()


if __name__ == "__main__":
    main()
