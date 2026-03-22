#!/usr/bin/env python3
"""Create a mini overfit dataset with rich VLM captions.

Selects representative samples from DIRECTOR-10K, re-captions them
with Qwen2-VL for rich descriptive text, and packages into a mini dataset.

Usage:
    python overfit_test/create_mini_dataset.py
"""

import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SRC_DIR = PROJECT_ROOT / "data" / "director_10k" / "output"
OUT_DIR = PROJECT_ROOT / "overfit_test" / "mini_dataset"

# Selected samples: seq_id -> desired overfit role
SELECTED_SAMPLES = [
    # A-category: zoom-in / forward (most common, 1094 samples)
    {"src": "seq_00021", "role": "zoom_in_A",     "category": "A", "camera": "forward"},
    # A-category: stationary (102 samples)
    {"src": "seq_00013", "role": "stationary_A",   "category": "A", "camera": "stationary"},
    # E-category: pan-left (980 samples)
    {"src": "seq_00018", "role": "pan_left_E",     "category": "E", "camera": "left,forward"},
    # E-category: pan-right (1033 samples)
    {"src": "seq_00028", "role": "pan_right_E",    "category": "E", "camera": "right,forward"},
    # E-category: zoom-out / backward (very rare, 1 sample)
    {"src": "seq_01914", "role": "zoom_out_E",     "category": "E", "camera": "backward"},
    # B-category: cross-cut (2000 samples)
    {"src": "seq_04323", "role": "crosscut_B",     "category": "B", "camera": "cross-cut"},
    # Additional B for variety
    {"src": "seq_05000", "role": "crosscut_B2",    "category": "B", "camera": "cross-cut"},
]


def load_vlm():
    """Load Qwen2-VL for rich captioning."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    print(f"Loading VLM: {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
    )
    model.eval()
    print("VLM loaded.")
    return model, processor


@torch.no_grad()
def caption_with_vlm(model, processor, frame_rgb: np.ndarray, anchor_rgb: np.ndarray,
                     camera_motion: str, category: str) -> dict:
    """Generate rich structured caption using VLM."""
    pil_frame = Image.fromarray(frame_rgb)
    pil_anchor = Image.fromarray(anchor_rgb)

    # Compose prompt based on category
    if category == "B":
        system = (
            "You are a professional cinematographer. You are analyzing a cross-cut transition in a video. "
            "A NEW character has appeared in the scene, replacing the previous one. "
            "Describe what you see in rich cinematic detail."
        )
        user_text = (
            "The first image is the CHARACTER REFERENCE (the new character who appears). "
            "The second image is a FRAME from the video showing this character. "
            "Write a JSON with:\n"
            '  "identity": Describe the character\'s appearance in detail (hair, clothing, build, distinguishing features).\n'
            '  "motion": Describe the scene, what the character is doing, and the camera movement. '
            "Be specific about actions and environment.\n"
            '  "transition": Describe this as a cross-cut transition where a different character appears.'
        )
    else:
        camera_desc = {
            "forward": "zooming in / dollying forward",
            "stationary": "static / locked off",
            "left,forward": "panning left while moving forward",
            "right,forward": "panning right while moving forward",
            "backward": "zooming out / pulling back",
        }.get(camera_motion, camera_motion)

        system = (
            "You are a professional cinematographer. "
            "Describe what you see in rich cinematic detail."
        )
        user_text = (
            "The first image is the CHARACTER REFERENCE (the main person in the scene). "
            "The second image is a FRAME from a video. "
            f"The camera is {camera_desc}. "
            "Write a JSON with:\n"
            '  "identity": Describe the character\'s appearance in detail (hair, clothing, build, distinguishing features).\n'
            '  "motion": Describe the scene environment, what the character is doing, '
            f"and the {camera_desc} camera movement. Be specific and cinematic."
        )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "image", "image": pil_anchor},
            {"type": "image", "image": pil_frame},
            {"type": "text", "text": user_text},
        ]},
    ]

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_prompt],
        images=[pil_anchor, pil_frame],
        return_tensors="pt",
        padding=True,
    ).to("cuda:0")

    output_ids = model.generate(**inputs, max_new_tokens=400, do_sample=False)
    gen_ids = output_ids[:, inputs.input_ids.shape[1]:]
    text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

    # Parse JSON from VLM output
    caption = _parse_vlm_json(text, camera_motion, category)
    return caption


def _parse_vlm_json(text: str, camera_motion: str, category: str) -> dict:
    """Parse VLM output into structured caption."""
    import re
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            identity = data.get("identity", "")
            motion = data.get("motion", "")
            transition = data.get("transition", "")
            # Handle nested dicts from VLM
            if isinstance(identity, dict):
                identity = " ".join(str(v) for v in identity.values())
            if isinstance(motion, dict):
                motion = " ".join(str(v) for v in motion.values())
            if isinstance(transition, dict):
                transition = " ".join(str(v) for v in transition.values())
            identity = str(identity).strip()
            motion = str(motion).strip()
            transition = str(transition).strip()

            if identity and motion:
                full = f"{identity} {motion}"
                if transition:
                    full = f"{transition} {identity} {motion}"
                return {"identity": identity, "motion": motion, "full": full}
    except json.JSONDecodeError:
        pass

    # Fallback: use the raw text
    text = text.strip()
    if not text:
        if category == "B":
            return {
                "identity": "A different character appears in the scene.",
                "motion": "Cross-cut transition to a new scene.",
                "full": "Cross-cut transition: a different character appears in a new scene."
            }
        return {
            "identity": "A person in the scene.",
            "motion": f"Camera: {camera_motion}",
            "full": f"A person in the scene. Camera: {camera_motion}"
        }

    return {"identity": text[:200], "motion": f"Camera: {camera_motion}", "full": text[:400]}


def extract_middle_frame(video_path: str) -> np.ndarray:
    """Extract the middle frame from a video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid = total // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Cannot read frame from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load VLM
    model, processor = load_vlm()

    metadata_entries = []

    for idx, sample in enumerate(SELECTED_SAMPLES, start=1):
        src_dir = SRC_DIR / sample["src"]
        new_seq_id = f"seq_{idx:05d}"
        dst_dir = OUT_DIR / new_seq_id

        print(f"\n{'='*60}")
        print(f"[{idx}/{len(SELECTED_SAMPLES)}] {sample['src']} -> {new_seq_id}")
        print(f"  Role: {sample['role']}, Category: {sample['category']}, Camera: {sample['camera']}")

        if not src_dir.exists():
            print(f"  WARNING: {src_dir} not found, skipping!")
            continue

        # Copy files
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)

        # Extract frames for VLM captioning
        video_path = str(dst_dir / "target_shot.mp4")
        mid_frame = extract_middle_frame(video_path)

        # Load anchor for VLM context
        anchor_path = dst_dir / "global_anchor_0.png"
        anchor_img = Image.open(str(anchor_path)).convert("RGBA")
        anchor_arr = np.array(anchor_img)
        # Composite on white for VLM
        alpha = anchor_arr[:, :, 3:4].astype(np.float32) / 255.0
        anchor_rgb = (anchor_arr[:, :, :3].astype(np.float32) * alpha +
                      255.0 * (1 - alpha)).astype(np.uint8)

        # Generate rich caption with VLM
        print("  Generating VLM caption...")
        caption = caption_with_vlm(
            model, processor, mid_frame, anchor_rgb,
            sample["camera"], sample["category"]
        )

        print(f"  Identity: {caption['identity'][:100]}...")
        print(f"  Motion:   {caption['motion'][:100]}...")
        print(f"  Full:     {caption['full'][:120]}...")

        # Save new caption
        with open(dst_dir / "caption.json", "w") as f:
            json.dump(caption, f, indent=2, ensure_ascii=False)

        # Metadata entry
        metadata_entries.append({
            "seq_id": new_seq_id,
            "original_seq_id": sample["src"],
            "source": "spatialvid",
            "category": sample["category"],
            "camera_motion": sample["camera"],
            "role": sample["role"],
            "num_characters": 1,
        })

    # Write metadata.jsonl
    meta_path = OUT_DIR / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\n{'='*60}")
    print(f"Mini dataset created: {OUT_DIR}")
    print(f"  {len(metadata_entries)} sequences")
    print(f"  Metadata: {meta_path}")

    # Cleanup VLM
    del model, processor
    torch.cuda.empty_cache()

    # Print all captions for review
    print(f"\n{'='*60}")
    print("=== ALL CAPTIONS ===")
    for idx, sample in enumerate(SELECTED_SAMPLES, start=1):
        new_seq_id = f"seq_{idx:05d}"
        cap_path = OUT_DIR / new_seq_id / "caption.json"
        if cap_path.exists():
            with open(cap_path) as f:
                cap = json.load(f)
            print(f"\n[{new_seq_id}] {sample['role']} ({sample['category']}):")
            print(f"  Full: {cap['full']}")


if __name__ == "__main__":
    main()
