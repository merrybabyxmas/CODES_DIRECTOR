#!/usr/bin/env python3
"""Create a dataset where pairs share identical captions.

Forces the model to use visual context (prev_frame + anchor) to distinguish
videos, since text alone is ambiguous.

Layout:
  - 6 samples total: 3 pairs × 2 videos each
  - Each pair shares the exact same caption
  - Videos within each pair have different visual content (different people, scenes)

This experiment proves the gate must open: with identical text, the only way
to reduce loss is to look at the visual context.
"""

import json
import shutil
from pathlib import Path

SRC = Path("overfit_test/mini_dataset")
DST = Path("overfit_test/shared_caption_dataset")

# Pair up samples with a shared generic caption
# Each pair: (sample_a, sample_b, shared_caption)
PAIRS = [
    {
        "a": "seq_00001",  # Blue puffer jacket person, zoom-in
        "b": "seq_00004",  # Dark jacket person, cobblestone alley
        "caption": {
            "identity": "A person walks through an urban environment.",
            "motion": "The camera moves forward, following the person as they walk.",
            "full": "A person walks through an urban environment. The camera moves forward, following the person as they walk."
        },
    },
    {
        "a": "seq_00003",  # Trench coat, dimly lit street
        "b": "seq_00005",  # Casual clothes, sunlit square
        "caption": {
            "identity": "A person stands in an outdoor setting.",
            "motion": "The camera pans across the scene, capturing the environment.",
            "full": "A person stands in an outdoor setting. The camera pans across the scene, capturing the environment."
        },
    },
    {
        "a": "seq_00006",  # Blue polo, cross-cut
        "b": "seq_00007",  # Dark jacket, city street
        "caption": {
            "identity": "A man appears in a new scene after a cut.",
            "motion": "Cross-cut transition to a different location with a new character.",
            "full": "A man appears in a new scene after a cut. Cross-cut transition to a different location with a new character."
        },
    },
]


def main():
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    metadata = []
    seq_idx = 0

    for pair_idx, pair in enumerate(PAIRS):
        for which, src_key in [("a", "a"), ("b", "b")]:
            seq_idx += 1
            src_dir = SRC / pair[src_key]
            new_id = f"seq_{seq_idx:05d}"
            dst_dir = DST / new_id

            if not src_dir.exists():
                print(f"WARNING: {src_dir} not found, skipping")
                continue

            # Copy all files
            shutil.copytree(src_dir, dst_dir)

            # Overwrite caption with shared caption
            with open(dst_dir / "caption.json", "w") as f:
                json.dump(pair["caption"], f, indent=2)

            metadata.append({
                "seq_id": new_id,
                "original_seq_id": pair[src_key],
                "pair_idx": pair_idx,
                "pair_role": which,
                "source": "shared_caption_test",
                "category": "A" if pair_idx < 2 else "B",
                "camera_motion": "forward",
                "num_characters": 1,
            })

            print(f"  {new_id} <- {pair[src_key]} (pair {pair_idx}, {which})")

    # Write metadata
    with open(DST / "metadata.jsonl", "w") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")

    print(f"\nCreated {len(metadata)} samples in {DST}")
    print(f"  {len(PAIRS)} pairs with shared captions")
    print(f"  Model MUST use visual context to distinguish within each pair")

    # Verify
    print("\nCaption verification:")
    for i in range(0, len(metadata), 2):
        a_cap = json.load(open(DST / metadata[i]["seq_id"] / "caption.json"))
        b_cap = json.load(open(DST / metadata[i+1]["seq_id"] / "caption.json"))
        match = "IDENTICAL" if a_cap["full"] == b_cap["full"] else "DIFFERENT"
        print(f"  Pair {i//2}: {metadata[i]['seq_id']} vs {metadata[i+1]['seq_id']} -> {match}")


if __name__ == "__main__":
    main()
