#!/usr/bin/env python3
"""
Convert nextgqa_grounding_val_test.jsonl (from build_grounding_manifest.py) into LLaVA
training JSON / JSONL with assistant replies that include <time_start> <t###> <time_end> <t###>.

Example:
  python nextgqa/jsonl_to_llava_grounding.py \\
    --manifest nextgqa/nextgqa_grounding_val_test.jsonl \\
    --output-json nextgqa/nextgqa_llava_grounding.json \\
    --num-bins 1000

  # Val only (3358 samples) for faster Colab runs:
  python nextgqa/jsonl_to_llava_grounding.py --splits val \\
    --output-json nextgqa/nextgqa_llava_grounding_val_only.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_TIME_END_TOKEN, DEFAULT_TIME_START_TOKEN
from llava.temporal_tokens import time_bin_vocab_list
from llava.time_quantization import segment_range_to_bins


def build_gpt_value(answer: str, start_bin: int, end_bin: int, num_bins: int) -> str:
    """Answer text first, then temporal bin span."""
    bins = time_bin_vocab_list(num_bins)
    ts, te = bins[start_bin], bins[end_bin]
    return (
        f"{answer.strip()} "
        f"Temporally, the relevant segment is {DEFAULT_TIME_START_TOKEN} {ts} "
        f"{DEFAULT_TIME_END_TOKEN} {te}."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path(__file__).parent / "nextgqa_grounding_val_test.jsonl")
    p.add_argument("--output-json", type=Path, default=Path(__file__).parent / "nextgqa_llava_grounding.json")
    p.add_argument("--output-jsonl", type=Path, default=None, help="Optional second copy as JSONL.")
    p.add_argument("--num-bins", type=int, default=1000)
    p.add_argument(
        "--splits",
        type=str,
        default="val,test",
        help="Comma-separated manifest splits to include: val, test (default both). Use e.g. 'val' only to save Colab time.",
    )
    p.add_argument(
        "--add-time-instruction-hint",
        action="store_true",
        help="If set, prepend a short hint that bins index the full video duration (for use with add_time_instruction).",
    )
    args = p.parse_args()

    allowed_splits = {s.strip().lower() for s in args.splits.split(",") if s.strip()}
    for s in allowed_splits:
        if s not in ("val", "test"):
            raise ValueError(f"Invalid split '{s}' in --splits; use val and/or test.")

    records_out = []
    with args.manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("split", "")).lower() not in allowed_splits:
                continue
            dur = float(row["duration_sec"])
            segs = row["temporal_segments_sec"]
            ib, ie = segment_range_to_bins(segs, dur, args.num_bins)
            vid_rel = row["video_relpath"].replace("\\", "/")

            q = row["question"].strip()
            human = f"{DEFAULT_IMAGE_TOKEN}\n{q}"
            if args.add_time_instruction_hint:
                human = (
                    f"{DEFAULT_IMAGE_TOKEN}\n"
                    f"This video has duration {dur:.2f} seconds in total; "
                    f"time bins are 0..{args.num_bins - 1} uniformly over that duration.\n"
                    f"{q}"
                )

            gpt = build_gpt_value(row["answer"], ib, ie, args.num_bins)
            item = {
                "id": f"{row['split']}_{row['video_id']}_{row['qid']}",
                "video": vid_rel,
                "duration_sec": dur,
                "video_id": row["video_id"],
                "qid": row["qid"],
                "split": row["split"],
                "conversations": [
                    {"from": "human", "value": human},
                    {"from": "gpt", "value": gpt},
                ],
            }
            records_out.append(item)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(records_out, f, ensure_ascii=False, indent=2)

    if args.output_jsonl:
        with args.output_jsonl.open("w", encoding="utf-8") as f:
            for item in records_out:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    cnt = Counter(str(x.get("split")) for x in records_out)
    print(f"Wrote {len(records_out)} samples to {args.output_json} (by split: {dict(cnt)})")
    if args.output_jsonl:
        print(f"Wrote JSONL to {args.output_jsonl}")


if __name__ == "__main__":
    main()
