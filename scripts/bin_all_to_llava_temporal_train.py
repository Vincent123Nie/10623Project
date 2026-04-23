#!/usr/bin/env python3
"""
Convert ``bin_all.json`` (``<start> [i] <end> [j]`` + ``<video>`` prompts) into
LLaVA temporal-LoRA training JSON with **vocabulary-aligned** bin tokens:

  <time_start> <t###> <time_end> <t###>

Assistant text: explanation tail (old span removed) + the same pattern as
``nextgqa/jsonl_to_llava_grounding.py``:

  {tail} Temporally, the relevant segment is <time_start> <tI> <time_end> <tJ>.

``start_bin`` / ``end_bin`` in JSON are authoritative.

Example:

  python scripts/bin_all_to_llava_temporal_train.py \\
    --in-json Dataset/bin_all.json \\
    --out-json Dataset/bin_all_llava_temporal_train.json \\
    --num-bins 1000
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_TIME_END_TOKEN, DEFAULT_TIME_START_TOKEN
from llava.temporal_tokens import time_bin_vocab_list
from llava.time_quantization import segment_range_to_bins

# Strip legacy bin_all assistant lead-in through ``<end> [n].``
_BIN_ALL_HEAD = re.compile(
    r"^(?:The event occurs at\s*)?<start>\s*\[\d+\]\s*<end>\s*\[\d+\]\.\s*",
    re.IGNORECASE | re.DOTALL,
)


def _extract_gpt_tail(conv_gpt: str) -> str:
    t = (conv_gpt or "").strip()
    t = _BIN_ALL_HEAD.sub("", t, count=1)
    return t.strip()


def _convert_item(row: Dict[str, Any], num_bins: int) -> Dict[str, Any]:
    sb = int(row["start_bin"])
    eb = int(row["end_bin"])
    dur = float(row["video_duration"])
    if sb < 0 or sb >= num_bins or eb < 0 or eb >= num_bins:
        s_sec = float(row.get("start_sec", 0.0))
        e_sec = float(row.get("end_sec", 0.0))
        sb, eb = segment_range_to_bins([[s_sec, e_sec]], dur, num_bins)

    bins = time_bin_vocab_list(num_bins)
    ts, te = bins[sb], bins[eb]

    raw_gpt = next(
        (c["value"] for c in row["conversations"] if c.get("from") == "gpt"),
        "",
    )
    tail = _extract_gpt_tail(raw_gpt)
    if not tail:
        tail = "The event is localized as follows."
    gpt_value = (
        f"{tail} "
        f"Temporally, the relevant segment is {DEFAULT_TIME_START_TOKEN} {ts} "
        f"{DEFAULT_TIME_END_TOKEN} {te}."
    )

    conv: List[Dict[str, str]] = []
    for c in row["conversations"]:
        if c.get("from") == "human":
            hv = c.get("value", "")
            if "<video>" in hv:
                hv = hv.replace("<video>", DEFAULT_IMAGE_TOKEN, 1)
            conv.append({"from": "human", "value": hv})
        elif c.get("from") == "gpt":
            conv.append({"from": "gpt", "value": gpt_value})
        else:
            conv.append(dict(c))

    out = {k: v for k, v in row.items() if k != "conversations"}
    out["conversations"] = conv
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-json", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--num-bins", type=int, default=1000)
    args = p.parse_args()

    with args.in_json.open("r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    out_list = [_convert_item(r, args.num_bins) for r in data]
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(out_list, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(out_list)} items to {args.out_json}")


if __name__ == "__main__":
    main()
