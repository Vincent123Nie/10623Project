from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _valid_segment(value: object) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and _is_finite_number(value[0])
        and _is_finite_number(value[1])
    )


def interval_iou(prediction: tuple[float, float], target: tuple[float, float]) -> float:
    pred_start, pred_end = sorted(prediction)
    target_start, target_end = sorted(target)
    intersection = max(0.0, min(pred_end, target_end) - max(pred_start, target_start))
    union = max(pred_end, target_end) - min(pred_start, target_start)
    return intersection / union if union > 0 else 0.0


def evaluate_records(records: Iterable[dict]) -> dict[str, float | int]:
    rows = list(records)
    ious: list[float] = []
    parseable = 0

    for row in rows:
        start = row.get("predictedStart", row.get("predicted_start"))
        end = row.get("predictedEnd", row.get("predicted_end"))
        targets = row.get("temporal_segments_sec") or row.get("targets") or []

        if _is_finite_number(start) and _is_finite_number(end) and end > start:
            parseable += 1
            valid_targets = [segment for segment in targets if _valid_segment(segment)]
            ious.append(max((interval_iou((start, end), (segment[0], segment[1])) for segment in valid_targets), default=0.0))
        else:
            ious.append(0.0)

    count = len(rows)
    return {
        "examples": count,
        "parseability": parseable / count if count else 0.0,
        "mIoU": sum(ious) / count if count else 0.0,
        "R@0.3": sum(score >= 0.3 for score in ious) / count if count else 0.0,
        "R@0.5": sum(score >= 0.5 for score in ious) / count if count else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VibeClip temporal-grounding JSONL predictions.")
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()

    with args.predictions.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    print(json.dumps(evaluate_records(records), indent=2))


if __name__ == "__main__":
    main()
