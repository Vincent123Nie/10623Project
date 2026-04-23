#!/usr/bin/env python3
"""
Build a JSONL manifest for NExT-GQA val/test rows that have temporal segments in gsub.

Each line is one QA sample with:
  - video_id, qid, split
  - question, answer, type, choices (a0..a4)
  - video_relpath (from map_vid_vidorID + .mp4)
  - video_abspath (optional, if --video-root is set)
  - duration_sec, fps, temporal_segments_sec (from gsub)
  - frame_indices_by_segment (from frame2time: frames whose mapped time falls in each segment)

Usage:
  python nextgqa/build_grounding_manifest.py \\
    --video-root "D:/NExTVideo/NExTVideo" \\
    -o nextgqa/nextgqa_grounding_val_test.jsonl

If --video-root is omitted, video_abspath is null (only video_relpath is filled).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def frames_in_segments(
    frame2time: list[float] | None, segments_sec: list[list[float]]
) -> list[list[int]]:
    """For each [t0, t1] segment, return frame indices i with t0 <= time[i] <= t1."""
    if not frame2time:
        return [[] for _ in segments_sec]
    out: list[list[int]] = []
    for seg in segments_sec:
        if len(seg) < 2:
            out.append([])
            continue
        t0, t1 = float(seg[0]), float(seg[1])
        if t0 > t1:
            t0, t1 = t1, t0
        idxs = [i for i, t in enumerate(frame2time) if t0 <= float(t) <= t1]
        out.append(idxs)
    return out


def process_split(
    csv_path: Path,
    gsub: dict,
    frame2time_map: dict,
    vid_map: dict,
    video_root: Path | None,
    split_name: str,
) -> tuple[list[dict], dict[str, int]]:
    stats = {
        "rows_total": 0,
        "rows_with_temporal": 0,
        "skip_no_video_in_gsub": 0,
        "skip_no_qid_in_location": 0,
        "skip_no_map_path": 0,
    }
    records: list[dict] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["rows_total"] += 1
            vid = str(row["video_id"]).strip()
            qid_str = str(int(row["qid"]))  # normalize "01" -> "1"

            g = gsub.get(vid)
            if not g:
                stats["skip_no_video_in_gsub"] += 1
                continue
            loc = g.get("location") or {}
            if qid_str not in loc:
                stats["skip_no_qid_in_location"] += 1
                continue

            segments = loc[qid_str]
            if not segments or not isinstance(segments, list):
                stats["skip_no_qid_in_location"] += 1
                continue

            rel = vid_map.get(vid)
            if not rel:
                stats["skip_no_map_path"] += 1
                continue
            rel_path = rel if str(rel).lower().endswith(".mp4") else f"{rel}.mp4"

            abspath: str | None = None
            if video_root is not None:
                abspath = str((video_root / rel_path).resolve())

            ft = frame2time_map.get(vid)
            if ft is not None and not isinstance(ft, list):
                ft = None

            frame_indices = frames_in_segments(ft, segments)

            rec = {
                "split": split_name,
                "video_id": vid,
                "qid": int(qid_str),
                "question": row["question"],
                "answer": row["answer"],
                "type": row.get("type", ""),
                "choices": {
                    "a0": row.get("a0", ""),
                    "a1": row.get("a1", ""),
                    "a2": row.get("a2", ""),
                    "a3": row.get("a3", ""),
                    "a4": row.get("a4", ""),
                },
                "frame_count_csv": int(row["frame_count"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
                "video_relpath": rel_path.replace("\\", "/"),
                "video_abspath": abspath,
                "duration_sec": float(g["duration"]),
                "fps": float(g.get("fps", 0) or 0),
                "temporal_segments_sec": segments,
                "frame2time_available": ft is not None,
                "frame_indices_by_segment": frame_indices,
            }
            records.append(rec)
            stats["rows_with_temporal"] += 1

    return records, stats


def main() -> None:
    p = argparse.ArgumentParser(description="Build NExT-GQA val/test grounding manifest (JSONL).")
    p.add_argument(
        "--nextgqa-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing csv/json annotations (default: this folder).",
    )
    p.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help="Root folder where videos live (e.g. D:/NExTVideo/NExTVideo). Sets video_abspath.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: <nextgqa-dir>/nextgqa_grounding_val_test.jsonl).",
    )
    args = p.parse_args()

    root: Path = args.nextgqa_dir
    out_path = args.output or (root / "nextgqa_grounding_val_test.jsonl")

    vid_map = load_json(root / "map_vid_vidorID.json")
    # Normalize keys to str
    vid_map = {str(k): v for k, v in vid_map.items()}

    gsub_val = load_json(root / "gsub_val.json")
    gsub_test = load_json(root / "gsub_test.json")
    f2t_val = load_json(root / "frame2time_val.json")
    f2t_test = load_json(root / "frame2time_test.json")
    f2t_val = {str(k): v for k, v in f2t_val.items()}
    f2t_test = {str(k): v for k, v in f2t_test.items()}

    video_root = args.video_root
    if video_root is not None:
        video_root = video_root.resolve()

    all_stats: dict[str, Any] = {}
    all_records: list[dict] = []

    for split, csv_name, gsub, f2t in (
        ("val", "val.csv", gsub_val, f2t_val),
        ("test", "test.csv", gsub_test, f2t_test),
    ):
        csv_path = root / csv_name
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        recs, st = process_split(csv_path, gsub, f2t, vid_map, video_root, split)
        all_records.extend(recs)
        all_stats[split] = st

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "output": str(out_path),
        "total_records": len(all_records),
        "video_root": str(video_root) if video_root else None,
        "by_split": all_stats,
    }
    summary_path = out_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
