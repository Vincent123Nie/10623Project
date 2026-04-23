#!/usr/bin/env python3
"""
Small launcher for LoRA on NExT-GQA temporal grounding JSON (nextgqa_llava_grounding*.json).

This does **not** reimplement the model or loss: it forwards to ``llava.train.train.train()``,
which uses the same ``LazySupervisedDataset``, collator, and ``LLaVATrainer`` as
``train_mem.py``. You get a **short argparse** instead of hundreds of CLI flags.

Example (single GPU):

  python nextgqa/train_nextgqa_temporal_lora.py \\
    --model /path/to/LLaVA-Next-Video-7B \\
    --data-json nextgqa/nextgqa_llava_grounding_val_only.json \\
    --video-folder /path/to/NExTVideo \\
    --image-folder /path/to/empty_images \\
    --output-dir ./outputs/nextgqa_lora_1

If ``--vision-tower`` is omitted, ``mm_vision_tower`` is read from the model ``config.json``.

**Colab:** do not ``subprocess.run([sys.executable, ...])`` unless that interpreter has
``llava`` installed. Prefer calling :func:`train_nextgqa_temporal_lora` in-process
(see docstring below).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Sequence


def _vision_tower_from_config(model_path: str) -> str:
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Expected config at {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    vt = cfg.get("mm_vision_tower")
    if not vt:
        raise ValueError(
            f"No mm_vision_tower in {cfg_path}; pass --vision-tower explicitly."
        )
    return vt


def build_argv(ns: argparse.Namespace) -> list[str]:
    model = os.path.abspath(ns.model)
    vision_tower = ns.vision_tower or _vision_tower_from_config(model)

    argv = [
        "train_nextgqa_temporal_lora.py",
        "--model_name_or_path",
        model,
        "--version",
        ns.version,
        "--vision_tower",
        vision_tower,
        "--data_path",
        os.path.abspath(ns.data_json),
        "--video_folder",
        os.path.abspath(ns.video_folder),
        "--image_folder",
        os.path.abspath(ns.image_folder),
        "--is_multimodal",
        "True",
        "--lazy_preprocess",
        "True",
        "--mm_use_temporal_ground_tokens",
        "True",
        "--mm_num_time_bins",
        str(ns.mm_num_time_bins),
        "--lora_enable",
        "True",
        "--lora_r",
        str(ns.lora_r),
        "--lora_alpha",
        str(ns.lora_alpha),
        "--lora_dropout",
        str(ns.lora_dropout),
        "--bf16",
        "True",
        "--output_dir",
        os.path.abspath(ns.output_dir),
        "--num_train_epochs",
        str(ns.num_train_epochs),
        "--per_device_train_batch_size",
        str(ns.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(ns.gradient_accumulation_steps),
        "--learning_rate",
        str(ns.learning_rate),
        "--weight_decay",
        "0.",
        "--warmup_ratio",
        str(ns.warmup_ratio),
        "--lr_scheduler_type",
        "cosine",
        "--model_max_length",
        str(ns.model_max_length),
        "--gradient_checkpointing",
        "True",
        "--save_steps",
        str(ns.save_steps),
        "--save_total_limit",
        str(ns.save_total_limit),
        "--logging_steps",
        str(ns.logging_steps),
        "--evaluation_strategy",
        "no",
        "--frames_upbound",
        str(ns.frames_upbound),
        "--add_time_instruction",
        "True",
        "--force_sample",
        "True",
        "--attn_implementation",
        ns.attn_implementation,
        "--dataloader_num_workers",
        str(ns.dataloader_num_workers),
        "--report_to",
        "none",
        "--mm_patch_merge_type",
        ns.mm_patch_merge_type,
        "--mm_newline_position",
        ns.mm_newline_position,
        "--mm_spatial_pool_stride",
        str(ns.mm_spatial_pool_stride),
        "--mm_spatial_pool_mode",
        ns.mm_spatial_pool_mode,
        "--image_aspect_ratio",
        ns.image_aspect_ratio,
        "--mm_use_im_patch_token",
        str(ns.mm_use_im_patch_token),
    ]
    return argv


def invoke_train(argv: list[str]) -> None:
    """Set ``sys.argv`` and run ``llava.train.train.train()`` (same process)."""
    print("[nextgqa] Entering llava.train.train (progress lines prefixed with [llava-train])", flush=True)
    print(
        "[nextgqa] Tip: if you use subprocess.run(..., capture_output=True), logs appear only after the process exits; "
        "omit capture_output to stream logs in Colab.",
        flush=True,
    )
    old = sys.argv[:]
    sys.argv = argv
    try:
        print(
            "[nextgqa] Importing llava.train.train (first time can take many minutes if code is on Google Drive) …",
            flush=True,
        )
        from llava.train.train import train

        print("[nextgqa] Import finished; calling train() …", flush=True)
        train()
    finally:
        sys.argv = old


_TRAIN_KW_DEFAULTS: dict[str, Any] = {
    "vision_tower": None,
    "version": "vicuna_v1",
    "num_train_epochs": 1.0,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.03,
    "model_max_length": 4096,
    "lora_r": 64,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "mm_num_time_bins": 1000,
    "frames_upbound": 16,
    "save_steps": 500,
    "save_total_limit": 2,
    "logging_steps": 10,
    "dataloader_num_workers": 2,
    "attn_implementation": "sdpa",
    "mm_patch_merge_type": "spatial_unpad",
    "mm_newline_position": "grid",
    "mm_spatial_pool_stride": 2,
    "mm_spatial_pool_mode": "average",
    "image_aspect_ratio": "square",
    "mm_use_im_patch_token": True,
}


def train_nextgqa_temporal_lora(
    *,
    model: str,
    data_json: str,
    video_folder: str,
    image_folder: str,
    output_dir: str,
    extra_argv: Sequence[str] | None = None,
    **kwargs: Any,
) -> None:
    """
    Run training in the **current** Python process (recommended for notebooks).

    Use this instead of ``subprocess.run`` so you use the same environment as
    ``%pip install`` / conda ``python``, and tracebacks print in the notebook.

    Optional hyperparameters match CLI flags (e.g. ``lora_r=32``, ``frames_upbound=8``).
    Pass ``extra_argv`` for any additional ``llava.train.train`` arguments.
    """
    cfg = {**_TRAIN_KW_DEFAULTS, **kwargs}
    unknown = set(cfg) - set(_TRAIN_KW_DEFAULTS)
    if unknown:
        raise TypeError(f"Unknown keyword arguments: {sorted(unknown)}")

    ns = argparse.Namespace(
        model=model,
        data_json=data_json,
        video_folder=video_folder,
        image_folder=image_folder,
        output_dir=output_dir,
        **cfg,
    )
    os.makedirs(ns.image_folder, exist_ok=True)
    os.makedirs(ns.output_dir, exist_ok=True)

    argv = build_argv(ns)
    if extra_argv:
        argv.extend(list(extra_argv))
    invoke_train(argv)


def main() -> None:
    print("[nextgqa] main() running — if you see this but nothing after, the hang is before argparse.", flush=True)
    p = argparse.ArgumentParser(
        description="NExT-GQA temporal grounding LoRA (thin wrapper → llava.train.train.train)"
    )
    p.add_argument("--model", required=True, help="HF folder or LLaVA checkpoint root (with config.json)")
    p.add_argument("--data-json", required=True, dest="data_json", help="List JSON from jsonl_to_llava_grounding.py")
    p.add_argument("--video-folder", required=True, dest="video_folder", help="Root joined with each sample's video relpath")
    p.add_argument(
        "--image-folder",
        required=True,
        dest="image_folder",
        help="Folder that must exist (use empty dir); required by trainer args",
    )
    p.add_argument("--output-dir", required=True, dest="output_dir")
    p.add_argument("--vision-tower", default=None, help="Override; default: mm_vision_tower from model config")
    p.add_argument("--version", default="vicuna_v1")
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--per-device-train-batch-size", type=int, default=1, dest="per_device_train_batch_size")
    p.add_argument("--gradient-accumulation-steps", type=int, default=8, dest="gradient_accumulation_steps")
    p.add_argument("--learning-rate", type=float, default=2e-4, dest="learning_rate")
    p.add_argument("--warmup-ratio", type=float, default=0.03, dest="warmup_ratio")
    p.add_argument("--model-max-length", type=int, default=4096, dest="model_max_length")
    p.add_argument("--lora-r", type=int, default=64, dest="lora_r")
    p.add_argument("--lora-alpha", type=int, default=16, dest="lora_alpha")
    p.add_argument("--lora-dropout", type=float, default=0.05, dest="lora_dropout")
    p.add_argument("--mm-num-time-bins", type=int, default=1000, dest="mm_num_time_bins")
    p.add_argument("--frames-upbound", type=int, default=16, dest="frames_upbound")
    p.add_argument("--save-steps", type=int, default=500, dest="save_steps")
    p.add_argument("--save-total-limit", type=int, default=2, dest="save_total_limit")
    p.add_argument("--logging-steps", type=int, default=10, dest="logging_steps")
    p.add_argument("--dataloader-num-workers", type=int, default=2, dest="dataloader_num_workers")
    p.add_argument("--attn-implementation", default="sdpa", dest="attn_implementation")
    p.add_argument("--mm-patch-merge-type", default="spatial_unpad", dest="mm_patch_merge_type")
    p.add_argument("--mm-newline-position", default="grid", dest="mm_newline_position")
    p.add_argument("--mm-spatial-pool-stride", type=int, default=2, dest="mm_spatial_pool_stride")
    p.add_argument("--mm-spatial-pool-mode", default="average", dest="mm_spatial_pool_mode")
    p.add_argument("--image-aspect-ratio", default="square", dest="image_aspect_ratio")
    p.set_defaults(mm_use_im_patch_token=True)
    p.add_argument(
        "--no-mm-use-im-patch-token",
        action="store_false",
        dest="mm_use_im_patch_token",
        help="Pass False for mm_use_im_patch_token if base config disables it",
    )
    p.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args passed through to llava.train.train (prefix with --)",
    )
    ns = p.parse_args()
    print("[nextgqa] argparse OK — building argv and entering invoke_train …", flush=True)
    os.makedirs(ns.image_folder, exist_ok=True)
    os.makedirs(ns.output_dir, exist_ok=True)

    argv = build_argv(ns)
    if ns.extra:
        if ns.extra and ns.extra[0] == "--":
            ns.extra = ns.extra[1:]
        argv.extend(ns.extra)

    invoke_train(argv)


if __name__ == "__main__":
    print("[nextgqa] __main__ — Python started this script (file is executing).", flush=True)
    main()
