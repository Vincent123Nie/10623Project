from __future__ import annotations

import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_TEMPORAL_SPAN_RE = re.compile(
    r"<time_start>\s*<t(\d+)>\s*<time_end>\s*<t(\d+)>"
)


def _parse_temporal_bin_spans(text: str) -> list[tuple[int, int]]:
    return [(int(match.group(1)), int(match.group(2))) for match in _TEMPORAL_SPAN_RE.finditer(text or "")]


def _bin_to_seconds_start(bin_index: int, duration_sec: float, num_bins: int) -> float:
    if num_bins <= 1:
        return 0.0
    bounded = max(0, min(num_bins - 1, int(bin_index)))
    return (bounded / (num_bins - 1)) * duration_sec


def _bin_to_seconds_end(bin_index: int, duration_sec: float, num_bins: int) -> float:
    if num_bins <= 1:
        return duration_sec
    bounded = max(0, min(num_bins - 1, int(bin_index)))
    return duration_sec if bounded == num_bins - 1 else ((bounded + 1) / (num_bins - 1)) * duration_sec


def _replace_temporal_span(text: str, duration_sec: float, num_bins: int) -> str:
    def replace(match: re.Match[str]) -> str:
        start_bin, end_bin = int(match.group(1)), int(match.group(2))
        if end_bin < start_bin:
            start_bin, end_bin = end_bin, start_bin
        start = _bin_to_seconds_start(start_bin, duration_sec, num_bins)
        end = _bin_to_seconds_end(end_bin, duration_sec, num_bins)
        return f"between {start:.1f} seconds and {end:.1f} seconds"

    return _TEMPORAL_SPAN_RE.sub(replace, text or "")


@dataclass(frozen=True)
class GroundingPrediction:
    predicted_start: float
    predicted_end: float
    reason: str
    raw_model_output: str
    sampled_frames: list[dict[str, float]]


def decode_temporal_prediction(text: str, duration_sec: float, num_bins: int = 1000) -> tuple[float, float]:
    if not math.isfinite(duration_sec) or duration_sec < 0:
        raise ValueError("duration_sec must be a finite, non-negative number.")
    if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins < 2:
        raise ValueError("num_bins must be an integer greater than or equal to 2.")

    spans = _parse_temporal_bin_spans(text)
    if not spans:
        raise ValueError("The model response did not contain a parseable temporal span.")

    start_bin, end_bin = spans[0]
    if end_bin < start_bin:
        start_bin, end_bin = end_bin, start_bin

    return (
        _bin_to_seconds_start(start_bin, duration_sec, num_bins),
        _bin_to_seconds_end(end_bin, duration_sec, num_bins),
    )


class LlavaVideoGrounder:
    """Load the LLaVA model once and run two-stage temporal grounding requests serially."""

    def __init__(
        self,
        model_path: str,
        model_base: str | None = None,
        num_time_bins: int = 1000,
        load_in_4bit: bool = True,
    ) -> None:
        import torch

        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model

        self._torch = torch
        self.num_time_bins = num_time_bins
        self._lock = threading.Lock()
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path,
            model_base,
            model_name,
            load_in_4bit=load_in_4bit,
            device_map="auto",
        )

    @classmethod
    def from_environment(cls) -> "LlavaVideoGrounder":
        model_path = os.environ.get("LLAVA_MODEL_PATH", "").strip()
        if not model_path:
            raise RuntimeError("LLAVA_MODEL_PATH is required.")
        return cls(
            model_path=model_path,
            model_base=os.environ.get("LLAVA_MODEL_BASE") or None,
            num_time_bins=int(os.environ.get("LLAVA_NUM_TIME_BINS", "1000")),
            load_in_4bit=os.environ.get("LLAVA_LOAD_IN_4BIT", "true").lower() != "false",
        )

    @staticmethod
    def _video_metadata(video_path: Path) -> tuple[float, int, float]:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        if total_frames <= 0 or fps <= 1e-6:
            raise ValueError(f"Could not read video metadata: {video_path}")
        return total_frames / fps, total_frames, fps

    @staticmethod
    def _frame_indices(
        total_frames: int,
        fps: float,
        count: int,
        focus: tuple[float, float] | None = None,
    ) -> tuple[list[int], set[int]]:
        import numpy as np

        coverage_count = count if focus is None else max(4, count // 2)
        coverage = np.linspace(0, total_frames - 1, coverage_count, dtype=int).tolist()
        focus_indices: list[int] = []

        if focus is not None:
            start_sec, end_sec = focus
            margin = max(1.0, (end_sec - start_sec) * 0.35)
            start_idx = max(0, int((start_sec - margin) * fps))
            end_idx = min(total_frames - 1, int((end_sec + margin) * fps))
            focus_indices = np.linspace(start_idx, end_idx, max(2, count - coverage_count), dtype=int).tolist()

        indices = sorted(set(coverage + focus_indices))
        return indices, set(focus_indices)

    @staticmethod
    def _read_frames(video_path: Path, indices: list[int]) -> tuple[list[Any], list[int]]:
        import cv2
        from PIL import Image

        cap = cv2.VideoCapture(str(video_path))
        frames: list[Any] = []
        decoded_indices: list[int] = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                decoded_indices.append(index)
        cap.release()
        if not frames:
            raise ValueError(f"No frames could be decoded from {video_path}")
        return frames, decoded_indices

    def _generate(self, frames: list[Any], query: str) -> str:
        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token

        image_tensor = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        image_tensor = image_tensor.half().to(self.model.device)

        instruction = (
            f"{query.strip()}\n"
            "Return the single best temporal interval using exactly "
            "<time_start> <t###> <time_end> <t###>, followed by a concise explanation."
        )
        conv = conv_templates["vicuna_v1"].copy()
        if self.model.config.mm_use_im_start_end:
            prompt_question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + instruction
        else:
            prompt_question = DEFAULT_IMAGE_TOKEN + "\n" + instruction
        conv.append_message(conv.roles[0], prompt_question)
        conv.append_message(conv.roles[1], None)
        input_ids = tokenizer_image_token(
            conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)

        with self._torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=256,
                use_cache=True,
            )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()

    def predict(self, video_path: Path, query: str, frame_budget: int = 16) -> GroundingPrediction:
        duration_sec, total_frames, fps = self._video_metadata(video_path)

        with self._lock:
            coarse_indices, _ = self._frame_indices(total_frames, fps, max(6, frame_budget // 2))
            coarse_frames, _ = self._read_frames(video_path, coarse_indices)
            coarse_output = self._generate(coarse_frames, query)
            coarse_interval = decode_temporal_prediction(coarse_output, duration_sec, self.num_time_bins)

            refined_indices, focus_indices = self._frame_indices(
                total_frames, fps, frame_budget, focus=coarse_interval
            )
            refined_frames, decoded_indices = self._read_frames(video_path, refined_indices)
            refined_output = self._generate(refined_frames, query)
            predicted_start, predicted_end = decode_temporal_prediction(
                refined_output, duration_sec, self.num_time_bins
            )

        reason = _replace_temporal_span(refined_output, duration_sec, self.num_time_bins)
        sampled_frames = [
            {
                "timestamp": round(index / fps, 3),
                "relevanceScore": 1.0 if index in focus_indices else 0.5,
            }
            for index in decoded_indices
        ]
        return GroundingPrediction(
            predicted_start=predicted_start,
            predicted_end=predicted_end,
            reason=reason,
            raw_model_output=refined_output,
            sampled_frames=sampled_frames,
        )
