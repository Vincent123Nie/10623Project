"""
Video QA with LLaVA-NeXT: optional argparse, optional <time_start>/<t###> -> 秒级自然语言
（与 bin_all 微调 / llava.temporal_text 一致的后处理）。
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.temporal_text import append_conversational_time_sentence


def _video_duration_sec(cap: cv2.VideoCapture) -> float:
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps > 1e-6 and n > 0:
        return n / fps
    return 0.0


def run_inference(
    model_path: str,
    video_path: str,
    question: str,
    max_frames_to_sample: int = 8,
    load_in_4bit: bool = True,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
    conversational_time: bool = False,
    num_time_bins: int = 1000,
) -> str:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频不存在: {video_path}")

    print(f"正在加载模型: {model_path} …")
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        None,
        model_name,
        load_in_4bit=load_in_4bit,
        device_map="auto",
    )

    cap = cv2.VideoCapture(video_path)
    duration_sec = _video_duration_sec(cap)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, max(total_frames - 1, 0), max_frames_to_sample, dtype=int)
    video_frames: list[Image.Image] = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames.append(Image.fromarray(frame))
    cap.release()
    if not video_frames:
        raise RuntimeError("未能从视频读取任何帧")

    image_tensor = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]
    image_tensor = image_tensor.half().to(model.device)

    conv = conv_templates["vicuna_v1"].copy()
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    stop_str = conv.sep if conv.sep_style != "two" else conv.sep2
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if output_text.endswith(stop_str):
        output_text = output_text[: -len(stop_str)]

    if conversational_time and duration_sec > 0:
        output_text = append_conversational_time_sentence(
            output_text, duration_sec, num_bins=num_time_bins
        )
    return output_text


def main() -> None:
    p = argparse.ArgumentParser(description="LLaVA-NeXT 视频推理（可选：时序 token → 秒级描述）")
    p.add_argument("--model-path", type=str, default=os.environ.get("LLAVA_MODEL", ""), help="模型/adapter 目录")
    p.add_argument("--video-path", type=str, required=True, help="单个 mp4 等视频文件路径")
    p.add_argument(
        "--question",
        type=str,
        default="Localize the event in the video and describe what happens in the relevant time span.",
        help="用户问题（与训练时风格接近更佳）",
    )
    p.add_argument("--max-frames", type=int, default=8, help="均匀采样帧数")
    p.add_argument(
        "--conversational-time",
        action="store_true",
        help="将 <time_start><t###><time_end><t###> 换成秒级自然语言（需视频可读时长；与 temporal 微调一致）",
    )
    p.add_argument("--num-time-bins", type=int, default=1000, help="与训练 mm_num_time_bins 一致")
    p.add_argument("--no-4bit", action="store_true", help="不用 4bit 量化加载")
    args = p.parse_args()
    if not args.model_path:
        print("请设置 --model-path 或环境变量 LLAVA_MODEL", file=sys.stderr)
        sys.exit(1)
    text = run_inference(
        model_path=args.model_path,
        video_path=args.video_path,
        question=args.question,
        max_frames_to_sample=args.max_frames,
        load_in_4bit=not args.no_4bit,
        conversational_time=args.conversational_time,
        num_time_bins=args.num_time_bins,
    )
    print(text)


if __name__ == "__main__":
    main()
