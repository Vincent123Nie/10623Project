"""Add <time_start>, <time_end>, and <t###> bin tokens to tokenizer + resize LLM embeddings."""

from __future__ import annotations

import torch

from llava.constants import DEFAULT_TIME_END_TOKEN, DEFAULT_TIME_START_TOKEN


def time_bin_vocab_list(num_bins: int) -> list[str]:
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")
    width = len(str(num_bins - 1))
    return [f"<t{i:0{width}d}>" for i in range(num_bins)]


def temporal_ground_token_list(num_bins: int) -> list[str]:
    return [DEFAULT_TIME_START_TOKEN, DEFAULT_TIME_END_TOKEN] + time_bin_vocab_list(num_bins)


def add_temporal_ground_tokens_to_model(tokenizer, model, num_bins: int = 1000) -> int:
    """
    Add missing temporal tokens and resize embeddings; mean-initialize new rows only.
    Safe to call multiple times (idempotent w.r.t. tokenizer vocab).
    Returns tokenizer.add_tokens count for this call (may be 0 if already present).
    """
    tokens = temporal_ground_token_list(num_bins)
    vocab = tokenizer.get_vocab()
    to_add = [t for t in tokens if t not in vocab]
    n_added = tokenizer.add_tokens(to_add, special_tokens=True) if to_add else 0

    cur = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != cur:
        # resize_token_embeddings creates new embeddings on CPU even if the model
        # is on GPU (device_map="auto"). Move embed/lm_head to CPU first, then back.
        in_emb     = model.get_input_embeddings()
        out_emb    = model.get_output_embeddings()
        in_device  = in_emb.weight.device
        out_device = out_emb.weight.device
        in_emb.to("cpu")
        out_emb.to("cpu")
        model.resize_token_embeddings(len(tokenizer))
        model.get_input_embeddings().to(in_device)
        model.get_output_embeddings().to(out_device)

    if n_added > 0:
        with torch.no_grad():
            w_in = model.get_input_embeddings().weight.data
            w_out = model.get_output_embeddings().weight.data
            old = w_in.shape[0] - n_added
            avg_in = w_in[:old].mean(dim=0, keepdim=True)
            avg_out = w_out[:old].mean(dim=0, keepdim=True)
            w_in[-n_added:] = avg_in
            w_out[-n_added:] = avg_out
    return n_added
