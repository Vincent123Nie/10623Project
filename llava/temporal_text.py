"""Parse and rewrite <time_start>/<t###>/<time_end> spans for user-facing (second-based) text."""

from __future__ import annotations

import re
from typing import List, Tuple

from llava.constants import DEFAULT_TIME_END_TOKEN, DEFAULT_TIME_START_TOKEN
from llava.time_quantization import bin_to_seconds_end, bin_to_seconds_start

# Model may emit <t0>..<t999> with fixed width; allow variable digits for robustness
_TEMPORAL_SPAN_RE = re.compile(
    re.escape(DEFAULT_TIME_START_TOKEN)
    + r"\s*<t(\d+)>\s*"
    + re.escape(DEFAULT_TIME_END_TOKEN)
    + r"\s*<t(\d+)>"
)


def parse_temporal_bin_spans(text: str) -> List[Tuple[int, int]]:
    """Return all (start_bin, end_bin) integer pairs in order of appearance."""
    out: List[Tuple[int, int]] = []
    for m in _TEMPORAL_SPAN_RE.finditer(text or ""):
        out.append((int(m.group(1)), int(m.group(2))))
    return out


def _format_conversational_range(
    t0_sec: float,
    t1_sec: float,
    duration_sec: float,
    point_threshold_sec: float = 0.75,
) -> str:
    """
    Phrasing aligned with ``conversational_all.json`` style: either a short point
    or a closed range in seconds.
    """
    T = max(1e-6, float(duration_sec))
    t0 = max(0.0, min(float(t0_sec), T))
    t1 = max(0.0, min(float(t1_sec), T))
    if t1 < t0:
        t0, t1 = t1, t0
    if (t1 - t0) <= point_threshold_sec:
        mid = 0.5 * (t0 + t1)
        return f"at {mid:.1f}s"
    return f"between {t0:.1f} seconds and {t1:.1f} seconds"


def replace_temporal_spans_with_conversational(
    text: str,
    duration_sec: float,
    num_bins: int = 1000,
    point_threshold_sec: float = 0.75,
) -> str:
    """
    Replace each ``<time_start> <tI> <time_end> <tJ>`` block with a single
    conversational English phrase in seconds (using the same bin↔time mapping
    as training: left edge of start bin, right edge of end bin).
    """

    def _sub(m: re.Match) -> str:
        ib, ie = int(m.group(1)), int(m.group(2))
        if num_bins < 1:
            return m.group(0)
        ib = max(0, min(num_bins - 1, ib))
        ie = max(0, min(num_bins - 1, ie))
        t0 = bin_to_seconds_start(ib, duration_sec, num_bins)
        t1 = bin_to_seconds_end(ie, duration_sec, num_bins)
        return _format_conversational_range(
            t0, t1, duration_sec, point_threshold_sec=point_threshold_sec
        )

    return _TEMPORAL_SPAN_RE.sub(_sub, text or "")


def append_conversational_time_sentence(
    text: str,
    duration_sec: float,
    num_bins: int = 1000,
    point_threshold_sec: float = 0.75,
) -> str:
    """
    If ``text`` still contains temporal span tokens, replace them; otherwise return
    ``text`` unchanged. Handy when you want to drop raw tokens from the final reply.
    """
    if not text or DEFAULT_TIME_START_TOKEN not in text:
        return text
    return replace_temporal_spans_with_conversational(
        text,
        duration_sec,
        num_bins=num_bins,
        point_threshold_sec=point_threshold_sec,
    )


def strip_leading_temporal_english_phrase(text: str) -> str:
    """Remove a leading 'Temporally, the relevant segment is ...' if present (nextgqa training style)."""
    if not text:
        return text
    s = text.lstrip()
    p = re.compile(
        r"^Temporally,\s*the relevant segment is\s*",
        re.IGNORECASE,
    )
    return p.sub("", s).lstrip()
