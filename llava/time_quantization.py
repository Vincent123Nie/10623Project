"""Map continuous time on [0, T] to discrete bin indices, consistent with I = floor((t/T) * (N-1)) clipped."""

from __future__ import annotations

import math
from typing import Tuple


def seconds_to_bin(t_sec: float, duration_sec: float, num_bins: int = 1000) -> int:
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")
    if num_bins == 1:
        return 0
    if duration_sec <= 0:
        return 0
    t = float(t_sec)
    t = max(0.0, min(t, float(duration_sec)))
    idx = int(math.floor((t / float(duration_sec)) * (num_bins - 1) + 1e-9))
    return max(0, min(num_bins - 1, idx))


def bin_to_seconds_start(bin_idx: int, duration_sec: float, num_bins: int = 1000) -> float:
    """Left edge of bin `bin_idx` on a uniform partition of [0, duration_sec]."""
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")
    if num_bins == 1:
        return 0.0
    b = max(0, min(num_bins - 1, int(bin_idx)))
    return (b / (num_bins - 1)) * float(duration_sec)


def bin_to_seconds_end(bin_idx: int, duration_sec: float, num_bins: int = 1000) -> float:
    """Right edge of bin `bin_idx` (start of next bin, or T for last bin)."""
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")
    if num_bins == 1:
        return float(duration_sec)
    b = max(0, min(num_bins - 1, int(bin_idx)))
    if b >= num_bins - 1:
        return float(duration_sec)
    return ((b + 1) / (num_bins - 1)) * float(duration_sec)


def segment_range_to_bins(
    segments_sec: list[list[float]],
    duration_sec: float,
    num_bins: int = 1000,
) -> Tuple[int, int]:
    """
    Merge all segments to global [t_min, t_max], quantize start/end to bins.
    Ensures end_bin >= start_bin.
    """
    if not segments_sec:
        return 0, 0
    t_min = min(float(s[0]) for s in segments_sec if len(s) >= 2)
    t_max = max(float(s[1]) for s in segments_sec if len(s) >= 2)
    if t_max < t_min:
        t_min, t_max = t_max, t_min
    ib = seconds_to_bin(t_min, duration_sec, num_bins)
    ie = seconds_to_bin(t_max, duration_sec, num_bins)
    if ie < ib:
        ib, ie = ie, ib
    return ib, ie
