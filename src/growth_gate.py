from __future__ import annotations

from typing import List

import numpy as np
from scipy.ndimage import binary_dilation


def build_allowed_growth_mask(bundle: dict, source_indices: List[int], dilation_iters: int = 0) -> np.ndarray:
    """
    Build a binary mask of anatomically/temporally allowed growth zones.
    source_indices are usually all sessions in the fit window (including baseline).
    """
    if not source_indices:
        raise ValueError("source_indices cannot be empty for growth gating.")

    label_bin = bundle["label_bin"]
    allowed = np.zeros_like(label_bin[0], dtype=bool)
    for idx in source_indices:
        allowed |= label_bin[idx] > 0

    if dilation_iters > 0:
        allowed = binary_dilation(allowed, iterations=int(dilation_iters))

    return allowed.astype(np.uint8)


def apply_growth_gate_to_mask(pred_mask: np.ndarray, allowed_mask: np.ndarray) -> np.ndarray:
    pred = pred_mask > 0
    allowed = allowed_mask > 0
    return np.logical_and(pred, allowed).astype(np.uint8)


def outside_growth_fraction(pred_mask: np.ndarray, allowed_mask: np.ndarray) -> float:
    pred = pred_mask > 0
    allowed = allowed_mask > 0
    outside = np.logical_and(pred, np.logical_not(allowed))
    return float(np.mean(outside.astype(np.float32)))
