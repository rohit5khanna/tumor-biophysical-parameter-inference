from __future__ import annotations

import numpy as np


def soft_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """
    Soft Dice over flattened arrays.
    pred and target should be float arrays in [0, 1].
    """
    p = pred.astype(np.float32).reshape(-1)
    t = target.astype(np.float32).reshape(-1)

    inter = float(np.sum(p * t))
    denom = float(np.sum(p) + np.sum(t))
    return (2.0 * inter + eps) / (denom + eps)


def hard_dice(pred_mask: np.ndarray, target_mask: np.ndarray, eps: float = 1e-6) -> float:
    p = (pred_mask > 0).astype(np.float32).reshape(-1)
    t = (target_mask > 0).astype(np.float32).reshape(-1)
    inter = float(np.sum(p * t))
    denom = float(np.sum(p) + np.sum(t))
    return (2.0 * inter + eps) / (denom + eps)


def hard_dice_from_field(field: np.ndarray, target_mask: np.ndarray, threshold: float = 0.25) -> float:
    pred = (field >= threshold).astype(np.uint8)
    return hard_dice(pred, target_mask)
