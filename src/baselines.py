from __future__ import annotations

from typing import Dict, List

import numpy as np

from .metrics import hard_dice


def evaluate_locf(bundle: Dict, baseline_idx: int, target_indices: List[int]) -> Dict:
    """
    Last Observation Carried Forward baseline:
    use baseline mask for all target future sessions.
    """
    pred = bundle["label_bin"][baseline_idx]
    gt = bundle["label_bin"]

    dices = [float(hard_dice(pred, gt[idx])) for idx in target_indices]
    pred_masks = [pred.copy() for _ in target_indices]
    return {
        "target_indices": list(target_indices),
        "dices": dices,
        "pred_masks": pred_masks,
        "mean_dice": float(np.mean(dices)) if dices else float("nan"),
    }
