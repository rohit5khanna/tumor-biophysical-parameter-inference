from __future__ import annotations

from typing import Dict, List

import numpy as np

from .growth_gate import apply_growth_gate_to_mask, outside_growth_fraction
from .metrics import hard_dice


def rollout_theta(
    simulator,
    bundle: Dict,
    theta: Dict[str, float],
    baseline_idx: int,
    target_indices: List[int],
    allowed_growth_mask: np.ndarray | None = None,
    apply_growth_gate: bool = False,
) -> Dict:
    sim = simulator.predict_for_indices(
        bundle=bundle,
        theta=theta,
        target_indices=target_indices,
        baseline_idx=baseline_idx,
    )

    gt = bundle["label_bin"]
    dices: List[float] = []
    outside_rates: List[float] = []
    pred_masks: List[np.ndarray] = []
    raw_pred_masks: List[np.ndarray] = []
    for i, idx in enumerate(target_indices):
        if not sim["success_flags"][i]:
            dices.append(0.0)
            outside_rates.append(0.0)
            pred_masks.append(np.zeros_like(gt[idx], dtype=np.uint8))
            raw_pred_masks.append(np.zeros_like(gt[idx], dtype=np.uint8))
        else:
            pred_raw = sim["pred_masks"][i].astype(np.uint8)
            raw_pred_masks.append(pred_raw)
            if allowed_growth_mask is not None:
                outside_rates.append(outside_growth_fraction(pred_raw, allowed_growth_mask))
                pred_eval = (
                    apply_growth_gate_to_mask(pred_raw, allowed_growth_mask)
                    if apply_growth_gate
                    else pred_raw
                )
            else:
                outside_rates.append(0.0)
                pred_eval = pred_raw

            pred_masks.append(pred_eval)
            dices.append(float(hard_dice(pred_eval, gt[idx])))

    return {
        "target_indices": list(target_indices),
        "dices": dices,
        "mean_dice": float(np.mean(dices)) if dices else float("nan"),
        "outside_rates": outside_rates,
        "mean_outside_rate": float(np.mean(outside_rates)) if outside_rates else 0.0,
        "pred_masks": pred_masks,
        "raw_pred_masks": raw_pred_masks,
        "success_flags": sim["success_flags"],
        "messages": sim["messages"],
    }


def rollout_ensemble(
    simulator,
    bundle: Dict,
    thetas: List[Dict[str, float]],
    baseline_idx: int,
    target_indices: List[int],
    allowed_growth_mask: np.ndarray | None = None,
    apply_growth_gate: bool = False,
) -> Dict:
    per_theta = []
    for theta in thetas:
        out = rollout_theta(
            simulator=simulator,
            bundle=bundle,
            theta=theta,
            baseline_idx=baseline_idx,
            target_indices=target_indices,
            allowed_growth_mask=allowed_growth_mask,
            apply_growth_gate=apply_growth_gate,
        )
        per_theta.append(out)

    if not per_theta:
        return {
            "per_theta": [],
            "dice_mean_per_session": [],
            "dice_std_per_session": [],
            "mean_dice": float("nan"),
        }

    dice_matrix = np.asarray([x["dices"] for x in per_theta], dtype=np.float32)
    mean_per_session = np.mean(dice_matrix, axis=0).tolist()
    std_per_session = np.std(dice_matrix, axis=0).tolist()
    return {
        "per_theta": per_theta,
        "target_indices": list(target_indices),
        "dice_mean_per_session": mean_per_session,
        "dice_std_per_session": std_per_session,
        "mean_dice": float(np.mean(dice_matrix)),
    }
