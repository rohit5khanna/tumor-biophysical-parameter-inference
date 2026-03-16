from __future__ import annotations

from typing import Dict, List

import numpy as np

from .metrics import hard_dice


def rollout_theta(
    simulator,
    bundle: Dict,
    theta: Dict[str, float],
    baseline_idx: int,
    target_indices: List[int],
) -> Dict:
    sim = simulator.predict_for_indices(
        bundle=bundle,
        theta=theta,
        target_indices=target_indices,
        baseline_idx=baseline_idx,
    )

    gt = bundle["label_bin"]
    dices: List[float] = []
    for i, idx in enumerate(target_indices):
        if not sim["success_flags"][i]:
            dices.append(0.0)
        else:
            dices.append(float(hard_dice(sim["pred_masks"][i], gt[idx])))

    return {
        "target_indices": list(target_indices),
        "dices": dices,
        "mean_dice": float(np.mean(dices)) if dices else float("nan"),
        "pred_masks": sim["pred_masks"],
        "success_flags": sim["success_flags"],
        "messages": sim["messages"],
    }


def rollout_ensemble(
    simulator,
    bundle: Dict,
    thetas: List[Dict[str, float]],
    baseline_idx: int,
    target_indices: List[int],
) -> Dict:
    per_theta = []
    for theta in thetas:
        out = rollout_theta(
            simulator=simulator,
            bundle=bundle,
            theta=theta,
            baseline_idx=baseline_idx,
            target_indices=target_indices,
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
