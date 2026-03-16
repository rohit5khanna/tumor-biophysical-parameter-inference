from __future__ import annotations

from typing import Dict, List

import numpy as np

from .metrics import hard_dice


def sample_theta(rng: np.random.Generator, bounds: Dict[str, List[float]]) -> Dict[str, float]:
    theta = {}
    for k, (lo, hi) in bounds.items():
        theta[k] = float(rng.uniform(lo, hi))
    return theta


def evaluate_theta_on_fit_window(
    simulator,
    bundle: Dict,
    theta: Dict[str, float],
    baseline_idx: int,
    fit_target_indices: List[int],
    failure_loss: float,
) -> Dict:
    sim = simulator.predict_for_indices(
        bundle=bundle,
        theta=theta,
        target_indices=fit_target_indices,
        baseline_idx=baseline_idx,
    )

    gt = bundle["label_bin"]
    dices: List[float] = []
    failed = False
    for i, idx in enumerate(fit_target_indices):
        if not sim["success_flags"][i]:
            failed = True
            dices.append(0.0)
        else:
            d = hard_dice(sim["pred_masks"][i], gt[idx])
            dices.append(float(d))

    if failed:
        loss = float(failure_loss)
    else:
        loss = float(np.mean([1.0 - d for d in dices]))

    return {
        "theta": theta,
        "loss": loss,
        "fit_dices": dices,
        "success_flags": sim["success_flags"],
        "messages": sim["messages"],
    }


def fit_patient_random_search(
    simulator,
    bundle: Dict,
    split: Dict,
    bounds: Dict[str, List[float]],
    n_starts: int,
    top_n: int,
    seed: int,
    failure_loss: float,
) -> Dict:
    # Stable per-patient RNG seed to make runs reproducible.
    patient_seed = (seed + abs(hash(bundle["patient_id"])) % (2**31 - 1)) % (2**31 - 1)
    rng = np.random.default_rng(patient_seed)

    trials = []
    for _ in range(n_starts):
        theta = sample_theta(rng, bounds)
        trial = evaluate_theta_on_fit_window(
            simulator=simulator,
            bundle=bundle,
            theta=theta,
            baseline_idx=split["baseline_idx"],
            fit_target_indices=split["fit_target_indices"],
            failure_loss=failure_loss,
        )
        trials.append(trial)

    trials = sorted(trials, key=lambda x: x["loss"])
    return {
        "patient_id": bundle["patient_id"],
        "best": trials[0],
        "top": trials[:top_n],
        "all_trials": trials,
        "split": split,
    }
