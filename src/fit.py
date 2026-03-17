from __future__ import annotations

from typing import Dict, List

import numpy as np

from .metrics import hard_dice


def sample_theta(rng: np.random.Generator, bounds: Dict[str, List[float]]) -> Dict[str, float]:
    theta = {}
    for k, (lo, hi) in bounds.items():
        theta[k] = float(rng.uniform(lo, hi))
    return theta


def _clip(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def baseline_center_pct(bundle: Dict, baseline_idx: int) -> Dict[str, float]:
    """
    Compute tumor center-of-mass from baseline binary mask and convert to pct in [0,1].
    If empty baseline mask, fall back to geometric center.
    """
    m = bundle["label_bin"][baseline_idx].astype(np.float32)
    shape = m.shape
    mass = np.sum(m)

    if mass <= 0:
        cx, cy, cz = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0, (shape[2] - 1) / 2.0
    else:
        xs, ys, zs = np.indices(shape)
        cx = float(np.sum(xs * m) / mass)
        cy = float(np.sum(ys * m) / mass)
        cz = float(np.sum(zs * m) / mass)

    return {
        "NxT1_pct": float(cx / max(shape[0] - 1, 1)),
        "NyT1_pct": float(cy / max(shape[1] - 1, 1)),
        "NzT1_pct": float(cz / max(shape[2] - 1, 1)),
    }


def sample_theta_com_guided(
    rng: np.random.Generator,
    bounds: Dict[str, List[float]],
    center_pct: Dict[str, float],
    jitter_pct: float,
) -> Dict[str, float]:
    """
    Sample Dw/rho uniformly from bounds, but sample seed location around baseline COM.
    """
    theta = {}
    for k, (lo, hi) in bounds.items():
        if k in ("NxT1_pct", "NyT1_pct", "NzT1_pct"):
            c = center_pct[k]
            span = (hi - lo) * jitter_pct
            v = float(rng.normal(loc=c, scale=span))
            theta[k] = _clip(v, lo, hi)
        else:
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
    seed_jitter_pct: float = 0.12,
) -> Dict:
    # Stable per-patient RNG seed to make runs reproducible.
    patient_seed = (seed + abs(hash(bundle["patient_id"])) % (2**31 - 1)) % (2**31 - 1)
    rng = np.random.default_rng(patient_seed)

    center_pct = baseline_center_pct(bundle, baseline_idx=split["baseline_idx"])

    trials = []
    for _ in range(n_starts):
        theta = sample_theta_com_guided(
            rng=rng,
            bounds=bounds,
            center_pct=center_pct,
            jitter_pct=seed_jitter_pct,
        )
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
        "seed_center_pct": center_pct,
        "best": trials[0],
        "top": trials[:top_n],
        "all_trials": trials,
        "split": split,
    }
