from __future__ import annotations

from typing import Dict, List

import numpy as np

from .growth_gate import apply_growth_gate_to_mask, build_allowed_growth_mask, outside_growth_fraction
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
    allowed_growth_mask: np.ndarray | None = None,
    apply_growth_gate: bool = False,
    outside_penalty_weight: float = 0.0,
) -> Dict:
    sim = simulator.predict_for_indices(
        bundle=bundle,
        theta=theta,
        target_indices=fit_target_indices,
        baseline_idx=baseline_idx,
    )

    gt = bundle["label_bin"]
    dices: List[float] = []
    outside_rates: List[float] = []
    failed = False
    for i, idx in enumerate(fit_target_indices):
        if not sim["success_flags"][i]:
            failed = True
            dices.append(0.0)
            outside_rates.append(0.0)
        else:
            pred = sim["pred_masks"][i].astype(np.uint8)
            if allowed_growth_mask is not None:
                outside_rates.append(outside_growth_fraction(pred, allowed_growth_mask))
                if apply_growth_gate:
                    pred = apply_growth_gate_to_mask(pred, allowed_growth_mask)
            else:
                outside_rates.append(0.0)
            d = hard_dice(pred, gt[idx])
            dices.append(float(d))

    if failed:
        loss = float(failure_loss)
    else:
        base_loss = float(np.mean([1.0 - d for d in dices]))
        outside_penalty = float(outside_penalty_weight * np.mean(outside_rates))
        loss = float(base_loss + outside_penalty)

    return {
        "theta": theta,
        "loss": loss,
        "base_loss": float(np.mean([1.0 - d for d in dices])) if dices else float("nan"),
        "outside_penalty": float(outside_penalty_weight * np.mean(outside_rates)) if outside_rates else 0.0,
        "outside_rates": outside_rates,
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
    growth_gate_enabled: bool = False,
    growth_gate_dilation: int = 0,
    growth_gate_apply_in_fit: bool = True,
    growth_gate_outside_penalty: float = 0.0,
) -> Dict:
    # Stable per-patient RNG seed to make runs reproducible.
    patient_seed = (seed + abs(hash(bundle["patient_id"])) % (2**31 - 1)) % (2**31 - 1)
    rng = np.random.default_rng(patient_seed)

    center_pct = baseline_center_pct(bundle, baseline_idx=split["baseline_idx"])
    allowed_growth_mask = None
    if growth_gate_enabled:
        source_indices = list(range(split["fit_last_idx"] + 1))
        allowed_growth_mask = build_allowed_growth_mask(
            bundle=bundle,
            source_indices=source_indices,
            dilation_iters=growth_gate_dilation,
        )

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
            allowed_growth_mask=allowed_growth_mask,
            apply_growth_gate=(growth_gate_enabled and growth_gate_apply_in_fit),
            outside_penalty_weight=(growth_gate_outside_penalty if growth_gate_enabled else 0.0),
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
        "growth_gate": {
            "enabled": bool(growth_gate_enabled),
            "dilation": int(growth_gate_dilation),
            "apply_in_fit": bool(growth_gate_apply_in_fit),
            "outside_penalty_weight": float(growth_gate_outside_penalty),
        },
        "growth_gate_allowed_mask": allowed_growth_mask,
    }
