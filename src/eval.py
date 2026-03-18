from __future__ import annotations

from typing import Dict

import numpy as np

from .baselines import evaluate_locf
from .rollout import rollout_ensemble, rollout_theta


def evaluate_patient(
    bundle: Dict,
    split: Dict,
    fit_result: Dict,
    simulator,
    growth_gate_allowed_mask=None,
    growth_gate_apply: bool = False,
) -> Dict:
    eval_indices = split["eval_indices"]
    baseline_idx = split["baseline_idx"]

    best_theta = fit_result["best"]["theta"]
    top_thetas = [x["theta"] for x in fit_result["top"]]

    best_eval = rollout_theta(
        simulator=simulator,
        bundle=bundle,
        theta=best_theta,
        baseline_idx=baseline_idx,
        target_indices=eval_indices,
        allowed_growth_mask=growth_gate_allowed_mask,
        apply_growth_gate=growth_gate_apply,
    )
    ensemble_eval = rollout_ensemble(
        simulator=simulator,
        bundle=bundle,
        thetas=top_thetas,
        baseline_idx=baseline_idx,
        target_indices=eval_indices,
        allowed_growth_mask=growth_gate_allowed_mask,
        apply_growth_gate=growth_gate_apply,
    )
    locf = evaluate_locf(bundle=bundle, baseline_idx=baseline_idx, target_indices=eval_indices)

    return {
        "patient_id": bundle["patient_id"],
        "fit_best_loss": float(fit_result["best"]["loss"]),
        "fit_best_mean_dice": float(np.mean(fit_result["best"]["fit_dices"])),
        "best_eval": best_eval,
        "ensemble_eval": ensemble_eval,
        "locf_eval": locf,
        "delta_best_vs_locf": float(best_eval["mean_dice"] - locf["mean_dice"]),
        "growth_gate_apply_in_eval": bool(growth_gate_apply),
    }
