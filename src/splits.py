from __future__ import annotations

from typing import Dict, List


def make_temporal_split(
    n_sessions: int,
    fit_sessions: int,
    eval_horizons: List[int],
) -> Dict:
    """
    Returns split dict:
      baseline_idx: int (always 0)
      fit_target_indices: list[int]   -> targets used in fitting objective
      fit_last_idx: int
      eval_indices: list[int]         -> future sessions for evaluation
    """
    if n_sessions < 3:
        raise ValueError(f"Need at least 3 sessions, got {n_sessions}.")
    if fit_sessions < 2:
        raise ValueError(f"fit_sessions must be >=2, got {fit_sessions}.")
    if fit_sessions >= n_sessions:
        raise ValueError(
            f"fit_sessions={fit_sessions} must be < n_sessions={n_sessions} "
            "to leave future sessions for evaluation."
        )

    baseline_idx = 0
    fit_target_indices = list(range(1, fit_sessions))
    fit_last_idx = fit_sessions - 1

    eval_indices = [fit_last_idx + h for h in eval_horizons if fit_last_idx + h < n_sessions]
    if not eval_indices:
        # Fallback: all remaining sessions after fit window.
        eval_indices = list(range(fit_sessions, n_sessions))
    if not eval_indices:
        raise ValueError("No valid eval indices available for this patient.")

    return {
        "baseline_idx": baseline_idx,
        "fit_target_indices": fit_target_indices,
        "fit_last_idx": fit_last_idx,
        "eval_indices": eval_indices,
    }
