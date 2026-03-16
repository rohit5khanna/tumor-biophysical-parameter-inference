from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .data_adapter import derive_brain_mask


def _ensure_tgtk_importable(tgtk_root: Optional[Path]) -> None:
    if tgtk_root is None:
        return
    root = str(tgtk_root.resolve())
    pkg = str((tgtk_root / "TumorGrowthToolkit").resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    if pkg not in sys.path:
        sys.path.insert(0, pkg)


def _build_solver_class(solver_name: str):
    if solver_name == "FK":
        from TumorGrowthToolkit.FK import Solver as _Solver

        return _Solver
    if solver_name == "FK_2c":
        from TumorGrowthToolkit.FK_2c import Solver as _Solver

        return _Solver
    raise ValueError(f"Unsupported solver_name='{solver_name}'. Use 'FK' or 'FK_2c'.")


class TGTKSimulator:
    """
    Thin adapter that:
      - constructs gm/wm from preprocessed image
      - calls TumorGrowthToolkit solver
      - returns binary masks at requested session days
    """

    def __init__(
        self,
        solver_name: str,
        tgtk_root: Optional[Path],
        resolution_factor: float,
        brain_threshold: float,
        mask_threshold: float,
        solver_static_params: Dict,
    ) -> None:
        _ensure_tgtk_importable(tgtk_root)
        self.Solver = _build_solver_class(solver_name)
        self.solver_name = solver_name
        self.resolution_factor = resolution_factor
        self.brain_threshold = brain_threshold
        self.mask_threshold = mask_threshold
        self.solver_static_params = dict(solver_static_params)

    def _prepare_gm_wm(self, bundle: Dict, baseline_idx: int) -> Dict[str, np.ndarray]:
        brain = derive_brain_mask(bundle, baseline_idx=baseline_idx, threshold=self.brain_threshold)
        # Minimal pragmatic choice: homogeneous WM over brain; GM left as zero.
        wm = brain.astype(np.float32)
        gm = np.zeros_like(wm, dtype=np.float32)
        return {"gm": gm, "wm": wm}

    def _field_to_mask(self, field: np.ndarray) -> np.ndarray:
        return (field >= self.mask_threshold).astype(np.uint8)

    def _solve_once(self, bundle: Dict, theta: Dict[str, float], baseline_idx: int, delta_days: float) -> Dict:
        tissue = self._prepare_gm_wm(bundle, baseline_idx=baseline_idx)
        params = {
            "Dw": float(theta["Dw"]),
            "rho": float(theta["rho"]),
            "NxT1_pct": float(theta["NxT1_pct"]),
            "NyT1_pct": float(theta["NyT1_pct"]),
            "NzT1_pct": float(theta["NzT1_pct"]),
            "gm": tissue["gm"],
            "wm": tissue["wm"],
            "resolution_factor": float(self.resolution_factor),
            "stopping_time": float(delta_days),
            "time_series_solution_Nt": None,
        }
        params.update(self.solver_static_params)

        solver = self.Solver(params)
        return solver.solve()

    def predict_for_indices(
        self,
        bundle: Dict,
        theta: Dict[str, float],
        target_indices: List[int],
        baseline_idx: int,
    ) -> Dict:
        days = bundle["days"]
        gt_bin = bundle["label_bin"]
        base_day = float(days[baseline_idx])
        base_mask = gt_bin[baseline_idx].astype(np.uint8)

        pred_masks: List[np.ndarray] = []
        raw_fields: List[np.ndarray] = []
        success_flags: List[bool] = []
        messages: List[str] = []

        for idx in target_indices:
            delta = float(days[idx]) - base_day
            if delta <= 0:
                pred_masks.append(base_mask.copy())
                raw_fields.append(base_mask.astype(np.float32))
                success_flags.append(True)
                messages.append("delta_days<=0, used baseline mask")
                continue

            try:
                res = self._solve_once(bundle=bundle, theta=theta, baseline_idx=baseline_idx, delta_days=delta)
                if not res.get("success", False):
                    pred_masks.append(np.zeros_like(base_mask))
                    raw_fields.append(np.zeros_like(base_mask, dtype=np.float32))
                    success_flags.append(False)
                    messages.append(str(res.get("error", "solver returned success=False")))
                    continue

                if self.solver_name == "FK":
                    field = np.asarray(res["final_state"], dtype=np.float32)
                else:
                    # FK_2c returns dict states; tumor estimate = proliferative + necrotic.
                    fs = res["final_state"]
                    field = np.asarray(fs["P"], dtype=np.float32) + np.asarray(fs["N"], dtype=np.float32)
                    field = np.clip(field, 0.0, 1.0)

                pred_masks.append(self._field_to_mask(field))
                raw_fields.append(field)
                success_flags.append(True)
                messages.append("ok")
            except Exception as exc:  # pragma: no cover - robust run loop
                pred_masks.append(np.zeros_like(base_mask))
                raw_fields.append(np.zeros_like(base_mask, dtype=np.float32))
                success_flags.append(False)
                messages.append(f"exception: {exc}")

        return {
            "target_indices": list(target_indices),
            "pred_masks": pred_masks,
            "raw_fields": raw_fields,
            "success_flags": success_flags,
            "messages": messages,
        }
