from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

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
        init_mode: str,
        init_smoothing_sigma: float,
        solver_static_params: Dict,
    ) -> None:
        _ensure_tgtk_importable(tgtk_root)
        self.Solver = _build_solver_class(solver_name)
        self.solver_name = solver_name
        self.resolution_factor = resolution_factor
        self.brain_threshold = brain_threshold
        self.mask_threshold = mask_threshold
        self.init_mode = init_mode
        self.init_smoothing_sigma = init_smoothing_sigma
        self.solver_static_params = dict(solver_static_params)

    def _prepare_gm_wm(self, bundle: Dict, baseline_idx: int) -> Dict[str, np.ndarray]:
        brain = derive_brain_mask(bundle, baseline_idx=baseline_idx, threshold=self.brain_threshold)
        # Minimal pragmatic choice: homogeneous WM over brain; GM left as zero.
        wm = brain.astype(np.float32)
        gm = np.zeros_like(wm, dtype=np.float32)
        return {"gm": gm, "wm": wm}

    def _field_to_mask(self, field: np.ndarray) -> np.ndarray:
        return (field >= self.mask_threshold).astype(np.uint8)

    def _build_initial_field(self, bundle: Dict, baseline_idx: int) -> np.ndarray:
        init = bundle["label_bin"][baseline_idx].astype(np.float32)
        if self.init_smoothing_sigma > 0:
            init = gaussian_filter(init, sigma=self.init_smoothing_sigma)
        if np.max(init) > 0:
            init = init / float(np.max(init))
        return np.clip(init, 0.0, 1.0)

    def _solve_fk_with_custom_init(
        self,
        bundle: Dict,
        theta: Dict[str, float],
        baseline_idx: int,
        delta_days: float,
    ) -> Dict:
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
        Dw = params["Dw"]
        rho = params["rho"]
        ratio = params.get("RatioDw_Dg", 10.0)
        th_matter = params.get("th_matter", 0.1)
        dx_mm = params.get("dx_mm", 1.0)
        dy_mm = params.get("dy_mm", 1.0)
        dz_mm = params.get("dz_mm", 1.0)
        stopping_time = params.get("stopping_time", 100.0)
        stopping_volume = params.get("stopping_volume", np.inf)

        sGM = params["gm"]
        sWM = params["wm"]
        res_factor = params["resolution_factor"]

        sGM_low = zoom(sGM, res_factor, order=1)
        sWM_low = zoom(sWM, res_factor, order=1)
        original_shape = sGM_low.shape
        new_shape = sGM.shape
        extrapolate_factor = tuple(new_sz / float(orig_sz) for new_sz, orig_sz in zip(new_shape, original_shape))

        # Use baseline tumor mask as initial field.
        init_full = self._build_initial_field(bundle, baseline_idx=baseline_idx)
        init_low = zoom(init_full, res_factor, order=1)
        if np.max(init_low) > 0:
            init_low = init_low / float(np.max(init_low))
        init_low = np.clip(init_low, 0.0, 1.0)

        dx = dx_mm / res_factor
        dy = dy_mm / res_factor
        dz = dz_mm / res_factor

        Nt = np.max(
            [
                stopping_time * Dw / np.power(np.min([dx, dy, dz]), 2) * 8 + 100,
                stopping_time * rho * 1.1,
            ]
        )
        dt = stopping_time / Nt
        N_steps = int(np.ceil(Nt))

        try:
            cropped_GM, cropped_WM, A, crop_coords = solver.crop_tissues_and_tumor(
                sGM_low,
                sWM_low,
                init_low,
                margin=2,
                threshold=0.5,
            )
            D_domain = solver.get_D(cropped_WM, cropped_GM, th_matter, Dw, ratio)

            final_time = None
            volume = 0.0
            for t in range(N_steps):
                A = solver.FK_update(A, D_domain, rho, dt, dx, dy, dz)
                volume = dx * dy * dz * np.sum(A)
                if volume >= stopping_volume:
                    final_time = t * dt
                    break
            if final_time is None:
                final_time = stopping_time

            A_full_low = solver.restore_tumor(sGM_low.shape, A, crop_coords)
            A_full = np.array(zoom(A_full_low, extrapolate_factor, order=1), dtype=np.float32)
            A_full = np.clip(A_full, 0.0, 1.0)
            init_full_rescaled = np.array(zoom(init_low, extrapolate_factor, order=1), dtype=np.float32)
            init_full_rescaled = np.clip(init_full_rescaled, 0.0, 1.0)

            return {
                "success": True,
                "initial_state": init_full_rescaled,
                "final_state": A_full,
                "final_time": float(final_time),
                "final_volume": float(volume),
                "stopping_criteria": "volume" if volume >= stopping_volume else "time",
                "Dw": Dw,
                "rho": rho,
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _solve_once(self, bundle: Dict, theta: Dict[str, float], baseline_idx: int, delta_days: float) -> Dict:
        if self.solver_name == "FK" and self.init_mode == "mask_field":
            return self._solve_fk_with_custom_init(
                bundle=bundle,
                theta=theta,
                baseline_idx=baseline_idx,
                delta_days=delta_days,
            )

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
