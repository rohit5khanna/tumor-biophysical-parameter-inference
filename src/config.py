from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


DEFAULT_PARAM_BOUNDS: Dict[str, List[float]] = {
    "Dw": [0.02, 1.50],
    "rho": [0.001, 0.25],
    "NxT1_pct": [0.20, 0.80],
    "NyT1_pct": [0.20, 0.80],
    "NzT1_pct": [0.20, 0.80],
}

DEFAULT_SOLVER_STATIC_PARAMS: Dict[str, Any] = {
    "RatioDw_Dg": 10.0,
    "th_matter": 0.1,
    "dx_mm": 1.0,
    "dy_mm": 1.0,
    "dz_mm": 1.0,
    "init_scale": 1.0,
    "verbose": False,
}


@dataclass
class ExperimentConfig:
    data_root: Path
    output_root: Path
    patient_ids: List[str]

    fit_sessions: int = 3
    eval_horizons: List[int] = field(default_factory=lambda: [1, 2])

    resolution_factor: float = 0.5
    solver_name: str = "FK"
    tgtk_root: Optional[Path] = None
    init_mode: str = "gaussian"
    init_smoothing_sigma: float = 1.0

    n_starts: int = 20
    top_n: int = 5
    seed: int = 42

    mask_threshold: float = 0.25
    mask_threshold_sweep: List[float] = field(default_factory=list)
    brain_threshold: float = 1e-6
    failure_loss: float = 2.0
    seed_jitter_pct: float = 0.12
    growth_gate_enabled: bool = False
    growth_gate_dilation: int = 0
    growth_gate_apply_in_fit: bool = True
    growth_gate_apply_in_eval: bool = True
    growth_gate_outside_penalty: float = 0.0

    param_bounds: Dict[str, List[float]] = field(default_factory=lambda: dict(DEFAULT_PARAM_BOUNDS))
    solver_static_params: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_SOLVER_STATIC_PARAMS))

    def validate(self) -> None:
        if not self.patient_ids:
            raise ValueError("patient_ids cannot be empty.")
        if self.fit_sessions < 2:
            raise ValueError("fit_sessions must be >= 2.")
        if self.top_n < 1:
            raise ValueError("top_n must be >= 1.")
        if self.n_starts < 1:
            raise ValueError("n_starts must be >= 1.")
        if self.top_n > self.n_starts:
            raise ValueError("top_n cannot be greater than n_starts.")
        if self.resolution_factor <= 0:
            raise ValueError("resolution_factor must be > 0.")
        if self.init_mode not in {"gaussian", "mask_field"}:
            raise ValueError("init_mode must be one of {'gaussian', 'mask_field'}.")
        if self.init_smoothing_sigma < 0:
            raise ValueError("init_smoothing_sigma must be >= 0.")
        if self.mask_threshold <= 0 or self.mask_threshold >= 1:
            raise ValueError("mask_threshold should be in (0, 1).")
        if self.mask_threshold_sweep:
            for t in self.mask_threshold_sweep:
                if t <= 0 or t >= 1:
                    raise ValueError("mask_threshold_sweep values should be in (0, 1).")
        if self.seed_jitter_pct <= 0:
            raise ValueError("seed_jitter_pct must be > 0.")
        if self.growth_gate_dilation < 0:
            raise ValueError("growth_gate_dilation must be >= 0.")
        if self.growth_gate_outside_penalty < 0:
            raise ValueError("growth_gate_outside_penalty must be >= 0.")
        if not self.eval_horizons:
            raise ValueError("eval_horizons cannot be empty.")
        if any(h < 1 for h in self.eval_horizons):
            raise ValueError("eval_horizons must contain positive integers.")
        for k in ("Dw", "rho", "NxT1_pct", "NyT1_pct", "NzT1_pct"):
            if k not in self.param_bounds:
                raise ValueError(f"Missing required param bound: {k}")
            lo, hi = self.param_bounds[k]
            if lo >= hi:
                raise ValueError(f"Invalid bounds for {k}: [{lo}, {hi}]")


def _resolve_tgtk_default(cfg_path: Path) -> Optional[Path]:
    """
    Best-effort default resolution:
      <repo>/../TumorGrowthToolkit
    where <repo> is the project root of this experiment package.
    """
    repo_root = cfg_path.resolve().parents[1]
    candidate = repo_root.parent / "TumorGrowthToolkit"
    if candidate.exists():
        return candidate
    return None


def load_config(path: Union[str, Path]) -> ExperimentConfig:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    param_bounds = dict(DEFAULT_PARAM_BOUNDS)
    param_bounds.update(raw.get("param_bounds", {}))

    solver_static_params = dict(DEFAULT_SOLVER_STATIC_PARAMS)
    solver_static_params.update(raw.get("solver_static_params", {}))

    tgtk_root_val = raw.get("tgtk_root", None)
    if tgtk_root_val is None:
        tgtk_root = _resolve_tgtk_default(cfg_path)
    else:
        tgtk_root = Path(tgtk_root_val)

    cfg = ExperimentConfig(
        data_root=Path(raw["data_root"]),
        output_root=Path(raw["output_root"]),
        patient_ids=list(raw["patient_ids"]),
        fit_sessions=int(raw.get("fit_sessions", 3)),
        eval_horizons=list(raw.get("eval_horizons", [1, 2])),
        resolution_factor=float(raw.get("resolution_factor", 0.5)),
        solver_name=str(raw.get("solver_name", "FK")),
        tgtk_root=tgtk_root,
        init_mode=str(raw.get("init_mode", "gaussian")),
        init_smoothing_sigma=float(raw.get("init_smoothing_sigma", 1.0)),
        n_starts=int(raw.get("n_starts", 20)),
        top_n=int(raw.get("top_n", 5)),
        seed=int(raw.get("seed", 42)),
        mask_threshold=float(raw.get("mask_threshold", 0.25)),
        mask_threshold_sweep=list(raw.get("mask_threshold_sweep", [])),
        brain_threshold=float(raw.get("brain_threshold", 1e-6)),
        failure_loss=float(raw.get("failure_loss", 2.0)),
        seed_jitter_pct=float(raw.get("seed_jitter_pct", 0.12)),
        growth_gate_enabled=bool(raw.get("growth_gate_enabled", False)),
        growth_gate_dilation=int(raw.get("growth_gate_dilation", 0)),
        growth_gate_apply_in_fit=bool(raw.get("growth_gate_apply_in_fit", True)),
        growth_gate_apply_in_eval=bool(raw.get("growth_gate_apply_in_eval", True)),
        growth_gate_outside_penalty=float(raw.get("growth_gate_outside_penalty", 0.0)),
        param_bounds=param_bounds,
        solver_static_params=solver_static_params,
    )

    cfg.validate()
    return cfg
