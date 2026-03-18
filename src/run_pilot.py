from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .config import load_config
from .data_adapter import load_many_patients
from .eval import evaluate_patient
from .fit import fit_patient_random_search
from .paths import prepare_output_dirs
from .solver_wrapper_tgtk import TGTKSimulator
from .splits import make_temporal_split
from .visualize import save_patient_eval_figures, save_patient_fit_figures


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        # Avoid serializing large prediction arrays.
        if obj.size > 2048:
            return {
                "__ndarray__": True,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "summary": {
                    "min": float(np.min(obj)),
                    "max": float(np.max(obj)),
                    "mean": float(np.mean(obj)),
                },
            }
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    return obj


def _save_json(path: Path, data: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data), f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="TumorGrowthToolkit pilot runner")
    parser.add_argument("--config", type=str, default="configs/pilot.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("[INFO] Loaded config")
    print(f"  data_root: {cfg.data_root}")
    print(f"  output_root: {cfg.output_root}")
    print(f"  tgtk_root: {cfg.tgtk_root}")
    print(f"  patients: {cfg.patient_ids}")
    print(
        "  growth_gate: "
        f"enabled={cfg.growth_gate_enabled}, "
        f"dilation={cfg.growth_gate_dilation}, "
        f"fit={cfg.growth_gate_apply_in_fit}, "
        f"eval={cfg.growth_gate_apply_in_eval}, "
        f"outside_penalty={cfg.growth_gate_outside_penalty}"
    )

    np.random.seed(cfg.seed)
    patients = load_many_patients(cfg.data_root, cfg.patient_ids)

    thresholds = cfg.mask_threshold_sweep if cfg.mask_threshold_sweep else [cfg.mask_threshold]
    sweep_rows = []

    for thr in thresholds:
        thr_tag = f"thr_{thr:.2f}".replace(".", "p")
        run_root = cfg.output_root if len(thresholds) == 1 else (cfg.output_root / thr_tag)
        out = prepare_output_dirs(run_root)

        print(f"[INFO] Running threshold={thr:.3f} -> output={run_root}")
        simulator = TGTKSimulator(
            solver_name=cfg.solver_name,
            tgtk_root=cfg.tgtk_root,
            resolution_factor=cfg.resolution_factor,
            brain_threshold=cfg.brain_threshold,
            mask_threshold=thr,
            init_mode=cfg.init_mode,
            init_smoothing_sigma=cfg.init_smoothing_sigma,
            solver_static_params=cfg.solver_static_params,
        )

        summary_rows = []
        for bundle in patients:
            pid = bundle["patient_id"]
            n_sessions = bundle["label"].shape[0]
            print(f"[INFO] Patient {pid} | sessions={n_sessions} | thr={thr:.3f}")

            split = make_temporal_split(
                n_sessions=n_sessions,
                fit_sessions=cfg.fit_sessions,
                eval_horizons=cfg.eval_horizons,
            )

            fit_result = fit_patient_random_search(
                simulator=simulator,
                bundle=bundle,
                split=split,
                bounds=cfg.param_bounds,
                n_starts=cfg.n_starts,
                top_n=cfg.top_n,
                seed=cfg.seed,
                failure_loss=cfg.failure_loss,
                seed_jitter_pct=cfg.seed_jitter_pct,
                growth_gate_enabled=cfg.growth_gate_enabled,
                growth_gate_dilation=cfg.growth_gate_dilation,
                growth_gate_apply_in_fit=cfg.growth_gate_apply_in_fit,
                growth_gate_outside_penalty=cfg.growth_gate_outside_penalty,
            )
            allowed_growth_mask = fit_result.get("growth_gate_allowed_mask", None)
            eval_result = evaluate_patient(
                bundle=bundle,
                split=split,
                fit_result=fit_result,
                simulator=simulator,
                growth_gate_allowed_mask=allowed_growth_mask,
                growth_gate_apply=(cfg.growth_gate_enabled and cfg.growth_gate_apply_in_eval),
            )
            save_patient_fit_figures(
                bundle=bundle,
                fit_result=fit_result,
                simulator=simulator,
                figures_dir=Path(out["figures"]),
                allowed_growth_mask=allowed_growth_mask,
                apply_growth_gate=(cfg.growth_gate_enabled and cfg.growth_gate_apply_in_fit),
            )
            save_patient_eval_figures(bundle=bundle, eval_result=eval_result, figures_dir=Path(out["figures"]))

            _save_json(Path(out["fits"]) / f"{pid}_fit.json", fit_result)
            _save_json(Path(out["metrics"]) / f"{pid}_metrics.json", eval_result)

            row = {
                "mask_threshold": thr,
                "patient_id": pid,
                "fit_best_loss": eval_result["fit_best_loss"],
                "fit_best_mean_dice": eval_result["fit_best_mean_dice"],
                "best_eval_mean_dice": eval_result["best_eval"]["mean_dice"],
                "ensemble_eval_mean_dice": eval_result["ensemble_eval"]["mean_dice"],
                "locf_eval_mean_dice": eval_result["locf_eval"]["mean_dice"],
                "delta_best_vs_locf": eval_result["delta_best_vs_locf"],
            }
            summary_rows.append(row)
            sweep_rows.append(row)

        summary_csv = Path(out["metrics"]) / "summary.csv"
        if summary_rows:
            with summary_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)
        _save_json(Path(out["metrics"]) / "summary.json", {"rows": summary_rows})
        print(f"[INFO] Done threshold={thr:.3f}. Summary: {summary_csv}")

    if len(thresholds) > 1 and sweep_rows:
        sweep_dir = prepare_output_dirs(cfg.output_root)
        sweep_csv = Path(sweep_dir["metrics"]) / "summary_sweep.csv"
        with sweep_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sweep_rows)
        _save_json(Path(sweep_dir["metrics"]) / "summary_sweep.json", {"rows": sweep_rows})
        print(f"[INFO] Sweep summary written to: {sweep_csv}")


if __name__ == "__main__":
    main()
