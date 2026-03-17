# Tumor Biophysical Parameter Inference (Pilot)

Minimal experiment pipeline to:
1. load TaDiff-preprocessed SAILOR patient data,
2. fit TumorGrowthToolkit PDE parameters on early sessions,
3. roll forward to future sessions,
4. evaluate Dice against ground truth and compare to LOCF baseline.

## Data format
Per patient, this pipeline expects:
- `sub-XX_image.npy`
- `sub-XX_label.npy`
- `sub-XX_days.npy`
- `sub-XX_treatment.npy`

## Configure
Edit `configs/pilot.yaml`:
- `data_root`: directory containing per-patient `.npy` files
- `output_root`: where metrics/fits/preds are saved
- `patient_ids`: pilot patients
- optional: `tgtk_root`

## Run
From repo root:

```bash
python -m src.run_pilot --config configs/pilot.yaml
```

Outputs go to:
- `outputs/.../fits`
- `outputs/.../metrics`
- `outputs/.../figures` (GT vs PDE vs LOCF visual comparisons)

Notes:
- Fit-window diagnostics are also saved as figures (`*_fit_session_*.png`).
- You can run threshold sweeps with `mask_threshold_sweep` in config.
