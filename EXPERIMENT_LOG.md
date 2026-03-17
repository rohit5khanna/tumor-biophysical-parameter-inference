# Experiment Log: Tumor Biophysical Parameter Inference

## Iteration 1 (2026-03-16)

### Goal
Run the first end-to-end pilot using TumorGrowthToolkit (`FK`) on TaDiff-preprocessed SAILOR data and verify:
1. Pipeline executes without errors.
2. Metrics and qualitative figures are generated.
3. Performance against LOCF baseline can be interpreted.

### Setup
1. Solver: `FK` (TumorGrowthToolkit).
2. Fit strategy: random search over parameters (`n_starts=20`, `top_n=5`).
3. Fitting objective: mean fit-window loss `mean(1 - Dice)`.
4. Data split: `fit_sessions=3`, `eval_horizons=[1,2]`.
5. Patients: `sub-01`, `sub-02`.
6. Resolution factor: `0.5`.
7. Mask threshold for PDE field to binary mask: `0.25`.
8. Baseline: LOCF (Last Observation Carried Forward).

### Results (summary.csv)
| patient_id | fit_best_loss | fit_best_mean_dice | best_eval_mean_dice | ensemble_eval_mean_dice | locf_eval_mean_dice | delta_best_vs_locf |
|---|---:|---:|---:|---:|---:|---:|
| sub-01 | 0.9999999999814229 | 1.8577207904547648e-11 | 0.2531319783917756 | 0.050626397132873535 | 0.11300722332109547 | 0.14012475507068012 |
| sub-02 | 0.8999633500757662 | 0.10003664992423374 | 0.1531532370253968 | 0.08331149816513062 | 0.411175186461564 | -0.2580219494361672 |

### Interpretation
1. Technical success: pipeline completed and produced metrics + figures.
2. Fit quality is weak overall:
   - `sub-01` fit Dice is effectively zero.
   - `sub-02` fit Dice is low (~0.10).
3. Future prediction is inconsistent:
   - `sub-01`: best PDE > LOCF.
   - `sub-02`: best PDE << LOCF.
4. Ensemble average is lower than best-model performance, indicating many poor parameter samples.
5. Current model assumptions are likely too coarse for robust fitting:
   - simplified anatomy proxy (`wm=brain mask`, `gm=0`),
   - fully random seed-location search,
   - limited search budget (`n_starts=20`).

### Decision
Proceed with a second FK iteration before attempting anisotropic/DTI methods.

### Next Iteration Plan (Iteration 2)
1. Improve initialization:
   - seed center around baseline tumor center-of-mass (not fully random).
2. Increase search quality:
   - `n_starts: 20 -> 80/100` for at least one patient.
3. Tune discretization threshold:
   - test `mask_threshold` in `[0.05, 0.1, 0.2, 0.3]`.
4. Add fit-window visual diagnostics (not only future-session visuals).
5. Keep LOCF as required baseline and continue go/no-go decisions using `delta_best_vs_locf`.

### Notes
1. DTI anisotropic solver (`FK_DTI`) is paused because current preprocessed TaDiff SAILOR files do not include diffusion tensor fields.
2. True patient-specific diffusion tensors require raw diffusion MRI (DWI + bvals/bvecs).

---

## Iteration 2 Code Update (Implemented, pending run)

### Changes applied
1. COM-guided seed sampling in random search:
   - baseline tumor center-of-mass used for `NxT1_pct/NyT1_pct/NzT1_pct`,
   - Gaussian jitter controlled by `seed_jitter_pct`.
2. Threshold sweep support:
   - optional `mask_threshold_sweep` list in config,
   - threshold-specific output folders to prevent overwrite.
3. Added fit-window visual diagnostics:
   - `*_fit_session_*.png` per patient/session,
   - compare best PDE vs GT vs baseline(session-0 carry-forward).
4. Runner now reports threshold in summary rows.

### New config knobs
1. `seed_jitter_pct` (default `0.12`)
2. `mask_threshold_sweep` (optional list, e.g. `[0.05, 0.1, 0.2, 0.3]`)

## Iteration 2 Run Results (2026-03-17)

### Setup used
1. Solver: `FK`
2. Patients: `sub-01`, `sub-02`
3. Seed strategy: COM-guided + jitter
4. Threshold sweep: `[0.05, 0.1, 0.2, 0.3]`
5. Baseline: LOCF

### Results (summary_sweep.csv)
| mask_threshold | patient_id | fit_best_loss | fit_best_mean_dice | best_eval_mean_dice | ensemble_eval_mean_dice | locf_eval_mean_dice | delta_best_vs_locf |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.05 | sub-01 | 0.6673142112679142 | 0.33268578873208576 | 0.446852265856689 | 0.39183419942855835 | 0.11300722332109547 | 0.33384504253559355 |
| 0.05 | sub-02 | 0.7994411245535140 | 0.20055887544648587 | 0.19989681415517815 | 0.15931102633476257 | 0.41117518646156400 | -0.21127837230638583 |
| 0.10 | sub-01 | 0.7301918263305839 | 0.26980817366941610 | 0.46015323731232470 | 0.39990460872650146 | 0.11300722332109547 | 0.34714601399122924 |
| 0.10 | sub-02 | 0.8073021093481857 | 0.19269789065181430 | 0.21815460778652650 | 0.17782208323478700 | 0.41117518646156400 | -0.19302057867503747 |
| 0.20 | sub-01 | 0.7900953629608074 | 0.20990463703919265 | 0.39326256189924164 | 0.39553529024124146 | 0.11300722332109547 | 0.28025533857814616 |
| 0.20 | sub-02 | 0.8231838145852319 | 0.17681618541476804 | 0.23850253331955407 | 0.19623425602912903 | 0.41117518646156400 | -0.17267265314200990 |
| 0.30 | sub-01 | 0.8318405516176814 | 0.16815944838231847 | 0.39743660984307650 | 0.39443454146385193 | 0.11300722332109547 | 0.28442938652198100 |
| 0.30 | sub-02 | 0.8384897831130229 | 0.16151021688697703 | 0.25440085305094995 | 0.18891364336013794 | 0.41117518646156400 | -0.15677433341061403 |

### Interpretation
1. Major improvement over Iteration 1 for `sub-01`:
   - PDE is consistently better than LOCF across all thresholds.
   - Best observed delta at threshold `0.10` (`+0.347`).
2. `sub-02` remains a hard case:
   - PDE is still below LOCF at every threshold.
   - Gap narrows as threshold increases (least negative at `0.30`).
3. Threshold matters substantially for reported Dice.
4. Average signal across these two patients is positive at threshold `0.10`, but not robust across patients yet.

### Decision
Proceed to Iteration 3 with targeted robustness improvements, not a full solver switch yet.

### Next Iteration Plan (Iteration 3)
1. Keep global default threshold at `0.10` for primary runs.
2. Deep-search hard case(s):
   - run `sub-02` with higher `n_starts` (e.g., 300) to test optimization ceiling.
3. Add treatment-phase parameterization:
   - first extension: piecewise `rho` by treatment phase (`rho_CRT`, `rho_TMZ`).
4. Keep LOCF as mandatory baseline and continue to track `delta_best_vs_locf`.

## Iteration 3 Code Update (Implemented, pending run)

### Changes applied
1. Added initialization mode switch in config:
   - `init_mode: gaussian`
   - `init_mode: mask_field`
2. Implemented mask-field initialization path for `FK`:
   - baseline tumor mask used as initial PDE field,
   - optional smoothing via `init_smoothing_sigma`,
   - same fitting/evaluation pipeline preserved for fair comparison.
3. Runner now passes init controls to simulator.

### New config knobs
1. `init_mode` (`gaussian` or `mask_field`)
2. `init_smoothing_sigma` (default `1.0`)
