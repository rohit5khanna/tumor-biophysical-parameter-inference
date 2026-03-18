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

## Iteration 3 Run Results (2026-03-17)

### Setup used
1. Solver: `FK`
2. Initialization: `init_mode=mask_field`, `init_smoothing_sigma=1.0`
3. Seed strategy: COM-guided + jitter (`seed_jitter_pct=0.12`)
4. Threshold: `mask_threshold=0.1`
5. Patients: `sub-01`, `sub-02`
6. Search budget: `n_starts=100`, `top_n=5`

### Results (summary.csv)
| mask_threshold | patient_id | fit_best_loss | fit_best_mean_dice | best_eval_mean_dice | ensemble_eval_mean_dice | locf_eval_mean_dice | delta_best_vs_locf |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.1 | sub-01 | 0.415094 | 0.584906 | 0.531048 | 0.524643 | 0.113007 | 0.418041 |
| 0.1 | sub-02 | 0.583757 | 0.416243 | 0.348149 | 0.281314 | 0.411175 | -0.063026 |

### Interpretation
1. Clear improvement over previous FK initialization:
   - `sub-01` improved strongly in both fit and future Dice.
   - `sub-02` improved substantially and is now close to LOCF.
2. `mask_field` initialization appears materially better than single-gaussian init for irregular tumor shapes.
3. Remaining gap: `sub-02` still under LOCF by a small margin (`-0.063`), indicating room for optimization/model refinement.

### Decision
Proceed to Iteration 4 focused on hard-case recovery:
1. Increase `n_starts` for `sub-02`.
2. Sweep initialization smoothing (`init_smoothing_sigma`) while keeping threshold fixed at `0.1`.

## Iteration 4 Run Results (2026-03-17)

### Setup used
1. Solver: `FK`
2. Patient: `sub-02` (hard-case focused run)
3. Initialization: `init_mode=mask_field`
4. Threshold: `mask_threshold=0.1`
5. Search budget: `n_starts=300`, `top_n=10`
6. Smoothing sweep: `init_smoothing_sigma in [0.5, 1.0, 1.5]`

### Results (aggregated table)
| patient_id | fit_best_mean_dice | best_eval_mean_dice | locf_eval_mean_dice | delta_best_vs_locf | run_path |
|---|---:|---:|---:|---:|---|
| sub-02 | 0.476842 | 0.363713 | 0.411175 | -0.047462 | `/content/drive/MyDrive/tumor_pde_outputs/pilot...` |
| sub-02 | 0.443114 | 0.337715 | 0.411175 | -0.073460 | `/content/drive/MyDrive/tumor_pde_outputs/pilot...` |
| sub-02 | 0.453890 | 0.381423 | 0.411175 | -0.029753 | `/content/drive/MyDrive/tumor_pde_outputs/pilot...` |

### Interpretation
1. Deep-search + smoothing sweep improved the hard case further:
   - previous best delta was `-0.063`,
   - now best observed delta is `-0.0298` (close to parity with LOCF).
2. Fit-window Dice and future Dice are not perfectly aligned (expected), suggesting mild overfit potential.
3. Current model remains slightly below LOCF for `sub-02`, but the residual gap is now small.

### Decision
Proceed to broader robustness check before major model rewrite.

### Next Iteration Plan (Iteration 5)
1. Run a batch evaluation on 6 patients using:
   - `init_mode=mask_field`,
   - `init_smoothing_sigma=1.5` (best from focused sweep),
   - `mask_threshold=0.1`,
   - moderate search budget (`n_starts=120`).
2. Use decision rule:
   - if >=4/6 patients beat LOCF (positive delta), continue optimizing this family;
   - otherwise prioritize model-structure changes.

## Iteration 5 Run Plan (Prepared)

### Planned command-level setup
1. Patients: `sub-01, sub-02, sub-03, sub-04, sub-06, sub-07`
2. Fit: `fit_sessions=3`, `eval_horizons=[1,2]`
3. Search: `n_starts=120`, `top_n=10`
4. Output root: `pilot_run_005_batch6_maskfield`

### Pending
Execution and result interpretation pending.

## Iteration 5 Run Results (2026-03-17, 3-patient subset)

### Setup used
1. Initialization: `mask_field`
2. Threshold: `0.1`
3. Patients evaluated: `sub-02`, `sub-03`, `sub-04`

### Results (summary.csv)
| mask_threshold | patient_id | fit_best_loss | fit_best_mean_dice | best_eval_mean_dice | ensemble_eval_mean_dice | locf_eval_mean_dice | delta_best_vs_locf |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.1 | sub-02 | 0.552732 | 0.447268 | 0.364501 | 0.308085 | 0.411175 | -0.046674 |
| 0.1 | sub-03 | 0.367093 | 0.632907 | 0.582452 | 0.516017 | 0.564696 | +0.017756 |
| 0.1 | sub-04 | 0.730699 | 0.269301 | 0.125737 | 0.197683 | 0.210893 | -0.085156 |

### Aggregate
1. Wins vs LOCF: `1 / 3`
2. Mean delta: `-0.0380`
3. Median delta: `-0.0467`
4. Best delta: `+0.0178`
5. Worst delta: `-0.0852`

### Qualitative observations from visual inspection
1. Positive/near-positive cases tend to have more connected, uniform spread.
2. Failure cases (`sub-02`, `sub-04`) show scattered or elongated/branch-like topology.
3. Current FK family often predicts compact/globular spread and can lock onto one component while missing dominant elongated branches.

### Decision
Proceed with failed-case-focused experiments on `sub-02` and `sub-04`, and prioritize structural initialization upgrades over pure random-search scaling.

### Next Iteration Plan (Iteration 6)
1. Run focused deep-search on failed cases (`sub-02`, `sub-04`) as a ceiling check.
2. Implement topology-aware initialization:
   - connected-component-preserving mask initialization (first),
   - optional explicit multi-seed initialization (second).
3. Re-evaluate against LOCF with the same protocol.

## Iteration 6 Run Results (2026-03-17, failed-case deep search)

### Setup used
1. Output root: `pilot_run_006_failedcases`
2. Patients: `sub-02`, `sub-04`
3. Fit/eval split: `fit_sessions=3`, `eval_horizons=[1,2]`
4. Initialization: `init_mode=mask_field`, `init_smoothing_sigma=1.5`
5. Threshold: `mask_threshold=0.1` (single-value sweep)
6. Search budget: `n_starts=400`, `top_n=10`
7. Seed jitter: `seed_jitter_pct=0.12`

### Results (summary.csv)
| mask_threshold | patient_id | fit_best_loss | fit_best_mean_dice | best_eval_mean_dice | ensemble_eval_mean_dice | locf_eval_mean_dice | delta_best_vs_locf |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.1 | sub-02 | 0.565008 | 0.434992 | 0.348431 | 0.295950 | 0.411175 | -0.062744 |
| 0.1 | sub-04 | 0.733021 | 0.266979 | 0.159035 | 0.223214 | 0.210893 | -0.051858 |

### Aggregate
1. Wins vs LOCF: `0 / 2`
2. Mean delta: `-0.0573`

### Qualitative observations from visual inspection
1. `sub-02` tumor appears dissipated, with one main glob-like mass thinly connected to scattered peripheral foci.
2. The model captures the main glob reasonably well (sometimes overestimates it), but tends to model scattered foci as additional globular growth.
3. This mismatch likely contributes to consistent negative Dice delta relative to LOCF on failed cases.

### Interpretation
1. Increasing search budget alone (`n_starts=400`) did not recover failed cases.
2. The dominant error mode appears structural (shape/topology mismatch), not just optimization insufficiency.
3. Next gains are more likely from model-structure changes (anisotropy/topology-aware initialization) than from further random-search scaling.

## Iteration 7 Run Results (2026-03-18, failed-case re-run)

### Setup used
1. Failed-case pair: `sub-02`, `sub-04`
2. Threshold: `mask_threshold=0.1`
3. Initialization: `mask_field` (sigma run, assumed `init_smoothing_sigma=1.5`)
4. Fit budget: high-search setting (same failed-case regime)
5. Runtime observation: this single run took ~3 hours in Colab.

### Results (summary.csv)
| mask_threshold | patient_id | fit_best_loss | fit_best_mean_dice | best_eval_mean_dice | ensemble_eval_mean_dice | locf_eval_mean_dice | delta_best_vs_locf |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.1 | sub-02 | 0.538224 | 0.461776 | 0.328991 | 0.231533 | 0.402329 | -0.073339 |
| 0.1 | sub-04 | 0.735340 | 0.264660 | 0.303972 | 0.279711 | 0.228869 | +0.075103 |

### Aggregate
1. Wins vs LOCF: `1 / 2`
2. Mean delta: `+0.000882` (near parity with LOCF overall)

### Interpretation
1. Mixed response: strong gain for `sub-04`, but `sub-02` remains below LOCF.
2. The pipeline can recover some irregular cases, but robustness across dissipated/scattered morphology is still limited.
3. Runtime remains a practical bottleneck; single long runs are vulnerable to Colab disconnects.

### Operational decision
1. Continue with one-sigma-per-run checkpoints to reduce failure risk.
2. Prioritize structural model changes for `sub-02` (anisotropy/topology-aware init) rather than only increasing search.

## Iteration 8 Plan (2026-03-18, sub-02 stability/growth-gating test)

### Motivation
1. Repeated failed-case results indicate `sub-02` has a structural mismatch with isotropic/globular FK behavior.
2. Visual review suggests over-growth in scattered/low-confidence peripheral regions that remain stable or weakly evolving across sessions.
3. A low-cost diagnostic is needed before major solver rewrites (anisotropy/tensor models).

### Hypothesis
If we constrain predicted growth to regions supported by early-session morphology, Dice on `sub-02` should improve relative to unconstrained FK rollout.

### Proposed approach
1. Use existing best-fit `sub-02` run outputs (`fit.json` + standard simulator rollout).
2. Build an "allowed growth zone" from the union of fit-window tumor masks.
3. Apply controlled morphological dilation to that zone (`dilation = 0..4`) to permit limited expansion.
4. Post-hoc gate predicted masks to allowed zones and recompute Dice on eval sessions.
5. Track outside-growth fraction (`pred ∩ ~allowed`) as a diagnostic.

### Why this is useful
1. It directly tests whether failure is due to uncontrolled peripheral spread rather than poor scalar parameter search.
2. It is computationally cheap and robust to Colab disconnects.
3. Positive signal would justify integrating a formal growth-gating penalty into fitting loss.

### Planned metrics
1. Mean Dice per dilation level (eval window).
2. Outside-growth fraction per dilation level.
3. Best dilation vs baseline (ungated) Dice delta.

### Status
Implementation started; numeric outcomes pending.

## Iteration 8 Results (2026-03-18, sub-02 growth-gating diagnostic)

### Setup
1. Patient: `sub-02`
2. Eval indices: `[3, 4]`
3. Allowed-growth zone: union of fit-window tumor masks.
4. Gating sweep: dilation iterations `0..4`.
5. Metrics: mean Dice after gating, per-session Dice, outside-growth fraction (`pred ∩ ~allowed`).

### Results
| dilation | mean_dice | outside_rate | dice_session_3 | dice_session_4 |
|---:|---:|---:|---:|---:|
| 0 | 0.4767 | 0.0044 | 0.4619 | 0.4916 |
| 1 | 0.4410 | 0.0022 | 0.3943 | 0.4878 |
| 2 | 0.4064 | 0.0008 | 0.3493 | 0.4635 |
| 3 | 0.3908 | 0.0001 | 0.3317 | 0.4499 |
| 4 | 0.3881 | 0.0000 | 0.3291 | 0.4472 |

### Interpretation
1. Strict gating (`dilation=0`) produced the highest Dice in this diagnostic.
2. As allowed zone expands (higher dilation), Dice drops toward ungated behavior.
3. This supports the hypothesis that excess peripheral spread is a key failure mode for `sub-02`.
4. Even small outside-growth fractions can be high-impact when morphology is sparse/scattered.

### Next action
1. Add an explicit comparison to ungated Dice in the same script context for exact parity checks.
2. If confirmed, integrate either:
   - a hard growth gate, or
   - a soft outside-growth penalty
   into fitting/evaluation for failed-case modes.

## Iteration 9 Code Update (2026-03-18, growth-gated fit/eval support)

### Implemented changes
1. Added optional growth-gating module to pipeline (`src/growth_gate.py`):
   - build allowed-growth mask from fit-window sessions,
   - optional dilation,
   - hard gate application,
   - outside-growth fraction metric.
2. Added new config flags:
   - `growth_gate_enabled`
   - `growth_gate_dilation`
   - `growth_gate_apply_in_fit`
   - `growth_gate_apply_in_eval`
   - `growth_gate_outside_penalty`
3. Updated fitting (`src/fit.py`):
   - optional hard gating during fit Dice computation,
   - optional outside-growth penalty in loss,
   - stores gate metadata and allowed mask in fit results.
4. Updated rollout/evaluation (`src/rollout.py`, `src/eval.py`):
   - optional gating during eval for best and ensemble rollouts,
   - logs outside-growth rates.
5. Updated fit-figure generation (`src/visualize.py`) to respect gated-fit mode.
6. Updated runner wiring (`src/run_pilot.py`) and default config template (`configs/pilot.yaml`).

### Status
Code implemented and compile-validated; `sub-02` proper gated fit run pending.
