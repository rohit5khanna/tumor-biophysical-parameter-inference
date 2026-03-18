from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .growth_gate import apply_growth_gate_to_mask
from .metrics import hard_dice


def _normalize_for_display(volume_3d: np.ndarray) -> np.ndarray:
    v = volume_3d.astype(np.float32)
    lo = float(np.percentile(v, 1))
    hi = float(np.percentile(v, 99))
    if hi <= lo:
        return np.zeros_like(v, dtype=np.float32)
    v = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    return v


def _best_slice_index(mask_3d: np.ndarray) -> int:
    z_mass = np.sum(mask_3d > 0, axis=(0, 1))
    if np.max(z_mass) == 0:
        return mask_3d.shape[2] // 2
    return int(np.argmax(z_mass))


def _overlay(base: np.ndarray, mask: np.ndarray, color: str = "r", alpha: float = 0.45):
    plt.imshow(base, cmap="gray", vmin=0.0, vmax=1.0)
    m = np.ma.masked_where(mask <= 0, mask)
    plt.imshow(m, cmap={"r": "Reds", "g": "Greens", "b": "Blues", "c": "Blues"}[color], alpha=alpha, vmin=0, vmax=1)


def save_patient_eval_figures(bundle: Dict, eval_result: Dict, figures_dir: Path) -> None:
    """
    Save one figure per eval session with:
      - GT mask
      - Best PDE prediction
      - LOCF prediction
      - Error maps (FP/FN) for best prediction
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    pid = bundle["patient_id"]
    gt_all = bundle["label_bin"]

    # Use first modality from same target session as background.
    # image_by_session shape: (T, M, H, W, D)
    image_by_session = bundle.get("image_by_session", None)

    target_indices = eval_result["best_eval"]["target_indices"]
    best_preds = eval_result["best_eval"]["pred_masks"]
    locf_preds = eval_result["locf_eval"]["pred_masks"]
    best_dices = eval_result["best_eval"]["dices"]
    locf_dices = eval_result["locf_eval"]["dices"]

    for i, sess_idx in enumerate(target_indices):
        gt = gt_all[sess_idx].astype(np.uint8)
        pred_best = best_preds[i].astype(np.uint8)
        pred_locf = locf_preds[i].astype(np.uint8)

        z = _best_slice_index(gt)

        if image_by_session is not None:
            bg = image_by_session[sess_idx, 0, :, :, :]  # modality 0
            bg = _normalize_for_display(bg)[:, :, z]
        else:
            bg = gt[:, :, z].astype(np.float32)

        gt2d = gt[:, :, z]
        best2d = pred_best[:, :, z]
        locf2d = pred_locf[:, :, z]

        fp = np.logical_and(best2d > 0, gt2d == 0).astype(np.uint8)
        fn = np.logical_and(best2d == 0, gt2d > 0).astype(np.uint8)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(
            f"{pid} | session {sess_idx} | "
            f"best Dice={best_dices[i]:.3f} | LOCF Dice={locf_dices[i]:.3f}",
            fontsize=11,
        )

        plt.sca(axes[0])
        _overlay(bg, gt2d, color="r", alpha=0.45)
        plt.title("GT mask")
        plt.axis("off")

        plt.sca(axes[1])
        plt.imshow(bg, cmap="gray", vmin=0.0, vmax=1.0)
        gtm = np.ma.masked_where(gt2d <= 0, gt2d)
        prm = np.ma.masked_where(best2d <= 0, best2d)
        plt.imshow(gtm, cmap="Reds", alpha=0.35, vmin=0, vmax=1)
        plt.imshow(prm, cmap="Blues", alpha=0.35, vmin=0, vmax=1)
        plt.title("Best PDE (blue) vs GT (red)")
        plt.axis("off")

        plt.sca(axes[2])
        plt.imshow(bg, cmap="gray", vmin=0.0, vmax=1.0)
        gtm = np.ma.masked_where(gt2d <= 0, gt2d)
        lfm = np.ma.masked_where(locf2d <= 0, locf2d)
        plt.imshow(gtm, cmap="Reds", alpha=0.35, vmin=0, vmax=1)
        plt.imshow(lfm, cmap="Greens", alpha=0.35, vmin=0, vmax=1)
        plt.title("LOCF (green) vs GT (red)")
        plt.axis("off")

        plt.sca(axes[3])
        plt.imshow(bg, cmap="gray", vmin=0.0, vmax=1.0)
        fpm = np.ma.masked_where(fp <= 0, fp)
        fnm = np.ma.masked_where(fn <= 0, fn)
        plt.imshow(fpm, cmap="Blues", alpha=0.45, vmin=0, vmax=1)
        plt.imshow(fnm, cmap="Reds", alpha=0.45, vmin=0, vmax=1)
        plt.title("Best errors: FP blue, FN red")
        plt.axis("off")

        fig.tight_layout()
        save_path = figures_dir / f"{pid}_session_{sess_idx:02d}.png"
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        plt.close(fig)


def save_patient_fit_figures(
    bundle: Dict,
    fit_result: Dict,
    simulator,
    figures_dir: Path,
    allowed_growth_mask: np.ndarray | None = None,
    apply_growth_gate: bool = False,
) -> None:
    """
    Save fit-window diagnostics for best theta:
      - GT
      - Best PDE vs GT
      - baseline (session 0) vs GT
      - FP/FN for best prediction
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    split = fit_result["split"]
    baseline_idx = split["baseline_idx"]
    target_indices: List[int] = split["fit_target_indices"]
    if not target_indices:
        return

    theta = fit_result["best"]["theta"]
    sim = simulator.predict_for_indices(
        bundle=bundle,
        theta=theta,
        target_indices=target_indices,
        baseline_idx=baseline_idx,
    )

    pid = bundle["patient_id"]
    gt_all = bundle["label_bin"]
    image_by_session = bundle.get("image_by_session", None)
    baseline_mask = gt_all[baseline_idx].astype(np.uint8)

    for i, sess_idx in enumerate(target_indices):
        gt = gt_all[sess_idx].astype(np.uint8)
        pred_best = sim["pred_masks"][i].astype(np.uint8) if sim["success_flags"][i] else np.zeros_like(gt)
        if allowed_growth_mask is not None and apply_growth_gate:
            pred_best = apply_growth_gate_to_mask(pred_best, allowed_growth_mask)
        pred_base = baseline_mask

        best_d = float(hard_dice(pred_best, gt))
        base_d = float(hard_dice(pred_base, gt))

        z = _best_slice_index(gt)
        if image_by_session is not None:
            bg = image_by_session[sess_idx, 0, :, :, :]
            bg = _normalize_for_display(bg)[:, :, z]
        else:
            bg = gt[:, :, z].astype(np.float32)

        gt2d = gt[:, :, z]
        best2d = pred_best[:, :, z]
        base2d = pred_base[:, :, z]
        fp = np.logical_and(best2d > 0, gt2d == 0).astype(np.uint8)
        fn = np.logical_and(best2d == 0, gt2d > 0).astype(np.uint8)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(
            f"{pid} | FIT session {sess_idx} | best Dice={best_d:.3f} | baseline Dice={base_d:.3f}",
            fontsize=11,
        )

        plt.sca(axes[0])
        _overlay(bg, gt2d, color="r", alpha=0.45)
        plt.title("GT mask")
        plt.axis("off")

        plt.sca(axes[1])
        plt.imshow(bg, cmap="gray", vmin=0.0, vmax=1.0)
        gtm = np.ma.masked_where(gt2d <= 0, gt2d)
        prm = np.ma.masked_where(best2d <= 0, best2d)
        plt.imshow(gtm, cmap="Reds", alpha=0.35, vmin=0, vmax=1)
        plt.imshow(prm, cmap="Blues", alpha=0.35, vmin=0, vmax=1)
        plt.title("Best PDE (blue) vs GT (red)")
        plt.axis("off")

        plt.sca(axes[2])
        plt.imshow(bg, cmap="gray", vmin=0.0, vmax=1.0)
        gtm = np.ma.masked_where(gt2d <= 0, gt2d)
        bsm = np.ma.masked_where(base2d <= 0, base2d)
        plt.imshow(gtm, cmap="Reds", alpha=0.35, vmin=0, vmax=1)
        plt.imshow(bsm, cmap="Greens", alpha=0.35, vmin=0, vmax=1)
        plt.title("Baseline session-0 (green) vs GT (red)")
        plt.axis("off")

        plt.sca(axes[3])
        plt.imshow(bg, cmap="gray", vmin=0.0, vmax=1.0)
        fpm = np.ma.masked_where(fp <= 0, fp)
        fnm = np.ma.masked_where(fn <= 0, fn)
        plt.imshow(fpm, cmap="Blues", alpha=0.45, vmin=0, vmax=1)
        plt.imshow(fnm, cmap="Reds", alpha=0.45, vmin=0, vmax=1)
        plt.title("Best errors: FP blue, FN red")
        plt.axis("off")

        fig.tight_layout()
        save_path = figures_dir / f"{pid}_fit_session_{sess_idx:02d}.png"
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
