from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import numpy as np


def _patient_file(root: Path, patient_id: str, suffix: str) -> Path:
    return root / f"{patient_id}_{suffix}.npy"


def _reshape_image_to_sessions(image: np.ndarray, n_sessions: int) -> np.ndarray:
    """
    TaDiff format: image shape = (M*T, H, W, D), modality-first stacking.
    Return shape: (T, M, H, W, D).
    """
    if image.ndim != 4:
        raise ValueError(f"image must be 4D, got shape={image.shape}")
    if image.shape[0] % n_sessions != 0:
        raise ValueError(
            f"image first dim must be modality x sessions. "
            f"Got image.shape[0]={image.shape[0]}, n_sessions={n_sessions}"
        )
    n_modalities = image.shape[0] // n_sessions
    # (M, T, H, W, D) -> (T, M, H, W, D)
    image_mt = image.reshape(n_modalities, n_sessions, *image.shape[1:])
    return np.transpose(image_mt, (1, 0, 2, 3, 4))


def validate_patient_bundle(bundle: Dict) -> None:
    label = bundle["label"]
    days = bundle["days"]
    treatment = bundle["treatment"]
    image = bundle["image"]

    if label.ndim != 4:
        raise ValueError(f"label must be 4D [T,H,W,D], got {label.shape}")
    if days.ndim != 1 or treatment.ndim != 1:
        raise ValueError("days and treatment must be 1D arrays.")

    n_sessions = label.shape[0]
    if n_sessions != len(days) or n_sessions != len(treatment):
        raise ValueError(
            "Session count mismatch among label/days/treatment: "
            f"{n_sessions}, {len(days)}, {len(treatment)}"
        )

    if image.shape[0] % n_sessions != 0:
        raise ValueError(
            f"image.shape[0]={image.shape[0]} is not divisible by n_sessions={n_sessions}"
        )

    if not np.all(np.isfinite(label)):
        raise ValueError("Label has non-finite values.")
    if not np.all(np.isfinite(image)):
        raise ValueError("Image has non-finite values.")
    if not np.all(np.diff(days) >= 0):
        raise ValueError("days must be nondecreasing.")


def load_patient_bundle(data_root: Union[str, Path], patient_id: str) -> Dict:
    root = Path(data_root)
    image_path = _patient_file(root, patient_id, "image")
    label_path = _patient_file(root, patient_id, "label")
    days_path = _patient_file(root, patient_id, "days")
    treatment_path = _patient_file(root, patient_id, "treatment")

    missing = [p for p in (image_path, label_path, days_path, treatment_path) if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing patient files for {patient_id}: {missing_str}")

    image = np.load(image_path)
    label = np.load(label_path)
    days = np.asarray(np.load(days_path), dtype=np.int64)
    treatment = np.asarray(np.load(treatment_path), dtype=np.int64)

    bundle = {
        "patient_id": patient_id,
        "image": image,
        "label": label,
        "label_bin": (label > 0).astype(np.uint8),
        "days": days,
        "treatment": treatment,
    }
    validate_patient_bundle(bundle)
    bundle["image_by_session"] = _reshape_image_to_sessions(bundle["image"], bundle["label"].shape[0])
    return bundle


def load_many_patients(data_root: Union[str, Path], patient_ids: List[str]) -> List[Dict]:
    return [load_patient_bundle(data_root, pid) for pid in patient_ids]


def derive_brain_mask(bundle: Dict, baseline_idx: int = 0, threshold: float = 1e-6) -> np.ndarray:
    """
    Approximate brain mask from baseline multiparametric MRI in TaDiff preprocessed data.
    """
    img_t = bundle["image_by_session"][baseline_idx]  # (M, H, W, D)
    brain = np.max(img_t, axis=0) > threshold

    # Fallback if image is empty for any reason: use union of label sessions.
    if not np.any(brain):
        brain = np.max(bundle["label_bin"], axis=0) > 0
    return brain.astype(np.float32)
