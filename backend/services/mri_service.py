from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import kurtosis, skew

ALLOWED_SHAPES = {(204, 256, 48), (228, 256, 54)}
ALLOWED_SPACING = (0.938, 0.938, 3.0)
SPACING_ATOL = 0.003
EXPECTED_FEATURES = 71


def load_volume(path):
    """Load a NIfTI volume using the same operations as the training notebook."""
    path = Path(path)
    if not (path.name.lower().endswith(".nii") or path.name.lower().endswith(".nii.gz")):
        raise ValueError("MRI accepts NIfTI .nii or .nii.gz files only.")

    image = nib.load(str(path))
    volume = np.nan_to_num(
        image.get_fdata(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    if volume.ndim == 4:
        volume = volume.mean(-1)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D or 4D NIfTI volume; received {volume.ndim} dimensions.")

    spacing = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    shape = tuple(image.shape[:3])
    return volume, spacing, shape


def normalise(volume):
    """Replicate the notebook's MRI intensity mask and normalization."""
    volume = np.asarray(volume, dtype=np.float32)
    mask = volume > 0.05 * np.percentile(volume, 99)
    if mask.sum() < 100:
        mask = volume > volume.min()
    if not mask.any():
        raise ValueError("MRI volume has no foreground voxels after masking.")

    lo, hi = np.percentile(volume[mask], [1, 99])
    volume = np.clip(volume, lo, hi)
    mean = volume[mask].mean()
    std = volume[mask].std() + 1e-6
    volume = np.where(mask, (volume - mean) / std, 0).astype(np.float32)
    return volume, mask


def volume_features(volume):
    """Return the exact 71 handcrafted features used by the trained MRI model."""
    volume, mask = normalise(volume)
    foreground = volume[mask]
    histogram, _ = np.histogram(foreground, bins=32, range=(-3, 3), density=True)

    gx, gy, gz = np.gradient(volume)
    gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)[mask]

    small = ndi.zoom(
        volume,
        (48 / volume.shape[0], 48 / volume.shape[1], 16 / volume.shape[2]),
        order=1,
    )
    regions = [
        region.mean()
        for part_x in np.array_split(small, 3, axis=0)
        for part_y in np.array_split(part_x, 3, axis=1)
        for region in np.array_split(part_y, 3, axis=2)
    ]

    features = np.concatenate([
        [
            foreground.mean(), foreground.std(), skew(foreground),
            kurtosis(foreground), mask.mean(),
            gradient_magnitude.mean(), gradient_magnitude.std(),
        ],
        np.percentile(foreground, [5, 25, 50, 75, 95]),
        histogram,
        regions,
    ]).astype(np.float32)

    if features.shape != (EXPECTED_FEATURES,):
        raise RuntimeError(f"MRI extractor generated {features.size} features; expected {EXPECTED_FEATURES}.")
    if not np.isfinite(features).all():
        raise ValueError("MRI feature vector contains NaN or infinity.")
    return features


def predict_mri(volume_path, model):
    """Predict using the matched-protocol MRI model; never infer missing metadata."""
    if model is None:
        raise RuntimeError("MRI classifier is not loaded.")

    volume, spacing, shape = load_volume(volume_path)
    if shape not in ALLOWED_SHAPES:
        raise ValueError(
            f"MRI shape {shape} is unsupported. This model accepts only {sorted(ALLOWED_SHAPES)}."
        )
    if not np.allclose(spacing, ALLOWED_SPACING, atol=SPACING_ATOL):
        raise ValueError(
            f"MRI voxel spacing {tuple(np.round(spacing, 4))} mm is unsupported; "
            f"expected approximately {ALLOWED_SPACING} mm (tolerance {SPACING_ATOL})."
        )

    features = volume_features(volume).reshape(1, -1)
    expected = getattr(model, "n_features_in_", EXPECTED_FEATURES)
    if features.shape[1] != expected:
        raise ValueError(f"MRI model expects {expected} features; received {features.shape[1]}.")

    prediction = int(model.predict(features)[0])
    class_names = {0: "Healthy", 1: "Parkinson's"}
    result = {
        "modality": "MRI",
        "class_id": prediction,
        "class_name": class_names.get(prediction, str(prediction)),
        "input_shape": list(shape),
        "voxel_spacing_mm": [float(x) for x in spacing],
    }
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        result["probabilities"] = {
            class_names.get(int(label), str(int(label))): float(probability)
            for label, probability in zip(model.classes_, probabilities)
        }
    return result
