
from pathlib import Path

import numpy as np
from PIL import Image

from scipy import ndimage as ndi
from skimage.feature import hog
from skimage.filters import threshold_otsu
from skimage.measure import label as cc_label
from skimage.morphology import skeletonize


def load_rgb(image_path):
    """Load an image and convert it to RGB."""
    image = Image.open(image_path)

    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        background.alpha_composite(image)
        image = background

    return image.convert("RGB")


def preprocess_drawing(image_path, size=224, margin=0.08):
    """Crop the drawing ink and pad it to a square."""
    rgb = load_rgb(image_path)
    gray = np.asarray(rgb.convert("L"), dtype=np.float32)

    threshold = threshold_otsu(gray)
    ink = gray < threshold

    if ink.mean() > 0.5:
        ink = ~ink

    ys, xs = np.where(ink)

    if len(ys) < 10:
        ys = np.array([0, gray.shape[0] - 1])
        xs = np.array([0, gray.shape[1] - 1])

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    height = y1 - y0
    width = x1 - x0

    side = int(max(height, width) * (1 + 2 * margin))
    canvas = np.full((side, side), 255, dtype=np.uint8)

    crop = np.where(ink, 0, 255).astype(np.uint8)[y0:y1, x0:x1]

    offset_y = (side - height) // 2
    offset_x = (side - width) // 2

    canvas[
        offset_y:offset_y + height,
        offset_x:offset_x + width
    ] = crop

    resized = Image.fromarray(canvas).resize(
        (size, size),
        Image.Resampling.BILINEAR
    )

    return np.asarray(resized)


def extract_handcrafted_features(image):
    """Extract HOG and stroke-related drawing features."""
    ink = image < 128
    skeleton = skeletonize(ink)

    distance = ndi.distance_transform_edt(ink)
    widths = distance[skeleton] * 2 if skeleton.any() else np.array([0.0])

    components = cc_label(ink, connectivity=2).max()

    gx, gy = np.gradient(image.astype(np.float32))
    magnitude = np.hypot(gx, gy)

    small = np.asarray(
        Image.fromarray(image).resize((128, 128))
    )

    hog_features = hog(
        small,
        orientations=9,
        pixels_per_cell=(32, 32),
        cells_per_block=(2, 2)
    )

    base_features = [
        ink.mean(),
        skeleton.sum() / max(ink.sum(), 1),
        widths.mean(),
        widths.std(),
        widths.std() / (widths.mean() + 1e-6),
        components,
        magnitude.mean(),
        magnitude.std()
    ]

    return np.concatenate([base_features, hog_features])


def predict_drawing(
    image_path,
    task,
    shape,
    model,
    feature_extractor
):
    """
    Predict from one drawing image.

    task: 1 or 2
    shape: 'spiral' or 'wave'
    """

    if task not in (1, 2):
        raise ValueError("Task must be 1 or 2.")

    if shape not in ("spiral", "wave"):
        raise ValueError("Shape must be 'spiral' or 'wave'.")

    if feature_extractor is None:
        raise RuntimeError("MobileNetV2 feature extractor is not loaded.")

    image = preprocess_drawing(image_path)

    # Training used grayscale images replicated across RGB channels.
    rgb_image = np.stack([image] * 3, axis=-1).astype(np.float32)

    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    model_input = preprocess_input(np.expand_dims(rgb_image, axis=0))

    embedding = feature_extractor.predict(
        model_input,
        verbose=0
    )[0]

    # Include handcrafted features only if the trained model used them.
    # The current exported configuration selects CNN embedding.
    features = np.concatenate([
        embedding,
        np.array([float(task), float(shape == "spiral")])
    ])

    features = features.reshape(1, -1)

    expected = getattr(model, "n_features_in_", None)

    if expected is not None and features.shape[1] != expected:
        raise ValueError(
            f"Drawing model expects {expected} features, "
            f"but preprocessing generated {features.shape[1]}."
        )

    if not np.isfinite(features).all():
        raise ValueError("Drawing features contain invalid values.")

    prediction = int(model.predict(features)[0])

    result = {
        "modality": "Drawing",
        "class_id": prediction,
        "class_name": (
            "Healthy" if prediction == 0 else "Parkinson's"
        )
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        class_names = {0: "Healthy", 1: "Parkinson's"}
        result["probabilities"] = {
            class_names.get(int(label), str(int(label))): float(probability)
            for label, probability in zip(model.classes_, probabilities)
        }

    return result
