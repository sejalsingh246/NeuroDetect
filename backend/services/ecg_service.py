
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def predict_ecg(image_path, model, feature_extractor):
    """
    Predict from an ECG image using the trained ECG classifier
    and the shared MobileNetV2 feature extractor.
    """

    if feature_extractor is None:
        raise RuntimeError("MobileNetV2 feature extractor is not loaded.")

    if model is None:
        raise RuntimeError("ECG classifier is not loaded.")

    image_path = Path(image_path)

    if image_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        raise ValueError("ECG accepts PNG, JPG or JPEG images only.")

    # Load original image as RGB.
    with Image.open(image_path) as image:
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGBA")
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            background.alpha_composite(image)
            image = background
        image = image.convert("RGB")

        # Match the training notebook's resize operation.
        image = image.resize((224, 224), Image.Resampling.BILINEAR)

        image_array = np.asarray(image)

    # MobileNetV2 preprocessing.
    image_array = image_array.astype(np.float32)
    image_array = preprocess_input(
        np.expand_dims(image_array, axis=0)
    )

    # Extract the 1280-dimensional embedding.
    embedding = feature_extractor.predict(
        image_array,
        verbose=0
    )

    expected_features = getattr(model, "n_features_in_", 1280)

    if embedding.shape != (1, expected_features):
        raise ValueError(
            f"Expected embedding shape (1, {expected_features}), "
            f"received {embedding.shape}."
        )

    if not np.isfinite(embedding).all():
        raise ValueError("ECG embedding contains invalid values.")

    prediction = int(model.predict(embedding)[0])

    class_names = {0: "Healthy", 1: "Parkinson's"}
    result = {
        "modality": "ECG",
        "class_id": prediction,
        "class_name": class_names.get(prediction, str(prediction))
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(embedding)[0]

        result["probabilities"] = {
            class_names.get(int(label), str(int(label))): float(probability)
            for label, probability in zip(model.classes_, probabilities)
        }

    return result
