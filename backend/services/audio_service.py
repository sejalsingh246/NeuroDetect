
from pathlib import Path

import joblib
import librosa
import numpy as np
import parselmouth

from parselmouth.praat import call
from scipy.stats import skew



BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "Audio" / "audio_final_model.pkl"



if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Audio model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)



def safe_stats(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]

    if values.size == 0:
        return [0.0, 0.0]

    return [
        float(np.mean(values)),
        float(np.std(values))
    ]


def praat_audio_features(y, sr):

    zeros = [0.0] * 9

    try:
        sound = parselmouth.Sound(
            np.asarray(y, dtype=np.float64),
            sampling_frequency=sr
        )

        pitch = sound.to_pitch(
            time_step=0.01,
            pitch_floor=75,
            pitch_ceiling=500
        )

        f0 = pitch.selected_array["frequency"]
        voiced = f0[f0 > 0]

        voiced_fraction = (
            float(np.mean(f0 > 0)) if f0.size else 0.0
        )

        f0_stats = safe_stats(voiced)

        point_process = call(
            sound,
            "To PointProcess (periodic, cc)",
            75,
            500
        )

        def safe_call(obj, command, *args):
            try:
                value = call(obj, command, *args)
                return float(value) if np.isfinite(value) else 0.0
            except Exception:
                return 0.0

        jitter_local = safe_call(
            point_process,
            "Get jitter (local)",
            0, 0, 0.0001, 0.02, 1.3
        )

        jitter_rap = safe_call(
            point_process,
            "Get jitter (rap)",
            0, 0, 0.0001, 0.02, 1.3
        )

        jitter_ppq5 = safe_call(
            point_process,
            "Get jitter (ppq5)",
            0, 0, 0.0001, 0.02, 1.3
        )

        shimmer_local = safe_call(
            [sound, point_process],
            "Get shimmer (local)",
            0, 0, 0.0001, 0.02, 1.3, 1.6
        )

        shimmer_apq3 = safe_call(
            [sound, point_process],
            "Get shimmer (apq3)",
            0, 0, 0.0001, 0.02, 1.3, 1.6
        )

        harmonicity = call(
            sound,
            "To Harmonicity (cc)",
            0.01, 75, 0.1, 1.0
        )

        hnr = safe_call(
            harmonicity,
            "Get mean",
            0, 0
        )

        return [
            jitter_local,
            jitter_rap,
            jitter_ppq5,
            shimmer_local,
            shimmer_apq3,
            hnr,
            *f0_stats,
            voiced_fraction
        ]

    except Exception:
        return zeros



def extract_audio_features(audio_path, sr=16000):

    y, sr = librosa.load(
        str(audio_path),
        sr=sr,
        mono=True
    )

    if y.size == 0:
        raise ValueError("Audio file is empty or unreadable.")

    # Remove silence
    y, _ = librosa.effects.trim(
        y,
        top_db=30
    )

    # Peak normalisation
    peak = np.max(np.abs(y)) if y.size else 0.0

    if peak > 0:
        y = y / peak

    duration = float(len(y) / sr)

    # Ensure enough samples for feature extraction
    if len(y) < 512:
        y = np.pad(y, (0, 512 - len(y)))

    n_fft = 1024
    hop = 256

    features = []

    # MFCC, delta and delta-delta
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20,
        n_fft=n_fft,
        hop_length=hop
    )

    delta_width = min(
        9,
        mfcc.shape[1] if mfcc.shape[1] % 2
        else mfcc.shape[1] - 1
    )

    delta_width = max(3, delta_width)

    delta = librosa.feature.delta(
        mfcc,
        width=delta_width,
        mode="nearest"
    )

    delta2 = librosa.feature.delta(
        mfcc,
        order=2,
        width=delta_width,
        mode="nearest"
    )

    for matrix in (mfcc, delta, delta2):
        for row in matrix:
            features.extend(safe_stats(row))

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_bands=6
    )

    for row in contrast:
        features.extend(safe_stats(row))

    # Spectral features
    scalar_features = (
        librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop
        )[0],

        librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop
        )[0],

        librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop
        )[0],

        librosa.feature.spectral_flatness(
            y=y, n_fft=n_fft, hop_length=hop
        )[0],

        librosa.feature.zero_crossing_rate(
            y, frame_length=n_fft, hop_length=hop
        )[0],

        librosa.feature.rms(
            y=y, frame_length=n_fft, hop_length=hop
        )[0]
    )

    for values in scalar_features:
        features.extend(safe_stats(values))

    # YIN pitch features
    try:
        yin = librosa.yin(
            y,
            fmin=75,
            fmax=500,
            sr=sr,
            frame_length=n_fft,
            hop_length=hop
        )

        yin = yin[np.isfinite(yin)]

    except Exception:
        yin = np.array([], dtype=np.float64)

    yin_mean, yin_std = safe_stats(yin)

    yin_rel_diff = (
        float(np.mean(np.abs(np.diff(yin))) / (yin_mean + 1e-8))
        if yin.size > 1 else 0.0
    )

    yin_skew = (
        float(skew(yin))
        if yin.size > 2 and np.std(yin) > 1e-8
        else 0.0
    )

    features.extend([
        yin_mean,
        yin_std,
        yin_rel_diff,
        yin_skew
    ])

    # Praat voice features
    features.extend(
        praat_audio_features(y, sr)
    )

    # Duration
    features.append(duration)

    result = np.asarray(features, dtype=np.float32)

    if result.shape != (160,):
        raise RuntimeError(
            f"Expected 160 audio features, got {result.size}."
        )

    if not np.isfinite(result).all():
        raise ValueError("Audio features contain invalid values.")

    return result



def predict_audio(audio_path, task, model):

    if model is None:
        raise RuntimeError("Audio classifier is not loaded.")

    if task not in (1, 2):
        raise ValueError("Task must be either 1 or 2.")

    if Path(audio_path).suffix.lower() != ".wav":
        raise ValueError("Only WAV audio files are supported.")

    audio_features = extract_audio_features(audio_path)

    # Append task indicator
    final_features = np.append(
        audio_features,
        float(task)
    ).reshape(1, -1)

    expected_features = model.n_features_in_

    if final_features.shape[1] != expected_features:
        raise ValueError(
            f"Model expects {expected_features} features, "
            f"but received {final_features.shape[1]}."
        )

    prediction = int(model.predict(final_features)[0])

    class_names = {0: "Healthy", 1: "Parkinson's"}
    result = {
        "modality": "Audio",
        "class_id": prediction,
        "class_name": class_names.get(prediction, str(prediction))
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(final_features)[0]

        result["probabilities"] = {
            class_names.get(int(label), str(int(label))): float(probability)
            for label, probability in zip(
                model.classes_,
                probabilities
            )
        }

    return result
