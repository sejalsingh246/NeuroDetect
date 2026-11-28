from pathlib import Path
from tempfile import NamedTemporaryFile
from collections import Counter
import shutil
import os

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import joblib
from fastapi.responses import FileResponse
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

from backend.services.audio_service import predict_audio
from backend.services.drawing_service import predict_drawing
from backend.services.ecg_service import predict_ecg
from backend.services.mri_service import predict_mri
from backend.services.assistant_service import ask_assistant

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")
MODELS_DIR = BASE_DIR / "models"
FRONTEND_PATH = BASE_DIR / "templates" / "index.html"
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
MODEL_PATHS = {
    "audio": MODELS_DIR / "Audio" / "audio_final_model.pkl",
    "drawing": MODELS_DIR / "Drawing" / "drawing_final_model.pkl",
    "ecg": MODELS_DIR / "ECG" / "ecg_final_model.pkl",
    "mri": MODELS_DIR / "MRI" / "mri_final_model.pkl",
}
FEATURE_EXTRACTOR_PATH = MODELS_DIR / "shared" / "mobilenetv2_imagenet_feature_extractor.keras"

app = FastAPI(
    title="Parkinson's Multimodal Detection API",
    description=(
        "Research prototype providing separate predictions from audio, drawing, "
        "ECG and MRI modalities. These outputs are not a combined diagnosis."
    ),
    version="1.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to the frontend origin before deployment.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = {}
feature_extractor = None
model_errors = {}


@app.on_event("startup")
def load_trained_models():
    global feature_extractor
    models.clear()
    model_errors.clear()

    for modality, model_path in MODEL_PATHS.items():
        try:
            if not model_path.is_file():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            models[modality] = joblib.load(model_path)
            print(f"[SUCCESS] {modality.upper()} model loaded.")
        except Exception as exc:
            model_errors[modality] = str(exc)
            print(f"[ERROR] Could not load {modality.upper()}: {exc}")

    try:
        if not FEATURE_EXTRACTOR_PATH.is_file():
            raise FileNotFoundError(f"Feature extractor not found: {FEATURE_EXTRACTOR_PATH}")
        feature_extractor = load_model(FEATURE_EXTRACTOR_PATH, compile=False)
        print("[SUCCESS] MobileNetV2 feature extractor loaded.")
    except Exception as exc:
        feature_extractor = None
        model_errors["feature_extractor"] = str(exc)
        print(f"[ERROR] Feature extractor could not load: {exc}")


def require_model(name):
    model = models.get(name)
    if model is None:
        detail = model_errors.get(name, "Model is not loaded.")
        raise HTTPException(status_code=503, detail=f"{name.title()} model unavailable: {detail}")
    return model



def run_with_upload(upload: UploadFile, allowed_suffixes, predictor):
    filename_lower = (upload.filename or "").lower()

    # Preserve the compound NIfTI suffix.
    suffix = (
        ".nii.gz"
        if filename_lower.endswith(".nii.gz")
        else Path(filename_lower).suffix
    )

    if not any(filename_lower.endswith(s) for s in allowed_suffixes):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Accepted: {', '.join(allowed_suffixes)}",
        )

    temp_path = None

    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp_path = Path(temp.name)
            shutil.copyfileobj(upload.file, temp)

        return predictor(temp_path)

    except HTTPException:
        raise

    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        print(f"[PREDICTION ERROR] {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check backend logs.",
        ) from exc

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

        upload.file.close()


def run_fusion_vote(predictions):
    """
    Experimental decision-level hard voting.
    Uses class_id from successful modality predictions.
    Does not average probabilities.
    """

    if len(predictions) < 2:
        raise HTTPException(
            status_code=422,
            detail="At least two successful modality predictions are required.",
        )

    votes = Counter(
        int(result["class_id"])
        for result in predictions.values()
    )

    highest_vote_count = max(votes.values())

    winners = [
        class_id
        for class_id, count in votes.items()
        if count == highest_vote_count
    ]

    if len(winners) != 1:
        consensus = None
        status = "no_consensus"
        label = "No consensus"
    else:
        consensus = winners[0]
        status = "consensus"
        label = "Parkinson's" if consensus == 1 else "Healthy"

    return {
        "status": status,
        "consensus_class_id": consensus,
        "consensus_label": label,
        "vote_counts": dict(votes),
        "modality_predictions": predictions,
        "warning": (
            "Experimental hard-voting result only. This is not a clinically "
            "validated multimodal diagnosis. Same-patient identity is based "
            "on user confirmation and is not independently verified."
        ),
    }


@app.get("/")
async def home(request: Request):
    if not FRONTEND_PATH.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Dashboard file not found: {FRONTEND_PATH}"
        )

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

@app.get("/health")
def health_check():
    return {
        "status": "running",
        "loaded_models": list(models.keys()),
        "feature_extractor_loaded": feature_extractor is not None,
        "model_errors": model_errors,
    }


@app.get("/models/status")
def get_model_status():
    return {
        name: ("loaded" if name in models else "not loaded")
        for name in ("audio", "drawing", "ecg", "mri")
    } | {"mobilenetv2": "loaded" if feature_extractor is not None else "not loaded"}


@app.get("/disclaimer")
def disclaimer():
    return {
        "notice": (
            "This application is an experimental research prototype. Outputs are not "
            "medical diagnoses and must not be used to make clinical decisions. "
            "Each modality produces an independent prediction; no patient-level fusion is performed."
        )
    }


@app.post("/predict/audio")
def audio_prediction(file: UploadFile = File(...), task: int = Form(...)):
    model = require_model("audio")
    return run_with_upload(file, (".wav",), lambda path: predict_audio(path, task, model))


@app.post("/predict/drawing")
def drawing_prediction(
    file: UploadFile = File(...),
    task: int = Form(...),
    shape: str = Form(...),
):
    model = require_model("drawing")
    if feature_extractor is None:
        raise HTTPException(status_code=503, detail="MobileNetV2 feature extractor is unavailable.")
    return run_with_upload(
        file, (".png", ".jpg", ".jpeg"),
        lambda path: predict_drawing(path, task, shape.lower(), model, feature_extractor),
    )


@app.post("/predict/ecg")
def ecg_prediction(file: UploadFile = File(...)):
    model = require_model("ecg")
    if feature_extractor is None:
        raise HTTPException(status_code=503, detail="MobileNetV2 feature extractor is unavailable.")
    return run_with_upload(
        file, (".png", ".jpg", ".jpeg"),
        lambda path: predict_ecg(path, model, feature_extractor),
    )


@app.post("/predict/mri")
def mri_prediction(file: UploadFile = File(...)):
    model = require_model("mri")
    return run_with_upload(
        file, (".nii", ".nii.gz"), lambda path: predict_mri(path, model)
    )


class AssistantMessage(BaseModel):
    role: str
    content: str = Field(min_length=1, max_length=4000)


class AssistantRequest(BaseModel):
    messages: list[AssistantMessage] = Field(min_length=1, max_length=12)


@app.post("/predict/fusion")
async def predict_fusion(
    same_patient_confirmed: bool = Form(...),

    audio_file: UploadFile = File(None),
    audio_task: int = Form(1),

    drawing_file: UploadFile = File(None),
    drawing_task: int = Form(1),
    drawing_shape: str = Form("spiral"),

    ecg_file: UploadFile = File(None),
    mri_file: UploadFile = File(None),
):
    if not same_patient_confirmed:
        raise HTTPException(
            status_code=400,
            detail="Please confirm that all uploaded samples belong to the same person."
        )

    uploads = {
        "Audio": (audio_file, {".wav"}),
        "Drawing": (drawing_file, {".png", ".jpg", ".jpeg"}),
        "ECG": (ecg_file, {".png", ".jpg", ".jpeg"}),
        "MRI": (mri_file, {".nii", ".nii.gz"}),
    }

    supplied = {
        name: item for name, item in uploads.items()
        if item[0] is not None
    }

    if len(supplied) < 2:
        raise HTTPException(
            status_code=422,
            detail="Upload at least two modality files."
        )

    temp_paths = {}
    predictions = {}
    errors = {}

    try:
        # Save all supplied uploads to temporary files.
        for name, (upload, allowed_suffixes) in supplied.items():
            filename = upload.filename or ""
            lower_name = filename.lower()

            if not any(lower_name.endswith(suffix) for suffix in allowed_suffixes):
                errors[name] = "Unsupported file format."
                continue

            suffix = ".nii.gz" if lower_name.endswith(".nii.gz") else Path(filename).suffix

            with NamedTemporaryFile(delete=False, suffix=suffix) as temp:
                shutil.copyfileobj(upload.file, temp)
                temp_paths[name] = temp.name

        # Run each modality independently so one failure doesn't stop the others.
        if "Audio" in temp_paths:
            try:
                predictions["Audio"] = predict_audio(
                    temp_paths["Audio"],
                    audio_task,
                    require_model("audio")
                )
            except Exception as exc:
                errors["Audio"] = str(exc)

        if "Drawing" in temp_paths:
            try:
                predictions["Drawing"] = predict_drawing(
                    temp_paths["Drawing"],
                    drawing_task,
                    drawing_shape.lower(),
                    require_model("drawing"),
                    feature_extractor
                )
            except Exception as exc:
                errors["Drawing"] = str(exc)

        if "ECG" in temp_paths:
            try:
                predictions["ECG"] = predict_ecg(
                    temp_paths["ECG"],
                    require_model("ecg"),
                    feature_extractor
                )
            except Exception as exc:
                errors["ECG"] = str(exc)

        if "MRI" in temp_paths:
            try:
                predictions["MRI"] = predict_mri(
                    temp_paths["MRI"],
                    require_model("mri")
                )
            except Exception as exc:
                errors["MRI"] = str(exc)

        result = run_fusion_vote(predictions)
        result["errors"] = errors
        result["same_patient_confirmed"] = True
        return result

    finally:
        for path in temp_paths.values():
            try:
                os.remove(path)
            except OSError:
                pass

        for upload, _ in supplied.values():
            await upload.close()


@app.post("/assistant/chat")
async def assistant_chat(request: AssistantRequest):
    """Answer an educational question using the fixed curated source pack."""
    messages = []
    for item in request.messages:
        if item.role not in ("user", "assistant"):
            raise HTTPException(status_code=422, detail="Message role must be user or assistant.")
        messages.append({"role": item.role, "content": item.content.strip()})

    if not messages or messages[-1]["role"] != "user":
        raise HTTPException(status_code=422, detail="The last message must be from the user.")

    try:
        return await ask_assistant(messages)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        print(f"[ASSISTANT PROVIDER ERROR] HTTP {status}: {exc.response.text[:500]}")
        if status in (401, 402, 403):
            detail = "OpenRouter rejected the request. Check the API key, account credits, and model access."
        elif status == 429:
            detail = "OpenRouter rate limit reached. Please wait and try again."
        else:
            detail = f"OpenRouter request failed (HTTP {status}). Please try again."
        raise HTTPException(status_code=502, detail=detail) from exc
    except httpx.HTTPError as exc:
        print(f"[ASSISTANT NETWORK ERROR] {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=502, detail="Could not reach OpenRouter. Check your internet connection and try again.") from exc
