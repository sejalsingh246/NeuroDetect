# NeuroDetect

### A Multimodal Parkinson's Disease Research Platform

**Machine Learning · Deep Learning · FastAPI · Source-grounded AI
Assistant**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange?logo=tensorflow)](https://www.tensorflow.org/)

[GitHub Repository](https://github.com/sejalsingh246/NeuroDetect)
:::

------------------------------------------------------------------------

## Overview

**NeuroDetect** is an academic research prototype exploring Parkinson's
disease classification across four data modalities: **audio, drawing,
ECG, and MRI**. Each modality has its own preprocessing,
feature-extraction, and inference pipeline. A FastAPI backend connects
these workflows to an interactive web dashboard.

The project also includes a **source-grounded educational assistant**
for general Parkinson's-related questions. It uses a curated collection
of health-information and research references and accesses a Google
Gemini model through OpenRouter.

> **Disclaimer:** NeuroDetect is an experimental academic prototype, not
> a medical device or clinically validated diagnostic system. Its model
> outputs are independent research predictions and must not be used to
> diagnose Parkinson's disease or guide treatment.

## Features

-   Four independent modality-specific prediction workflows.
-   Audio classification using extracted acoustic features.
-   Drawing analysis using MobileNetV2 image embeddings and task/shape
    metadata.
-   ECG analysis using MobileNetV2 image embeddings.
-   MRI classification using handcrafted volumetric features.
-   Interactive dashboard for submitting inputs and viewing model
    responses.
-   Educational Parkinson's assistant using a curated source pack.
-   FastAPI endpoints for model inference and assistant interaction.
-   Explicit communication of research limitations; no unsupported
    combined diagnosis.

## Architecture

``` mermaid
flowchart TD
    A[User] --> B[NeuroDetect Dashboard]
    B --> C[FastAPI Backend]
    C --> D[Audio Pipeline]
    C --> E[Drawing Pipeline]
    C --> F[ECG Pipeline]
    C --> G[MRI Pipeline]
    C --> H[Educational AI Assistant]
    D --> I[Audio Classifier]
    E --> J[MobileNetV2 Features + Classifier]
    F --> K[MobileNetV2 Features + Classifier]
    G --> L[Handcrafted Features + Classifier]
    H --> M[Curated Parkinson's Sources]
    H --> N[OpenRouter / Gemini]
    I --> O[Independent Prediction]
    J --> O
    K --> O
    L --> O
    N --> P[Source-referenced Answer]
    O --> B
    P --> B
```

### Modality pipelines

  -----------------------------------------------------------------------
  Modality                Feature representation  Classifier
  ----------------------- ----------------------- -----------------------
  Audio                   Extracted acoustic      ExtraTrees
                          features                

  Drawing                 MobileNetV2             Logistic Regression
                          embeddings + task/shape 
                          metadata                

  ECG                     MobileNetV2 embeddings  Logistic Regression

  MRI                     Handcrafted volumetric  Logistic Regression
                          features                
  -----------------------------------------------------------------------

Drawing and ECG use the shared MobileNetV2 feature extractor included in
the backend model assets.

**Important:** These models operate independently. NeuroDetect does not
currently perform verified patient-level multimodal fusion. Predictions
from unrelated samples must not be averaged or combined into a single
disease probability.

## Technology Stack

  Area                                 Technologies
  ------------------------------------ --------------------------------------------------
  Language                             Python 3.11
  Backend                              FastAPI, Uvicorn
  Machine learning                     scikit-learn
  Deep learning / feature extraction   TensorFlow, Keras, MobileNetV2
  Frontend                             HTML, CSS, JavaScript
  AI assistant                         OpenAI Python SDK, OpenRouter API, Google Gemini
  Configuration                        Environment variables, `.env`

## Repository Structure

``` text
NeuroDetect/
├── backend/
│   ├── knowledge/
│   │   └── parkinsons_sources.py
│   ├── models/
│   │   ├── Audio/
│   │   ├── Drawing/
│   │   ├── ECG/
│   │   ├── MRI/
│   │   └── shared/
│   ├── services/
│   │   ├── assistant_service.py
│   │   ├── audio_service.py
│   │   ├── drawing_service.py
│   │   ├── ecg_service.py
│   │   └── mri_service.py
│   ├── templates/
│   │   └── index.html
│   ├── main.py
│   └── requirements.txt
├── .env.example
├── .gitignore
└── main.py
```
## Multimodal Fusion

NeuroDetect includes an experimental hard-voting fusion mechanism
that combines predictions from multiple modalities:

- Audio
- Drawing
- ECG
- MRI

The fusion mechanism:
- Requires confirmation that uploaded samples belong to the same person.
- Requires at least two successful modality predictions.
- Uses majority voting based on predicted classes.
- Excludes failed modality predictions.
- Returns no consensus in case of a tie.

Note: This is an experimental implementation and has not been
clinically validated. The available datasets do not establish
verified patient-level alignment across all modalities.

## Getting Started

### Prerequisites

-   Python 3.11
-   Git
-   An OpenRouter API key (required for the AI assistant)

### 1. Clone the repository

``` bash
git clone https://github.com/sejalsingh246/NeuroDetect.git
cd NeuroDetect
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

``` powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
```

If PowerShell activation is restricted, use Command Prompt:

``` bat
.venv311\Scripts\activate.bat
```

### 3. Install dependencies

``` bash
pip install -r backend/requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` in the project root and add your own
OpenRouter key:

``` env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=google/gemini-2.5-flash
OPENROUTER_SITE_URL=http://localhost:8000
```

The model ID is an example configuration; check OpenRouter for current
model availability and pricing.

**Never commit `.env`, expose API keys in frontend code, or share your
secret key publicly.** The committed `.env.example` should contain
placeholders only.

### 5. Start the server

From the repository root:

``` bash
python -m uvicorn backend.main:app --reload
```

Open the dashboard at:

``` text
http://127.0.0.1:8000
```

Interactive API documentation:

``` text
http://127.0.0.1:8000/docs
```

## API Endpoints

  Method   Endpoint             Purpose
  -------- -------------------- ----------------------------------
  `GET`    `/`                  Serves the dashboard
  `GET`    `/health`            Application health check
  `GET`    `/models/status`     Model loading/status information
  `GET`    `/disclaimer`        Project disclaimer
  `POST`   `/predict/audio`     Audio inference
  `POST`   `/predict/drawing`   Drawing inference
  `POST`   `/predict/ecg`       ECG inference
  `POST`   `/predict/mri`       MRI inference

The assistant is also integrated into the backend. Use `/docs` on a
running instance to inspect the exact assistant route and request
schema.

## Educational AI Assistant

The assistant is intended for **general Parkinson's education**, not
personal diagnosis or treatment advice.

-   Uses a curated source pack containing Parkinson's Foundation, WHO,
    and research references.
-   Calls a Google Gemini model through OpenRouter.
-   Keeps the API key on the backend.
-   Provides source references with its responses.
-   Is instructed to acknowledge when the curated material does not
    support an answer.

**Current grounding method:** curated source context is supplied to the
assistant. A full retrieval-augmented generation (RAG) pipeline with
document chunking, embeddings, vector search, and retrieval evaluation
is **not yet implemented**.

## Research References

The following papers and resources were supplied for the project's
curated knowledge collection:

1.  [Springer --- DOI:
    10.1007/s44163-025-00241-9](https://link.springer.com/10.1007/s44163-025-00241-9)
2.  [Diagnostics --- 2023, 13(11),
    1924](https://www.mdpi.com/2075-4418/13/11/1924)
3.  [Expert Systems with Applications --- article
    S0957417423010436](https://www.sciencedirect.com/science/article/pii/S0957417423010436)
4.  [Scientific Reports ---
    2024](https://www.nature.com/articles/s41598-024-64004-9)
5.  [Procedia Computer Science --- article
    S1877050923000078](https://www.sciencedirect.com/science/article/pii/S1877050923000078)
6.  [Scientific Reports ---
    2025](https://www.nature.com/articles/s41598-025-96575-6)
7.  [Diagnostics --- 2022, 12(8),
    2003](https://www.mdpi.com/2075-4418/12/8/2003)
8.  [Frontiers in Artificial Intelligence ---
    2023](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1084001/full)
9.  [npj Parkinson's Disease ---
    2022](https://www.nature.com/articles/s41531-022-00288-w)
10. [Journal of Data Science and Intelligent
    Systems](https://ojs.bonviewpress.com/index.php/jdsis/article/view/831)

Additional educational sources:

-   [Parkinson's Foundation](https://www.parkinson.org/)
-   [World Health Organization --- Parkinson
    disease](https://www.who.int/news-room/fact-sheets/detail/parkinson-disease)
-   [National Institute of Neurological Disorders and
    Stroke](https://www.ninds.nih.gov/health-information/disorders/parkinsons-disease)

Refer to the original publications for their methods, datasets, and
conclusions. Their inclusion here does not mean that every paper's
method is implemented in NeuroDetect.

## Limitations

-   The four modalities use independent models and are not linked
    through verified shared patient identifiers.
-   Patient-level multimodal fusion is not implemented.
-   The MRI matched-protocol evaluation subset is small; successful
    processing of samples does not establish model accuracy or clinical
    validity.
-   Dataset size, acquisition differences, class imbalance, and possible
    dataset-specific shortcuts may affect performance.
-   Model scores are experimental outputs, not established or calibrated
    clinical probabilities.
-   The assistant uses curated context rather than a full RAG retrieval
    pipeline and may provide incomplete or incorrect information.
-   No prospective clinical validation or regulatory approval is
    claimed.

## Future Scope

-   Build a verified patient-level dataset linking the different
    modalities.
-   Evaluate feature-level or decision-level fusion only when patient
    linkage and study design support it.
-   Implement RAG with document ingestion, chunking, embeddings, vector
    retrieval, and source-level evaluation.
-   Expand validation with subject-independent splits and external
    datasets.
-   Improve explainability, uncertainty reporting, and probability
    calibration.
-   Add automated tests, experiment tracking, and reproducible
    model-training documentation.

## Responsible Use

NeuroDetect is for academic learning and research demonstrations only.

-   It does not diagnose Parkinson's disease.
-   It must not be used to make treatment, medication, or care
    decisions.
-   Model predictions are not a substitute for assessment by a qualified
    healthcare professional.
-   Do not upload identifiable health records or sensitive personal
    information to the assistant.
-   Anyone concerned about symptoms should consult a qualified
    healthcare professional.

## Contributing

Bug reports, suggestions, and contributions are welcome. Please describe
the change, its motivation, and tests performed. For model-related
contributions, document the dataset split and evaluation method; do not
report performance without sufficient experimental context.

## Author

**Sejal Singh** · [GitHub
@sejalsingh246](https://github.com/sejalsingh246)

------------------------------------------------------------------------

::: 
*NeuroDetect --- Multimodal Parkinson's Research Prototype*
:::
