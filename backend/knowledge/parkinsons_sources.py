"""Curated Parkinson's disease educational and research source pack.

The assistant receives these concise source notes as fixed context on each
request. This is curated-context prompting, not RAG or live retrieval.
"""

SOURCES = [
    {
        "id": 1,
        "title": "What is Parkinson's? — Parkinson's Foundation",
        "url": "https://www.parkinson.org/understanding-parkinsons/what-is-parkinsons",
        "summary": ("Parkinson's disease is a progressive neurodegenerative brain disorder "
                    "that predominantly affects dopamine-producing neurons. Motor symptoms "
                    "can include resting tremor, bradykinesia, rigidity, and gait or balance "
                    "difficulties. Non-motor symptoms may include sleep problems, mood changes, "
                    "constipation, loss of smell, and cognitive changes. Symptoms vary. "
                    "Treatments can help manage symptoms, but this source does not present a cure.")
    },
    {
        "id": 2,
        "title": "Parkinson disease — World Health Organization",
        "url": "https://www.who.int/news-room/fact-sheets/detail/parkinson-disease",
        "summary": ("WHO describes Parkinson disease as a brain condition associated with "
                    "movement symptoms and a range of non-motor effects. Care may include "
                    "medicines, rehabilitation, and in selected cases surgery. Rehabilitation "
                    "can support mobility, balance, strength, and daily functioning.")
    },
    {
        "id": 3,
        "title": "Getting Diagnosed — Parkinson's Foundation",
        "url": "https://www.parkinson.org/understanding-parkinsons/getting-diagnosed",
        "summary": ("Diagnosis is based on clinical history, symptoms, and neurological/physical "
                    "examination. No single laboratory or imaging test by itself confirms "
                    "Parkinson's disease. A neurologist, ideally a movement-disorders specialist, "
                    "assesses the overall clinical picture; investigations may help evaluate "
                    "other possible causes.")
    },
    {
        "id": 4,
        "title": "10 Early Signs — Parkinson's Foundation",
        "url": "https://www.parkinson.org/understanding-parkinsons/10-early-signs",
        "summary": ("The Foundation discusses possible early signs such as tremor, smaller "
                    "handwriting, loss of smell, sleep changes, stiffness, slowness, walking "
                    "changes, constipation, softer voice, and masked facial expression. No "
                    "single sign means a person has Parkinson's; concerning symptoms need "
                    "professional assessment.")
    },
    {
        "id": 5,
        "title": "A review of machine learning and deep learning algorithms for Parkinson's disease detection using handwriting and voice datasets — Heliyon (2024)",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10865258/",
        "summary": ("This review surveys ML/DL approaches for Parkinson's detection using "
                    "handwriting and voice data, including common feature and classification "
                    "approaches. It is a research review, not a clinical guideline. Results from "
                    "individual datasets should not be interpreted as diagnostic performance "
                    "for an individual person.")
    },
    {
        "id": 6,
        "title": "A review of machine learning and deep learning for Parkinson's disease detection — Discover Artificial Intelligence (2025)",
        "url": "https://link.springer.com/10.1007/s44163-025-00241-9",
        "summary": ("Rabie and Akhloufi review ML/DL methods and datasets used for Parkinson's "
                    "detection and progression monitoring, including audio, gait, and medical "
                    "imaging. They discuss SVM, random forests, and CNNs, alongside limitations "
                    "such as dataset diversity, data quality, privacy, and the need for explainable "
                    "AI. Reported high accuracies across studies are dataset-specific and are "
                    "not directly comparable or proof of clinical readiness.")
    },
    {
        "id": 7,
        "title": "Automatic and Early Detection of Parkinson's Disease by Analyzing Acoustic Signals Using Classification Algorithms Based on Recursive Feature Elimination — Diagnostics (2023)",
        "url": "https://www.mdpi.com/2075-4418/13/11/1924",
        "summary": ("This study evaluates classification of Parkinson's disease from acoustic/voice "
                    "signals and applies recursive feature elimination for feature selection. "
                    "It compares machine-learning classifiers and reports performance within "
                    "its experimental dataset. The results are research findings tied to that "
                    "dataset and validation design, not a validated clinical screening test.")
    },
    {
        "id": 8,
        "title": "Deep learning and wearable sensors for the diagnosis and monitoring of Parkinson's disease: A systematic review — Expert Systems with Applications (2023)",
        "url": "https://www.sciencedirect.com/science/article/pii/S0957417423010436",
        "summary": ("Sigcha and colleagues systematically review deep-learning and wearable-sensor "
                    "approaches for Parkinson's diagnosis and monitoring. The review covers "
                    "sensor-derived movement/gait information and highlights that wearable and "
                    "deep-learning approaches remain under development, with methodological and "
                    "validation challenges that affect translation into practice.")
    },
    {
        "id": 9,
        "title": "Detection of Parkinson disease using multiclass machine learning approach — Scientific Reports (2024)",
        "url": "https://www.nature.com/articles/s41598-024-64004-9",
        "summary": ("Srinivasan and colleagues investigate machine-learning and feed-forward neural "
                    "network approaches for distinguishing Parkinson's and healthy groups using "
                    "voice-signal characteristics. The article describes K-nearest-neighbour and "
                    "feed-forward neural-network methods and feature/model-selection procedures. "
                    "Its reported results apply to the study dataset and do not establish a "
                    "standalone clinical diagnosis from voice.")
    },
    {
        "id": 10,
        "title": "Early detection of Parkinson's disease using machine learning — Procedia Computer Science (2023)",
        "url": "https://www.sciencedirect.com/science/article/pii/S1877050923000078",
        "summary": ("Govindu and Palwe present a machine-learning approach for early detection "
                    "of Parkinson's disease. The paper is a conference-proceedings research "
                    "study; its model findings should be interpreted in the context of the "
                    "dataset, feature engineering, and evaluation protocol used by the authors, "
                    "rather than as evidence of clinical diagnostic capability.")
    },
    {
        "id": 11,
        "title": "Explainable artificial intelligence to diagnose early Parkinson's disease via voice analysis — Scientific Reports (2025)",
        "url": "https://www.nature.com/articles/s41598-025-96575-6",
        "summary": ("Shen and colleagues study voice-based early Parkinson's classification using "
                    "81 recordings and a hybrid CNN/RNN/multiple-kernel-learning/MLP approach. "
                    "They analyse acoustic features including MFCCs, jitter, and shimmer, and "
                    "use SHAP to explain feature contributions. The sample is small and the "
                    "reported performance is study-specific; it should not be generalized to "
                    "clinical diagnosis or other populations without independent validation.")
    },
    {
        "id": 12,
        "title": "Imperative Role of Machine Learning Algorithm for Detection of Parkinson's Disease: Review, Challenges and Recommendations — Diagnostics (2022)",
        "url": "https://www.mdpi.com/2075-4418/12/8/2003",
        "summary": ("Rana, Dumka, and Singh review 112 research papers covering Parkinson's "
                    "datasets, targets, biomarkers, and machine-learning algorithms. The review "
                    "discusses the promise of ML and biomarkers as decision-support research, "
                    "alongside challenges and future recommendations. It is a literature review, "
                    "not a clinical protocol or endorsement of any one model.")
    },
    {
        "id": 13,
        "title": "Machine learning approaches to identify Parkinson's disease using voice signal features — Frontiers in Artificial Intelligence (2023)",
        "url": "https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1084001/full",
        "summary": ("Alshammri and colleagues evaluate SVM, random forest, decision tree, KNN, "
                    "and MLP models using voice features from a UCI dataset described as 195 "
                    "recordings from 31 people. They report their best results for selected "
                    "models with SMOTE and GridSearchCV. The small, repeated-recording dataset "
                    "makes subject-level separation and external validation important when "
                    "interpreting performance; results are not clinical diagnostic evidence.")
    },
    {
        "id": 14,
        "title": "Multi-modality machine learning predicting Parkinson's disease — npj Parkinson's Disease (2022)",
        "url": "https://www.nature.com/articles/s41531-022-00288-w",
        "summary": ("Makarious and colleagues investigate multimodal, primarily multi-omic data "
                    "from PPMI and validate a selected model in the PDBP cohort. They report "
                    "AUC values for internal development and external validation and discuss "
                    "how combining data modalities can add predictive information. The study's "
                    "data modalities and cohort design differ from this dashboard's four "
                    "independent experimental classifiers; it does not validate this project's "
                    "fusion or establish patient-level linkage in its datasets.")
    },
    {
        "id": 15,
        "title": "Voice Biomarkers for Parkinson's Disease Prediction Using Machine Learning Models with Improved Feature Reduction Techniques — Journal of Data Science and Intelligent Systems (2023)",
        "url": "https://ojs.bonviewpress.com/index.php/jdsis/article/view/831",
        "summary": ("Chintalapudi and colleagues study voice recordings as potential biomarkers "
                    "and compare SVM, KNN, and random-forest models with SMOTE and dimensionality "
                    "reduction. The article reports its best classification result for random "
                    "forest on its experimental data. These results depend on the dataset and "
                    "evaluation design and should not be interpreted as clinical validation.")
    },
]


def format_source_context():
    return "\n\n".join(
        f"[{s['id']}] {s['title']}\nURL: {s['url']}\nSource notes: {s['summary']}"
        for s in SOURCES
    )


def public_sources():
    return [{"id": s["id"], "title": s["title"], "url": s["url"]} for s in SOURCES]
