"""Source-grounded Parkinson's educational assistant using OpenRouter."""
import os
import httpx
from backend.knowledge.parkinsons_sources_backup import format_source_context, public_sources

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """You are NeuroDetect Assistant, an educational assistant for a student/research
prototype about Parkinson's disease.

SOURCE-GROUNDING RULES
- Answer using ONLY the curated source notes supplied below for factual claims about
  Parkinson's disease. The notes are short summaries of the named official sources.
- Cite factual statements inline using [1], [2], etc. These numbers refer to the source
  list returned with the answer. Do not invent citations or sources.
- If the supplied notes do not contain enough information, say that the current source
  pack does not cover it. Do not fill gaps with guesses or unsupported medical claims.
- Explain technical concepts in accessible language. Clearly distinguish established
  source facts from general explanations.
- Never claim that this prototype, its classifiers, or a probability score diagnoses
  Parkinson's disease. Its Audio, Drawing, ECG, and MRI outputs are independent
  experimental model predictions; they are not patient-level fusion and are not
  clinically validated.
- Do not diagnose, estimate an individual's risk, interpret personal symptoms as
  Parkinson's, or recommend a medication/dose or medication change. Encourage a
  qualified clinician/neurologist for personal medical concerns.
- If a user describes sudden severe symptoms or an emergency, advise seeking urgent
  local medical care rather than continuing a chatbot discussion.
- Be calm, respectful, concise, and transparent. Do not pretend to be a doctor.

CURATED SOURCE NOTES:
{source_context}
"""

async def ask_assistant(messages):
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured. Add your OpenRouter key to the backend .env file."
        )

    model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash").strip()
    system_message = SYSTEM_PROMPT.format(source_context=format_source_context())
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_message}, *messages],
        "temperature": 0.2,
        "max_tokens": 900,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8000"),
        "X-Title": "NeuroDetect Parkinson's Research Assistant",
    }
    timeout = httpx.Timeout(45.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(OPENROUTER_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("The language model returned an unexpected response.") from exc
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("The language model returned an empty answer.")
    return {"answer": answer.strip(), "sources": public_sources(), "model": model}
