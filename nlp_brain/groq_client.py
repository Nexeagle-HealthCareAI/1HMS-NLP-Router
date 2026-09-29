"""Groq API call: the piece that plays the same role matching.py + the
sklearn model's .predict() step play in the original pipeline, but for the
Groq-backed classifier. Takes an already-cleaned, already-gibberish-checked
segment and an allowed specialist list; returns a ranked list of
(label, score) pairs plus an overall confidence, or None if every retry
failed.

Kept separate from groq_classifier.py for the same reason matching.py and
features.py are kept separate from classifier.py there: the "how do we
decide" step (GroqSymptomClassifier.predict()/_predict_segment()) shouldn't
need to know the specifics of the Groq wire format, retry policy, or JSON
schema -- and vice versa, this module doesn't need to know what a
PredictionResult is.
"""
import json
from typing import List, Optional

from groq import Groq

from .config import GROQ_MAX_RETRIES, GROQ_MODEL


def _build_system_prompt(specialists: List[str]) -> str:
    return (
        "You classify short Hinglish (Hindi+English) descriptions of "
        "physical symptoms into medical specialists.\n\n"
        f"Allowed specialists (choose ONLY from this exact list): "
        f"{json.dumps(specialists)}\n\n"
        "Respond with ONLY a JSON object, no other text, in this exact "
        "shape:\n"
        '{"ranked_labels": [{"specialist": "<one of the allowed '
        'specialists>", "score": <0-1 float>}, ...], '
        '"confidence": <0-1 float, your overall confidence that the top '
        "label is correct>}\n\n"
        "ranked_labels should be ordered highest score first. Include at "
        "least 3 entries whenever plausible (even lower-confidence "
        "guesses), and at most 5. If the text doesn't clearly describe a "
        "symptom matching any allowed specialist, return "
        '{"ranked_labels": [], "confidence": 0.0}.'
    )


def _validate_response(parsed: dict, specialists: List[str]) -> dict:
    """Drops any label the model invented that isn't in the allowed list
    (guards against Groq hallucinating a specialist name), and coerces
    confidence/score into floats in [0, 1] in case the model returns them
    as strings or out-of-range values."""
    ranked = []
    for entry in parsed.get("ranked_labels", []):
        label = entry.get("specialist")
        if label in specialists:
            score = max(0.0, min(1.0, float(entry.get("score", 0))))
            ranked.append((label, score))
    confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0))))
    return {"ranked_labels": ranked, "confidence": confidence}


def classify_segment(
    client: Groq,
    segment: str,
    specialists: List[str],
    model: str = GROQ_MODEL,
    max_retries: int = GROQ_MAX_RETRIES,
) -> Optional[dict]:
    """Classifies one already-cleaned, already-gibberish-checked segment.

    Returns {"ranked_labels": [(label, score), ...], "confidence": float}
    on success, or None if every attempt failed (bad API key, rate limit, a
    decommissioned model ID, malformed JSON, etc.). Callers should treat
    None the same as "the model couldn't help" rather than raise on it, so
    one bad segment doesn't take down a multi-symptom query -- see
    GroqSymptomClassifier._predict_segment()."""
    system_prompt = _build_system_prompt(specialists)
    last_error = None

    for _ in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": segment},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            parsed = json.loads(response.choices[0].message.content)
            return _validate_response(parsed, specialists)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            last_error = e
            continue
        except Exception as e:  # network/auth/rate-limit/model errors from Groq
            last_error = e
            continue

    # Every attempt failed -- surface the real reason instead of letting the
    # caller silently treat this the same as low confidence.
    if last_error is not None:
        print(f"[groq_client] classify_segment failed after "
              f"{max_retries + 1} attempt(s): {last_error}")
    return None
