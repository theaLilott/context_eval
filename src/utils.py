# src/utils.py

from src.config import ALL_MODELS, MODEL_API_KEYS
from src.logger import log

def get_models_for_row(ranking_type, ai_model):
    """
    Dispatcher: decides which models to use for this row.
    - If ranking_type == 'AI', only use the given ai_model (string)
    - Else, use all models from config.ALL_MODELS
    """
    if ranking_type and ranking_type.lower() == "ai" and ai_model:
        return [ai_model]
    return ALL_MODELS

def call_model_api(model_name, prompt, api_keys=MODEL_API_KEYS, max_retries=3):
    """
    Call the specified model via the relevant API.
    Fallback: Retry up to max_retries times if API call fails.
    Returns the model response (string), or None on total failure.
    """
    # Pseudocode: you need to implement the API calls for each provider
    import time

    for attempt in range(1, max_retries+1):
        try:
            # Example: route to correct provider based on model_name prefix
            if model_name.startswith("openai/"):
                # (Fill in OpenAI API logic here)
                response = f"FAKE OpenAI response for: {prompt[:40]}"
            elif model_name.startswith("anthropic/"):
                # (Fill in Anthropic API logic here)
                response = f"FAKE Anthropic response for: {prompt[:40]}"
            elif model_name.startswith("mistral/"):
                # (Fill in Mistral API logic here)
                response = f"FAKE Mistral response for: {prompt[:40]}"
            else:
                raise ValueError(f"Unknown model/provider: {model_name}")

            log.info(f"[{model_name}] Model call success on attempt {attempt}")
            return response
        except Exception as e:
            log.error(f"[{model_name}] Model call failed on attempt {attempt}: {e}")
            time.sleep(1)  # basic backoff

    log.error(f"[{model_name}] All retries failed.")
    return None

def parse_json_response(response_text):
    """
    Utility to parse a JSON string returned by an LLM (removes code fences if present).
    Returns a dict, or None if parsing fails.
    """
    import json
    import re

    # Remove possible ```json fences
    cleaned = response_text.strip()
    cleaned = re.sub(r"^```json|^```|```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except Exception as e:
        log.error(f"JSON parsing failed: {e}\nResponse: {response_text}")
        return None
