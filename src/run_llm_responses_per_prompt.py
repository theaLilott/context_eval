import os
import pandas as pd
import requests
import time
from dotenv import load_dotenv

# =============== CONFIGURATION ===============
INPUT_CSV = "eval_dataset/evaluation_prompts.csv"
OUTPUT_CSV = "eval_dataset/llm_responses_final.csv"
load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
# List of models to run when ai_model is "All"
MODELS_FOR_ALL = [
    "anthropic/claude-sonnet-4",
    "google/gemini-2.5-pro",
    "openai/gpt-5"
]

API_URL = "https://openrouter.ai/api/v1/chat/completions"

def get_model_response(prompt, model, apikey=OPENROUTER_API_KEY):
    headers = {"Authorization": f"Bearer {apikey}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0 if model in ["google/gemini-2.5-pro", "anthropic/claude-sonnet-4"] else None
    }
    # Remove temp if not set
    if payload["temperature"] is None:
        del payload["temperature"]
    try:
        r = requests.post(API_URL, json=payload, headers=headers)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error with model {model}: {e} | Response: {getattr(r, 'text', '')}")
        return f"ERROR: {e}"

def main():
    if not OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY not set in environment or .env!")

    # Load main prompts
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    # Setup output file and resume (skip prompt_id + used_llm combos already done)
    if os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0:
        done = pd.read_csv(OUTPUT_CSV)
        done_keys = set(zip(done["prompt_id"], done["used_llm"]))
        print(f"Resuming: {len(done_keys)} prompt_id+used_llm combos already done, will skip.")
    else:
        done_keys = set()

    for i, row in df.iterrows():
        prompt_id = row.get("prompt_id", row.get("request_id", i))
        prompt = row["final_prompt"]
        ai_model = row["ai_model"]
        models_to_run = []

        # Main logic: which model(s) to use?
        if ai_model.strip().lower() == "all":
            models_to_run = MODELS_FOR_ALL
        else:
            models_to_run = [ai_model.strip()]

        for model in models_to_run:
            key = (prompt_id, model)
            if key in done_keys:
                print(f"Skipping {prompt_id} [{model}] (already done)")
                continue
            print(f"[{i+1}/{len(df)}] Running: prompt_id={prompt_id}, model={model}, theme={row.get('theme', '')}")
            response = get_model_response(prompt, model)
            time.sleep(1.5)
            out_row = row.to_dict()
            out_row["used_llm"] = model
            out_row["model_response"] = response
            # Write to output
            write_header = not os.path.isfile(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0
            pd.DataFrame([out_row]).to_csv(
                OUTPUT_CSV,
                mode="a",
                header=write_header,
                index=False
            )
            # Add to done_keys for future skips in this run
            done_keys.add(key)

    print(f"\n✅ Done! All responses saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
