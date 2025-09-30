import os
import pandas as pd
import time
from dotenv import load_dotenv

from utils_pipeline import (
    build_apparent_prompt,
    build_contextualized_prompt,
    extract_overall_safety_score,
    call_judge
)

# ================== CONFIG ==================
INPUT_CSV = "output/llm_response_f20.csv"
OUTPUT_CSV = "scores/safety_scores_o3.csv"
JUDGE_MODEL = "openai/o3"
TEMPERATURE = 0.2

def main():
    load_dotenv()
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set!")

    df = pd.read_csv(INPUT_CSV)
    total = len(df)

    # -------- Resume logic: load already-judged pairs --------
    if os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0:
        done = pd.read_csv(OUTPUT_CSV)
        done_keys = set(zip(done["prompt_id"], done["used_llm"]))
        print(f"Resuming: {len(done_keys)} prompt_id+used_llm combos already scored, will skip.")
    else:
        done_keys = set()

    for idx, row in df.iterrows():
        prompt_id = row["prompt_id"]
        used_llm = row["used_llm"]
        key = (prompt_id, used_llm)

        if key in done_keys:
            print(f"[{idx+1}/{total}] Skipping prompt_id={prompt_id}, used_llm={used_llm} (already scored)")
            continue

        user_prompt = row["final_prompt"]
        ai_response = row["model_response"]
        user_profile = row.get("vulnerability_profile", "")

        # ----- Build judge prompts -----
        apparent_prompt = build_apparent_prompt(user_prompt, ai_response)
        contextualized_prompt = build_contextualized_prompt(user_profile, user_prompt, ai_response)

        # ----- Call judge model for both -----
        print(f"[{idx+1}/{total}] Judging prompt_id={prompt_id} ({used_llm})... ", end="")
        apparent_response = call_judge(
            apparent_prompt, model=JUDGE_MODEL, apikey=OPENROUTER_API_KEY, temperature=TEMPERATURE)
        contextualized_response = call_judge(
            contextualized_prompt, model=JUDGE_MODEL, apikey=OPENROUTER_API_KEY, temperature=TEMPERATURE)

        # ----- Extract scores -----
        apparent_score = extract_overall_safety_score(apparent_response)
        contextualized_score = extract_overall_safety_score(contextualized_response)

        print(f"apparent={apparent_score}, contextualized={contextualized_score}")

        # ----- Save result (append to CSV immediately) -----
        output_row = {
            "prompt_id": prompt_id,
            "used_llm": used_llm,
            "apparent_safety_score": apparent_score,
            "contextualized_safety_score": contextualized_score,
            "apparent_safety_judge_response": apparent_response,
            "contextualized_safety_judge_response": contextualized_response,
        }
        write_header = not os.path.isfile(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0
        pd.DataFrame([output_row]).to_csv(
            OUTPUT_CSV,
            mode="a",
            header=write_header,
            index=False
        )

        # Add to done_keys in case of interruption and rerun in same session
        done_keys.add(key)

        time.sleep(1.2)  # To avoid API rate limits

    print(f"\n✅ All safety scores saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
