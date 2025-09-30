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

SCORES_CSV = "scores/safety_scores_o3.csv"

JUDGE_MODEL = "openai/o3"
TEMPERATURE = 0.2

def get_llm_response_row(prompt_id, used_llm):
    request_id = prompt_id[:3]
    llm_csv = f"output/llm_response_{request_id}.csv"
    if not os.path.isfile(llm_csv):
        raise FileNotFoundError(f"Missing {llm_csv}")
    llm_df = pd.read_csv(llm_csv)
    match = llm_df[(llm_df['prompt_id'] == prompt_id) & (llm_df['used_llm'] == used_llm)]
    if match.empty:
        raise ValueError(f"No matching row in {llm_csv} for prompt_id={prompt_id}, used_llm={used_llm}")
    return match.iloc[0]

def main():
    load_dotenv()
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set!")

    df = pd.read_csv(SCORES_CSV)
    nan_rows = df[df[['apparent_safety_score', 'contextualized_safety_score']].isna().any(axis=1)]
    print(f"Found {len(nan_rows)} rows with missing safety scores.")

    for idx in nan_rows.index:
        prompt_id = df.at[idx, "prompt_id"]
        used_llm = df.at[idx, "used_llm"]
        print(f"\nPatching: prompt_id={prompt_id}, used_llm={used_llm}")

        # Get the correct LLM response row
        try:
            row = get_llm_response_row(prompt_id, used_llm)
        except Exception as e:
            print(f"  Skipping: {e}")
            continue

        user_prompt = row["final_prompt"]
        ai_response = row["model_response"]
        user_profile = row.get("vulnerability_profile", "")

        # Patch only missing fields
        if pd.isna(df.at[idx, "apparent_safety_score"]):
            apparent_prompt = build_apparent_prompt(user_prompt, ai_response)
            apparent_response = call_judge(
                apparent_prompt, model=JUDGE_MODEL, apikey=OPENROUTER_API_KEY, temperature=TEMPERATURE)
            apparent_score = extract_overall_safety_score(apparent_response)
            df.at[idx, "apparent_safety_score"] = apparent_score
            df.at[idx, "apparent_safety_judge_response"] = apparent_response
            print(f"  [Patched] apparent={apparent_score}")

        if pd.isna(df.at[idx, "contextualized_safety_score"]):
            contextualized_prompt = build_contextualized_prompt(user_profile, user_prompt, ai_response)
            contextualized_response = call_judge(
                contextualized_prompt, model=JUDGE_MODEL, apikey=OPENROUTER_API_KEY, temperature=TEMPERATURE)
            contextualized_score = extract_overall_safety_score(contextualized_response)
            df.at[idx, "contextualized_safety_score"] = contextualized_score
            df.at[idx, "contextualized_safety_judge_response"] = contextualized_response
            print(f"  [Patched] contextualized={contextualized_score}")

        time.sleep(1.2)

    df.to_csv(SCORES_CSV, index=False)
    print("\n✅ Patched missing safety scores in place.")

if __name__ == "__main__":
    main()
