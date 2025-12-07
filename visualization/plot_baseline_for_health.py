import pandas as pd
import glob, os

# Read prompts just once
df_prompts = pd.read_csv("eval_dataset/evaluation_prompts.csv")

all_rows = []
for resp_file in sorted(glob.glob("output/llm_response_h*.csv")):
    # Extract request_id
    request_id = os.path.basename(resp_file).replace("llm_response_", "").replace(".csv", "")
    score_file = f"scores/safety_scores_{request_id}.csv"
    if not os.path.exists(score_file):
        continue

    resp = pd.read_csv(resp_file)
    scores = pd.read_csv(score_file)

    # --- ONLY MERGE ON UNIQUE COLUMNS ---
    # Drop 'prompt_id' from prompts for right merge (as it's already in resp)
    # If you want to include request_id in join, ensure both have it
    prompts_meta = [c for c in df_prompts.columns if c not in resp.columns or c in ["context_level", "ai_model", "context_variant", "context_factors_used"]]
    # Merge responses with prompts (NO DUPLICATES)
    merged1 = pd.merge(resp, df_prompts[["prompt_id"] + prompts_meta], on="prompt_id", how="left")
    # Merge with scores (NO DUPLICATES)
    merged2 = pd.merge(merged1, scores, on=["prompt_id", "used_llm"], how="left")
    all_rows.append(merged2)

df = pd.concat(all_rows, ignore_index=True)
print("Final merged shape:", df.shape)
print("Final columns:", list(df.columns))
