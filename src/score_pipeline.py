import pandas as pd
import glob
import os

MAGIC_ROW_COUNT = 621  # Change if needed per file

# Scan all output/llm_response_h*.csv to cover all your IDs
input_files = sorted(glob.glob("output/llm_response_h*.csv"))

for input_path in input_files:
    base = os.path.basename(input_path).replace("llm_response_", "").replace(".csv", "")
    score_path = f"scores/safety_scores_{base}.csv"
    if not os.path.exists(score_path):
        print(f"\n{score_path}: [MISSING] No scores file for this input, skipping.")
        continue

    # Load input and output
    df_in = pd.read_csv(input_path)
    df_out = pd.read_csv(score_path)

    # Merge to get the correct ranking_type for each output row
    merged = df_out.merge(
        df_in[["prompt_id", "used_llm", "ranking_type"]],
        on=["prompt_id", "used_llm"],
        how="left"
    )

    # Only check non-AI rows
    non_ai = merged[merged["ranking_type"] != "AI"].reset_index(drop=True)
    n_total = len(non_ai)
    n_nan_apparent = non_ai["apparent_safety_score"].isna().sum()
    n_nan_context = non_ai["contextualized_safety_score"].isna().sum()

    print(f"\n{score_path}")
    print(f"  Non-AI rows: {n_total}")
    print(f"  NaN in 'apparent_safety_score': {n_nan_apparent}")
    print(f"  NaN in 'contextualized_safety_score': {n_nan_context}")

    if n_total != MAGIC_ROW_COUNT:
        print(f"  ⚠️ Row count mismatch! Expected {MAGIC_ROW_COUNT}.")
    if n_nan_apparent == 0 and n_nan_context == 0 and n_total == MAGIC_ROW_COUNT:
        print("  ✅ All non-AI rows have scores. [COMPLETE!]")
    else:
        print("  ❌ Some non-AI rows missing scores or count mismatch. [INCOMPLETE!]")
