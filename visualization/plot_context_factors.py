import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

# Parameters
target_model = "google/gemini-2.5-pro"    # Or whichever you want
target_theme = "Health"                   # Optional: filter on this
target_vuln = "high"                      # Optional: filter on this

# Load eval prompts
eval_prompts = pd.read_csv("eval_dataset/evaluation_prompts.csv")

# Aggregate and merge
all_rows = []
for resp_path in sorted(glob.glob("output/llm_response_h*.csv")):
    base = os.path.basename(resp_path).replace("llm_response_", "").replace(".csv", "")
    score_path = f"scores/safety_scores_{base}.csv"
    if not os.path.exists(score_path):
        continue
    resp = pd.read_csv(resp_path)
    scores = pd.read_csv(score_path)
    merged = pd.merge(resp, scores, on=["prompt_id", "used_llm"], how="inner", suffixes=("", "_score"))
    merged = pd.merge(merged, eval_prompts, on="prompt_id", how="left", suffixes=("", "_eval"))
    all_rows.append(merged)
df = pd.concat(all_rows, ignore_index=True)

# Filter for your plot
df = df[df["used_llm"] == target_model]
if target_theme:
    df = df[df["theme"] == target_theme]
if target_vuln:
    df = df[df["vulnerability_profile_level"] == target_vuln]

# Map for display
ranking_types = {
    "context-blind": "Context-Blind",
    "relevance": "Relevance",
    "likelihood": "Likelihood"
}
plot_colors = {
    "context-blind": "#2C318C",
    "relevance": "#E37222",
    "likelihood": "#28B463"
}

# For each context_level and ranking_type, compute mean safety score
context_levels = sorted(df["context_level"].unique())
lines = {}
for rt in ranking_types.keys():
    y = []
    for cl in context_levels:
        sub = df[(df["ranking_type"].str.lower() == rt) & (df["context_level"] == cl)]
        if rt == "context-blind":
            val = sub["apparent_safety_score"].mean()
        else:
            val = sub["contextualized_safety_score"].mean()
        y.append(val)
    lines[rt] = y

# --- Plot ---
plt.figure(figsize=(7, 5))
for rt, y in lines.items():
    plt.plot(context_levels, y, marker="o", label=ranking_types[rt], color=plot_colors[rt])
plt.xlabel("Context Factors Included in Prompt (context_level)")
plt.ylabel("Safety Score")
plt.title(f"Safety vs. Context Inclusion for {target_theme}, {target_vuln.title()} Vulnerability ({target_model})")
plt.ylim(1, 7)
plt.xticks(context_levels)
plt.grid(axis="y", linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("plots/RQ3_safety_vs_context_gemini.png", dpi=300)
plt.close()
