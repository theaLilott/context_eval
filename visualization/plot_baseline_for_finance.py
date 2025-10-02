import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import glob
import os

# --- Pretty labels & colors
model_map = {
    "openai/gpt-5": "GPT-5",
    "anthropic/claude-sonnet-4": "Claude",
    "google/gemini-2.5-pro": "Gemini"
}
profiles = ["context-blind", "low", "medium", "high"]
colors = ["#364B9A", "#4AABAF", "#7CBC42", "#EE8F71"]

matplotlib.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.2,
})

# --- Aggregate all fXX files ---
response_files = sorted(glob.glob("output/llm_response_f*.csv"))
scores_files = sorted(glob.glob("scores/safety_scores_f*.csv"))
all_rows = []

for resp_file, score_file in zip(response_files, scores_files):
    resp = pd.read_csv(resp_file)
    scores = pd.read_csv(score_file)
    merged = pd.merge(
        resp,
        scores,
        on=["prompt_id", "used_llm"],
        how="inner",
        suffixes=("", "_score")
    )
    # -- Check: Must have exactly 27 Baseline rows (9 per used_llm)
    baseline_rows = merged[merged['ranking_type'].str.lower() == 'baseline']
    n_baseline = len(baseline_rows)
    if n_baseline != 27:
        print(f"{resp_file}: ❌ Only {n_baseline} Baseline rows found (should be 27)")
        continue
    else:
        print(f"{resp_file}: ✅ 27 Baseline rows found")
    all_rows.append(merged)

if not all_rows:
    print("No valid data files found with 27 Baseline rows each. Exiting.")
    exit(0)

df = pd.concat(all_rows, ignore_index=True)

models = ["openai/gpt-5", "anthropic/claude-sonnet-4", "google/gemini-2.5-pro"]
bar_data = []

for model in models:
    # 1. Context-blind: AVG apparent_safety_score among Baseline rows
    base_rows = df[(df['used_llm'] == model) & (df['ranking_type'].str.lower() == 'baseline')]
    if not base_rows.empty:
        context_blind_score = base_rows['apparent_safety_score'].mean()
        bar_data.append({"model": model, "profile": "context-blind", "score": context_blind_score})

    # 2–4. Baseline, contextualized by vuln level (low, medium, high)
    for vuln in ["low", "medium", "high"]:
        vuln_rows = base_rows[base_rows["vulnerability_profile_level"] == vuln]
        if not vuln_rows.empty:
            mean_ctx_score = vuln_rows['contextualized_safety_score'].mean()
            bar_data.append({"model": model, "profile": vuln, "score": mean_ctx_score})

bar_df = pd.DataFrame(bar_data)

# --- Save CSV summary ---
os.makedirs("plots", exist_ok=True)
csv_out = "plots/AVG_apparent_contextualized_safety_by_model_baseline_finance.csv"
bar_df.to_csv(csv_out, index=False)
print(f"Saved CSV to {csv_out}")

pivot = bar_df.pivot(index="model", columns="profile", values="score")
pivot = pivot[[c for c in profiles if c in pivot.columns]]
pivot.index = [model_map.get(x, x) for x in pivot.index]

# --- Plot
fig, ax = plt.subplots(figsize=(8.6, 4.6))

pivot.plot(kind="bar", width=0.75, edgecolor="black", ax=ax, color=colors)

# Labels & title
ax.set_ylabel("Safety Score (1–7)", labelpad=8)
ax.set_xlabel("Model", labelpad=8)
ax.set_title(
    "Avg Apparent (Context-Blind) & Contextualized Safety by Model (Baseline Only) [Finance]",
    fontsize=15, pad=12, loc='center'
)

# Force scale 1–7
ax.set_ylim(1, 7)
ax.set_yticks(range(1, 8))

# Grid and spines
ax.set_axisbelow(True)
ax.grid(axis="y", linestyle=":", alpha=0.5)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# X tick labels
ax.set_xticklabels(pivot.index, rotation=15, ha='right')

# --- Legend OUTSIDE on the right
ax.legend(
    title="Profile",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    ncol=1,
    frameon=False,
)

plt.tight_layout(rect=[0, 0, 0.88, 1])  # leave space for legend on right

# Save plot
out_path = "plots/AVG_apparent_contextualized_safety_by_model_baseline_finance.png"
plt.savefig(out_path, dpi=400, bbox_inches='tight')
plt.close()
print(f"Saved plot to {out_path}")
