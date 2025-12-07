import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
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

# --- Load data
resp = pd.read_csv("output/llm_response_f20.csv")
prompts = pd.read_csv("eval_dataset/evaluation_prompts.csv")
scores = pd.read_csv("scores/safety_scores_o3.csv")

# --- Join
df = pd.merge(resp, prompts, on="prompt_id", suffixes=("", "_meta"))
df = pd.merge(df, scores, on=["prompt_id", "used_llm"], how="inner")

# --- Build table for bars
models = ["openai/gpt-5", "anthropic/claude-sonnet-4", "google/gemini-2.5-pro"]
bar_data = []

for model in models:
    # Context-blind: pick the HIGHEST apparent safety among Baseline rows
    base_rows = df[(df['used_llm'] == model) & (df['ranking_type'] == 'Baseline')]
    if not base_rows.empty:
        context_blind_score = base_rows['apparent_safety_score'].max()
        bar_data.append({"model": model, "profile": "context-blind", "score": context_blind_score})

    # Contextualized: for each vul level, average across the 3 user profiles
    for vuln in ["low", "medium", "high"]:
        ctx_rows = df[
            (df['used_llm'] == model) &
            (df['vulnerability_profile_level'] == vuln) &
            (df['ranking_type'] != 'Baseline')
        ]
        ctx_profile_means = (
            ctx_rows.groupby('vulnerability_profile')['contextualized_safety_score']
            .first()  # one per profile
            .tolist()
        )
        if len(ctx_profile_means) > 0:
            mean_ctx_score = sum(ctx_profile_means) / len(ctx_profile_means)
            bar_data.append({"model": model, "profile": vuln, "score": mean_ctx_score})

bar_df = pd.DataFrame(bar_data)
pivot = bar_df.pivot(index="model", columns="profile", values="score")
pivot = pivot[[c for c in profiles if c in pivot.columns]]
pivot.index = [model_map.get(x, x) for x in pivot.index]

# --- Plot
os.makedirs("plots", exist_ok=True)
fig, ax = plt.subplots(figsize=(8.6, 4.6))

pivot.plot(kind="bar", width=0.75, edgecolor="black", ax=ax, color=colors)

# Labels & title
ax.set_ylabel("Safety Score", labelpad=8)
ax.set_xlabel("Model", labelpad=8)
ax.set_title(
    "Apparent (Context-Blind) & Avg Contextualized Safety by Model",
    fontsize=15, pad=12, loc='center'
)

# Force scale 0–5
ax.set_ylim(0, 5)
ax.set_yticks(range(0, 6))

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

# Save
out_path = "plots/neurips_MAX_apparent_AVG_contextualized_safety_by_model_profile.png"
plt.savefig(out_path, dpi=400, bbox_inches='tight')
plt.close()
print(f"Saved to {out_path}")
