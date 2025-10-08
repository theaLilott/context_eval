import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import glob
import os
from collections import defaultdict

# =========================
# Config & Styling
# =========================
matplotlib.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.2,
})

# Pretty labels & colors
MODEL_MAP = {
    "openai/gpt-5": "GPT-5",
    "anthropic/claude-sonnet-4": "Claude Sonnet 4",
    "google/gemini-2.5-pro": "Gemini 2.5 Pro",
}
MODEL_ORDER = ["openai/gpt-5", "anthropic/claude-sonnet-4", "google/gemini-2.5-pro"]

# Panel 1 (4 bars)
COLORS_P1 = ["#364B9A", "#4AABAF", "#7CBC42", "#EE8F71"]
P1_LABELS = ["context-blind", "low vulnerability", "medium vulnerability", "high vulnerability"]

# Panel 2 (2 bars)
COLORS_P2 = ["#364B9A", "#EE8F71"]
P2_LABELS = ["context-blind", "context-aware"]

OUT_DIR = "plot_for_paper"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Helpers
# =========================
def _safe_lower(s):
    return s.astype(str).str.lower()

def load_domain_df(domain):
    if domain == "finance":
        resp_glob = "output/llm_response_f*.csv"
        score_glob = "scores/safety_scores_f*.csv"
    else:
        resp_glob = "output/llm_response_h*.csv"
        score_glob = "scores/safety_scores_h*.csv"

    response_files = sorted(glob.glob(resp_glob))
    scores_files = sorted(glob.glob(score_glob))

    if not response_files or not scores_files:
        print(f"[{domain}] No files found.")
        return pd.DataFrame()

    all_rows = []
    for resp_file, score_file in zip(response_files, scores_files):
        try:
            resp = pd.read_csv(resp_file)
            scores = pd.read_csv(score_file)
        except Exception as e:
            print(f"[{domain}] Read error: {resp_file} / {score_file}: {e}")
            continue

        for df in (resp, scores):
            if "prompt_id" in df.columns:
                df["prompt_id"] = df["prompt_id"].astype(str).str.strip()
            if "used_llm" in df.columns:
                df["used_llm"] = df["used_llm"].astype(str).str.strip()

        merged = pd.merge(
            resp,
            scores,
            on=["prompt_id", "used_llm"],
            how="inner",
            suffixes=("", "_score"),
        )

        if "ranking_type" not in merged.columns and "ranking_type_score" in merged.columns:
            merged["ranking_type"] = merged["ranking_type_score"]

        if "ranking_type" not in merged.columns:
            print(f"[{domain}] Skipping {resp_file}: no ranking_type.")
            continue

        baseline_rows = merged[_safe_lower(merged["ranking_type"]) == "baseline"]
        if len(baseline_rows) != 27:
            print(f"[{domain}] {os.path.basename(resp_file)}: ❌ {len(baseline_rows)} baseline rows (need 27)")
            continue
        else:
            print(f"[{domain}] {os.path.basename(resp_file)}: ✅ 27 baseline rows")
        all_rows.append(merged)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)

def aggregate_panel1(df):
    out = []
    if df.empty:
        return pd.DataFrame(columns=["model", "profile", "score"])
    base = df[_safe_lower(df["ranking_type"]) == "baseline"].copy()
    for model in MODEL_ORDER:
        m = base[base["used_llm"] == model]
        if m.empty:
            continue
        cb = m["apparent_safety_score"].mean()
        out.append({"model": model, "profile": "context-blind", "score": cb})
        for lvl in ["low", "medium", "high"]:
            mlvl = m[_safe_lower(m["vulnerability_profile_level"]) == lvl]
            if not mlvl.empty:
                out.append({
                    "model": model,
                    "profile": f"{lvl} vulnerability",
                    "score": mlvl["contextualized_safety_score"].mean(),
                })
    return pd.DataFrame(out)

def aggregate_panel2(df):
    out = []
    if df.empty:
        return pd.DataFrame(columns=["model", "profile", "score"])
    base = df[_safe_lower(df["ranking_type"]) == "baseline"].copy()
    for model in MODEL_ORDER:
        m = base[base["used_llm"] == model]
        if m.empty:
            continue
        cb = m["apparent_safety_score"].mean()
        out.append({"model": model, "profile": "context-blind", "score": cb})
        ca_vals = []
        for lvl in ["low", "medium", "high"]:
            mlvl = m[_safe_lower(m["vulnerability_profile_level"]) == lvl]
            if not mlvl.empty:
                ca_vals.append(mlvl["contextualized_safety_score"].mean())
        if ca_vals:
            out.append({
                "model": model,
                "profile": "context-aware",
                "score": sum(ca_vals) / len(ca_vals),
            })
    return pd.DataFrame(out)

def plot_panel1(ax, tidy_df, title_str):
    import numpy as np
    profile_order = ["context-blind", "low vulnerability", "medium vulnerability", "high vulnerability"]
    data = []
    for model in MODEL_ORDER:
        row = []
        for prof in profile_order:
            val = tidy_df[(tidy_df["model"] == model) & (tidy_df["profile"] == prof)]["score"]
            row.append(float(val.iloc[0]) if not val.empty else float("nan"))
        data.append(row)
    data = np.array(data, dtype=float)
    n_models, n_profiles = data.shape
    x = np.arange(n_models)
    width = 0.18
    for i in range(n_profiles):
        bars = ax.bar(x + (i - (n_profiles-1)/2)*width, data[:, i],
                      width=width, edgecolor="black", color=COLORS_P1[i], label=P1_LABELS[i])
        for b in bars:
            if not np.isnan(b.get_height()):
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.03,
                        f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=11)
    ax.set_title(title_str, pad=10)
    ax.set_ylabel("Safety Score (1–7)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_MAP.get(m, m) for m in MODEL_ORDER], rotation=15, ha="right")
    ax.set_ylim(1, 7)
    ax.set_yticks(range(1, 8))
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

def plot_panel2(ax, tidy_df, title_str):
    import numpy as np
    profile_order = ["context-blind", "context-aware"]
    data = []
    for model in MODEL_ORDER:
        row = []
        for prof in profile_order:
            val = tidy_df[(tidy_df["model"] == model) & (tidy_df["profile"] == prof)]["score"]
            row.append(float(val.iloc[0]) if not val.empty else float("nan"))
        data.append(row)
    data = np.array(data, dtype=float)
    n_models, n_profiles = data.shape
    x = np.arange(n_models)
    width = 0.32
    for i in range(n_profiles):
        bars = ax.bar(x + (i - 0.5)*width, data[:, i],
                      width=width, edgecolor="black", color=COLORS_P2[i], label=P2_LABELS[i])
        for b in bars:
            if not np.isnan(b.get_height()):
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.03,
                        f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=11)
    ax.set_title(title_str, pad=10)
    ax.set_ylabel("Safety Score (1–7)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_MAP.get(m, m) for m in MODEL_ORDER], rotation=15, ha="right")
    ax.set_ylim(1, 7)
    ax.set_yticks(range(1, 8))
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

# =========================
# Load data
# =========================
df_fin = load_domain_df("finance")
df_health = load_domain_df("health")

# =========================
# Panel 1: Profiles (4 bars)
# =========================
p1_fin = aggregate_panel1(df_fin)
p1_health = aggregate_panel1(df_health)

fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
plot_panel1(axes1[0], p1_health, "Health Advice")
plot_panel1(axes1[1], p1_fin, "Financial Advice")

handles, labels = axes1[1].get_legend_handles_labels()
fig1.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06))
fig1.tight_layout(rect=[0, 0, 1, 0.98])
panel1_path = os.path.join(OUT_DIR, "panel1_profiles.pdf")
fig1.savefig(panel1_path, dpi=300, bbox_inches="tight")
plt.close(fig1)
print(f"Saved Panel 1 to {panel1_path}")

# =========================
# Panel 2: Context-blind vs Context-aware (2 bars)
# =========================
p2_fin = aggregate_panel2(df_fin)
p2_health = aggregate_panel2(df_health)

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
plot_panel2(axes2[0], p2_health, "Health Advice")
plot_panel2(axes2[1], p2_fin, "Financial Advice")

handles2, labels2 = axes2[1].get_legend_handles_labels()
fig2.legend(handles2, labels2, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
fig2.tight_layout(rect=[0, 0, 1, 0.98])
panel2_path = os.path.join(OUT_DIR, "panel2_cbca.pdf")
fig2.savefig(panel2_path, dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"Saved Panel 2 to {panel2_path}")
