import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import glob, os, ast

# ---------------- Config ----------------
MODEL_MAP = {
    "openai/gpt-5": "GPT-5",
    "anthropic/claude-sonnet-4": "Claude Sonnet 4",
    "google/gemini-2.5-pro": "Gemini 2.5 Pro",
}
RANKING_TYPES = ["baseline", "relevance", "likelihood"]
COLORS = {"baseline": "#555555", "relevance": "#E07B39", "likelihood": "#1C90F3"}
VULN_LEVELS = ["low", "medium", "high"]
OUT_DIR = "plot_for_paper_context_factors"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.1,
})

# ---------------- Helpers ----------------
def count_factors(val):
    if pd.isnull(val) or val in ['', 'nan', 'none']:
        return 0
    if isinstance(val, int):
        return val
    try:
        factors = ast.literal_eval(val) if isinstance(val, str) and val.startswith("[") else val
        if isinstance(factors, list):
            return len(factors)
        if isinstance(factors, str):
            return 1
    except Exception:
        pass
    return 0

def load_domain_df(domain):
    prefix = "f" if domain == "finance" else "h"
    prompts = pd.read_csv("eval_dataset/evaluation_prompts.csv")
    llm_responses = pd.concat([pd.read_csv(f) for f in glob.glob(f"output/llm_response_{prefix}*.csv")], ignore_index=True)
    safety_scores = pd.concat([pd.read_csv(f) for f in glob.glob(f"scores/safety_scores_{prefix}*.csv")], ignore_index=True)

    df = llm_responses.merge(prompts, on="prompt_id", suffixes=("", "_prompt"))
    df = df.merge(safety_scores, on=["prompt_id", "used_llm"], suffixes=("", "_score"))

    for col in ["used_llm", "ranking_type", "vulnerability_profile_level", "topic"]:
        df[col] = df[col].astype(str).str.lower().str.strip()

    df = df[df["topic"] == domain]
    df["n_factors"] = df["context_factors_used"].apply(count_factors)
    return df

def build_curves(df_sub):
    curves = {}
    for r in RANKING_TYPES:
        d = df_sub[df_sub["ranking_type"] == r]
        x_vals = sorted(d["n_factors"].unique())
        app = d.groupby("n_factors")["apparent_safety_score"].mean()
        ctx = d.groupby("n_factors")["contextualized_safety_score"].mean()
        curves[r] = {
            "x": x_vals,
            "apparent": [app.get(x, float("nan")) for x in x_vals],
            "contextualized": [ctx.get(x, float("nan")) for x in x_vals],
        }
    return curves

def plot_three_panels(domain, model_id, out_prefix):
    df = load_domain_df(domain)
    if df.empty:
        print(f"[{domain}] No data.")
        return

    pretty_model = MODEL_MAP.get(model_id, model_id)
    domain_title = "Health Advice" if domain == "health" else "Financial Advice"

    d_model = df[(df["used_llm"] == model_id) | (df["used_llm"] == "all")]
    if d_model.empty:
        print(f"[{domain}] No rows for model {model_id}.")
        return

    # Figure: three subplots (low/medium/high), shared y for consistent scale
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    # Common title at top
    fig.suptitle(f"{domain_title} | {pretty_model}", y=0.97, fontsize=14)

    # Collect union X for consistent ticks across subplots
    all_x_union = set()
    panel_data = []
    for vlevel in VULN_LEVELS:
        d_v = d_model[d_model["vulnerability_profile_level"] == vlevel]
        curves = build_curves(d_v)
        panel_data.append((vlevel, curves))
        for r in RANKING_TYPES:
            all_x_union.update(curves[r]["x"])
    all_x = sorted(all_x_union) if all_x_union else []

    for ax, (vlevel, curves) in zip(axes, panel_data):
        # --- draw all lines first ---
        for r in RANKING_TYPES:
            cx = curves[r]["x"]
            cy_app = curves[r]["apparent"]
            cy_ctx = curves[r]["contextualized"]
            color = COLORS[r]
            ax.plot(cx, cy_app, linestyle="--", color=color, marker="o", label=r.capitalize())
            ax.plot(cx, cy_ctx, linestyle="-",  color=color, marker="o")

            # connect within same ranking_type (context-aware baseline 0→1)
            if 0 in cx and 1 in cx:
                y0 = cy_ctx[cx.index(0)]
                y1 = cy_ctx[cx.index(1)]
                ax.plot([0, 1], [y0, y1], color=color, linewidth=1.0, alpha=0.55)

        # --- connect Baseline → others (both dashed & solid) ---
        cx_base = curves["baseline"]["x"]
        cy_base_app = curves["baseline"]["apparent"]
        cy_base_ctx = curves["baseline"]["contextualized"]

        for r in ["relevance", "likelihood"]:
            cx_r = curves[r]["x"]
            cy_r_app = curves[r]["apparent"]
            cy_r_ctx = curves[r]["contextualized"]
            color = COLORS[r]
            if 0 in cx_base and 1 in cx_r:
                # dashed connection: baseline (blind) → relevance/likelihood (blind)
                y0 = cy_base_app[cx_base.index(0)]
                y1 = cy_r_app[cx_r.index(1)]
                ax.plot([0, 1], [y0, y1], color=color, linestyle="--", alpha=0.6, linewidth=1.0)
                # solid connection: baseline (aware) → relevance/likelihood (aware)
                y0 = cy_base_ctx[cx_base.index(0)]
                y1 = cy_r_ctx[cx_r.index(1)]
                ax.plot([0, 1], [y0, y1], color=color, linestyle="-", alpha=0.6, linewidth=1.0)

        # per-panel formatting
        ax.set_title(f"{vlevel} vulnerability", pad=4)
        ax.set_xlabel("# Context factors")
        if ax is axes[0]:
            ax.set_ylabel("Safety score")
        ax.set_xticks(all_x)
        ax.set_ylim(2.5, 6.5)
        ax.set_yticks([3, 4, 5, 6])
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


    # ---------- Shared legend at the top ----------
    # First row: colors -> ranking types (solid samples)
    color_handles = [
        Line2D([0], [0], color=COLORS["baseline"],   lw=2.2, linestyle="-", marker="o", label="Baseline"),
        Line2D([0], [0], color=COLORS["relevance"],  lw=2.2, linestyle="-", marker="o", label="Relevance"),
        Line2D([0], [0], color=COLORS["likelihood"], lw=2.2, linestyle="-", marker="o", label="Likelihood"),
    ]
    # Second row: line-style semantics (black samples)
    style_handles = [
        Line2D([0], [0], color="#000000", lw=2.2, linestyle="--", label="context-blind"),
        Line2D([0], [0], color="#000000", lw=2.2, linestyle="-",  label="context-aware"),
    ]
    handles = color_handles + style_handles
    labels  = [h.get_label() for h in handles]

    # Place legend above plots
    # fig.legend(handles, labels,
    #                  loc="upper center", bbox_to_anchor=(0.5, 1.06),
    #                  ncol=5, frameon=False, handlelength=2.8, columnspacing=1.4)
    fig.legend(handles, labels,
           loc="upper center", bbox_to_anchor=(0.5, 1.12),
           ncol=5, frameon=False, handlelength=2.8, columnspacing=1.4)

    fig.tight_layout(rect=[0, 0, 1, 0.90])  # leave room for suptitle & legend
    out_path = os.path.join(OUT_DIR, f"{out_prefix}_{domain}.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved panel: {out_path}")


# ---------------- Generate: one model per domain ----------------
# Example: make Gemini the main text; switch model_id as needed.
for model_id in ["google/gemini-2.5-pro", "openai/gpt-5", "anthropic/claude-sonnet-4"]:
    plot_three_panels("health",  model_id, out_prefix=f"cf_panel_{MODEL_MAP.get(model_id, model_id).replace(' ', '_')}")
    plot_three_panels("finance", model_id, out_prefix=f"cf_panel_{MODEL_MAP.get(model_id, model_id).replace(' ', '_')}")
