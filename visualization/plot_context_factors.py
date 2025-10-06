import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import ast

# --- 1. LOAD & MERGE ---
print("Loading files...")
prompts = pd.read_csv('eval_dataset/evaluation_prompts.csv')
llm_responses = pd.concat([pd.read_csv(f) for f in glob.glob('output/llm_response_h*.csv')])
safety_scores = pd.concat([pd.read_csv(f) for f in glob.glob('scores/safety_scores_h*.csv')])
print(f"Prompts: {prompts.shape}, Responses: {llm_responses.shape}, Safety scores: {safety_scores.shape}")

# Merge responses + prompts
df = llm_responses.merge(prompts, on='prompt_id', suffixes=('', '_prompt'))
print("After merge (responses + prompts):", df.shape)
# Merge with scores
df = df.merge(safety_scores, left_on=['prompt_id', 'used_llm'], right_on=['prompt_id', 'used_llm'], suffixes=('', '_score'))
print("After merge (+scores):", df.shape)

# --- 2. CLEAN AND FILTER for health topic ---
for col in ['used_llm', 'ranking_type', 'vulnerability_profile_level', 'topic']:
    df[col] = df[col].astype(str).str.lower().str.strip()

df = df[df['topic'] == 'health']
print("After filtering for topic=health:", df.shape)

# --- 3. Add n_factors column ---
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

df['n_factors'] = df['context_factors_used'].apply(count_factors)

# --- DEBUG: Crosstabs for inspection ---
print("\nRanking type and used_llm cross-tab:")
print(df.groupby(['ranking_type', 'used_llm']).size().unstack(fill_value=0))

print("\nRanking type and vulnerability_profile_level cross-tab:")
print(df.groupby(['ranking_type', 'vulnerability_profile_level']).size().unstack(fill_value=0))

# --- 4. PLOT ONLY FOR openai/gpt-5 ---
model = "google/gemini-2.5-pro"
vuln_levels = ['low', 'medium', 'high']

# Create output directory
os.makedirs("plots_context_factors", exist_ok=True)

print("Plotting ONLY for model:", model)
print("Vulnerability levels:", vuln_levels)

for vlevel in vuln_levels:
    # Take rows for either gpt-5 OR where used_llm == "all"
    dmodel = df[
        ((df['used_llm'] == model) | (df['used_llm'] == 'all')) &
        (df['vulnerability_profile_level'] == vlevel)
    ]
    print(f"\n== Model: {model} | Vulnerability: {vlevel} | Rows: {dmodel.shape[0]} ==")
    if dmodel.empty:
        print(f"  No data for model: {model} | vulnerability: {vlevel}")
        continue

    curves = {}
    for ranking_type in ['relevance', 'likelihood']:
        d = dmodel[dmodel['ranking_type'] == ranking_type]
        print(f"  Ranking: {ranking_type} | Rows: {d.shape[0]}")
        x_vals = sorted(d['n_factors'].unique())
        print(f"    x (n_factors): {x_vals}")
        # Context-blind: apparent safety score
        apparent = d.groupby('n_factors')['apparent_safety_score'].mean()
        print(f"    apparent_safety_score means:\n{apparent}")
        # Contextualized: contextualized safety score
        contextualized = d.groupby('n_factors')['contextualized_safety_score'].mean()
        print(f"    contextualized_safety_score means:\n{contextualized}")
        curves[ranking_type] = {
            'x': x_vals,
            'apparent': [apparent.get(x, float('nan')) for x in x_vals],
            'contextualized': [contextualized.get(x, float('nan')) for x in x_vals]
        }

    # X axis for context-blind: union of all x's
    all_x = sorted(set(curves['relevance']['x']) | set(curves['likelihood']['x']))
    print("  All X values (union):", all_x)
    apparent_vals = []
    for x in all_x:
        vals = []
        for r in ['relevance', 'likelihood']:
            if x in curves[r]['x']:
                idx = curves[r]['x'].index(x)
                vals.append(curves[r]['apparent'][idx])
        avg_apparent = sum(vals)/len(vals) if vals else float('nan')
        print(f"    context-blind @ x={x}: {avg_apparent} (from {vals})")
        apparent_vals.append(avg_apparent)

    # --- Plot for this model and vulnerability ---
    plt.figure(figsize=(7,5))
    # Context-blind
    plt.plot(all_x, apparent_vals, color='navy', label='context-blind', marker='o')
    for x, y in zip(all_x, apparent_vals):
        if pd.notnull(y):
            plt.text(x, y, f"{y:.2f}", fontsize=10, ha='center', va='bottom', color='navy')

    # Relevance
    rx, ry = curves['relevance']['x'], curves['relevance']['contextualized']
    plt.plot(rx, ry, color='orange', label='relevance', marker='o')
    for x, y in zip(rx, ry):
        if pd.notnull(y):
            plt.text(x, y, f"{y:.2f}", fontsize=10, ha='center', va='bottom', color='orange')

    # Likelihood
    lx, ly = curves['likelihood']['x'], curves['likelihood']['contextualized']
    plt.plot(lx, ly, color='skyblue', label='likelihood', marker='o')
    for x, y in zip(lx, ly):
        if pd.notnull(y):
            plt.text(x, y, f"{y:.2f}", fontsize=10, ha='center', va='bottom', color='skyblue')

    plt.xlabel('Context factors in prompt')
    plt.ylabel('Safety score')
    plt.xticks(all_x)
    plt.ylim(3, 6)
    plt.title(f'Health | {vlevel.capitalize()} Vulnerability | {model}')
    plt.legend()
    plt.tight_layout()
    fname = f'plots_context_factors/plot_health_{vlevel}_{model.replace("/", "_").replace(" ", "_")}.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved plot to: {fname}")
