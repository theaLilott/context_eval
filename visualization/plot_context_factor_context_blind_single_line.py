import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import ast

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

for topic, prefix in [('finance', 'f'), ('health', 'h')]:
    print(f"\n=== Processing topic: {topic} ===")
    prompts = pd.read_csv('eval_dataset/evaluation_prompts.csv')
    llm_responses = pd.concat([pd.read_csv(f) for f in glob.glob(f'output/llm_response_{prefix}*.csv')])
    safety_scores = pd.concat([pd.read_csv(f) for f in glob.glob(f'scores/safety_scores_{prefix}*.csv')])

    df = llm_responses.merge(prompts, on='prompt_id', suffixes=('', '_prompt'))
    df = df.merge(safety_scores, left_on=['prompt_id', 'used_llm'], right_on=['prompt_id', 'used_llm'], suffixes=('', '_score'))

    for col in ['used_llm', 'ranking_type', 'vulnerability_profile_level', 'topic']:
        df[col] = df[col].astype(str).str.lower().str.strip()

    df = df[df['topic'] == topic]
    df['n_factors'] = df['context_factors_used'].apply(count_factors)

    vuln_levels = ['low', 'medium', 'high']
    os.makedirs(f"plots_context_factors/{topic}", exist_ok=True)

    all_models = sorted([m for m in df['used_llm'].unique() if m != "all"])

    for model in all_models:
        for vlevel in vuln_levels:
            dmodel = df[
                ((df['used_llm'] == model) | (df['used_llm'] == 'all')) &
                (df['vulnerability_profile_level'] == vlevel)
            ]
            if dmodel.empty:
                continue

            nrows_used = dmodel.shape[0]
            print(f"Model: {model:<22} | Vuln: {vlevel:<6} | Used rows: {nrows_used:>4}")

            d_baseline = dmodel[dmodel['ranking_type'] == 'baseline']
            d_relevance = dmodel[dmodel['ranking_type'] == 'relevance']
            d_likelihood = dmodel[dmodel['ranking_type'] == 'likelihood']

            n_factors_all = sorted(set(dmodel['n_factors'].unique()))

            # --- Calculate context-blind line ---
            context_blind_x = []
            context_blind_y = []
            for n in n_factors_all:
                if n == 0:
                    val = d_baseline[d_baseline['n_factors'] == 0]['apparent_safety_score'].mean()
                    context_blind_x.append(0)
                    context_blind_y.append(val)
                else:
                    vals = []
                    for df_rank in [d_relevance, d_likelihood]:
                        v = df_rank[df_rank['n_factors'] == n]['apparent_safety_score'].mean()
                        if not pd.isnull(v):
                            vals.append(v)
                    if vals:
                        avg_val = sum(vals) / len(vals)
                        context_blind_x.append(n)
                        context_blind_y.append(avg_val)

            # --- Contextualized curves ---
            rel_ctx = d_relevance.groupby('n_factors')['contextualized_safety_score'].mean()
            lik_ctx = d_likelihood.groupby('n_factors')['contextualized_safety_score'].mean()

            plt.figure(figsize=(7,5))

            # --- Plot context-blind (navy) ---
            plt.plot(context_blind_x, context_blind_y, color='navy', marker='o', label='context-blind (apparent)')
            for x, y in zip(context_blind_x, context_blind_y):
                if pd.notnull(y):
                    plt.text(x, y, f"{y:.2f}", fontsize=10, ha='center', va='bottom', color='navy')

            # --- Plot contextualized - relevance (orange) ---
            rx, ry = rel_ctx.index.values, rel_ctx.values
            plt.plot(rx, ry, color='orange', marker='o', label='relevance (contextualized)')
            for x, y in zip(rx, ry):
                if pd.notnull(y):
                    plt.text(x, y, f"{y:.2f}", fontsize=10, ha='center', va='bottom', color='orange')

            # --- Plot contextualized - likelihood (skyblue) ---
            lx, ly = lik_ctx.index.values, lik_ctx.values
            plt.plot(lx, ly, color='skyblue', marker='o', label='likelihood (contextualized)')
            for x, y in zip(lx, ly):
                if pd.notnull(y):
                    plt.text(x, y, f"{y:.2f}", fontsize=10, ha='center', va='bottom', color='skyblue')

            # --- Plot baseline contextualized (gray square at n=0) ---
            b0 = d_baseline[d_baseline['n_factors'] == 0]['contextualized_safety_score'].mean()
            if not pd.isnull(b0):
                plt.plot([0], [b0], color='gray', marker='s', markersize=9, linestyle='None', label='baseline (contextualized)')
                plt.text(0, b0, f"{b0:.2f}", fontsize=10, ha='center', va='top', color='gray')

            plt.xlabel('Context factors in prompt')
            plt.ylabel('Safety score')
            plt.xticks(n_factors_all)
            plt.ylim(2, 6)
            plt.title(f'{topic.capitalize()} | {vlevel.capitalize()} Vulnerability | {model}')
            plt.legend(fontsize=9, loc='best')
            plt.tight_layout()
            fname = f'plots_context_factors/{topic}/plot_{topic}_{vlevel}_{model.replace("/", "_").replace(" ", "_")}_contextblind.png'
            plt.savefig(fname, dpi=150)
            plt.close()
            print(f"  Saved plot to: {fname}")
