import os, json, time, logging
import pandas as pd
from typing import Dict, List, Tuple, Optional
from openai import OpenAI

from src.utils.constants import (
    FACTORS, ID_TO_TOPIC_THEME, OUTPUT_COLUMNS, VARIANT_BUILDER_PROMPT_L3, 
    VARIANT_BUILDER_PROMPT_L5,
)
from src.utils.profile_utils import load_profiles_from_raw

# ---------------- Paths ----------------
DATA_DIR  = "eval_dataset"
PLAIN_PATH = os.path.join(DATA_DIR, "eval_dataset.csv")
REL_PATH   = os.path.join(DATA_DIR, "relevance_rankings.csv")
LIK_PATH   = os.path.join(DATA_DIR, "likelihood_rankings.csv")
AI_PATH    = os.path.join(DATA_DIR, "ai_rankings.csv")
PROF_XLSX  = os.path.join(DATA_DIR, "vulnerability_profiles_raw.xlsx")
OUT_PATH   = os.path.join(DATA_DIR, "evaluation_prompts.csv")

# clause cache
CLAUSE_CSV = os.path.join(DATA_DIR, "clause_cache.csv")

# ------- Auth / client -------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("Missing OPENAI_API_KEY.")
client = OpenAI(api_key=OPENAI_API_KEY)
VARIANT_MODEL = "gpt-4o-mini"

# ------- Logging -------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# --- Usage tracking for variant calls ---
VARIANT_USAGE_CSV = os.path.join(DATA_DIR, "variant_usage.csv")
VARIANT_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "successful_api_calls": 0,
}

def accumulate_variant_usage(resp) -> None:
    try:
        u = getattr(resp, "usage", None) or {}
        pt = int(getattr(u, "prompt_tokens", 0) or u.get("prompt_tokens", 0) or 0)
        ct = int(getattr(u, "completion_tokens", 0) or u.get("completion_tokens", 0) or 0)
        tt = int(getattr(u, "total_tokens", 0) or u.get("total_tokens", 0) or (pt + ct))
        VARIANT_USAGE["prompt_tokens"] += pt
        VARIANT_USAGE["completion_tokens"] += ct
        VARIANT_USAGE["total_tokens"] += tt
        VARIANT_USAGE["successful_api_calls"] += 1
    except Exception:
        pass

def append_variant_usage_row(extra: dict):
    import pandas as pd
    from datetime import datetime
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": VARIANT_MODEL,
        "successful_api_calls": VARIANT_USAGE["successful_api_calls"],
        "prompt_tokens": VARIANT_USAGE["prompt_tokens"],
        "completion_tokens": VARIANT_USAGE["completion_tokens"],
        "total_tokens": VARIANT_USAGE["total_tokens"],
    }
    row.update(extra or {})
    df = pd.DataFrame([row])
    if not os.path.exists(VARIANT_USAGE_CSV):
        df.to_csv(VARIANT_USAGE_CSV, index=False)
    else:
        df.to_csv(VARIANT_USAGE_CSV, mode="a", index=False, header=False)


# ------- Globals resolved via shared loader -------
LEVEL_COL: Optional[str] = None
FACTOR_COL_MAP: Dict[str, str] = {}

def sanitize_value(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()

def profile_lookup(prof: dict, factor: str) -> str:
    col = FACTOR_COL_MAP.get(factor, factor)
    if col in prof and pd.notna(prof[col]):
        return sanitize_value(prof[col])
    return ""

def join_clauses(clauses: List[str]) -> str:
    if not clauses:
        return ""
    if len(clauses) == 1:
        return f"{clauses[0]}."
    return (", ".join(clauses[:-1]) + f", and {clauses[-1]}.")

# --------- Stage 1:CLAUSE LOOKUP from cache CSV ----------
if not os.path.exists(CLAUSE_CSV):
    raise FileNotFoundError(
        f"Missing clause cache at {CLAUSE_CSV}. Run generate_clauses.py first."
    )
_clause_df = pd.read_csv(CLAUSE_CSV)
CLAUSE_LOOKUP = {(row["factor"], row["value"]): row["clause"] for _, row in _clause_df.iterrows()}

def get_clause(factor, value):
    # Robust fallback if a pair is missing from the cache
    return CLAUSE_LOOKUP.get((factor, value), f"I have {factor.lower()} = {value}")

# ---------------- Stage 2: build level3/level5 variants ----------------
def parse_variant_response(raw: str, expect_key: str) -> List[Dict[str, object]]:
    """
    Accepts:
      - {"level3":[{"clauses":[...],"sentence":"..."}, ...]}
      - {"level5":[{"clauses":[...],"sentence":"..."}, ...]}
      - Backward-compat: lists of lists [["...","...",...], ...]
    Returns: [{"clauses":[...], "sentence":"..."}, ...]
    """
    data = json.loads(raw)
    out = data.get(expect_key, data) if isinstance(data, dict) else data
    if not isinstance(out, list):
        raise ValueError("Invalid JSON shape for variants")

    norm: List[Dict[str, object]] = []
    for item in out:
        if isinstance(item, dict) and "clauses" in item:
            clauses = item.get("clauses", [])
            sent = item.get("sentence", "")
            if isinstance(clauses, list) and all(isinstance(x, str) for x in clauses):
                norm.append({"clauses": clauses, "sentence": str(sent or "").strip()})
        elif isinstance(item, list) and all(isinstance(x, str) for x in item):
            norm.append({"clauses": item, "sentence": ""})
    return norm

def build_level3_variants_with_llm(
    theme: str,
    ranking_key: str,
    top5: List[str],
    factor_to_clause: Dict[str, str]
) -> Dict[str, List[Dict[str, object]]]:
    clauses_json = json.dumps({f: factor_to_clause[f] for f in top5}, ensure_ascii=False)
    prompt = VARIANT_BUILDER_PROMPT_L3.format(
        top5=json.dumps(top5),
        clauses_json=clauses_json
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=VARIANT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )
            accumulate_variant_usage(resp)
            raw = (resp.choices[0].message.content or "").strip()
            return {"level3": parse_variant_response(raw, expect_key="level3")}
        except Exception as e:
            logging.warning(f"[{VARIANT_MODEL}] L3 variant builder error (attempt {attempt+1}/3): {e}")
            time.sleep(1.5 * (attempt + 1))
    # Hard fallback for L3 (dict objects; empty sentence -> we will join later)
    c = [factor_to_clause[f] for f in top5]
    fallback = [
        {"clauses": c[:3], "sentence": ""},
        {"clauses": [c[0], c[1], c[3]], "sentence": ""},
        {"clauses": [c[0], c[2], c[4]], "sentence": ""},
        {"clauses": [c[1], c[3], c[4]], "sentence": ""},
        {"clauses": [c[2], c[3], c[4]], "sentence": ""},
    ]
    return {"level3": fallback}

def build_level5_variants_with_llm(
    theme: str,
    ranking_key: str,
    top5: List[str],
    factor_to_clause: Dict[str, str]
) -> Dict[str, List[Dict[str, object]]]:
    clauses_json = json.dumps({f: factor_to_clause[f] for f in top5}, ensure_ascii=False)
    prompt = VARIANT_BUILDER_PROMPT_L5.format(
        top5=json.dumps(top5),
        clauses_json=clauses_json
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=VARIANT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )
            accumulate_variant_usage(resp)
            raw = (resp.choices[0].message.content or "").strip()
            return {"level5": parse_variant_response(raw, expect_key="level5")}
        except Exception as e:
            logging.warning(f"[{VARIANT_MODEL}] L5 variant builder error (attempt {attempt+1}/3): {e}")
            time.sleep(1.5 * (attempt + 1))
    # Hard fallback for L5 (dict objects; empty sentence -> we will join later)
    c = [factor_to_clause[f] for f in top5]
    fallback = [
        {"clauses": c, "sentence": ""},
        {"clauses": c[::-1], "sentence": ""},
        {"clauses": [c[1], c[3], c[0], c[4], c[2]], "sentence": ""},
        {"clauses": [c[2], c[0], c[4], c[1], c[3]], "sentence": ""},
        {"clauses": [c[3], c[4], c[1], c[2], c[0]], "sentence": ""},
    ]
    return {"level5": fallback}

def dedupe_variant_objects(
    variant_items: List[Dict[str, object]],
    need: int,
    k: int,
    pool: List[str],
) -> List[Dict[str, object]]:
    """
    Deduplicate by clause tuple, keep LLM-provided sentence when available.
    If fewer than `need`, fill with permutations from `pool` (no sentence).
    """
    from itertools import permutations

    # Map original items by their clause tuple to preserve LLM sentences
    orig_map: Dict[Tuple[str, ...], str] = {}
    for it in variant_items:
        clauses = it.get("clauses", [])
        sent = it.get("sentence", "")
        if isinstance(clauses, list) and len(clauses) == k and all(isinstance(x, str) for x in clauses):
            t = tuple(clauses)
            if t not in orig_map:
                orig_map[t] = str(sent or "").strip()

    uniq: List[Dict[str, object]] = []
    seen: set = set()

    # Keep unique items in given order
    for t, sent in orig_map.items():
        if t not in seen:
            uniq.append({"clauses": list(t), "sentence": sent})
            seen.add(t)
            if len(uniq) >= need:
                return uniq[:need]

    # Fill with permutations if needed (no sentence -> will fallback to join_clauses)
    for cand in permutations(pool, k):
        if len(uniq) >= need:
            break
        if cand not in seen:
            uniq.append({"clauses": list(cand), "sentence": ""})
            seen.add(cand)

    return uniq[:need]

# ---------------- Rankings loaders ----------------
def load_single_ranking_map(df: pd.DataFrame, kind: str) -> Dict[str, Dict]:
    out = {}
    needed = {"id","topic","theme","1","2","3","4","5"}
    if df is None or df.empty or not set(needed).issubset(df.columns):
        raise ValueError(f"{kind} rankings CSV is missing or malformed. Need columns {sorted(needed)}.")
    for _, r in df.iterrows():
        out[str(r["theme"])] = {
            "id": r["id"], "topic": r["topic"], "theme": r["theme"],
            "1": r["1"], "2": r["2"], "3": r["3"], "4": r["4"], "5": r["5"],
        }
    return out

# ---------------- Main ----------------
def main():
    # Load core data
    plain = pd.read_csv(PLAIN_PATH)
    ai_df  = pd.read_csv(AI_PATH)
    rel_df = pd.read_csv(REL_PATH)
    lik_df = pd.read_csv(LIK_PATH)

    # Use the SAME loader/mapping as in generate_clauses.py
    profiles_df, level_col, factor_col_map = load_profiles_from_raw(
        PROF_XLSX, FACTORS, ID_TO_TOPIC_THEME
    )
    global LEVEL_COL, FACTOR_COL_MAP
    LEVEL_COL = level_col
    FACTOR_COL_MAP = factor_col_map

    # Build ranking maps
    rel_map = load_single_ranking_map(rel_df, "Relevance")
    lik_map = load_single_ranking_map(lik_df, "Likelihood")

    # AI rankings
    ai_by_model: Dict[str, Dict[str, Dict]] = {}
    models = ai_df["model"].dropna().unique().tolist()
    for m in models:
        ai_by_model[m] = {}
        for _, r in ai_df[ai_df["model"] == m].iterrows():
            ai_by_model[m][str(r["theme"])] = {
                "id": r["id"], "topic": r["topic"], "theme": r["theme"],
                "1": r["1"], "2": r["2"], "3": r["3"], "4": r["4"], "5": r["5"],
            }

    # Group profiles by theme
    profs_by_theme: Dict[str, List[Dict]] = {}
    for _, r in profiles_df.iterrows():
        profs_by_theme.setdefault(r["theme"], []).append(r.to_dict())

    rows = []
    VARIANT_CACHE: Dict[Tuple[str, str, str, str], Dict[str, List[Dict[str, object]]]] = {}
    # key = ( "L3"/"L5", theme, rank_key, prof_uid )
    # value = {"level3":[{...}], "level5":[{...}]} but in our use we store one level per key

    # ---- Level 0 
    for _, base in plain.iterrows():
        request_id, topic, theme, question = base["id"], base["topic"], base["theme"], base["question"]
        theme_profiles = profs_by_theme.get(theme, [])
        if not theme_profiles:
            logging.warning(f"No profiles found for theme '{theme}'.")
            continue
        for prof in theme_profiles:
            level_val = str(prof.get(LEVEL_COL, "")).strip().lower()
            prof_uid  = prof.get("profile_id") or prof.get("ID")
            vprof = {k: v for k, v in prof.items() if k not in {"ID", "vulnerability"}}
            prof_json = json.dumps(vprof, ensure_ascii=False)
            pid0 = f"{request_id}_none_{level_val}_{prof_uid}_0_v0"
            rows.append({
                "prompt_id": pid0,
                "request_id": request_id,
                "topic": topic,
                "theme": theme,
                "vulnerability_profile_level": level_val,
                "vulnerability_profile": prof_json,
                "ranking_type": "None",
                "ai_model": "",
                "context_level": 0,
                "context_variant": 0,
                "context_factors_used": "[]",
                "final_prompt": question.strip(),
                "deduped_context": "",
            })

    # ---- Contexted prompts
    for _, base in plain.iterrows():
        request_id, topic, theme, question = base["id"], base["topic"], base["theme"], base["question"]

        # common rankings for this theme (Relevance & Likelihood)
        if theme not in rel_map or theme not in lik_map:
            logging.warning(f"Missing Relevance/Likelihood ranking for theme='{theme}'. Skipping those.")
            common_rankings: List[Tuple[str, Dict]] = []
        else:
            common_rankings = [("Relevance", rel_map[theme]), ("Likelihood", lik_map[theme])]

        theme_profiles = profs_by_theme.get(theme, [])
        if not theme_profiles:
            logging.warning(f"No profiles found for theme '{theme}'.")
            continue

        # --- Relevance & Likelihood 
        for kind, rk in common_rankings:
            top5 = [rk["1"], rk["2"], rk["3"], rk["4"], rk["5"]]
            for prof in theme_profiles:
                level_val = str(prof.get(LEVEL_COL, "")).strip().lower()
                prof_uid  = prof.get("profile_id") or prof.get("ID")
                vprof = {k: v for k, v in prof.items() if k not in {"ID", "vulnerability"}}
                prof_json = json.dumps(vprof, ensure_ascii=False)

                # Build clauses via cache
                factor_to_clause = {f: get_clause(f, profile_lookup(prof, f)) for f in top5}

                # L1
                l1_clause = factor_to_clause[top5[0]].rstrip(".")
                final_prompt = f"{l1_clause}. {question}".strip()
                pid = f"{request_id}_{kind.lower()}_{level_val}_{prof_uid}_1_v0"
                rows.append({
                    "prompt_id": pid,
                    "request_id": request_id,
                    "topic": topic,
                    "theme": theme,
                    "vulnerability_profile_level": level_val,
                    "vulnerability_profile": prof_json,
                    "ranking_type": kind,
                    "ai_model": "",
                    "context_level": 1,
                    "context_variant": 0,
                    "context_factors_used": json.dumps([top5[0]], ensure_ascii=False),
                    "final_prompt": final_prompt,
                    "deduped_context": "",
                })

                # L3/L5 
                rank_key = f"{kind}:NA:{tuple(top5)}"

                # L3
                cache_key_l3 = ("L3", theme, rank_key, prof_uid)
                if cache_key_l3 not in VARIANT_CACHE:
                    VARIANT_CACHE[cache_key_l3] = build_level3_variants_with_llm(theme, rank_key, top5, factor_to_clause)
                varset_l3 = VARIANT_CACHE[cache_key_l3]

                pool = [factor_to_clause[f] for f in top5]
                orig_l3_map = {tuple(item["clauses"]): item.get("sentence","") for item in varset_l3.get("level3", [])}
                lvl3_items = dedupe_variant_objects(varset_l3.get("level3", []), need=5, k=3, pool=pool)

                for vi, item in enumerate(lvl3_items, start=1):
                    t = tuple(item["clauses"])
                    deduped_flag = "llm" if t in orig_l3_map else "deduped"
                    sentence = item["sentence"] or join_clauses([c.rstrip(".") for c in item["clauses"]])
                    pid = f"{request_id}_{kind.lower()}_{level_val}_{prof_uid}_3_v{vi}"
                    rows.append({
                        "prompt_id": pid,
                        "request_id": request_id,
                        "topic": topic,
                        "theme": theme,
                        "vulnerability_profile_level": level_val,
                        "vulnerability_profile": prof_json,
                        "ranking_type": kind,
                        "ai_model": "",
                        "context_level": 3,
                        "context_variant": vi,
                        "context_factors_used": json.dumps(top5, ensure_ascii=False),
                        "final_prompt": f"{sentence} {question}".strip(),
                        "deduped_context": deduped_flag,
                    })

                cache_key_l5 = ("L5", theme, rank_key, prof_uid)
                if cache_key_l5 not in VARIANT_CACHE:
                    VARIANT_CACHE[cache_key_l5] = build_level5_variants_with_llm(theme, rank_key, top5, factor_to_clause)
                varset_l5 = VARIANT_CACHE[cache_key_l5]

                orig_l5_map = {tuple(item["clauses"]): item.get("sentence","") for item in varset_l5.get("level5", [])}
                lvl5_items = dedupe_variant_objects(varset_l5.get("level5", []), need=5, k=5, pool=pool)

                for vi, item in enumerate(lvl5_items, start=1):
                    t = tuple(item["clauses"])
                    deduped_flag = "llm" if t in orig_l5_map else "deduped"
                    sentence = item["sentence"] or join_clauses([c.rstrip(".") for c in item["clauses"]])
                    pid = f"{request_id}_{kind.lower()}_{level_val}_{prof_uid}_5_v{vi}"
                    rows.append({
                        "prompt_id": pid,
                        "request_id": request_id,
                        "topic": topic,
                        "theme": theme,
                        "vulnerability_profile_level": level_val,
                        "vulnerability_profile": prof_json,
                        "ranking_type": kind,
                        "ai_model": "",
                        "context_level": 5,
                        "context_variant": vi,
                        "context_factors_used": json.dumps(top5, ensure_ascii=False),
                        "final_prompt": f"{sentence} {question}".strip(),
                        "deduped_context": deduped_flag,
                    })

        # --- AI (per model)
        for model, theme_map in ai_by_model.items():
            if theme not in theme_map:
                logging.warning(f"Missing AI ranking for theme='{theme}' (model={model}). Skipping.")
                continue
            rk = theme_map[theme]
            top5 = [rk["1"], rk["2"], rk["3"], rk["4"], rk["5"]]

            for prof in theme_profiles:
                level_val = str(prof.get(LEVEL_COL, "")).strip().lower()
                prof_uid  = prof.get("profile_id") or prof.get("ID")
                vprof = {k: v for k, v in prof.items() if k not in {"ID", "vulnerability"}}
                prof_json = json.dumps(vprof, ensure_ascii=False)

                factor_to_clause = {f: get_clause(f, profile_lookup(prof, f)) for f in top5}

                # L1
                l1_clause = factor_to_clause[top5[0]].rstrip(".")
                pid = f"{request_id}_ai-{model}_{level_val}_{prof_uid}_1_v0"
                rows.append({
                    "prompt_id": pid,
                    "request_id": request_id,
                    "topic": topic,
                    "theme": theme,
                    "vulnerability_profile_level": level_val,
                    "vulnerability_profile": prof_json,
                    "ranking_type": "AI",
                    "ai_model": model,
                    "context_level": 1,
                    "context_variant": 0,
                    "context_factors_used": json.dumps([top5[0]], ensure_ascii=False),
                    "final_prompt": f"{l1_clause}. {question}".strip(),
                    "deduped_context": "",
                })

                # L3/L5
                rank_key = f"AI:{model}:{tuple(top5)}"

                # L3
                cache_key_l3 = ("L3", theme, rank_key, prof_uid)
                if cache_key_l3 not in VARIANT_CACHE:
                    VARIANT_CACHE[cache_key_l3] = build_level3_variants_with_llm(theme, rank_key, top5, factor_to_clause)
                varset_l3 = VARIANT_CACHE[cache_key_l3]

                pool = [factor_to_clause[f] for f in top5]
                orig_l3_map = {tuple(item["clauses"]): item.get("sentence","") for item in varset_l3.get("level3", [])}
                lvl3_items = dedupe_variant_objects(varset_l3.get("level3", []), need=5, k=3, pool=pool)
                for vi, item in enumerate(lvl3_items, start=1):
                    t = tuple(item["clauses"])
                    deduped_flag = "llm" if t in orig_l3_map else "deduped"
                    sentence = item["sentence"] or join_clauses([c.rstrip(".") for c in item["clauses"]])
                    pid = f"{request_id}_ai-{model}_{level_val}_{prof_uid}_3_v{vi}"
                    rows.append({
                        "prompt_id": pid,
                        "request_id": request_id,
                        "topic": topic,
                        "theme": theme,
                        "vulnerability_profile_level": level_val,
                        "vulnerability_profile": prof_json,
                        "ranking_type": "AI",
                        "ai_model": model,
                        "context_level": 3,
                        "context_variant": vi,
                        "context_factors_used": json.dumps(top5, ensure_ascii=False),
                        "final_prompt": f"{sentence} {question}".strip(),
                        "deduped_context": deduped_flag,
                    })

                # L5
                cache_key_l5 = ("L5", theme, rank_key, prof_uid)
                if cache_key_l5 not in VARIANT_CACHE:
                    VARIANT_CACHE[cache_key_l5] = build_level5_variants_with_llm(theme, rank_key, top5, factor_to_clause)
                varset_l5 = VARIANT_CACHE[cache_key_l5]

                orig_l5_map = {tuple(item["clauses"]): item.get("sentence","") for item in varset_l5.get("level5", [])}
                lvl5_items = dedupe_variant_objects(varset_l5.get("level5", []), need=5, k=5, pool=pool)
                for vi, item in enumerate(lvl5_items, start=1):
                    t = tuple(item["clauses"])
                    deduped_flag = "llm" if t in orig_l5_map else "deduped"
                    sentence = item["sentence"] or join_clauses([c.rstrip(".") for c in item["clauses"]])
                    pid = f"{request_id}_ai-{model}_{level_val}_{prof_uid}_5_v{vi}"
                    rows.append({
                        "prompt_id": pid,
                        "request_id": request_id,
                        "topic": topic,
                        "theme": theme,
                        "vulnerability_profile_level": level_val,
                        "vulnerability_profile": prof_json,
                        "ranking_type": "AI",
                        "ai_model": model,
                        "context_level": 5,
                        "context_variant": vi,
                        "context_factors_used": json.dumps(top5, ensure_ascii=False),
                        "final_prompt": f"{sentence} {question}".strip(),
                        "deduped_context": deduped_flag,
                    })

    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    # Sanity logs
    n_questions = len(plain)
    n_models = len(models)
    logging.info(f"Themes: {plain['theme'].nunique()} | Questions: {n_questions} | Models: {n_models}")
    logging.info(f"Profiles total: {sum(len(v) for v in (profs_by_theme.values()))}")
    logging.info(f"Rows built: {len(out_df)}  (expected ≈35,712 if inputs complete)")

    out_df.to_csv(OUT_PATH, index=False)
    logging.info(f"Wrote {len(out_df)} prompts to {OUT_PATH}")

    # Add usage summary 
    extra = {
        "themes": plain["theme"].nunique(),
        "questions": len(plain),
        "models": len(models),
        "variant_sets_built": len(VARIANT_CACHE),
    }
    append_variant_usage_row(extra)
    logging.info(f"Variant token usage saved to {VARIANT_USAGE_CSV}: {VARIANT_USAGE} | extra={extra}")

if __name__ == "__main__":
    main()
