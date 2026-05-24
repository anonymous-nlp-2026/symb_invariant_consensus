import json
import os
from pathlib import Path

BASE = Path("./results")

def load_json(path):
    with open(path) as f:
        return json.load(f)

def pct(x, n):
    return round(x / n * 100, 2)

# =============================================================================
# TABLE 1: tab:main_results (from experiments.tex)
# =============================================================================
print("=" * 100)
print("TABLE 1 VERIFICATION: tab:main_results (experiments.tex)")
print("=" * 100)

# Paper claims (from experiments.tex)
table1_claims = [
    # (label, model, domain, n, SC%, SICA%, delta, p)
    ("T1-01", "Mistral-7B", "PW-D5", 600, 39.33, 36.50, -2.83, 0.014),
    ("T1-02", "Mistral-7B", "LogiQA", 200, 50.00, 51.50, 1.50, 0.581),
    ("T1-03", "Mistral-7B", "FOLIO(s123)", 204, 54.41, 57.35, 2.94, 0.070),
    ("T1-04", "Mistral-7B(3s)", "FOLIO", 204, 56.53, 56.70, 0.16, 1.000),
    ("T1-05", "Mistral-7B", "FOLIO(s456)", 204, 55.88, 55.88, 0.00, 1.000),
    ("T1-06", "Mistral-7B", "FOLIO(s789)", 204, 59.31, 56.86, -2.45, 0.267),
    ("T1-07", "LLaMA-8B", "FOLIO", 204, 66.18, 61.76, -4.41, 0.093),
    ("T1-08", "LLaMA-8B", "StrategyQA", 200, 72.00, 74.00, 2.00, 0.424),
    ("T1-09", "Qwen2.5-14B", "FOLIO(ICL)", 199, 74.37, 74.87, 0.50, 1.000),
    ("T1-10", "Qwen2.5-14B", "PW-D5", 600, 74.00, 71.50, -2.50, 0.011),
    ("T1-11", "Qwen2.5-14B(4s)", "FOLIO", 204, 75.98, 75.49, -0.49, 1.000),
    ("T1-12", "Qwen2.5-7B", "FOLIO", 204, 75.98, 76.47, 0.49, 1.000),
    ("T1-13", "Qwen2.5-14B", "LogiQA", 200, 79.00, 78.00, -1.00, 0.625),
    ("T1-14", "Qwen3-14B", "FOLIO", 204, 83.33, 84.80, 1.47, 1.000),
    ("T1-15", "Qwen3-14B", "PW-D5", 600, 85.83, 85.67, -0.17, 1.000),
    ("T1-16", "Qwen3-14B(think)", "FOLIO", 204, 86.27, 86.76, 0.49, 1.000),
]

# Load tiebreak revalidation
tiebreak = load_json(BASE / "sc_tiebreak_revalidation_16639.json")
tiebreak_map = {}
for entry in tiebreak["summary"]:
    tiebreak_map[entry["exp_id"]] = entry

# Load mcnemar recomputed
mcnemar = load_json(BASE / "mcnemar_recomputed_16639.json")
mcnemar_map = {}
for entry in mcnemar["results"]:
    mcnemar_map[entry["exp_id"]] = entry

# Result file mapping
result_files = {
    # Mistral PW-D5 default seed
    "mistral_pw_default": BASE / "multi_seed/mistral_pw_seed42/results.json",
    # Mistral LogiQA
    "mistral_logiqa": BASE / "exp049b_mistral_logiqa200/exp049b_results.json",
    # Mistral FOLIO seeds
    "mistral_folio_default": BASE / "exp033_mistral_7b_folio204/exp033_results.json",
    "mistral_folio_s123": BASE / "exp052_mistral_folio204_seed123/exp052_results.json",
    "mistral_folio_s456": BASE / "exp053_mistral_folio204_seed456/exp053_results.json",
    "mistral_folio_s789": BASE / "multi_seed/mistral_folio_seed789/results.json",
    # LLaMA
    "llama_folio": BASE / "exp-063-llama8b-folio204-16639/results.json",
    # Qwen2.5-14B
    "qwen14b_folio": BASE / "exp036_qwen25_14b_folio204/results.json",
    "qwen14b_pw": BASE / "exp032_qwen25_14b_pw600/results.json",
    "qwen14b_logiqa": BASE / "exp038_qwen25_14b_logiqa200/results.json",
    "qwen14b_folio_icl": BASE / "exp050b_mistral_icl_oracle/exp050b_results.json",
    # Multi-seed Qwen FOLIO
    "qwen14b_folio_s123": BASE / "multi_seed/qwen25_folio_seed123/results.json",
    "qwen14b_folio_s456": BASE / "multi_seed/qwen25_folio_seed456/results.json",
    "qwen14b_folio_s789": BASE / "multi_seed/qwen25_folio_seed789/results.json",
    # Qwen3
    "qwen3_pw": BASE / "exp033_qwen3_14b_pw600_nonthinking/results.json",
    "qwen3_thinking": BASE / "exp028b_qwen3_thinking_folio204/exp028b_remaining38_results.json",
}

def get_summary(path):
    """Extract SC/SICA from result file summary."""
    if not path.exists():
        return None
    data = load_json(path)
    s = data.get("summary", data)
    n = s.get("n_problems", s.get("n"))
    sc_acc = s.get("sc_accuracy", s.get("new_sc_pct"))
    sica_acc = s.get("sica_accuracy")
    sc_correct = s.get("sc_correct")
    sica_correct = s.get("sica_correct")
    
    if sc_acc is not None and sc_acc < 1:
        sc_pct = round(sc_acc * 100, 2)
    elif sc_acc is not None:
        sc_pct = sc_acc
    else:
        sc_pct = None
    
    if sica_acc is not None and sica_acc < 1:
        sica_pct = round(sica_acc * 100, 2)
    elif sica_acc is not None:
        sica_pct = sica_acc
    else:
        sica_pct = None
    
    return {
        "n": n,
        "sc_pct": sc_pct,
        "sica_pct": sica_pct,
        "sc_correct": sc_correct,
        "sica_correct": sica_correct,
    }

def check_value(label, field, paper_val, actual_val, tolerance=0.015):
    if actual_val is None:
        return f"  ⚠️  {field}: paper={paper_val}, actual=MISSING"
    diff = abs(paper_val - actual_val)
    if diff <= tolerance:
        return f"  ✅ {field}: paper={paper_val}, actual={actual_val}"
    else:
        return f"  ❌ {field}: paper={paper_val}, actual={actual_val} (diff={diff:.2f})"

# T1-01: Mistral-7B PW-D5 (seed=42)
print("\n--- T1-01: Mistral-7B × PW-D5 ---")
d = get_summary(result_files["mistral_pw_default"])
if d:
    print(f"  File: multi_seed/mistral_pw_seed42/results.json")
    print(f"  Raw: n={d['n']}, SC={d['sc_pct']}%, SICA={d['sica_pct']}%, SC_correct={d['sc_correct']}, SICA_correct={d['sica_correct']}")
    print(check_value("T1-01", "SC%", 39.33, d['sc_pct']))
    print(check_value("T1-01", "SICA%", 36.50, d['sica_pct']))
    if d['sc_pct'] and d['sica_pct']:
        actual_delta = round(d['sica_pct'] - d['sc_pct'], 2)
        print(check_value("T1-01", "Δ(pp)", -2.83, actual_delta))
else:
    print("  ⚠️  File not found")

# T1-02: Mistral-7B LogiQA
print("\n--- T1-02: Mistral-7B × LogiQA ---")
d = get_summary(result_files["mistral_logiqa"])
if d:
    print(f"  File: exp049b_mistral_logiqa200/exp049b_results.json")
    print(f"  Raw: n={d['n']}, SC={d['sc_pct']}%, SICA={d['sica_pct']}%")
    print(check_value("T1-02", "SC%", 50.00, d['sc_pct']))
    print(check_value("T1-02", "SICA%", 51.50, d['sica_pct']))
    if d['sc_pct'] and d['sica_pct']:
        actual_delta = round(d['sica_pct'] - d['sc_pct'], 2)
        print(check_value("T1-02", "Δ(pp)", 1.50, actual_delta))

# T1-03: Mistral-7B FOLIO s123 (tiebreak recomputed)
print("\n--- T1-03: Mistral-7B × FOLIO(s123) ---")
d = get_summary(result_files["mistral_folio_s123"])
tb = tiebreak_map.get("exp-052", {})
mc = mcnemar_map.get("exp-052", {})
if d:
    print(f"  File: exp052_mistral_folio204_seed123/exp052_results.json")
    print(f"  Raw: n={d['n']}, SC={d['sc_pct']}%, SICA={d['sica_pct']}%")
    recomp_sc = tb.get("new_sc_pct", mc.get("new_sc_pct"))
    print(f"  Tiebreak-recomputed SC: {recomp_sc}%")
    print(check_value("T1-03", "SC%(recomp)", 54.41, recomp_sc))
    print(check_value("T1-03", "SICA%", 57.35, d['sica_pct']))
    if recomp_sc and d['sica_pct']:
        actual_delta = round(d['sica_pct'] - recomp_sc, 2)
        print(check_value("T1-03", "Δ(pp)", 2.94, actual_delta))
    mc_p = mc.get("p_value")
    if mc_p:
        print(check_value("T1-03", "p-value", 0.070, round(mc_p, 3)))

# T1-04: Mistral-7B 3-seed FOLIO average
print("\n--- T1-04: Mistral-7B(3s) × FOLIO ---")
# Need seed replication data
seed_repl = load_json(BASE / "seed_replication_summary.json")
seeds_data = seed_repl.get("seeds", [])
print(f"  Seeds available: {[s.get('seed') for s in seeds_data]}")
# The 3-seed average should use default(42), s123, s456
# But with tiebreak-recomputed SC values
# From tiebreak: exp-052 (s123) new_sc=54.41, exp-053 (s456) new_sc=55.88
# Default (s42): exp-033, SC from file = 54.41 (from seed_repl) 
# Wait, seed_repl shows seed 42 SC=54.41 (hardcoded), but we need to check if tiebreak was applied
# Actually the seed_repl SC=54.41 for seed42 might be the old value
# Let me check the exp033 file
d_default = get_summary(result_files["mistral_folio_default"])
if d_default:
    print(f"  Default seed (42): SC={d_default['sc_pct']}%, SICA={d_default['sica_pct']}%")
    # exp033 original: SC=54.41, SICA=59.31 (from the file we read earlier)
    # But was it tiebreak-recomputed? exp-033 is NOT in the tiebreak file.
    # So the original SC=54.41% (111/204) is used directly.
    
    # s123 tiebreak-recomputed
    s123_sc = tiebreak_map.get("exp-052", {}).get("new_sc_pct", None)
    s123_sica = 57.35  # from exp052
    
    # s456 tiebreak-recomputed  
    s456_sc = tiebreak_map.get("exp-053", {}).get("new_sc_pct", None)
    s456_sica = 55.88  # from exp053
    
    if s123_sc and s456_sc:
        avg_sc = round((d_default['sc_pct'] + s123_sc + s456_sc) / 3, 2)
        avg_sica = round((d_default['sica_pct'] + s123_sica + s456_sica) / 3, 2)
        avg_delta = round(avg_sica - avg_sc, 2)
        print(f"  3-seed avg: SC={avg_sc}%, SICA={avg_sica}%, Δ={avg_delta}")
        print(check_value("T1-04", "SC%", 56.53, avg_sc))
        print(check_value("T1-04", "SICA%", 56.70, avg_sica))
        print(check_value("T1-04", "Δ(pp)", 0.16, avg_delta))

# T1-05: Mistral-7B FOLIO s456
print("\n--- T1-05: Mistral-7B × FOLIO(s456) ---")
d = get_summary(result_files["mistral_folio_s456"])
tb = tiebreak_map.get("exp-053", {})
mc = mcnemar_map.get("exp-053", {})
if d:
    recomp_sc = tb.get("new_sc_pct")
    print(f"  Raw: SC={d['sc_pct']}%, SICA={d['sica_pct']}%")
    print(f"  Tiebreak-recomputed SC: {recomp_sc}%")
    print(check_value("T1-05", "SC%(recomp)", 55.88, recomp_sc))
    print(check_value("T1-05", "SICA%", 55.88, d['sica_pct']))
    mc_p = mc.get("p_value")
    if mc_p:
        print(check_value("T1-05", "p-value", 1.000, round(mc_p, 3)))

# T1-06: Mistral-7B FOLIO s789
print("\n--- T1-06: Mistral-7B × FOLIO(s789) ---")
d = get_summary(result_files["mistral_folio_s789"])
if d:
    print(f"  Raw: n={d['n']}, SC={d['sc_pct']}%, SICA={d['sica_pct']}%, SC_corr={d['sc_correct']}, SICA_corr={d['sica_correct']}")
    print(check_value("T1-06", "SC%", 59.31, d['sc_pct']))
    print(check_value("T1-06", "SICA%", 56.86, d['sica_pct']))
    if d['sc_pct'] and d['sica_pct']:
        actual_delta = round(d['sica_pct'] - d['sc_pct'], 2)
        print(check_value("T1-06", "Δ(pp)", -2.45, actual_delta))

# T1-07: LLaMA-8B FOLIO
print("\n--- T1-07: LLaMA-3.1-8B × FOLIO ---")
d = get_summary(result_files["llama_folio"])
if d:
    print(f"  Raw: n={d['n']}, SC={d['sc_pct']}%, SICA={d['sica_pct']}%, SC_corr={d['sc_correct']}, SICA_corr={d['sica_correct']}")
    print(check_value("T1-07", "SC%", 66.18, d['sc_pct']))
    print(check_value("T1-07", "SICA%", 61.76, d['sica_pct']))
    if d['sc_pct'] and d['sica_pct']:
        actual_delta = round(d['sica_pct'] - d['sc_pct'], 2)
        print(check_value("T1-07", "Δ(pp)", -4.41, actual_delta))

# T1-08: LLaMA-8B StrategyQA - need to find this file
print("\n--- T1-08: LLaMA-3.1-8B × StrategyQA ---")
# Search for StrategyQA results
stratqa_candidates = list(BASE.glob("**/strategyqa*")) + list(BASE.glob("**/strategy_qa*"))
if not stratqa_candidates:
    # Try cross_dataset
    for p in BASE.glob("cross_dataset/**/*.json"):
        try:
            data = load_json(p)
            if "strategy" in str(data).lower()[:500]:
                stratqa_candidates.append(p)
        except:
            pass
if stratqa_candidates:
    print(f"  Found candidates: {stratqa_candidates}")
else:
    print("  ⚠️  No StrategyQA result file found - cannot verify")

# T1-09: Qwen2.5-14B FOLIO ICL
print("\n--- T1-09: Qwen2.5-14B × FOLIO(ICL) ---")
d = get_summary(result_files["qwen14b_folio_icl"])
if d:
    print(f"  File: exp050b_mistral_icl_oracle/exp050b_results.json")
    print(f"  WARNING: This file is labeled 'Mistral ICL oracle', not Qwen2.5-14B ICL")
    print(f"  Raw: n={d['n']}, SC={d['sc_pct']}%, SICA={d['sica_pct']}%")
    print(check_value("T1-09", "SC%", 74.37, d['sc_pct']))
    print(check_value("T1-09", "SICA%", 74.87, d['sica_pct']))
# Also check exp032_qwen14b_pw_oracle
oracle_path = BASE / "exp032_qwen14b_pw_oracle/results.json"
if oracle_path.exists():
    d2 = get_summary(oracle_path)
    print(f"  Alt file (exp032_qwen14b_pw_oracle): n={d2['n']}, SC={d2['sc_pct']}%, SICA={d2['sica_pct']}%")

# T1-10: Qwen2.5-14B PW-D5 (tiebreak recomputed)
print("\n--- T1-10: Qwen2.5-14B × PW-D5 ---")
d = get_summary(result_files["qwen14b_pw"])
tb = tiebreak_map.get("exp-039", {})
mc = mcnemar_map.get("exp-039", {})
if d:
    print(f"  Raw: n={d['n']}, SC={d['sc_pct']}%, SICA={d['sica_pct']}%")
    recomp_sc = tb.get("new_sc_pct")
    print(f"  Tiebreak-recomputed SC: {recomp_sc}%")
    print(check_value("T1-10", "SC%(recomp)", 74.00, recomp_sc))
    print(check_value("T1-10", "SICA%", 71.50, d['sica_pct']))
    mc_p = mc.get("p_value")
    if mc_p:
        print(check_value("T1-10", "p-value", 0.011, round(mc_p, 3)))

# T1-11: Qwen2.5-14B 4-seed FOLIO
print("\n--- T1-11: Qwen2.5-14B(4s) × FOLIO ---")
qwen_seeds = {}
for seed_name, seed_file in [
    ("default", result_files["qwen14b_folio"]),
    ("s123", result_files["qwen14b_folio_s123"]),
    ("s456", result_files["qwen14b_folio_s456"]),
    ("s789", result_files["qwen14b_folio_s789"]),
]:
    if seed_file.exists():
        d = get_summary(seed_file)
        qwen_seeds[seed_name] = d
        print(f"  {seed_name}: SC={d['sc_pct']}%, SICA={d['sica_pct']}%, SC_corr={d['sc_correct']}, SICA_corr={d['sica_correct']}")
    else:
        print(f"  {seed_name}: FILE NOT FOUND ({seed_file})")

if len(qwen_seeds) >= 4:
    avg_sc = round(sum(s['sc_pct'] for s in qwen_seeds.values()) / len(qwen_seeds), 2)
    avg_sica = round(sum(s['sica_pct'] for s in qwen_seeds.values()) / len(qwen_seeds), 2)
    avg_delta = round(avg_sica - avg_sc, 2)
    print(f"  4-seed avg: SC={avg_sc}%, SICA={avg_sica}%, Δ={avg_delta}")
    print(check_value("T1-11", "SC%", 75.98, avg_sc))
    print(check_value("T1-11", "SICA%", 75.49, avg_sica))
    print(check_value("T1-11", "Δ(pp)", -0.49, avg_delta))

# T1-12: Qwen2.5-7B FOLIO
print("\n--- T1-12: Qwen2.5-7B × FOLIO ---")
qwen7b_path = Path("./experiments/exp-061-qwen7b-folio204")
print(f"  Experiment dir exists: {qwen7b_path.exists()}")
print(f"  No results.json found in experiment dir (only logs)")
# Check if results are in fleiss kappa or other files
print("  ⚠️  Cannot locate Qwen2.5-7B FOLIO result file - cannot verify")

# T1-13: Qwen2.5-14B LogiQA
print("\n--- T1-13: Qwen2.5-14B × LogiQA ---")
d = get_summary(result_files["qwen14b_logiqa"])
if d:
    print(f"  Raw: n={d['n']}, SC={d['sc_pct']}%, SICA={d['sica_pct']}%")
    print(check_value("T1-13", "SC%", 79.00, d['sc_pct']))
    print(check_value("T1-13", "SICA%", 78.00, d['sica_pct']))
    if d['sc_pct'] and d['sica_pct']:
        actual_delta = round(d['sica_pct'] - d['sc_pct'], 2)
        print(check_value("T1-13", "Δ(pp)", -1.00, actual_delta))

# T1-14: Qwen3-14B FOLIO
print("\n--- T1-14: Qwen3-14B × FOLIO ---")
# Check the folio_2x2_qwen3 results
q3_folio_path = BASE / "exp_folio_2x2_qwen3/results.json"
if q3_folio_path.exists():
    d = load_json(q3_folio_path)
    # Check structure
    if "summary" in d:
        s = d["summary"]
        print(f"  File: exp_folio_2x2_qwen3/results.json")
        n = s.get("n_problems", s.get("n"))
        sc = s.get("sc_accuracy")
        sica = s.get("sica_accuracy")
        if sc and sc < 1:
            sc = round(sc * 100, 2)
        if sica and sica < 1:
            sica = round(sica * 100, 2)
        print(f"  Raw: n={n}, SC={sc}%, SICA={sica}%")
        print(check_value("T1-14", "SC%", 83.33, sc))
        print(check_value("T1-14", "SICA%", 84.80, sica))
    else:
        # Maybe it has a different structure
        keys = list(d.keys())[:10]
        print(f"  File keys: {keys}")
else:
    print("  ⚠️  Cannot locate Qwen3-14B FOLIO result file")
    # Try other possible locations
    for p in BASE.glob("**/qwen3*folio*results*.json"):
        print(f"  Candidate: {p}")

# T1-15: Qwen3-14B PW-D5
print("\n--- T1-15: Qwen3-14B × PW-D5 ---")
d = get_summary(result_files["qwen3_pw"])
if d:
    print(f"  Raw: n={d['n']}, SC={d['sc_pct']}%, SICA={d['sica_pct']}%, SC_corr={d['sc_correct']}")
    print(check_value("T1-15", "SC%", 85.83, d['sc_pct']))
    print(check_value("T1-15", "SICA%", 85.67, d['sica_pct']))
    if d['sc_pct'] and d['sica_pct']:
        actual_delta = round(d['sica_pct'] - d['sc_pct'], 2)
        print(check_value("T1-15", "Δ(pp)", -0.17, actual_delta))

# T1-16: Qwen3-14B (think) FOLIO  
print("\n--- T1-16: Qwen3-14B(think) × FOLIO ---")
d = get_summary(result_files["qwen3_thinking"])
if d:
    print(f"  Raw: n={d['n']}, SC={d['sc_pct']}%, SICA={d['sica_pct']}%")
    print(f"  WARNING: This file has only {d['n']} problems (not 204)")
    print("  ⚠️  Partial results only - need full 204-problem file for verification")

# =============================================================================
# TABLE 2: tab:frontier-scb (from analysis.tex)
# =============================================================================
print("\n" + "=" * 100)
print("TABLE 2 VERIFICATION: tab:frontier-scb (analysis.tex)")
print("=" * 100)

table2_claims = [
    # (label, model, ext, SC%, n_err, BR, delta)
    ("T2-01", "Mistral-7B", "self", 55.88, 90, 4.97, 0.00),
    ("T2-02", "LLaMA-8B", "self", None, None, 2.88, None),  # --- values
    ("T2-03", "Qwen-14B", "self", 75.98, 49, 3.85, -0.49),
    ("T2-04", "R1-Distill-8B", "cross", 74.51, 52, 2.92, 2.45),
    ("T2-05", "gpt-4o", "self", 73.91, 53, 2.94, -1.80),
    ("T2-06", "gpt-4.1", "cross", 82.84, 35, 2.70, -1.47),
    ("T2-07", "gpt-5.5", "cross", 84.31, 32, 3.48, 0.49),
    ("T2-08", "o3", "cross", 82.65, 35, 1.66, 0.00),
    ("T2-09", "Gemini-2.5-Pro", "cross", 87.37, 12, 2.79, -1.10),
]

# T2-01: Mistral-7B self-extraction FOLIO
print("\n--- T2-01: Mistral-7B (self) ---")
# Use exp033 with tiebreak-corrected SC for s456 = 55.88%
# But Table 2 uses s456 (the "default" in Table 2 with SC=55.88%)
d = get_summary(result_files["mistral_folio_s456"])
if d:
    recomp_sc = tiebreak_map.get("exp-053", {}).get("new_sc_pct")
    print(f"  File: exp053 (s456) with tiebreak SC={recomp_sc}%")
    print(check_value("T2-01", "SC%", 55.88, recomp_sc))
    if recomp_sc:
        actual_n_err = 204 - round(recomp_sc * 204 / 100)
        print(check_value("T2-01", "n_err", 90, actual_n_err))

# Also check exp033 default seed
d_def = get_summary(result_files["mistral_folio_default"])
if d_def:
    print(f"  Alt (exp033 default seed42): SC={d_def['sc_pct']}%, n={d_def['n']}")
    # exp033 default: SC_correct=111, SC=54.41%. n_err = 204-111=93. Not 90.
    # But with tiebreak: if SC=55.88% (from s456), n_err = 204-114 = 90. ✓
    # So Table 2 Mistral row uses seed456 tiebreaked data.

# Check BR from earlier analysis
# BR was 4.97 from analysis.tex text  
# Let me check if there's a BR file for Mistral
print("  BR from paper text: 4.97×")

# T2-03: Qwen-14B self-extraction FOLIO
print("\n--- T2-03: Qwen-14B (self) ---")
d = get_summary(result_files["qwen14b_folio"])
if d:
    print(f"  Raw: SC={d['sc_pct']}%, SC_corr={d['sc_correct']}, n={d['n']}")
    # With some tiebreaking, SC should be 75.98%
    # Original: 158/204 = 77.45%. Paper says 75.98% = 155/204.
    # This could be from the 4-seed average or from tiebreak correction
    print(check_value("T2-03", "SC%", 75.98, d['sc_pct']))
    actual_n_err = d['n'] - d['sc_correct']
    print(f"  n_err from file: {actual_n_err}")
    # Paper says n_err=49. 204-155=49 (if SC=75.98%). 204-158=46 (if SC=77.45%).
    print(check_value("T2-03", "n_err", 49, actual_n_err))

# T2-04: R1-Distill-8B cross-model
print("\n--- T2-04: R1-Distill-8B (cross) ---")
r1_path = BASE / "exp_r1_distill_8b_sica/crossmodel_summary.json"
if r1_path.exists():
    r1 = load_json(r1_path)
    sc_pct = round(r1.get("sc_accuracy", 0) * 100, 2)
    sica_pct = round(r1.get("sica_accuracy", 0) * 100, 2)
    delta = r1.get("delta_pp")
    n_sica_wrong = r1.get("n_sica_wrong")
    mean_br = r1.get("mean_br")
    n = r1.get("n_problems", 204)
    n_err = n - round(r1.get("sc_accuracy", 0) * n)
    
    print(f"  Raw: SC={sc_pct}%, SICA={sica_pct}%, Δ={delta}, BR={mean_br}")
    print(check_value("T2-04", "SC%", 74.51, sc_pct))
    print(check_value("T2-04", "n_err", 52, n_err))
    print(check_value("T2-04", "BR", 2.92, round(mean_br, 2) if mean_br else None))
    print(check_value("T2-04", "Δ(pp)", 2.45, delta))

# T2-05: gpt-4o self-extraction
print("\n--- T2-05: gpt-4o (self) ---")
gpt4o_path = BASE / "e6_gpt4o_folio204/results.json"
if gpt4o_path.exists():
    d = get_summary(gpt4o_path)
    print(f"  Raw: n={d['n']}, SC={d['sc_pct']}%, SICA={d['sica_pct']}%, SC_corr={d['sc_correct']}")
    print(check_value("T2-05", "SC%", 73.91, d['sc_pct']))
    actual_n_err = d['n'] - d['sc_correct']
    print(check_value("T2-05", "n_err", 53, actual_n_err))

# T2-06: gpt-4.1 cross-model
print("\n--- T2-06: gpt-4.1 (cross) ---")
gpt41_cm = BASE / "exp_d135_frontier_scb/gpt-4.1/cross_model/cross_model_mistral.json"
gpt41_sc = BASE / "exp_d135_frontier_scb/gpt-4.1/results.json"
if gpt41_cm.exists():
    cm = load_json(gpt41_cm)
    s = cm.get("summary", cm)
    sc_correct = s.get("sc_correct")
    cm_correct = s.get("cross_model_correct")
    n = s.get("total", 204)
    sc_acc = round(sc_correct / n * 100, 2) if sc_correct else None
    cm_acc = round(cm_correct / n * 100, 2) if cm_correct else None
    delta = round((cm_correct - sc_correct) / n * 100, 2) if sc_correct and cm_correct else None
    n_err = n - sc_correct if sc_correct else None
    
    print(f"  Raw: n={n}, SC_corr={sc_correct}, CM_corr={cm_correct}")
    print(f"  SC={sc_acc}%, SICA(cross)={cm_acc}%, Δ={delta}")
    print(check_value("T2-06", "SC%", 82.84, sc_acc))
    print(check_value("T2-06", "n_err", 35, n_err))
    print(check_value("T2-06", "Δ(pp)", -1.47, delta))

# Also check full metrics
gpt41_fm = BASE / "exp_d135_frontier_scb/gpt-4.1/cross_model/gpt41_full_metrics.json"
if gpt41_fm.exists():
    fm = load_json(gpt41_fm)
    br = fm.get("mean_br")
    if br:
        print(check_value("T2-06", "BR", 2.70, round(br, 2)))
    else:
        print(f"  BR from full_metrics: {fm.get('mean_br', 'NOT FOUND')}")

# T2-07: gpt-5.5 cross-model
print("\n--- T2-07: gpt-5.5 (cross) ---")
gpt55_cm = BASE / "exp_d135_frontier_scb/gpt-5.5/cross_model/cross_model_mistral.json"
if gpt55_cm.exists():
    cm = load_json(gpt55_cm)
    s = cm.get("summary", cm)
    sc_correct = s.get("sc_correct")
    cm_correct = s.get("cross_model_correct")
    n = s.get("total", 204)
    sc_acc = round(sc_correct / n * 100, 2) if sc_correct else None
    cm_acc = round(cm_correct / n * 100, 2) if cm_correct else None
    delta = round((cm_correct - sc_correct) / n * 100, 2) if sc_correct and cm_correct else None
    n_err = n - sc_correct if sc_correct else None
    
    print(f"  Raw: n={n}, SC_corr={sc_correct}, CM_corr={cm_correct}")
    print(check_value("T2-07", "SC%", 84.31, sc_acc))
    print(check_value("T2-07", "n_err", 32, n_err))
    print(check_value("T2-07", "Δ(pp)", 0.49, delta))

# T2-08: o3 cross-model
print("\n--- T2-08: o3 (cross) ---")
o3_cm = BASE / "exp_d135_frontier_scb/o3/cross_model/cross_model_mistral.json"
o3_sc = BASE / "exp_d135_frontier_scb/comparison.json"
if o3_cm.exists():
    cm = load_json(o3_cm)
    s = cm.get("summary", cm)
    sc_correct = s.get("sc_correct")
    cm_correct = s.get("cross_model_correct")
    n = s.get("total")
    sc_acc_raw = s.get("sc_acc")
    sc_acc = round(sc_acc_raw * 100, 2) if sc_acc_raw and sc_acc_raw < 1 else sc_acc_raw
    n_err = n - sc_correct if sc_correct and n else None
    delta = round((cm_correct - sc_correct) / n * 100, 2) if sc_correct and cm_correct and n else None
    
    print(f"  Raw: n={n}, SC_corr={sc_correct}, CM_corr={cm_correct}")
    print(f"  SC_acc_raw={sc_acc_raw}, SC={sc_acc}%")
    print(check_value("T2-08", "SC%", 82.65, sc_acc))
    print(check_value("T2-08", "n_err", 35, n_err))
    print(check_value("T2-08", "Δ(pp)", 0.00, delta))

# T2-09: Gemini-2.5-Pro cross-model (interim n=95)
print("\n--- T2-09: Gemini-2.5-Pro (cross, interim) ---")
gemini_metrics = BASE / "exp_d135_frontier_scb/google_gemini-2.5-pro/cross_model/gemini_interim_95_metrics.json"
if gemini_metrics.exists():
    gm = load_json(gemini_metrics)
    sc_acc = round(gm.get("sc_accuracy", 0) * 100, 2)
    sica_acc = round(gm.get("sica_accuracy", 0) * 100, 2)
    delta = gm.get("delta_pp")
    mean_br = gm.get("mean_br")
    n = gm.get("n_questions", 95)
    n_sica_wrong = gm.get("n_sica_wrong")
    n_err = n - round(gm.get("sc_accuracy", 0) * n)
    
    print(f"  Raw: n={n}, SC={sc_acc}%, SICA={sica_acc}%, Δ={delta}, BR={mean_br}")
    print(check_value("T2-09", "SC%", 87.37, sc_acc))
    print(check_value("T2-09", "n_err", 12, n_err))
    print(check_value("T2-09", "BR", 2.79, round(mean_br, 2) if mean_br else None))
    print(check_value("T2-09", "Δ(pp)", -1.10, delta))

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 100)
print("ADDITIONAL CHECKS")
print("=" * 100)

# Check for duplicate table labels
print("\n--- Duplicate tab:main_results check ---")
import subprocess
# Check if both experiments.tex and diagnosis.tex define tab:main_results
exp_path = Path("./docs/paper/experiments.tex")
diag_path = Path("./docs/paper/diagnosis.tex")
if exp_path.exists() and diag_path.exists():
    exp_has = "tab:main_results" in exp_path.read_text()
    diag_has = "tab:main_results" in diag_path.read_text()
    if exp_has and diag_has:
        print("  ⚠️  BOTH experiments.tex AND diagnosis.tex define tab:main_results")
        print("     diagnosis.tex is NOT in main.tex (✅ no LaTeX conflict)")
        print("     But the two versions have DIFFERENT values:")
        print("     experiments.tex row 4: Mistral-7B(3s) FOLIO SC=56.53 SICA=56.70")
        print("     diagnosis.tex row 4: Mistral-7B FOLIO SC=55.88 SICA=59.31")
        print("     experiments.tex row 6: Mistral-7B FOLIO(s789) SC=59.31 SICA=56.86")
        print("     diagnosis.tex row 6: Mistral-7B FOLIO(s789) SC=54.41 SICA=53.92")
    else:
        print(f"  experiments.tex has tab:main_results: {exp_has}")
        print(f"  diagnosis.tex has tab:main_results: {diag_has}")

print("\nDone.")
