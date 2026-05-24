import json
import numpy as np
from collections import Counter
import os

BASE = "./results"

###############################################################################
# CIL computation: I(V; Y | G) — conditional mutual information
###############################################################################
def compute_cil(v_outputs, ground_truths, g_answers):
    """Compute CIL = I(V; Y | G) in bits using empirical distributions."""
    n = len(v_outputs)
    assert len(ground_truths) == n == len(g_answers)
    
    g_values = set(g_answers)
    cil = 0.0
    for g in g_values:
        mask = [i for i in range(n) if g_answers[i] == g]
        if not mask:
            continue
        p_g = len(mask) / n
        
        joint = Counter()
        v_marginal = Counter()
        y_marginal = Counter()
        for i in mask:
            joint[(v_outputs[i], ground_truths[i])] += 1
            v_marginal[v_outputs[i]] += 1
            y_marginal[ground_truths[i]] += 1
        
        n_g = len(mask)
        mi_g = 0.0
        for (v, y), count in joint.items():
            p_vy = count / n_g
            p_v = v_marginal[v] / n_g
            p_y = y_marginal[y] / n_g
            if p_vy > 0 and p_v > 0 and p_y > 0:
                mi_g += p_vy * np.log2(p_vy / (p_v * p_y))
        cil += p_g * mi_g
    return cil

def compute_delta_pp(v_outputs, ground_truths, g_answers):
    """Compute SICA/V accuracy - SC accuracy in pp."""
    v_correct = sum(1 for v, y in zip(v_outputs, ground_truths) if v == y)
    g_correct = sum(1 for g, y in zip(g_answers, ground_truths) if g == y)
    n = len(v_outputs)
    return (v_correct - g_correct) / n * 100

###############################################################################
# Task 1: Load per-question data for self-extraction conditions
###############################################################################
print("=" * 80)
print("TASK 1: CIL = I(V; Y | G) for all conditions")
print("=" * 80)

results_table = []

# --- Self-extraction conditions ---
self_configs = [
    ("Mistral-7B × FOLIO (self)", f"{BASE}/multi_seed/mistral_folio_seed123/results.json"),
    ("Mistral-7B × PW (self)", f"{BASE}/multi_seed/mistral_pw_seed42/results.json"),
    ("Qwen2.5-14B × FOLIO (self)", f"{BASE}/multi_seed/qwen25_folio_seed123/results.json"),
    ("Qwen2.5-14B × PW (self)", f"{BASE}/multi_seed/qwen25_pw_seed123/results.json"),
    ("LLaMA-8B × FOLIO (self)", f"{BASE}/multi_seed/llama8b_folio_seed123/results.json"),
    ("LLaMA-8B × PW (self)", f"{BASE}/multi_seed/llama8b_pw_seed42/results.json"),
]

for name, path in self_configs:
    try:
        with open(path) as f:
            data = json.load(f)
        questions = data["results"]
        v = [q["sica_answer"] for q in questions]
        g = [q["sc_answer"] for q in questions]
        y = [q["ground_truth"] for q in questions]
        cil = compute_cil(v, y, g)
        delta = compute_delta_pp(v, y, g)
        n = len(questions)
        results_table.append(("Self", name, n, cil, delta))
        print(f"  {name}: n={n}, CIL={cil:.4f} bits, Δ={delta:+.2f}pp")
    except Exception as e:
        print(f"  {name}: ERROR - {e}")

# --- Cross-architecture: Zero-shot DeBERTa × FOLIO (complementarity only) ---
print("\n  [Cross-arch: Zero-shot NLI on FOLIO — from complementarity counts]")
try:
    with open(f"{BASE}/folio_zeroshot_nli_combo_d061.json") as f:
        folio_zs = json.load(f)
    for gen_key, gen_label in [("mistral", "Mistral-7B"), ("llama8b", "LLaMA-8B"),
                                ("qwen14b", "Qwen2.5-14B"), ("qwen3", "Qwen3-14B")]:
        if gen_key in folio_zs:
            c = folio_zs[gen_key]["complementarity"]
            br = c["both_right"]; vrgw = c["nli_right_sc_wrong"]
            grvw = c["sc_right_nli_wrong"]; bw = c["both_wrong"]
            n = br + vrgw + grvw + bw
            # Binary CIL from 2×2 table
            v_bin = []
            g_bin = []
            y_bin = []
            for _ in range(br):
                v_bin.append("correct"); g_bin.append("correct"); y_bin.append("correct")
            for _ in range(vrgw):
                v_bin.append("correct"); g_bin.append("wrong"); y_bin.append("correct")
            for _ in range(grvw):
                v_bin.append("wrong"); g_bin.append("correct"); y_bin.append("correct")
            for _ in range(bw):
                v_bin.append("wrong"); g_bin.append("wrong"); y_bin.append("correct")
            cil = compute_cil(v_bin, y_bin, g_bin)
            delta_w1 = folio_zs[gen_key].get("w1", {}).get("delta_pp", folio_zs[gen_key].get("w3", {}).get("delta_pp", 0))
            name = f"{gen_label} × FOLIO + DeBERTa-ZS"
            results_table.append(("Cross-ZS", name, n, cil, delta_w1))
            print(f"  {name}: n={n}, CIL_binary={cil:.4f} bits, Δ_w1={delta_w1:+.2f}pp")
except Exception as e:
    print(f"  FOLIO ZS combo error: {e}")

# --- Cross-architecture: Zero-shot DeBERTa × PW (per-question data) ---
print("\n  [Cross-arch: Zero-shot NLI on PW — per-question data]")

# r11 canonical has per-question data for Mistral × {DeBERTa, RoBERTa, BART}
try:
    with open(f"{BASE}/r11_3verifier_canonical.json") as f:
        r11 = json.load(f)
    pq = r11["per_question"]
    for verifier_key, verifier_label in [("deberta", "DeBERTa-lg"), ("roberta", "RoBERTa-lg"), ("bart", "BART-lg")]:
        v = [pq[qid][verifier_key] for qid in pq]
        g = [pq[qid]["sc_answer"] for qid in pq]
        y = [pq[qid]["gold"] for qid in pq]
        cil = compute_cil(v, y, g)
        # Get delta_pp from combo
        delta_w1 = r11["individual_verifiers"][verifier_key]["combos"]["w1"]["delta_pp"]
        name = f"Mistral-7B × PW + {verifier_label} ZS"
        n = len(v)
        results_table.append(("Cross-ZS", name, n, cil, delta_w1))
        print(f"  {name}: n={n}, CIL={cil:.4f} bits, Δ_w1={delta_w1:+.2f}pp")
except Exception as e:
    print(f"  r11 per-question error: {e}")

# exp092: LLaMA × DeBERTa ZS per-question
try:
    with open(f"{BASE}/exp092_llama8b_pw_nli_combo/results.json") as f:
        exp092 = json.load(f)
    pq = exp092["per_question"]
    v = [pq[qid]["deberta_pred"] for qid in pq]
    g = [pq[qid]["sc_answer"] for qid in pq]
    y = [pq[qid]["gold"] for qid in pq]
    cil = compute_cil(v, y, g)
    delta = exp092["combo"]["w1"]["delta_pp"]
    name = "LLaMA-8B × PW + DeBERTa-lg ZS"
    results_table.append(("Cross-ZS", name, len(v), cil, delta))
    print(f"  {name}: n={len(v)}, CIL={cil:.4f} bits, Δ_w1={delta:+.2f}pp")
except Exception as e:
    print(f"  exp092 error: {e}")

# qwen14b PW NLI combo
try:
    with open(f"{BASE}/qwen14b_pw_nli_combo/results.json") as f:
        qw14 = json.load(f)
    pq = qw14["per_question"]
    v = [pq[qid]["deberta_pred"] for qid in pq]
    g = [pq[qid]["sc_answer"] for qid in pq]
    y = [pq[qid]["gold"] for qid in pq]
    cil = compute_cil(v, y, g)
    delta = qw14["combo"]["w1"]["delta_pp"]
    name = "Qwen2.5-14B × PW + DeBERTa-lg ZS"
    results_table.append(("Cross-ZS", name, len(v), cil, delta))
    print(f"  {name}: n={len(v)}, CIL={cil:.4f} bits, Δ_w1={delta:+.2f}pp")
except Exception as e:
    print(f"  qwen14b combo error: {e}")

# d116: Qwen3 × DeBERTa ZS
try:
    with open(f"{BASE}/d116_qwen3_pw_deberta_large/results.json") as f:
        d116 = json.load(f)
    pq = d116["per_question"]
    v = [pq[qid]["deberta_pred"] for qid in pq]
    g = [pq[qid]["sc_answer"] for qid in pq]
    y = [pq[qid]["gold"] for qid in pq]
    cil = compute_cil(v, y, g)
    delta = d116["combo"]["w1"]["delta_pp"]
    name = "Qwen3-14B × PW + DeBERTa-lg ZS"
    results_table.append(("Cross-ZS", name, len(v), cil, delta))
    print(f"  {name}: n={len(v)}, CIL={cil:.4f} bits, Δ_w1={delta:+.2f}pp")
except Exception as e:
    print(f"  d116 error: {e}")

# --- Cross-architecture: Fine-tuned DeBERTa × PW ---
print("\n  [Cross-arch: Fine-tuned DeBERTa × PW]")
# eval_finetuned has per-question NLI predictions; need to match with SC answers from generators
try:
    with open(f"{BASE}/eval_finetuned_deberta-large-pw-owa/results.json") as f:
        ft_data = json.load(f)
    ft_pq = ft_data["per_question"]  # {qid: {gold, nli_pred, ...}}
    
    # Match with SC answers from different generators
    gen_configs = [
        ("Mistral-7B", f"{BASE}/multi_seed/mistral_pw_seed42/results.json"),
        ("LLaMA-8B", f"{BASE}/multi_seed/llama8b_pw_seed42/results.json"),
        ("Qwen2.5-14B", f"{BASE}/multi_seed/qwen25_pw_seed123/results.json"),
    ]
    
    for gen_label, gen_path in gen_configs:
        try:
            with open(gen_path) as f:
                gen_data = json.load(f)
            gen_questions = {q["problem_id"]: q for q in gen_data["results"]}
            
            v_list, g_list, y_list = [], [], []
            for qid in ft_pq:
                if qid in gen_questions:
                    v_list.append(ft_pq[qid]["nli_pred"])
                    g_list.append(gen_questions[qid]["sc_answer"])
                    y_list.append(ft_pq[qid]["gold"])
            
            cil = compute_cil(v_list, y_list, g_list)
            delta = compute_delta_pp(v_list, y_list, g_list)
            name = f"{gen_label} × PW + DeBERTa-FT"
            results_table.append(("Cross-FT", name, len(v_list), cil, delta))
            print(f"  {name}: n={len(v_list)}, CIL={cil:.4f} bits, Δ={delta:+.2f}pp")
        except Exception as e:
            print(f"  {gen_label} FT combo error: {e}")
except Exception as e:
    print(f"  FT DeBERTa error: {e}")


###############################################################################
# Summary table
###############################################################################
print("\n" + "=" * 80)
print("CIL SUMMARY TABLE")
print("=" * 80)
print(f"{'Condition':<45} {'Type':<10} {'N':>4} {'CIL (bits)':>12} {'Δ(pp)':>8} {'CIL>0↔Δ>0?':>12}")
print("-" * 95)
for typ, name, n, cil, delta in results_table:
    # Check if CIL sign matches delta sign (excluding ~0 cases)
    if abs(cil) < 0.001 and abs(delta) < 0.5:
        match = "≈0"
    elif (cil > 0 and delta > 0) or (cil <= 0 and delta <= 0):
        match = "✓"
    else:
        match = "✗"
    print(f"{name:<45} {typ:<10} {n:>4} {cil:>12.4f} {delta:>+8.2f} {match:>12}")


###############################################################################
# Task 2 (Q5): BR on SC-correct subset
###############################################################################
print("\n" + "=" * 80)
print("TASK 2 (Q5): BR on SC-correct subset")
print("=" * 80)

def compute_br_from_questions(questions):
    """Compute BR = P(SICA agrees with SC) / P(SICA disagrees with SC)
    using sica_scores. 
    BR > 1 means constraints tend to support the SC majority answer."""
    agree = 0
    disagree = 0
    for q in questions:
        sc_ans = q["sc_answer"]
        sica_scores = q.get("sica_scores", {})
        if not sica_scores:
            continue
        sc_score = sica_scores.get(sc_ans, 0)
        other_score = sum(v for k, v in sica_scores.items() if k != sc_ans)
        if sc_score > other_score:
            agree += 1
        elif other_score > sc_score:
            disagree += 1
        # tie: neither
    return agree, disagree

for name, path in self_configs:
    try:
        with open(path) as f:
            data = json.load(f)
        questions = data["results"]
        
        # Full set BR
        agree_all, disagree_all = compute_br_from_questions(questions)
        br_all = agree_all / max(disagree_all, 1)
        
        # SC-correct subset
        sc_correct_qs = [q for q in questions if q["sc_answer"] == q["ground_truth"]]
        agree_correct, disagree_correct = compute_br_from_questions(sc_correct_qs)
        br_correct = agree_correct / max(disagree_correct, 1)
        
        # SC-wrong subset
        sc_wrong_qs = [q for q in questions if q["sc_answer"] != q["ground_truth"]]
        agree_wrong, disagree_wrong = compute_br_from_questions(sc_wrong_qs)
        br_wrong = agree_wrong / max(disagree_wrong, 1)
        
        print(f"\n  {name}:")
        print(f"    Full set:     n={len(questions):>4}, BR={br_all:.2f} (agree={agree_all}, disagree={disagree_all})")
        print(f"    SC-correct:   n={len(sc_correct_qs):>4}, BR={br_correct:.2f} (agree={agree_correct}, disagree={disagree_correct})")
        print(f"    SC-wrong:     n={len(sc_wrong_qs):>4}, BR={br_wrong:.2f} (agree={agree_wrong}, disagree={disagree_wrong})")
    except Exception as e:
        print(f"  {name}: ERROR - {e}")


###############################################################################
# Task 3 (Q6): Training-free remediation conditions with p_raw < 0.05
###############################################################################
print("\n" + "=" * 80)
print("TASK 3 (Q6): Training-free conditions with raw p < 0.05 (before BH)")
print("=" * 80)

all_conditions = []

# From r11_3verifier_canonical.json
try:
    with open(f"{BASE}/r11_3verifier_canonical.json") as f:
        r11 = json.load(f)
    for vname, vdata in r11["individual_verifiers"].items():
        for wk, wdata in vdata["combos"].items():
            all_conditions.append({
                "label": f"Mistral × {vname} × {wk}",
                "generator": "Mistral-7B",
                "verifier": vname,
                "weight": wk,
                "dataset": "PW-600",
                "delta_pp": wdata["delta_pp"],
                "p_raw": wdata["mcnemar_p"],
                "type": "zero-shot NLI"
            })
    # Ensemble methods
    for ens_key, ens_label in [("ensemble_majority", "Majority-3V"), ("additive_3verifier", "Additive-3V")]:
        if ens_key in r11:
            for wk, wdata in r11[ens_key]["combos"].items():
                all_conditions.append({
                    "label": f"Mistral × {ens_label} × {wk}",
                    "generator": "Mistral-7B",
                    "verifier": ens_label,
                    "weight": wk,
                    "dataset": "PW-600",
                    "delta_pp": wdata["delta_pp"],
                    "p_raw": wdata["mcnemar_p"],
                    "type": "zero-shot NLI ensemble"
                })
except Exception as e:
    print(f"  r11 error: {e}")

# From exp092 (LLaMA × DeBERTa ZS)
try:
    with open(f"{BASE}/exp092_llama8b_pw_nli_combo/results.json") as f:
        exp092 = json.load(f)
    for wk in ["w1", "w3", "w5"]:
        if wk in exp092["combo"]:
            wdata = exp092["combo"][wk]
            all_conditions.append({
                "label": f"LLaMA-8B × DeBERTa-lg × {wk}",
                "generator": "LLaMA-8B",
                "verifier": "DeBERTa-large",
                "weight": wk,
                "dataset": "PW-600",
                "delta_pp": wdata["delta_pp"],
                "p_raw": wdata["mcnemar_p"],
                "type": "zero-shot NLI"
            })
except Exception as e:
    print(f"  exp092 error: {e}")

# From qwen14b_pw_nli_combo
try:
    with open(f"{BASE}/qwen14b_pw_nli_combo/results.json") as f:
        qw14 = json.load(f)
    for wk in ["w1", "w3", "w5"]:
        if wk in qw14["combo"]:
            wdata = qw14["combo"][wk]
            all_conditions.append({
                "label": f"Qwen2.5-14B × DeBERTa-lg × {wk}",
                "generator": "Qwen2.5-14B",
                "verifier": "DeBERTa-large",
                "weight": wk,
                "dataset": "PW-600",
                "delta_pp": wdata["delta_pp"],
                "p_raw": wdata["mcnemar_p"],
                "type": "zero-shot NLI"
            })
except Exception as e:
    print(f"  qw14 error: {e}")

# From d116 (Qwen3 × DeBERTa ZS)
try:
    with open(f"{BASE}/d116_qwen3_pw_deberta_large/results.json") as f:
        d116 = json.load(f)
    for wk in ["w1", "w3", "w5"]:
        if wk in d116["combo"]:
            wdata = d116["combo"][wk]
            all_conditions.append({
                "label": f"Qwen3-14B × DeBERTa-lg × {wk}",
                "generator": "Qwen3-14B",
                "verifier": "DeBERTa-large",
                "weight": wk,
                "dataset": "PW-600",
                "delta_pp": wdata["delta_pp"],
                "p_raw": wdata["mcnemar_p"],
                "type": "zero-shot NLI"
            })
except Exception as e:
    print(f"  d116 error: {e}")

# From folio_zeroshot_nli_combo_d061.json
try:
    with open(f"{BASE}/folio_zeroshot_nli_combo_d061.json") as f:
        folio_zs = json.load(f)
    for gen_key, gen_label in [("mistral", "Mistral-7B"), ("llama8b", "LLaMA-8B"),
                                ("qwen14b", "Qwen2.5-14B"), ("qwen3", "Qwen3-14B")]:
        if gen_key in folio_zs:
            gen_data = folio_zs[gen_key]
            for wk in ["w1", "w3", "w5"]:
                if wk in gen_data:
                    wdata = gen_data[wk]
                    all_conditions.append({
                        "label": f"{gen_label} × DeBERTa-lg-ZS × {wk} / FOLIO",
                        "generator": gen_label,
                        "verifier": "DeBERTa-large-ZS",
                        "weight": wk,
                        "dataset": "FOLIO-204",
                        "delta_pp": wdata["delta_pp"],
                        "p_raw": wdata["mcnemar_p"],
                        "type": "zero-shot NLI"
                    })
except Exception as e:
    print(f"  FOLIO ZS error: {e}")

# Print all conditions sorted by p_raw
all_conditions.sort(key=lambda x: x["p_raw"])
print(f"\nTotal training-free conditions found: {len(all_conditions)}")
print(f"\n{'Label':<50} {'Δpp':>6} {'p_raw':>12} {'p<0.05':>8} {'Δ>0':>5}")
print("-" * 85)

sig_count = 0
sig_positive = 0
for c in all_conditions:
    is_sig = c["p_raw"] < 0.05
    is_positive = c["delta_pp"] > 0
    sig_mark = "***" if c["p_raw"] < 0.001 else ("**" if c["p_raw"] < 0.01 else ("*" if c["p_raw"] < 0.05 else ""))
    print(f"{c['label']:<50} {c['delta_pp']:>+6.2f} {c['p_raw']:>12.6f} {sig_mark:>8} {'Y' if is_positive else 'N':>5}")
    if is_sig:
        sig_count += 1
        if is_positive:
            sig_positive += 1

print(f"\nSignificant (p_raw < 0.05): {sig_count}/{len(all_conditions)}")
print(f"  Of which positive (Δ > 0): {sig_positive}")

# Analyze common characteristics of significant positive conditions
print("\n--- Common characteristics of significant positive conditions (p_raw < 0.05, Δ > 0) ---")
sig_pos = [c for c in all_conditions if c["p_raw"] < 0.05 and c["delta_pp"] > 0]
if sig_pos:
    # By generator
    gen_counts = Counter(c["generator"] for c in sig_pos)
    print(f"  By generator: {dict(gen_counts)}")
    # By weight
    wt_counts = Counter(c["weight"] for c in sig_pos)
    print(f"  By weight: {dict(wt_counts)}")
    # By dataset
    ds_counts = Counter(c["dataset"] for c in sig_pos)
    print(f"  By dataset: {dict(ds_counts)}")
    # By verifier
    vr_counts = Counter(c["verifier"] for c in sig_pos)
    print(f"  By verifier: {dict(vr_counts)}")

print("\n\nDONE.")
