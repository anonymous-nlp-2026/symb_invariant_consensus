import json
import numpy as np
from collections import Counter
import math

BASE = "/root/symb_invariant_consensus/results"

def compute_cil(v_outputs, ground_truths, g_answers):
    """Compute CIL = I(V; Y | G) in bits."""
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
                mi_g += p_vy * math.log2(p_vy / (p_v * p_y))
        cil += p_g * mi_g
    return cil

def delta_pp(v, y, g):
    v_c = sum(1 for a, b in zip(v, y) if a == b)
    g_c = sum(1 for a, b in zip(g, y) if a == b)
    return (v_c - g_c) / len(v) * 100

def load_sica_results(path, results_key="results"):
    """Load per-question SICA results."""
    with open(path) as f:
        data = json.load(f)
    questions = data[results_key]
    v = [q["sica_answer"] for q in questions]
    g = [q["sc_answer"] for q in questions]
    y = [q["ground_truth"] for q in questions]
    return v, g, y, questions

###############################################################################
print("=" * 80)
print("TASK 1: CIL = I(V; Y | G)")
print("=" * 80)
###############################################################################

rows = []

# === Self-extraction ===
self_configs = [
    ("Mistral Ã— FOLIO (self)",    f"{BASE}/multi_seed/mistral_folio_seed123/results.json"),
    ("Mistral Ã— PW (self)",       f"{BASE}/multi_seed/mistral_pw_seed42/results.json"),
    ("Qwen2.5 Ã— FOLIO (self)",   f"{BASE}/multi_seed/qwen25_folio_seed123/results.json"),
    ("Qwen2.5 Ã— PW (self)",     f"{BASE}/exp032_qwen25_14b_pw600/results.json"),
    ("LLaMA-8B Ã— FOLIO (self)",  f"{BASE}/multi_seed/llama8b_folio_seed123/results.json"),
    ("LLaMA-8B Ã— PW (self)",    f"{BASE}/exp048_llama8b_pw600/exp048_results.json"),
]

for name, path in self_configs:
    try:
        v, g, y, qs = load_sica_results(path)
        cil = compute_cil(v, y, g)
        d = delta_pp(v, y, g)
        rows.append(("Self", name, len(v), cil, d))
    except Exception as e:
        print(f"  ERROR {name}: {e}")

# === Cross-arch: ZS NLI on PW (per-question) ===

# r11: Mistral Ã— {DeBERTa, RoBERTa, BART} on PW
try:
    with open(f"{BASE}/r11_3verifier_canonical.json") as f:
        r11 = json.load(f)
    pq = r11["per_question_preds"]
    for vk, vl in [("deberta", "DeBERTa-lg"), ("roberta", "RoBERTa-lg"), ("bart", "BART-lg")]:
        v = [pq[qid][vk] for qid in pq]
        g = [pq[qid]["sc_answer"] for qid in pq]
        y = [pq[qid]["gold"] for qid in pq]
        cil = compute_cil(v, y, g)
        d = delta_pp(v, y, g)
        rows.append(("Cross-ZS", f"Mistral Ã— PW + {vl}", len(v), cil, d))
except Exception as e:
    print(f"  ERROR r11: {e}")

# exp092: LLaMA Ã— DeBERTa ZS on PW
try:
    with open(f"{BASE}/exp092_llama8b_pw_nli_combo/results.json") as f:
        exp092 = json.load(f)
    pq = exp092["per_question"]
    v = [pq[qid]["deberta_pred"] for qid in pq]
    g = [pq[qid]["sc_answer"] for qid in pq]
    y = [pq[qid]["gold"] for qid in pq]
    cil = compute_cil(v, y, g)
    d = delta_pp(v, y, g)
    rows.append(("Cross-ZS", "LLaMA-8B Ã— PW + DeBERTa-lg", len(v), cil, d))
except Exception as e:
    print(f"  ERROR exp092: {e}")

# Qwen2.5 Ã— DeBERTa ZS on PW
try:
    with open(f"{BASE}/qwen14b_pw_nli_combo/results.json") as f:
        qw14 = json.load(f)
    pq = qw14["per_question"]
    v = [pq[qid]["deberta_pred"] for qid in pq]
    g = [pq[qid]["sc_answer"] for qid in pq]
    y = [pq[qid]["gold"] for qid in pq]
    cil = compute_cil(v, y, g)
    d = delta_pp(v, y, g)
    rows.append(("Cross-ZS", "Qwen2.5 Ã— PW + DeBERTa-lg", len(v), cil, d))
except Exception as e:
    print(f"  ERROR qw14: {e}")

# Qwen3 Ã— DeBERTa ZS on PW
try:
    with open(f"{BASE}/d116_qwen3_pw_deberta_large/results.json") as f:
        d116 = json.load(f)
    pq = d116["per_question"]
    v = [pq[qid]["deberta_pred"] for qid in pq]
    g = [pq[qid]["sc_answer"] for qid in pq]
    y = [pq[qid]["gold"] for qid in pq]
    cil = compute_cil(v, y, g)
    d = delta_pp(v, y, g)
    rows.append(("Cross-ZS", "Qwen3 Ã— PW + DeBERTa-lg", len(v), cil, d))
except Exception as e:
    print(f"  ERROR d116: {e}")

# === Cross-arch: Fine-tuned DeBERTa Ã— PW ===
try:
    with open(f"{BASE}/eval_finetuned_deberta-large-pw-owa/results.json") as f:
        ft = json.load(f)
    ft_pq = ft["per_question"]

    gen_configs = [
        ("Mistral",   f"{BASE}/multi_seed/mistral_pw_seed42/results.json"),
        ("LLaMA-8B",  f"{BASE}/exp048_llama8b_pw600/exp048_results.json"),
        ("Qwen2.5",   f"{BASE}/exp032_qwen25_14b_pw600/results.json"),
    ]
    for gl, gp in gen_configs:
        with open(gp) as f:
            gd = json.load(f)
        gq = {q["problem_id"]: q for q in gd["results"]}
        vl, gl2, yl = [], [], []
        for qid in ft_pq:
            if qid in gq:
                vl.append(ft_pq[qid]["nli_pred"])
                gl2.append(gq[qid]["sc_answer"])
                yl.append(ft_pq[qid]["gold"])
        cil = compute_cil(vl, yl, gl2)
        d = delta_pp(vl, yl, gl2)
        rows.append(("Cross-FT", f"{gl} Ã— PW + DeBERTa-FT", len(vl), cil, d))
except Exception as e:
    print(f"  ERROR FT: {e}")

# === FOLIO ZS NLI (per-question from r11 preds if available, else complementarity) ===
# The folio_zeroshot_nli_combo has no per-question data, only complementarity counts.
# We note: binary complementarity MI is NOT the same as I(V;Y|G).
# Skip these for now â€” mark as "N/A (no per-question data)".

# === Print CIL Table ===
print(f"\n{'Condition':<40} {'Type':<10} {'N':>4} {'CIL(bits)':>10} {'Î”(pp)':>8} {'Match':>6}")
print("-" * 82)
for typ, name, n, cil, d in rows:
    if abs(cil) < 0.005 and abs(d) < 0.5:
        m = "â‰ˆ0"
    elif (cil > 0.005 and d > 0) or (cil <= 0.005 and d <= 0):
        m = "âœ“"
    else:
        m = "âœ—"
    print(f"{name:<40} {typ:<10} {n:>4} {cil:>10.4f} {d:>+8.2f} {m:>6}")

# Compute mean self vs cross
self_cils = [r[3] for r in rows if r[0] == "Self"]
cross_zs_cils = [r[3] for r in rows if r[0] == "Cross-ZS"]
cross_ft_cils = [r[3] for r in rows if r[0] == "Cross-FT"]
print(f"\nMean CIL â€” Self: {np.mean(self_cils):.4f} | Cross-ZS: {np.mean(cross_zs_cils):.4f} | Cross-FT: {np.mean(cross_ft_cils):.4f}")
print(f"Ratio Cross-ZS/Self: {np.mean(cross_zs_cils)/np.mean(self_cils):.1f}x | Cross-FT/Self: {np.mean(cross_ft_cils)/np.mean(self_cils):.1f}x")


###############################################################################
print("\n" + "=" * 80)
print("TASK 2 (Q5): BR > 1 on SC-correct subset?")
print("=" * 80)
###############################################################################

all_q5_data = []

for name, path in self_configs:
    try:
        _, _, _, questions = load_sica_results(path)
        
        subsets = {
            "All":        questions,
            "SC-correct": [q for q in questions if q["sc_answer"] == q["ground_truth"]],
            "SC-wrong":   [q for q in questions if q["sc_answer"] != q["ground_truth"]],
        }
        
        print(f"\n  {name}:")
        for subset_name, qs in subsets.items():
            agree = disagree = tie = 0
            for q in qs:
                sc = q["sc_answer"]
                scores = q.get("sica_scores", {})
                if not scores:
                    continue
                sc_s = scores.get(sc, 0)
                other_s = sum(v for k, v in scores.items() if k != sc)
                if sc_s > other_s:
                    agree += 1
                elif other_s > sc_s:
                    disagree += 1
                else:
                    tie += 1
            br = agree / max(disagree, 1)
            print(f"    {subset_name:<12} n={len(qs):>4}  BR={br:.2f}  (agree={agree}, disagree={disagree}, tie={tie})")
            if subset_name in ["SC-correct", "SC-wrong"]:
                all_q5_data.append((name, subset_name, len(qs), br))
    except Exception as e:
        print(f"  ERROR {name}: {e}")

# Summary
print("\n  Q5 Summary: BR on SC-correct subset")
print(f"  {'Condition':<35} {'n_correct':>10} {'BR_correct':>12} {'_wrong':>10} {'BR_wrong':>12} {'BR_c>1?':>8}")
print("  " + "-" * 90)
for i in range(0, len(all_q5_data), 2):
    correct = all_q5_data[i]
    wrong = all_q5_data[i+1] if i+1 < len(all_q5_data) else None
    br_c = correct[3]
    print(f"  {correct[0]:<35} {correct[2]:>10} {br_c:>12.2f} {wrong[2] if wrong else 'N/A':>10} {wrong[3] if wrong else 0:>12.2f} {'YES' if br_c > 1 else 'NO':>8}")


###############################################################################
print("\n" + "=" * 80)
print("TASK 3 (Q6): Training-free conditions with raw p < 0.05")
print("=" * 80)
###############################################################################

all_conds = []

# r11 canonical (Mistral Ã— 3 verifiers Ã— 3 weights + 2 ensemble Ã— 3 weights)
with open(f"{BASE}/r11_3verifier_canonical.json") as f:
    r11 = json.load(f)
for vn, vd in r11["individual_verifiers"].items():
    for wk, wd in vd["combos"].items():
        all_conds.append((f"Mistral Ã— {vn} Ã— {wk}", wd["delta_pp"], wd["mcnemar_p"], "PW"))
for ek, el in [("ensemble_majority", "Majority-3V"), ("additive_3verifier", "Additive-3V")]:
    if ek in r11:
        for wk, wd in r11[ek]["combos"].items():
            all_conds.append((f"Mistral Ã— {el} Ã— {wk}", wd["delta_pp"], wd["mcnemar_p"], "PW"))

# exp092 (LLaMA Ã— DeBERTa ZS)
with open(f"{BASE}/exp092_llama8b_pw_nli_combo/results.json") as f:
    exp092 = json.load(f)
for wk in ["w1", "w3", "w5"]:
    wd = exp092["combo"][wk]
    all_conds.append((f"LLaMA Ã— DeBERTa-lg Ã— {wk}", wd["delta_pp"], wd["mcnemar_p"], "PW"))

# Qwen2.5 Ã— DeBERTa ZS
with open(f"{BASE}/qwen14b_pw_nli_combo/results.json") as f:
    qw14 = json.load(f)
for wk in ["w1", "w3", "w5"]:
    wd = qw14["combo"][wk]
    all_conds.append((f"Qwen2.5 Ã— DeBERTa-lg Ã— {wk}", wd["delta_pp"], wd["mcnemar_p"], "PW"))

# Qwen3 Ã— DeBERTa ZS
with open(f"{BASE}/d116_qwen3_pw_deberta_large/results.json") as f:
    d116 = json.load(f)
for wk in ["w1", "w3", "w5"]:
    wd = d116["combo"][wk]
    all_conds.append((f"Qwen3 Ã— DeBERTa-lg Ã— {wk}", wd["delta_pp"], wd["mcnem\—Ü—K”ÈŠJB‚ˆÈ“ÓSÈ”ÈÛÛX›ÜÂÚ]Ü[ŠˆžÐTÑ_KÙ›Û[×Þ™\›ÜÚÝÛ›WØÛÛX›×ÙŒKšœÛÛˆŠH\ÈŽ‚ˆžœÈHœÛÛ‹›ØY
ŠB™›ÜˆÚËÛ[ˆÊ›Z\Ý˜[‹“Z\Ý˜[ŠK
›[XNˆ‹“SPHŠK
œ]Ù[ŒMˆ‹”]Ù[Œ‹HŠK
œ]Ù[ŒÈ‹”]Ù[ŒÈŠWN‚ˆYˆÚÈ[ˆžœÎ‚ˆ›ÜˆÚÈ[ˆÈÌH‹ÌÈ‹ÍH—N‚ˆYˆÚÈ[ˆžœÖÙÚ×N‚ˆÙHžœÖÙÚ×VÝÚ×Bˆ[ØÛÛ™Ë˜\[™

ˆžÙÛH0åÈP‘T•KV”È0åÈÝÚßH
“ÊH‹ÙÈ™[WÜ—KÙÈ›XÛ™[X\—Ü—K‘“ÓSÈŠJB‚˜[ØÛÛ™ËœÛÜ
Ù^O[[X™HˆÌ—JB‚œš[
ˆ—•Ý[ÛÛ™][ÛœÎˆÛ[Š[ØÛÛ™Ê_HŠBœš[
ˆ—žÉÓX™[	Î_HÉó¥	ÎßHÉÜÜ˜]ÉÎŒLŸHÉÔÚYÉÎ_HÉó¥Œ	Î_HŠBœš[
‹Hˆ
ˆÎ
B‚œÚY×ÜÜÈH×B™›ÜˆX™[È[ˆ[ØÛÛ™Î‚ˆÚYÈHŠŠŠˆˆYˆŒH[ÙH
ŠŠˆˆYˆŒH[ÙH
ŠˆˆYˆŒH[ÙHˆŠJBˆÜÈH–HˆYˆˆ[ÙH
HˆYˆOH[ÙH“ˆŠBˆš[
ˆžÛX™[_HÙŠÍËŒ™ŸHÜŒL‹™ŸHÜÚYÎ_HÙÜÎ_HŠBˆYˆŒH[™ˆ‚ˆÚY×ÜÜË˜\[™

X™[ÊJB‚œš[
ˆ—”ÚYÛšYšXØ[ÜÚ]]™H
Ü˜]ÈŒK3¥ˆ
NˆÛ[ŠÚY×ÜÜÊ_KÞÛ[Š[ØÛÛ™Ê_HŠBœš[
—‹KKHÛÛ[[ÛˆÚ\˜XÝ\š\ÝXÜÈKKHŠB™Ù[—ØÈHÛÝ[\Š
BÝØÈHÛÝ[\Š
B™×ØÈHÛÝ[\Š
B™›ÜˆX™[È[ˆÚY×ÜÜÎ‚ˆÈ^˜XÝÙ[™\˜]Ü‚ˆÙ[ˆHX™[œÜ]
ˆ0åÈŠVÌBˆÙ[—ØÖÙÙ[—H
ÏHBˆÈ^˜XÝÙZYÚˆ›ÜˆÈ[ˆÈÌH‹ÌÈ‹ÍH—N‚ˆYˆÈ[ˆX™[‚ˆÝØÖÝ×H
ÏHBˆœ™XZÂˆ×ØÖÙ×H
ÏHB‚œš[
ˆˆÙ[™\˜]ÜœÎˆÙXÝ
Ù[—ØÊ_HŠBœš[
ˆˆÙZYÚÎˆÙXÝ
ÝØÊ_HŠBœš[
ˆˆ]\Ù]ÎˆÙXÝ
×ØÊ_HŠBœš[
ˆ—ˆÙ^H]\›ŽˆÛ[ŠÚY×ÜÜÊ_HÛÛ™][ÛœÈ\™HSÜÚ]]™H
3¥Œ
KˆŠBœš[
ˆˆ[ÛˆËMŒÚ]Z\Ý˜[ÜˆSPH
ÙXZÙ\ˆÙ[™\˜]ÜœÊKˆŠBœš[
ˆˆYÚ\ˆÙZYÚÈ
ÌËÍJHÛZ[˜]H8 %[Ü™H“H[™›Y[˜ÙH8¡¤ˆ[Ü™HØZ[‹ˆŠB‚œš[
—‘Ó‘KˆŠB