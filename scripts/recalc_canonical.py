#!/usr/bin/env python3
"""Recalculate FT DeBERTa combo and DRC tau=0.7 using canonical SC votes.

ID_FIX: proofwriter_full.json uses _Q8 for two questions, but the SC votes
file uses _Q4. Apply mapping: _Q8 -> _Q4 for vote lookup.

Combo method: HARD vote -- add w to NLI predicted class (not soft probs).
"""

import json

CLASSES = sorted(["True", "False", "Unknown"])
WEIGHTS = [1, 3, 5]

ID_FIX = {
    "ProofWriter_AttNeg-OWA-D5-1176_Q8": "ProofWriter_AttNeg-OWA-D5-1176_Q4",
    "ProofWriter_AttNoneg-OWA-D5-1284_Q8": "ProofWriter_AttNoneg-OWA-D5-1284_Q4",
}

with open("/root/symb_invariant_consensus/data/proofwriter_full.json") as f:
    pw_data = json.load(f)
with open("/root/symb_invariant_consensus/data/mistral_pw600_sc_votes.json") as f:
    sc_votes_raw = json.load(f)
with open("/root/symb_invariant_consensus/results/deberta_ft_combo_all_fixed/results.json") as f:
    ft_results = json.load(f)
with open("/root/symb_invariant_consensus/results/r11_3verifier_canonical.json") as f:
    r11 = json.load(f)

gt = {q["id"]: q["answer"] for q in pw_data}
pids = sorted(gt.keys())
n = len(pids)
assert n == 600

def get_sc_votes(pid):
    lookup = ID_FIX.get(pid, pid)
    v = sc_votes_raw.get(lookup, [0, 0, 0])
    return {"True": v[0], "False": v[1], "Unknown": v[2]}

def sc_answer_alpha(votes):
    max_v = max(votes.get(c, 0) for c in CLASSES)
    tied = [c for c in CLASSES if votes.get(c, 0) == max_v]
    return sorted(tied)[0]

def hard_combo(sv, nli_pred, w):
    combo_v = {c: sv.get(c, 0) for c in CLASSES}
    combo_v[nli_pred] = combo_v.get(nli_pred, 0) + w
    max_cv = max(combo_v.get(c, 0) for c in CLASSES)
    tied_c = [c for c in CLASSES if combo_v.get(c, 0) == max_cv]
    return sorted(tied_c)[0]

sc_votes = {pid: get_sc_votes(pid) for pid in pids}
sc_answers = {pid: sc_answer_alpha(sc_votes[pid]) for pid in pids}
sc_correct_n = sum(1 for pid in pids if sc_answers[pid] == gt[pid])
sc_acc = sc_correct_n / n
print(f"Canonical SC: {sc_correct_n}/{n} = {sc_acc*100:.2f}%")
assert sc_correct_n == 236, f"Expected 236, got {sc_correct_n}"

# Verify against r11 per-question sc_answers
r11_pq = r11["per_question_preds"]
mismatches = 0
for pid in pids:
    if sc_answers[pid] != r11_pq[pid]["sc_answer"]:
        mismatches += 1
        print(f"  SC MISMATCH: {pid}: computed={sc_answers[pid]} r11={r11_pq[pid]['sc_answer']}")
assert mismatches == 0, f"SC mismatches: {mismatches}"

# Verify zero-shot DeBERTa hard-vote combo w=3 = 276/600 (r11 canonical)
deb_preds = {pid: r11_pq[pid]["deberta"] for pid in pids}
deb_combo_correct = sum(1 for pid in pids
    if hard_combo(sc_votes[pid], deb_preds[pid], 3) == gt[pid])
assert deb_combo_correct == 276, f"Expected 276, got {deb_combo_correct}"
print(f"Verified: zero-shot DeBERTa w=3 hard-combo = {deb_combo_correct}/600 = 46.00%")

# ============ TASK 1: FT DeBERTa combo ============
print("\n" + "="*60)
print("TASK 1: FT DeBERTa Combo (canonical SC votes)")
print("="*60)

ft_pq = ft_results["per_question"]
ft_preds = {}
for pid in pids:
    if pid in ft_pq:
        ft_preds[pid] = ft_pq[pid]["nli_pred"]
    else:
        ft_preds[pid] = None
        print(f"  WARNING: {pid} not in FT results")

ft_correct = sum(1 for pid in pids if ft_preds[pid] is not None and ft_preds[pid] == gt[pid])
print(f"FT DeBERTa standalone: {ft_correct}/{n} ({ft_correct/n*100:.2f}%)")

nli_right_sc_wrong = sum(1 for pid in pids
    if ft_preds[pid] is not None and ft_preds[pid] == gt[pid] and sc_answers[pid] != gt[pid])
sc_right_nli_wrong = sum(1 for pid in pids
    if sc_answers[pid] == gt[pid] and ft_preds[pid] is not None and ft_preds[pid] != gt[pid])
print(f"Complementarity: {nli_right_sc_wrong}:{sc_right_nli_wrong}")

print("\nCombo results (canonical SC + FT DeBERTa):")
old_vals = {1: (42.33, 3.00), 3: (52.17, 12.83), 5: (65.50, 26.17)}
for w in WEIGHTS:
    combo_correct = 0
    for pid in pids:
        sv = sc_votes[pid]
        pred = ft_preds[pid]
        if pred is not None:
            combo_ans = hard_combo(sv, pred, w)
        else:
            combo_ans = sc_answer_alpha(sv)
        if combo_ans == gt[pid]:
            combo_correct += 1
    new_acc = combo_correct / n * 100
    new_delta = new_acc - sc_acc * 100
    old_acc, old_delta = old_vals[w]
    change = new_acc - old_acc
    print(f"  w={w}: {new_acc:.2f}% ({combo_correct}/{n}), delta={new_delta:+.2f}pp"
          f"  [old: {old_acc:.2f}% d={old_delta:+.2f} | change={change:+.2f}pp]")

# ============ TASK 2: DRC tau=0.7 ============
print("\n" + "="*60)
print("TASK 2: DRC tau=0.7 (zero-shot DeBERTa MNLI, w=3, hard-vote)")
print("="*60)

# Full hard-vote combo (zero-shot DeBERTa, w=3)
full_combo_answers = {}
for pid in pids:
    full_combo_answers[pid] = hard_combo(sc_votes[pid], deb_preds[pid], 3)

full_combo_correct = sum(1 for pid in pids if full_combo_answers[pid] == gt[pid])
full_combo_acc = full_combo_correct / n
print(f"Full combo (zero-shot DeBERTa w=3): {full_combo_correct}/{n} = {full_combo_acc*100:.2f}%")

# DRC sweep
print("\nDRC tau sweep:")
for tau in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    drc_correct = 0
    nli_calls = 0
    for pid in pids:
        sv = sc_votes[pid]
        K = sum(sv.values())
        max_count = max(sv.values()) if K > 0 else 0
        agreement = max_count / K if K > 0 else 0

        if agreement >= tau:
            pred = sc_answers[pid]
        else:
            pred = full_combo_answers[pid]
            nli_calls += 1

        if pred == gt[pid]:
            drc_correct += 1

    drc_acc = drc_correct / n
    nli_cost = nli_calls / n * 100
    delta_drc = (drc_acc - sc_acc) * 100
    delta_full = (full_combo_acc - sc_acc) * 100
    recovery = delta_drc / delta_full * 100 if delta_full > 0 else 0
    marker = " <---" if tau == 0.7 else ""
    print(f"  tau={tau:.1f}: {drc_acc*100:.2f}% ({drc_correct}/{n}), "
          f"NLI={nli_calls}/{n} ({nli_cost:.1f}%), Recovery={recovery:.1f}%{marker}")

# tau=0.7 comparison
tau = 0.7
drc_correct = 0
nli_calls = 0
for pid in pids:
    sv = sc_votes[pid]
    K = sum(sv.values())
    max_count = max(sv.values()) if K > 0 else 0
    agreement = max_count / K if K > 0 else 0
    if agreement >= tau:
        pred = sc_answers[pid]
    else:
        pred = full_combo_answers[pid]
        nli_calls += 1
    if pred == gt[pid]:
        drc_correct += 1

drc_acc = drc_correct / n
nli_cost_pct = nli_calls / n * 100
delta_drc = (drc_acc - sc_acc) * 100
delta_full = (full_combo_acc - sc_acc) * 100
recovery = delta_drc / delta_full * 100 if delta_full > 0 else 0

print(f"\nDRC tau=0.7 comparison:")
print(f"  Accuracy:  old=45.33%    -> new={drc_acc*100:.2f}%")
print(f"  NLI calls: old=233/600   -> new={nli_calls}/{n}")
print(f"  NLI cost:  old=38.8%     -> new={nli_cost_pct:.1f}%")
print(f"  Recovery:  old=90.0%     -> new={recovery:.1f}%")
print(f"  Full combo: old=46.00%   -> new={full_combo_acc*100:.2f}%")

print("\nDone.")
