"""FOLIO zero-shot DeBERTa combo audit with D061 tiebreaking."""
import json, time, os, sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.stats import binomtest
from collections import Counter

FOLIO_DATA = "/root/symb_invariant_consensus/data/folio_full.json"
DEBERTA_PATH = "/root/autodl-tmp/models/deberta-large-mnli"
CLASSES = sorted(["True", "False", "Unknown"])
WEIGHTS = [1, 3, 5]

FOLIO_TRACES = {
    "mistral": {
        "path": "/root/symb_invariant_consensus/results/exp033_mistral_7b_folio204/exp033_results.json",
        "name": "Mistral-7B",
    },
    "llama8b": {
        "path": "/root/symb_invariant_consensus/results/exp-063-llama8b-folio204-16639/results.json",
        "name": "LLaMA-3.1-8B",
    },
    "qwen14b": {
        "path": "/root/symb_invariant_consensus/results/exp036_qwen25_14b_folio204/results.json",
        "name": "Qwen2.5-14B",
    },
    "qwen3": {
        "path": "/root/symb_invariant_consensus/results/exp_folio_2x2_qwen3/results.json",
        "name": "Qwen3-14B",
    },
}

def split_folio_problem(text):
    for marker in [
        "Determine whether the following conclusion is true, false, or uncertain:\n",
        "Determine whether the following conclusion is True, False, or Unknown:\n",
        "Determine whether the following conclusion is true, false, or unknown:\n",
    ]:
        idx = text.find(marker)
        if idx != -1:
            premise = text[:idx].strip()
            for prefix in ["Given the following premises:\n", "Given the following premises:"]:
                if premise.startswith(prefix):
                    premise = premise[len(prefix):].strip()
            hypothesis = text[idx + len(marker):].strip()
            return premise, hypothesis
    return text, ""

def d061_sc(votes):
    if not votes:
        return None
    valid = {k: v for k, v in votes.items() if v > 0}
    if not valid:
        return None
    max_v = max(valid.values())
    tied = sorted([k for k, v in valid.items() if v == max_v])
    return tied[0]

def mcnemar_exact(a, b):
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    return binomtest(n01, n, 0.5).pvalue, n01, n10

def main():
    with open(FOLIO_DATA) as f:
        folio = json.load(f)
    gt = {p["id"]: p["answer"] for p in folio}
    problems = {p["id"]: p["problem"] for p in folio}
    print(f"Loaded {len(folio)} FOLIO problems", flush=True)

    print(f"\nLoading DeBERTa-large-MNLI from {DEBERTA_PATH}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(DEBERTA_PATH)
    model.eval()

    label_map = {0: "False", 1: "Unknown", 2: "True"}

    nli_preds = {}
    nli_probs_all = {}
    t0 = time.time()

    for i, pid in enumerate(sorted(gt.keys())):
        premise, hypothesis = split_folio_problem(problems[pid])
        if not hypothesis:
            print(f"  WARNING: could not parse {pid}", flush=True)
            continue
        
        inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_idx = probs.argmax().item()
        pred = label_map[pred_idx]
        
        nli_preds[pid] = pred
        nli_probs_all[pid] = {label_map[j]: round(probs[j].item(), 4) for j in range(3)}
        
        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(gt)}", flush=True)

    elapsed = time.time() - t0
    n_nli = len(nli_preds)
    nli_correct = sum(1 for pid in nli_preds if nli_preds[pid] == gt[pid])
    print(f"\nDeBERTa-large-MNLI zero-shot on FOLIO: {nli_correct}/{n_nli} = {100*nli_correct/n_nli:.2f}%")
    print(f"Inference time: {elapsed:.1f}s ({elapsed/n_nli:.3f}s/sample)")

    nli_dist = Counter(nli_preds.values())
    gt_dist = Counter(gt.values())
    print(f"NLI pred distribution: {dict(sorted(nli_dist.items()))}")
    print(f"Gold distribution: {dict(sorted(gt_dist.items()))}")

    print("\n" + "=" * 130)
    print(f"{'Model':<20} {'SC(D061)':>10} {'NLI':>10} {'combo_w1':>10} {'Dw1':>8} {'pw1':>10} {'combo_w3':>10} {'Dw3':>8} {'pw3':>10} {'combo_w5':>10} {'Dw5':>8} {'pw5':>10}")
    print("=" * 130)

    all_results = {
        "nli_standalone": {
            "model": "DeBERTa-large-MNLI (zero-shot)",
            "n": n_nli,
            "accuracy": round(nli_correct / n_nli, 4),
            "correct": nli_correct,
            "inference_time_s": round(elapsed, 1),
            "pred_distribution": dict(sorted(nli_dist.items())),
        }
    }

    for key, info in FOLIO_TRACES.items():
        if not os.path.exists(info["path"]):
            print(f"  {info['name']:<18} FILE NOT FOUND: {info['path']}")
            continue
        
        with open(info["path"]) as f:
            data = json.load(f)
        items = data["results"]
        
        sc_votes_by_id = {}
        for item in items:
            pid = item.get("problem_id", "")
            votes = item.get("sc_vote_distribution", item.get("sc_votes", {}))
            norm_votes = {}
            for k, v in votes.items():
                if k in CLASSES:
                    norm_votes[k] = norm_votes.get(k, 0) + v
            sc_votes_by_id[pid] = norm_votes
        
        common_pids = sorted(set(sc_votes_by_id.keys()) & set(nli_preds.keys()) & set(gt.keys()))
        n_common = len(common_pids)
        
        sc_correct_list = []
        combo_correct = {w: [] for w in WEIGHTS}
        
        for pid in common_pids:
            gold = gt[pid]
            votes = sc_votes_by_id.get(pid, {})
            sc_pred = d061_sc(votes) if votes else None
            nli_pred = nli_preds[pid]
            
            sc_ok = sc_pred == gold if sc_pred else False
            sc_correct_list.append(sc_ok)
            
            for w in WEIGHTS:
                combo_votes = {}
                for c in CLASSES:
                    combo_votes[c] = votes.get(c, 0) + w * (1 if nli_pred == c else 0)
                combo_pred = d061_sc(combo_votes)
                combo_ok = combo_pred == gold if combo_pred else False
                combo_correct[w].append(combo_ok)
        
        sc_acc = sum(sc_correct_list) / n_common
        
        model_result = {
            "generator": info["name"],
            "n": n_common,
            "sc_baseline_d061": round(sc_acc, 4),
            "sc_correct": sum(sc_correct_list),
            "complementarity": {
                "nli_right_sc_wrong": sum(1 for pid in common_pids if nli_preds[pid] == gt[pid] and d061_sc(sc_votes_by_id.get(pid, {})) != gt[pid]),
                "sc_right_nli_wrong": sum(1 for pid in common_pids if nli_preds[pid] != gt[pid] and d061_sc(sc_votes_by_id.get(pid, {})) == gt[pid]),
                "both_right": sum(1 for pid in common_pids if nli_preds[pid] == gt[pid] and d061_sc(sc_votes_by_id.get(pid, {})) == gt[pid]),
                "both_wrong": sum(1 for pid in common_pids if nli_preds[pid] != gt[pid] and d061_sc(sc_votes_by_id.get(pid, {})) != gt[pid]),
            },
        }
        
        for w in WEIGHTS:
            combo_acc = sum(combo_correct[w]) / n_common
            delta = combo_acc - sc_acc
            p, n01, n10 = mcnemar_exact(sc_correct_list, combo_correct[w])
            model_result[f"w{w}"] = {
                "accuracy": round(combo_acc, 4),
                "correct": sum(combo_correct[w]),
                "delta_pp": round(delta * 100, 2),
                "mcnemar_p": round(p, 6),
                "discordant": [n01, n10],
            }
        
        all_results[key] = model_result
        
        w1 = model_result["w1"]
        w3 = model_result["w3"]
        w5 = model_result["w5"]
        
        print(f"{info['name']:<20} {sc_acc*100:>10.2f} {nli_correct/n_nli*100:>10.2f} {w1['accuracy']*100:>10.2f} {w1['delta_pp']:>+8.2f} {w1['mcnemar_p']:>10.4f} {w3['accuracy']*100:>10.2f} {w3['delta_pp']:>+8.2f} {w3['mcnemar_p']:>10.4f} {w5['accuracy']*100:>10.2f} {w5['delta_pp']:>+8.2f} {w5['mcnemar_p']:>10.4f}")

    output_path = "/root/symb_invariant_consensus/results/folio_zeroshot_nli_combo_d061.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")

if __name__ == "__main__":
    main()
