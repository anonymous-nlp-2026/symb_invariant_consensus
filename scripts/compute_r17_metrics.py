import json, os, math

RESULTS_DIR = "./results"

CONDITIONS = {
    "Mistral-7B x FOLIO": os.path.join(RESULTS_DIR, "exp033_mistral_7b_folio204/exp033_results.json"),
    "Mistral-7B x PW": os.path.join(RESULTS_DIR, "multi_seed/mistral_pw_seed42/results.json"),
    "Qwen2.5-14B x FOLIO": os.path.join(RESULTS_DIR, "exp036_qwen25_14b_folio204/results.json"),
    "Qwen2.5-14B x PW": os.path.join(RESULTS_DIR, "exp032_qwen25_14b_pw600/results.json"),
    "LLaMA-8B x FOLIO": os.path.join(RESULTS_DIR, "exp-063-llama8b-folio204-16639/results.json"),
    "LLaMA-8B x PW": os.path.join(RESULTS_DIR, "exp048_llama8b_pw600/exp048_results.json"),
}

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def median_val(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0: return 0.0
    if n % 2 == 1: return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0

def pearson_r(xs, ys):
    n = len(xs)
    if n < 2: return float('nan')
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0: return float('nan')
    return num / (dx * dy)

def get_argmax(d):
    if not d: return None
    max_v = max(d.values())
    return sorted([k for k, v in d.items() if v == max_v])[0]

print("=" * 70)
print("TASK 1: Argmax Agreement Rate (SICA argmax vs SC argmax)")
print("=" * 70)

all_agreement = {}
for cond, fpath in CONDITIONS.items():
    if not os.path.exists(fpath):
        print("  %s: FILE NOT FOUND (%s)" % (cond, fpath))
        continue
    with open(fpath) as f:
        data = json.load(f)
    results = data["results"]
    n = len(results)
    agree = 0
    n_valid = 0
    for r in results:
        sica_scores = r.get("sica_scores", {})
        sc_dist = r.get("sc_vote_distribution", {})
        if not sica_scores or not sc_dist:
            continue
        n_valid += 1
        sica_argmax = get_argmax(sica_scores)
        sc_argmax = get_argmax(sc_dist)
        if sica_argmax == sc_argmax:
            agree += 1
    rate = agree / n_valid * 100 if n_valid else 0
    all_agreement[cond] = {"rate": rate, "agree": agree, "n": n_valid, "total": n}
    print("  %s: %.1f%% (%d/%d)" % (cond, rate, agree, n_valid))

total_agree = sum(v["agree"] for v in all_agreement.values())
total_n = sum(v["n"] for v in all_agreement.values())
print("\n  OVERALL: %.1f%% (%d/%d)" % (total_agree/total_n*100, total_agree, total_n))

print("\n" + "=" * 70)
print("TASK 2: Base-Rate Predicted BR vs Observed BR (SC-incorrect only)")
print("=" * 70)

task2_summary = {}
for cond, fpath in CONDITIONS.items():
    if not os.path.exists(fpath):
        continue
    with open(fpath) as f:
        data = json.load(f)
    results = data["results"]
    predicted_brs = []
    observed_brs = []
    n_incorrect = 0
    n_unanimous = 0
    for r in results:
        if r.get("sc_correct", True):
            continue
        n_incorrect += 1
        sica_scores = r.get("sica_scores", {})
        sc_dist = r.get("sc_vote_distribution", {})
        if not sica_scores or not sc_dist:
            continue
        sc_majority = get_argmax(sc_dist)
        n_majority = sc_dist.get(sc_majority, 0)
        other_counts = [v for k, v in sc_dist.items() if k != sc_majority]
        if not other_counts or max(other_counts) == 0:
            n_unanimous += 1
            continue
        max_other_count = max(other_counts)
        pred_br = n_majority / max_other_count
        score_majority = sica_scores.get(sc_majority, 0)
        other_scores = [v for k, v in sica_scores.items() if k != sc_majority]
        if not other_scores or max(other_scores) == 0:
            continue
        max_other_score = max(other_scores)
        obs_br = score_majority / max_other_score
        predicted_brs.append(pred_br)
        observed_brs.append(obs_br)
    n_valid = len(predicted_brs)
    if n_valid > 0:
        pred_m = mean(predicted_brs)
        obs_m = mean(observed_brs)
        excess = obs_m - pred_m
        corr = pearson_r(predicted_brs, observed_brs)
        task2_summary[cond] = {
            "n_incorrect": n_incorrect,
            "n_unanimous": n_unanimous,
            "n_valid": n_valid,
            "pred_mean": round(pred_m, 4),
            "obs_mean": round(obs_m, 4),
            "excess": round(excess, 4),
            "corr": round(corr, 4),
            "pred_median": round(median_val(predicted_brs), 4),
            "obs_median": round(median_val(observed_brs), 4),
        }
        print("  %s:" % cond)
        print("    N_incorrect=%d, N_unanimous=%d, N_valid=%d" % (n_incorrect, n_unanimous, n_valid))
        print("    Predicted BR (mean): %.4f  (median: %.4f)" % (pred_m, median_val(predicted_brs)))
        print("    Observed BR  (mean): %.4f  (median: %.4f)" % (obs_m, median_val(observed_brs)))
        print("    Excess (obs - pred): %.4f" % excess)
        print("    Pearson r:           %.4f" % corr)
    else:
        print("  %s: no valid (N_incorrect=%d, N_unanimous=%d)" % (cond, n_incorrect, n_unanimous))

print("\n" + "=" * 70)
print("BONUS: When SICA and SC disagree, who is correct?")
print("=" * 70)

for cond, fpath in CONDITIONS.items():
    if not os.path.exists(fpath):
        continue
    with open(fpath) as f:
        data = json.load(f)
    results = data["results"]
    sica_wins = sc_wins = both_wrong = n_disagree = 0
    for r in results:
        sica_scores = r.get("sica_scores", {})
        sc_dist = r.get("sc_vote_distribution", {})
        if not sica_scores or not sc_dist:
            continue
        sica_argmax = get_argmax(sica_scores)
        sc_argmax = get_argmax(sc_dist)
        if sica_argmax != sc_argmax:
            n_disagree += 1
            gt = r.get("ground_truth", "")
            if sica_argmax == gt:
                sica_wins += 1
            elif sc_argmax == gt:
                sc_wins += 1
            else:
                both_wrong += 1
    if n_disagree > 0:
        print("  %s: %d disagree | SICA correct: %d, SC correct: %d, both wrong: %d" % (cond, n_disagree, sica_wins, sc_wins, both_wrong))
    else:
        print("  %s: 0 disagreements" % cond)

output = {"argmax_agreement": all_agreement, "br_decomposition": task2_summary}
out_path = os.path.join(RESULTS_DIR, "r17_argmax_br_metrics.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print("\nSaved to %s" % out_path)
