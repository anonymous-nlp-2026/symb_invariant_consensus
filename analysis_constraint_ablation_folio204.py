"""
Constraint Ablation + SICA K-Ablation on FOLIO 204.
Phase 1: Extract per-trace constraints from stored traces (needs LLM).
Phase 2: Run three aggregation comparisons + K ablation (CPU-only).

Usage:
  python analysis_constraint_ablation_folio204.py --extract   # Phase 1: extract constraints
  python analysis_constraint_ablation_folio204.py --analyze   # Phase 2: run ablations
  python analysis_constraint_ablation_folio204.py --all       # Both phases
"""
import json
import os
import sys
import time
import random
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

random.seed(42)

INTERMEDIATES_DIR = "./results/folio_204_14b/intermediates"
RESULTS_FILE = "./results/folio_204_14b/folio_204_results.json"
CONSTRAINTS_DIR = "./results/folio_204_14b/per_trace_constraints"
K_ABLATION_DIR = "./results/k_ablation_folio204"
CONSTRAINT_ABLATION_DIR = "./results/constraint_ablation_folio204"

API_BASE = "http://localhost:8001/v1"


def normalize_logic_answer(ans):
    ans = str(ans).strip().lower()
    if ans in ("true", "yes", "t"): return "True"
    elif ans in ("false", "no", "f"): return "False"
    elif ans in ("unknown", "uncertain", "u", "undetermined"): return "Unknown"
    return ans.capitalize()


def load_problems():
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    problems = []
    for r in data["results"]:
        pid = r["problem_id"]
        with open(os.path.join(INTERMEDIATES_DIR, f"{pid}.json")) as f:
            intermed = json.load(f)
        problems.append({
            "pid": pid,
            "gt": normalize_logic_answer(r["ground_truth"]),
            "dataset": r["dataset"],
            "traces": intermed["sica_result"]["traces"],
            "sica_answer": normalize_logic_answer(r["sica_answer"]),
            "sica_correct": r["sica_correct"],
        })
    return problems


# ---- Phase 1: Extract per-trace constraints ----

def extract_constraints_for_trace(trace_text, api_base=API_BASE):
    """Extract constraints from a single trace using vLLM API."""
    from sica.constraint_extractor import LOGIC_EXTRACTION_PROMPT
    import httpx

    prompt = LOGIC_EXTRACTION_PROMPT.format(trace=trace_text)
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.1,
    }
    try:
        resp = httpx.post(f"{api_base}/chat/completions", json=payload, timeout=60)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        return parse_extraction_response(raw)
    except Exception as e:
        return {"error": str(e), "constraints": []}


def parse_extraction_response(raw):
    """Parse LLM extraction response into constraints list."""
    import re
    text = raw.strip()
    # Strip thinking tags if present
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Find JSON
    json_match = re.search(r'\{[\s\S]*\}', text)
    if not json_match:
        return {"error": "no JSON found", "constraints": []}
    try:
        data = json.loads(json_match.group())
        constraints = data.get("constraints", [])
        answer = data.get("answer", "")
        return {"constraints": constraints, "extracted_answer": answer}
    except json.JSONDecodeError:
        return {"error": "JSON parse failed", "constraints": []}


def extract_all_constraints(problems, max_workers=2):
    """Extract constraints from all traces of all problems."""
    os.makedirs(CONSTRAINTS_DIR, exist_ok=True)

    total = sum(len(p["traces"]) for p in problems)
    done = 0
    skipped = 0
    t_start = time.time()

    for pi, p in enumerate(problems):
        pid = p["pid"]
        out_file = os.path.join(CONSTRAINTS_DIR, f"{pid}.json")
        if os.path.exists(out_file):
            skipped += len(p["traces"])
            continue

        per_trace = []
        for ti, trace in enumerate(p["traces"]):
            trace_text = trace.get("trace", "")
            if not trace_text.strip():
                per_trace.append({"trace_idx": ti, "answer": trace.get("answer", ""), "constraints": [], "error": "empty trace"})
                done += 1
                continue
            result = extract_constraints_for_trace(trace_text)
            per_trace.append({
                "trace_idx": ti,
                "answer": trace.get("answer", ""),
                "constraints": result.get("constraints", []),
                "extracted_answer": result.get("extracted_answer", ""),
                "error": result.get("error"),
            })
            done += 1

        with open(out_file, "w") as f:
            json.dump({"pid": pid, "gt": p["gt"], "per_trace": per_trace}, f, indent=2)

        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done - skipped) / rate if rate > 0 else 0
        print(f"  [{pi+1}/{len(problems)}] {pid} done ({done + skipped}/{total} traces, {rate:.1f}/s, ETA {eta:.0f}s)")

    print(f"Extraction complete: {done} extracted, {skipped} skipped (cached)")


# ---- Phase 2: Analysis ----

def run_k_ablation_sica(problems):
    """K ablation using SICA (constraint-weighted) and SC baselines."""
    from sica.z3_maxsat import ConstraintDeduplicator, MaxSATSolver
    from sica.scorer import InvariantScorer

    K_VALUES = [3, 4, 6, 8, 11, 12]
    results = {}

    for k in K_VALUES:
        sica_correct = 0
        count_only_correct = 0
        sc_correct = 0
        per_problem = []

        for p in problems:
            pid = p["pid"]
            gt = p["gt"]
            traces_k = p["traces"][:k]

            # SC baseline
            answers = [normalize_logic_answer(t.get("answer", "")) for t in traces_k if t.get("answer")]
            if not answers:
                continue
            vote_counts = Counter(answers)
            sc_ans = max(sorted(vote_counts.keys()), key=lambda x: vote_counts[x])
            sc_ok = (sc_ans == gt)
            if sc_ok:
                sc_correct += 1

            # Load constraints for first-K traces
            cfile = os.path.join(CONSTRAINTS_DIR, f"{pid}.json")
            if not os.path.exists(cfile):
                sica_correct += int(sc_ok)
                count_only_correct += int(sc_ok)
                continue

            with open(cfile) as f:
                cdata = json.load(f)

            per_trace_k = cdata["per_trace"][:k]
            all_constraints = [t.get("constraints", []) for t in per_trace_k]
            trace_dicts = [{"trace_idx": t["trace_idx"], "answer": normalize_logic_answer(t.get("answer", ""))} for t in per_trace_k]
            candidates = sorted(set(td["answer"] for td in trace_dicts if td["answer"]))

            if not candidates:
                sica_correct += int(sc_ok)
                count_only_correct += int(sc_ok)
                continue

            deduplicator = ConstraintDeduplicator()
            unique = deduplicator.deduplicate(all_constraints)

            # SICA (MaxSAT)
            solver = MaxSATSolver()
            ms_result = solver.solve(unique, timeout_ms=10000)
            scorer = InvariantScorer(alpha=0.5)
            ms_scores = scorer.score(ms_result, trace_dicts, candidates)
            sica_ans = scorer.select_answer(ms_scores, vote_counts)
            sica_ok = (sica_ans == gt)
            if sica_ok:
                sica_correct += 1

            # Count-only (all constraints satisfied)
            from sica.z3_maxsat import MaxSATResult
            co_result = MaxSATResult(
                satisfied=list(unique),
                excluded=[],
                total_weight=sum(uc.weight for uc in unique),
                solve_time_ms=0.0,
            )
            co_scores = scorer.score(co_result, trace_dicts, candidates)
            co_ans = scorer.select_answer(co_scores, vote_counts)
            co_ok = (co_ans == gt)
            if co_ok:
                count_only_correct += 1

        n = len(problems)
        results[k] = {
            "sica_acc": round(sica_correct / n, 4),
            "count_only_acc": round(count_only_correct / n, 4),
            "sc_acc": round(sc_correct / n, 4),
            "sica_correct": sica_correct,
            "count_only_correct": count_only_correct,
            "sc_correct": sc_correct,
        }

    return results


def run_constraint_ablation(problems):
    """Compare MaxSAT vs count-only vs random-weight aggregation."""
    from sica.z3_maxsat import ConstraintDeduplicator, MaxSATSolver, MaxSATResult, UniqueConstraint
    from sica.scorer import InvariantScorer

    maxsat_correct = 0
    count_correct = 0
    sc_correct_count = 0
    total = 0
    per_problem = []
    agreement_maxsat_count = 0
    random_correct_trials = []
    agreement_maxsat_random = []

    for p in problems:
        pid = p["pid"]
        gt = p["gt"]

        cfile = os.path.join(CONSTRAINTS_DIR, f"{pid}.json")
        if not os.path.exists(cfile):
            continue

        with open(cfile) as f:
            cdata = json.load(f)

        per_trace = cdata["per_trace"]
        all_constraints = [t.get("constraints", []) for t in per_trace]
        trace_dicts = [{"trace_idx": t["trace_idx"], "answer": normalize_logic_answer(t.get("answer", ""))} for t in per_trace]
        answers = [td["answer"] for td in trace_dicts if td["answer"]]
        if not answers:
            continue

        vote_counts = Counter(answers)
        candidates = sorted(set(answers))
        sc_ans = max(sorted(vote_counts.keys()), key=lambda x: vote_counts[x])
        sc_ok = (sc_ans == gt)
        sc_correct_count += int(sc_ok)

        deduplicator = ConstraintDeduplicator()
        unique = deduplicator.deduplicate(all_constraints)

        solver = MaxSATSolver()
        ms_result = solver.solve(unique, timeout_ms=10000)
        scorer = InvariantScorer(alpha=0.5)
        ms_scores = scorer.score(ms_result, trace_dicts, candidates)
        maxsat_ans = scorer.select_answer(ms_scores, vote_counts)
        maxsat_ok = (maxsat_ans == gt)
        maxsat_correct += int(maxsat_ok)

        co_result = MaxSATResult(
            satisfied=list(unique),
            excluded=[],
            total_weight=sum(uc.weight for uc in unique),
            solve_time_ms=0.0,
        )
        co_scores = scorer.score(co_result, trace_dicts, candidates)
        count_ans = scorer.select_answer(co_scores, vote_counts)
        count_ok = (count_ans == gt)
        count_correct += int(count_ok)

        if maxsat_ans == count_ans:
            agreement_maxsat_count += 1

        # Random weight trials
        trial_correct = []
        trial_agree = []
        for _ in range(10):
            random_unique = []
            for uc in unique:
                rw = random.random() * 10
                random_unique.append(UniqueConstraint(
                    expression=uc.expression,
                    z3_formula=uc.z3_formula,
                    weight=rw,
                    source_traces=list(uc.source_traces),
                ))
            rr = solver.solve(random_unique, timeout_ms=10000)
            rr_scores = scorer.score(rr, trace_dicts, candidates)
            rr_ans = scorer.select_answer(rr_scores, vote_counts)
            trial_correct.append(int(rr_ans == gt))
            trial_agree.append(int(rr_ans == maxsat_ans))

        random_correct_trials.append(trial_correct)
        agreement_maxsat_random.append(trial_agree)

        total += 1
        per_problem.append({
            "pid": pid,
            "gt": gt,
            "maxsat_answer": maxsat_ans,
            "count_answer": count_ans,
            "sc_answer": sc_ans,
        })

    random_accs = [sum(t[i] for t in random_correct_trials) / total for i in range(10)]
    random_agree = [a / total for a in agreement_maxsat_random]

    return {
        "n": total,
        "maxsat_accuracy": maxsat_correct / total if total else 0,
        "count_only_accuracy": count_correct / total if total else 0,
        "random_weight_mean_accuracy": sum(random_accs) / len(random_accs),
        "sc_accuracy": sc_correct_count / total if total else 0,
        "maxsat_correct": maxsat_correct,
        "count_only_correct": count_correct,
        "sc_correct": sc_correct_count,
        "agreement_maxsat_vs_countonly": f"{agreement_maxsat_count}/{total}",
        "agreement_maxsat_vs_randomwt_mean": f"{sum(agreement_maxsat_random)/len(agreement_maxsat_random):.0f}/{total}",
        "per_problem": per_problem,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true", help="Phase 1: extract constraints")
    parser.add_argument("--analyze", action="store_true", help="Phase 2: run ablations")
    parser.add_argument("--all", action="store_true", help="Both phases")
    args = parser.parse_args()

    if not (args.extract or args.analyze or args.all):
        args.all = True

    problems = load_problems()
    print(f"Loaded {len(problems)} problems")

    if args.extract or args.all:
        print("\n=== Phase 1: Extracting per-trace constraints ===")
        extract_all_constraints(problems)

    if args.analyze or args.all:
        # Check if constraints are available
        available = sum(1 for p in problems if os.path.exists(os.path.join(CONSTRAINTS_DIR, f"{p['pid']}.json")))
        print(f"\n=== Phase 2: Analysis ({available}/{len(problems)} problems have constraints) ===")

        if available == 0:
            print("ERROR: No constraints extracted. Run --extract first.")
            return

        # K Ablation with SICA
        print("\n--- K Ablation (SICA + SC) ---")
        k_results = run_k_ablation_sica(problems)

        print(f"\n{'K':>4} | {'SICA':>8} | {'Count-only':>10} | {'SC':>8} | {'Δ(SICA-SC)':>10}")
        print("-" * 55)
        for k in sorted(k_results.keys()):
            r = k_results[k]
            delta = (r["sica_acc"] - r["sc_acc"]) * 100
            print(f"{k:>4} | {r['sica_acc']:.4f}   | {r['count_only_acc']:.4f}     | {r['sc_acc']:.4f}   | {delta:+.2f}pp")

        os.makedirs(K_ABLATION_DIR, exist_ok=True)
        k_out = {str(k): {kk: vv for kk, vv in v.items() if kk != "per_problem"} for k, v in k_results.items()}
        with open(os.path.join(K_ABLATION_DIR, "k_ablation_sica_results.json"), "w") as f:
            json.dump(k_out, f, indent=2)

        # Constraint Ablation
        print("\n--- Constraint Ablation (MaxSAT vs Count-only vs Random-weight) ---")
        ablation = run_constraint_ablation(problems)

        print(f"\n{'Method':>15} | {'Accuracy':>8} | {'Correct':>8}")
        print("-" * 40)
        print(f"{'MaxSAT (SICA)':>15} | {ablation['maxsat_accuracy']:.4f}   | {ablation['maxsat_correct']}")
        print(f"{'Count-only':>15} | {ablation['count_only_accuracy']:.4f}   | {ablation['count_only_correct']}")
        print(f"{'Random-weight':>15} | {ablation['random_weight_mean_accuracy']:.4f}   | -")
        print(f"{'SC baseline':>15} | {ablation['sc_accuracy']:.4f}   | {ablation['sc_correct']}")
        print(f"\nAgreement MaxSAT vs Count-only: {ablation['agreement_maxsat_vs_countonly']}")
        print(f"Agreement MaxSAT vs Random-wt:  {ablation['agreement_maxsat_vs_randomwt_mean']}")

        os.makedirs(CONSTRAINT_ABLATION_DIR, exist_ok=True)
        abl_out = {k: v for k, v in ablation.items() if k != "per_problem"}
        with open(os.path.join(CONSTRAINT_ABLATION_DIR, "constraint_ablation_results.json"), "w") as f:
            json.dump(abl_out, f, indent=2)

        print(f"\nResults saved to {K_ABLATION_DIR} and {CONSTRAINT_ABLATION_DIR}")


if __name__ == "__main__":
    main()
