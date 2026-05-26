# Beyond Self-Consistency: Self-Extracted Constraints Reduce to Majority Voting under Zero-Shot Extraction

Code and data for studying symbolic confirmation bias in LLM self-verification.

## Overview

Self-Consistency (SC) improves LLM reasoning by majority voting over chain-of-thought traces. A natural extension extracts logical constraints from traces and aggregates them with a symbolic solver. We test this through Symbolic Invariant Consensus Aggregation (SICA), which extracts first-order logic constraints under zero-shot prompting and solves weighted partial MAX-SAT via Z3.

**Key finding:** Self-extracted constraints systematically confirm their source trace's answer rather than providing independent verification -- a phenomenon we term *symbolic confirmation bias* (SCB). Under zero-shot self-extraction, constraint-based aggregation degenerates to majority voting. This bias persists across nine models from 7B to frontier scale. Twenty-seven training-free remediation strategies fail to break this bottleneck. Only a structurally independent cross-architecture verifier restores corrective signal.

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Data Preparation

Download the raw datasets and run the preparation scripts:

```bash
# ProofWriter
python data/prepare_proofwriter.py

# FOLIO
python data/prepare_folio.py

# LogiQA
python data/prepare_logiqa.py
```

See `data/README.md` for dataset sources and details.

## Running Experiments

### Main SICA Pipeline

```bash
# Run SICA with vLLM backend
python run_full_mvp.py --mode vllm --k 12

# Resume from checkpoint
python run_full_mvp.py --mode vllm --k 12 --resume
```

### Independent Constraint Generation (ICG)

```bash
python run_icg.py --dataset folio --model mistral-7b --k 12
```

### Baselines

```bash
# Debate baseline
python run_debate.py --dataset folio --model mistral-7b

# Contrastive verification
python run_contrastive.py --dataset proofwriter --model mistral-7b
```

### Cross-Architecture NLI Verification

```bash
python direction_r_nli_verifier.py --dataset proofwriter --model mistral-7b
```

### Cross-Dataset Replication

```bash
python run_cross_dataset_replication.py --dataset folio --k 12
```

### Analysis

```bash
# Confirmation bias analysis
python analysis_confirmation_bias.py

# K ablation study
python analysis_k_ablation_folio204.py

# Constraint ablation
python analysis_constraint_ablation_folio204.py

# Process-level kappa
python compute_process_kappa_multi.py

# Statistical significance (McNemar's test)
python mcnemar_recompute.py
```

## Project Structure

```
.
├── sica/                    # Core SICA library
│   ├── pipeline.py          # Main SICA pipeline
│   ├── trace_generator.py   # LLM trace generation (vLLM/API)
│   ├── constraint_extractor.py  # Symbolic constraint extraction
│   ├── z3_maxsat.py         # Z3-based MAX-SAT solver
│   ├── scorer.py            # Scoring and evaluation
│   └── z3_feedback.py       # Z3 feedback loop
├── baselines/               # Baseline methods
│   └── self_consistency.py  # Self-Consistency (majority vote)
├── utils/                   # Utility functions
│   └── math_equiv.py        # Mathematical equivalence checking
├── data/                    # Data preparation scripts
├── scripts/                 # Experiment and analysis scripts
├── figures/                 # Figure generation scripts
├── docs/paper/              # Paper LaTeX source
├── run_full_mvp.py          # Main experiment entry point
├── run_icg.py               # ICG experiments
├── run_debate.py            # Debate baseline
├── run_contrastive.py       # Contrastive verification
├── direction_r_nli_verifier.py  # Cross-architecture NLI verification
└── ...                      # Additional experiment scripts
```

## License

This code is provided for review purposes.
