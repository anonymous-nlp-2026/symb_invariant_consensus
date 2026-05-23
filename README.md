# Symbolic Invariant Constraint Aggregation (SICA)

Code for studying self-verification in LLMs through symbolic invariant constraints.

## Overview

SICA is a method that improves LLM reasoning accuracy by:
1. Generating multiple reasoning traces via sampling
2. Extracting symbolic invariant constraints from each trace
3. Using Z3 MAX-SAT solving to find the answer that satisfies the most constraints
4. Aggregating results through constraint-level consensus rather than answer-level voting

This approach decouples verification from generation, enabling cross-model verification and providing interpretable explanations for answer selection.

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

### Cross-Dataset Replication

```bash
python run_cross_dataset_replication.py --dataset folio --k 12
```

### Analysis

```bash
# K ablation study
python analysis_k_ablation_folio204.py

# Constraint ablation
python analysis_constraint_ablation_folio204.py

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
├── run_full_mvp.py          # Main experiment entry point
├── run_icg.py               # ICG experiments
├── run_debate.py            # Debate baseline
├── run_contrastive.py       # Contrastive verification
└── ...                      # Additional experiment scripts
```

## License

This code is provided for review purposes.
