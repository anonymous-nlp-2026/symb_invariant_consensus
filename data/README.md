# Data

This directory contains data preparation scripts for the datasets used in our experiments.

## Datasets

### ProofWriter
- **Source**: [ProofWriter dataset](https://allenai.org/data/proofwriter)
- **Task**: Logical reasoning over natural language rules and facts
- **Preparation**: Run `python data/prepare_proofwriter.py` to generate the formatted dataset

### FOLIO
- **Source**: [FOLIO dataset](https://github.com/Yale-LILY/FOLIO)
- **Task**: First-order logic natural language inference
- **Preparation**: Run `python data/prepare_folio.py` to generate the formatted dataset

### LogiQA
- **Source**: [LogiQA dataset](https://github.com/lgw863/LogiQA-dataset)
- **Task**: Logical reasoning in reading comprehension
- **Preparation**: Run `python data/prepare_logiqa.py` to generate the formatted dataset

## Notes

- Raw datasets are not included in this repository. Please download them from the sources above.
- After downloading, place the raw data files in this directory and run the corresponding preparation scripts.
- The preparation scripts will generate JSON files in the format expected by the experiment scripts.
