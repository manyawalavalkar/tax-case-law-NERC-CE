# Named Entity Recognition and Classification and Concept Extraction in the Legal and Tax Domain

This repository contains the code used to conduct the experiments and produce the results presented in the research master thesis *"Named Entity Recognition and Classification and Concept Extraction in the Legal and Tax Domain."*

The existing datasets used in these experiments can be found on https://github.com/terenceau1/E-NER-Dataset & https://huggingface.co/datasets/AjayMukundS/Indian_Legal_NER_Dataset. As the test set and some code (for the company baseline) used were provided by the IBFD, they cannot be shared or displayed in the output. 

## Repository Structure

The code is organized into the following categories:

### Datasets

| File | Description |
|---|---|
| `caselaw_stats.ipynb` | Creates the combined existing legal NERC dataset |
| `ibfd_dataset.ipynb` | Combines the test subsets into the final test set |
| `final_statistics.ipynb` | Calculates descriptive statistics for the NERC datasets |
| `soup_making_notebook.ipynb` | Runs an LLM to annotate target domain data |

### Gazetteers and Baseline

| File | Description |
|---|---|
| `heuristic_gazeteer.ipynb` | Creates the heuristics-based gazetteer |
| `baseline_comps` | Runs the heuristics-based gazetteer, company gazetteer, and company baseline |

### LLM

| File | Description |
|---|---|
| `LLMs_inference_zeroshot_complete.ipynb` | Performs zero-shot prompting for NERC/CE |
| `LLMs_inference_fewshot_complete.ipynb` | Performs few-shot prompting for NERC/CE |

### Encoders

**RoBERTa**
| File | Description |
|---|---|
| `RoBERTa_finetuning_final.ipynb` | Fine-tunes RoBERTa and evaluates fine-tuned variants |
| `finetune_roberta_ald.py` | Fine-tunes RoBERTa on the approximate domain legal NERC dataset |
| `finetune_roberta_llm.py` | Fine-tunes RoBERTa on LLM-annotated data |
| `finetune_roberta_mixed.py` | Fine-tunes RoBERTa on existing legal data + LLM-annotated data |

**LegalBERT**
| File | Description |
|---|---|
| `LegalBERT_finetuning_testing_final.ipynb` | Fine-tunes LegalBERT and evaluates fine-tuned variants |
| `finetune_legalbert_ald.py` | Fine-tunes LegalBERT on the approximate domain legal  NERC dataset |
| `finetune_legalbert_llm.py` | Fine-tunes LegalBERT on LLM-annotated data |
| `finetune_legalbert_mixed.py` | Fine-tunes LegalBERT on existing legal data + LLM-annotated data |

**GLiNER**
| File | Description |
|---|---|
| `GLiNER_finetuning_testing.ipynb` | Fine-tunes GLiNER and evaluates fine-tuned variants |
| `finetune_gliner1_ald.py` | Fine-tunes GLiNER on the approximate domain legal NERC dataset |
| `finetune_gliner1_llm.py` | Fine-tunes GLiNER on LLM-annotated data |
| `finetune_gliner1_mixed.py` | Fine-tunes GLiNER on existing legal data + LLM-annotated data |

### Evaluation and Error Analysis

| File | Description |
|---|---|
| `FINAL_EVALUATION.ipynb` | Calculates SEQEVAL and NERVALUATE metrics on the model predictions.  Also performs a token- and span-level analysis on the predictions made by GLiNER |
| `IAA_calculator.ipynb` | Calculates the inter-annotator agreement between the test set annotators |