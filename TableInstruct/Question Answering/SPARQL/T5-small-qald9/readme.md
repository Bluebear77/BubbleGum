# SPARQL-5-small-qald9

This directory contains SPARQL query outputs generated using the `yazdipour/text-to-sparql-t5-small-qald9` model, applied to random samples from the TableInstruct benchmark.

## Dataset Description

The inputs are 50 randomly sampled questions from various subsets of the TableInstruct dataset, which comprises QA pairs grounded in structured tables. Files include:

- `fetaqa_test-50.csv`
- `fetaqa_train_7325-50.csv`
- `hitab_test-50.csv`
- `hitab_train_7417-50.csv`
- `hybridqa_eval-...`

Each CSV includes:
- `question`: natural language question.
- `sparql`: model-generated SPARQL query.

## Model Overview

- **Model**: `yazdipour/text-to-sparql-t5-small-qald9`
- **Architecture**: T5-small (60M parameters)
- **Training Data**: QALD-9 (primarily) and LC-QuAD
- **Task**: English-to-SPARQL semantic parsing
- **Target KB**: DBpedia

## Purpose

This data was produced to evaluate the generalization capacity of a QALD-trained SPARQL semantic parser on tabular QA inputs. It supports analysis of model robustness across domains and schema.

## Generation Details

SPARQL queries were generated via a Python script using Hugging Face Transformers, processing each file independently and saving the results with identical filenames in this directory.

## Dependencies

- Python ≥ 3.8  
- `transformers`, `torch`, `pandas`, `tqdm`  
- Model: [`yazdipour/text-to-sparql-t5-small-qald9`](https://huggingface.co/yazdipour/text-to-sparql-t5-small-qald9)


