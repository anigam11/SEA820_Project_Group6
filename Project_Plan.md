# SEA820 Final Project Plan – Phase 2: Transformer Fine-Tuning

**Project Title**: Detecting AI-Generated Text Using Transformer Models  
**Dataset**: Preprocessed CSV (`NLP/preprocessed_dataset.csv`)  
**Baseline Model**: Naive Bayes (F1-score: 0.9685)


## Project Overview

This phase explores transformer-based models (**DistilBERT** and **RoBERTa**) to classify text as **human-written** or **AI-generated**. The objective is to **outperform our Naive Bayes baseline**, which demonstrated stronger real-world generalization despite slightly lower test set metrics compared to logistic regression. This phase also includes performance analysis, error inspection, and ethical considerations related to AI-generated content detection.


## Task Assignment & Plan (2 Weeks)

| Week | Task                      | Description                                                                 | Assigned To    |
|------|---------------------------|-----------------------------------------------------------------------------|----------------|
| 1    | Dataset Preparation       | Load preprocessed CSV into HuggingFace `DatasetDict`                        | Both           |
| 1    | Tokenizer Setup           | Apply padding/truncation using FastTokenizers for both models               | Both           |
| 1    | Fine-Tuning DistilBERT    | Train `distilbert-base-uncased` on 100k–150k balanced subset                | **Abhi**       |
| 1    | Fine-Tuning RoBERTa       | Train `roberta-base` on the same balanced dataset                           | **Arnav**      |
| 2    | Evaluation & Metrics      | Independently evaluate both models (acc, precision, recall, F1)             | Both (separate)|
| 2    | Error Analysis            | Identify misclassifications, inspect false positives/negatives              | **Abhi**       |
| 2    | Ethics Discussion         | Analyze model risks, fairness, and unintended harms                         | **Arnav**      |
| 2    | Model Comparison          | Compare transformer results to Naive Bayes baseline                         | Both           |
| 2    | Report Writing            | Jointly draft final report (methods, graphs, analysis, discussion)          | Both           |
| 2    | Code Cleanup & README     | Finalize codebase, document structure, add usage guide                      | Both           |
| 2    | Presentation (10-12 mins)   | Build slides and rehearse presentation together                             | Both           |


## To look and compare with:
- **Metrics**: Accuracy, Precision, Recall, F1
- **Baseline**: Naive Bayes (TF-IDF-based model)



## Deliverables

- Evaluation outputs on validation and real-world samples  
- Error analysis and ethical reflection writeups  
- Final `report` and `ppt`  
- `README.md` with environment setup and training instructions


## Success Criteria

- Transformer models match or exceed the Naive Bayes baseline (F1 ≥ 0.9685)  
- Each member independently completes assigned model training and evaluation  
- Final report includes metrics, error insights, and ethical analysis  
- Sampling-based validation shows generalization to human-authored and AI content  
- Presentation clearly communicates motivation, approach, baseline, and outcomes
