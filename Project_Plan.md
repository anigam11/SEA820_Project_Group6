
# **SEA820 Final Project Plan (Phase 2: Transformer Fine-Tuning)**

**Project Title**: Detecting AI-Generated Text Using Transformer Models  
**Dataset**: Preprocessed (CSV), located in `NLP/preprocessed_dataset.csv` (Google Drive)  
**Baseline Chosen**: Naive Bayes - 97.07% Acccuracy

### Phase 2 Deliverables

- Fine-tuned DistilBERT model  
- Evaluation and baseline comparison  
- Error analysis  
- Ethical considerations  
- Final report and code  
- Presentation slides

### Task Breakdown and Timeline

| Week | Task                             | Details                                                  | Assigned To |
|------|----------------------------------|----------------------------------------------------------|--------------|
| 2    | Load Dataset into HuggingFace   | Convert CSV into DatasetDict                             | Abhi         |
| 2    | Model Selection 1               | Use `DistilBERT`;                                        | Arnav        |
| 2    | Model Selection 2               | Use `Roberta`                                            | Abhi         |
| 2    | Fine-tuning Experiments         | Tune learning rate, batch size, number of epochs         | Both         |
| 2–3  | Evaluation                      | Calculate accuracy, precision, recall, and F1            | Both         |
| 3    | Error Analysis                  | Analyze common false positives and false negatives       | Abhi         |
| 3    | Ethics Report                   | Discuss risks, fairness, bias, and potential harm        | Arnav        |
| 3    | Final Report and Code           | Clean and comment code, write `README.md`, `report.md`   | Both         |
| 3    | Presentation                    | Create 10 minute summary slides and prepare presentation | Both         |

---

### Technical Stack

- **Metrics**: Accuracy, Precision, Recall, F1  
- **Baseline for Comparison**: Naive bayes (Accuracy: 97.07%)

---

### Success Criteria

- Transformer model must match or exceed Logistic Regression baseline performance  
- Codebase must be modular, documented, and reproducible  
- Report includes methodology, results, error analysis, and ethical reflection  
- Clear examples of classification errors with interpretation  
