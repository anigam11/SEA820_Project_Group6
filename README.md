# SEA820_Project_Group6
NLP final project to detect AI-generated text

# **AI vs Human Text Classification Project**

## **Overview**
This project detects whether a given text is **human-written** or **AI-generated**, using both:
1. **Classical ML Baseline Models** (Logistic Regression, Naive Bayes with TF-IDF)
2. **Transformer Models** (DistilBERT, RoBERTa fine-tuned with Hugging Face)

We use the **AI vs Human Text** dataset from Kaggle (preprocessed CSV stored in Google Drive).  
The goal is to **outperform the Naive Bayes baseline** from Phase 1 in Phase 2.

---

## **Repository Structure**
```
project/
│
├── phase1_project.ipynb             # Classical ML (Logistic Regression, Naive Bayes)
├── NLP_Part2_DistillBert.ipynb       # Phase 2 – DistilBERT Fine-Tuning
├── NLP_part2_Roberta.ipynb           # Phase 2 – RoBERTa Fine-Tuning
├── data/                             # Dataset folder
│   └── preprocessed_dataset.csv
├── models/                           # Saved fine-tuned models
│   ├── Model_distill/
│   └── Model_roberta/
```

All results are attached in the report, as well as the presentation pdf and also the notebooks consist of them

---

## **Requirements**
Install dependencies in Google Colab or locally:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install transformers datasets
pip install tabulate
```

If using Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## **Dataset**
- **File:** `/content/drive/MyDrive/NLP/preprocessed_dataset.csv`
- **Columns:**
  - `processed_text`: Cleaned text
  - `generated`: Label (0 = Human, 1 = AI)
  - `text_length`: Character length
  - `text`: Original raw text

---

## **Phase 1 – Baseline Models**
1. **Load Dataset**
   ```python
   import pandas as pd
   df = pd.read_csv("/content/drive/MyDrive/NLP/preprocessed_dataset.csv")
   ```

2. **Preprocess with TF-IDF**
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   X = TfidfVectorizer(max_features=15000).fit_transform(df["processed_text"])
   y = df["generated"]
   ```

3. **Train Naive Bayes**
   ```python
   from sklearn.naive_bayes import MultinomialNB
   model_nb = MultinomialNB()
   model_nb.fit(X_train, y_train)
   ```

4. **Evaluate**
   ```python
   from sklearn.metrics import classification_report
   y_pred = model_nb.predict(X_test)
   print(classification_report(y_test, y_pred))
   ```

---

## **Phase 2 – Transformer Fine-Tuning**

### **Step 1: Downsample for Fine-Tuning**
```python
from sklearn.model_selection import train_test_split

df_sampled, _ = train_test_split(df, train_size=100000, stratify=df['generated'], random_state=42)
df_sampled.to_csv("balanced_100k.csv", index=False)
```
We use **100k samples** to balance **training time**.

---

### **Step 2: Tokenization**
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    texts = [str(x) for x in batch["processed_text"]]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=256)
```

---

### **Step 3: Fine-Tune**
#### **DistilBERT**
```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./finetuned-distilbert",
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none"
)
```

#### **RoBERTa**
```python
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

training_args = TrainingArguments(
    output_dir="./finetuned-roberta",
    learning_rate=3e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=2,
    weight_decay=0.05,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none"
)
```

---

## **Evaluation & Error Analysis**
```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

predictions = trainer.predict(ds_val)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

print(classification_report(y_true, y_pred, target_names=["Human", "AI"]))
print(confusion_matrix(y_true, y_pred))
```

---

## **Testing on External Samples**
```python
samples = [
    "The school times were the best...",
    "Climate change continues to pose..."
]
encoded = tokenizer(samples, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
outputs = model(**encoded)
probs = torch.softmax(outputs.logits, dim=1)
```

---

## **Saving and Loading Models**
```python
# Save
trainer.save_model("/content/drive/MyDrive/NLP/Model_distill")
tokenizer.save_pretrained("/content/drive/MyDrive/NLP/Model_distill")

# Load
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained("/content/drive/MyDrive/NLP/Model_distill")
tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/NLP/Model_distill")
```

---

## **Results Summary**
- **Baseline Naive Bayes:** 97.07% Accuracy  
- **DistilBERT Fine-Tuned:** ~99.71% Accuracy  
- **RoBERTa Fine-Tuned:** ~99.66% Accuracy  
- Outperformed baseline significantly in **all metrics**.

---

## **Ethical Considerations**
- Potential bias against **non-native speakers**
- Misuse for censorship or false accusations
- Important to **evaluate on diverse datasets**

---

## **How to Run**
1. Open the `.ipynb` file in **Google Colab**  
2. Mount Google Drive  
3. Install dependencies  
4. Load dataset  
5. Run cells in sequence for chosen model (DistilBERT / RoBERTa)  
6. Evaluate & save results

---

## **Authors**
- **Abhi Patel**
- **Arnav Nigam**

