import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import joblib
import re
from sklearn.pipeline import Pipeline

# --- Data Loading and Preprocessing ---
# Load data
data = pd.read_csv('/Users/jiahui/helpdesk/MainApp/issues_dataset_balanced.csv')

# Normalize text: remove special characters, handle case, and remove extra spaces
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.lower()

data['description'] = data['description'].apply(preprocess_text)

# Encode labels as integers
label_to_index = {label: idx for idx, label in enumerate(data['issue_type'].unique())}
index_to_label = {idx: label for label, idx in label_to_index.items()}
data['label'] = data['issue_type'].map(label_to_index)

# Split data
X = data['description']
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# --- Random Forest Approach ---
# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1, 2))

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(zip(np.unique(y_train), class_weights))

# Define pipeline with SMOTE and RandomForest
rf_pipeline = ImbPipeline([
    ('tfidf', tfidf),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(class_weight=class_weights_dict, n_estimators=200, random_state=42))
])

# Train the pipeline
rf_pipeline.fit(X_train, y_train)

# Evaluate on test data
rf_y_pred = rf_pipeline.predict(X_test)
print("Random Forest Results:")
print(classification_report(y_test, rf_y_pred, target_names=[index_to_label[i] for i in range(len(index_to_label))]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_y_pred))

# Save the pipeline
joblib.dump(rf_pipeline, 'random_forest_issue_classifier.pkl')

# --- BERT Fine-Tuning ---
class IssueDataset(Dataset):
    def __init__(self, descriptions, labels, tokenizer, max_len):
        self.descriptions = descriptions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        desc = str(self.descriptions[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            desc,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Tokenizer and Dataset Preparation
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
max_len = 128

train_dataset = IssueDataset(X_train.tolist(), y_train.tolist(), tokenizer, max_len)
test_dataset = IssueDataset(X_test.tolist(), y_test.tolist(), tokenizer, max_len)

# Model Setup
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=len(label_to_index))

# Training Arguments
training_args = TrainingArguments(
    output_dir='./bert_model',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=2,
    remove_unused_columns=False,
    load_best_model_at_end=True,  # Automatically load the best model
    metric_for_best_model="eval_loss",  # Use evaluation loss to select the best model
    report_to="none",
)

# Define metrics for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        'accuracy': (predictions == labels).mean(),
        'f1': classification_report(labels, predictions, target_names=[index_to_label[i] for i in range(len(index_to_label))], output_dict=True)['weighted avg']['f1-score']
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
print("BERT Evaluation:")
eval_results = trainer.evaluate()
print(eval_results)

# Save the model
model.save_pretrained('./bert_issue_classifier')
tokenizer.save_pretrained('./bert_issue_classifier')

# --- Summary ---
# Two models:
# - Random Forest: Simpler deployment.
# - BERT: Better contextual understanding for production use.