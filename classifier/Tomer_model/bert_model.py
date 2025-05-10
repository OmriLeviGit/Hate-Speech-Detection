import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Step 1: Load Data
# Note that BERT gets raw text as it knows to make tokenization and builds embeddings
df = pd.read_csv("results.csv")
# Use only relevant columns
df = df[['content', 'sentiment']].dropna()

# Optional: Binary sentiment
# df['sentiment'] = df['sentiment'].apply(lambda x: 'Positive' if x == 'Positive' else 'Negative')

# Encode labels
label_map = {label: i for i, label in enumerate(df['sentiment'].unique())}
df['label'] = df['sentiment'].map(label_map)

# Step 2: Train-Test Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['content'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# Step 3: Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Step 4: Dataset
class TweetDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = TweetDataset(train_encodings, train_labels)
val_dataset = TweetDataset(val_encodings, val_labels)

# Step 5: Model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(label_map)
)
model.to(device)

# Step 6: Training Setup
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

optimizer = AdamW(model.parameters(), lr=2e-5)

# Step 7: Training Loop
for epoch in range(3):  # train for 3 epochs
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# Step 8: Evaluation
model.eval()
preds = []
true = []

with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true.extend(batch['labels'].cpu().numpy())

print("\nClassification Report:")
print(classification_report(true, preds, target_names=label_map.keys()))

print("Confusion Matrix:")
print(confusion_matrix(true, preds))

# Step 9: Save Model
model.save_pretrained("bert_sentiment_model")
tokenizer.save_pretrained("bert_sentiment_model")
