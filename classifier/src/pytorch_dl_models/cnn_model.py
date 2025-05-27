import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import spacy
import re
import pandas as pd
from collections import Counter
import emoji

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report


# Load CSV
df = pd.read_csv("results.csv")

# Keep only tweets with Positive or Negative sentiment
df = df[df["sentiment"].isin(["Positive", "Negative"])]

# Convert string sentiment to a number: Positive - 1, Negative - 0
df["label"] = df["sentiment"].map({"Positive": 1, "Negative": 0})

# Balance the dataset
min_count = df["label"].value_counts().min()

balanced_df = pd.concat([
    df[df["label"] == 0].sample(min_count, random_state=42),
    df[df["label"] == 1].sample(min_count, random_state=42)
])

# Shuffle and split
balanced_df = shuffle(balanced_df, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    balanced_df["content"].values,
    balanced_df["label"].values,
    test_size=0.2,
    random_state=42
)

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # Faster

def clean_mentions_urls(text):
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    return text


def strip_hashtags(text):
    return re.sub(r"#(\w+)", r"\1", text)


def convert_emojis(text):
    return emoji.demojize(text, delimiters=(" ", " "))


def preprocess(text):
    text = clean_mentions_urls(text)
    text = strip_hashtags(text)
    text = convert_emojis(text)
    return text

def spacy_tokenizer(text):
    text = preprocess(text)
    doc = nlp(text.lower())
    return [token.text for token in doc if not token.is_space and not token.is_punct]

print("Preprocessing text...")
# Tokenize the training data
tokenized_train = [spacy_tokenizer(text) for text in X_train]

# Build vocab with min_freq to avoid rare words
min_freq = 2
special_tokens = ["<pad>", "<unk>"]
counter = Counter(token for tweet in tokenized_train for token in tweet)

# Vocab dictionary: token → index
vocab = {token: idx for idx, token in enumerate(special_tokens)}
for token, freq in counter.items():
    if freq >= min_freq:
        vocab[token] = len(vocab)

pad_idx = vocab["<pad>"]
unk_idx = vocab["<unk>"]

def encode_tokens(tokens, vocab, max_len):
    ids = [vocab.get(token, unk_idx) for token in tokens]
    ids = ids[:max_len] + [pad_idx] * max(0, max_len - len(ids))
    return ids

MAX_LEN = 50

# Encode training tweets
X_train_encoded = torch.tensor([
    encode_tokens(spacy_tokenizer(text), vocab, MAX_LEN)
    for text in X_train
])
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

# Encode test tweets
X_test_encoded = torch.tensor([
    encode_tokens(spacy_tokenizer(text), vocab, MAX_LEN)
    for text in X_test
])
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


class TweetDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


BATCH_SIZE = 32

train_dataset = TweetDataset(X_train_encoded, y_train_tensor)
test_dataset = TweetDataset(X_test_encoded, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_classes=2, pad_idx=0):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.conv = nn.Conv1d(embed_dim, 100, kernel_size=3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # [B, E, L]
        x = F.relu(self.conv(x)).max(dim=2).values  # [B, 100]
        x = self.dropout(x)
        return self.fc(x)


vocab_size = len(vocab)
model = TextCNN(vocab_size=vocab_size, pad_idx=pad_idx)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)


def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss, total_correct = 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)  # → [batch_size, 2]

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / len(dataloader.dataset)
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, total_correct = 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / len(dataloader.dataset)
    return avg_loss, accuracy

NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")


def get_predictions_and_labels(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return all_preds, all_labels

# Get predictions
preds, labels = get_predictions_and_labels(model, test_loader)

# Print classification report
print(classification_report(labels, preds, target_names=["Negative", "Positive"]))
