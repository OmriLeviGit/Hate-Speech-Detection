import os
import numpy as np
from collections import Counter

import torch
import torch.nn as nn

from .base_dl_model import BaseDLModel


# Dynamically resolve the GloVe path relative to this file
base_dir = os.path.dirname(os.path.abspath(__file__))
glove_path = os.path.join(base_dir, "..", "GloVe", "glove.6B.300d.txt")
glove_path = os.path.abspath(glove_path)


class LSTMModel(BaseDLModel):

    def __init__(self, params):
        super().__init__(params)
        self.word2idx = {}
        self.embedding_matrix = None
        self.vocab_size = None

    # Tokenizes text by simple whitespace split (assumes normalized input)
    def tokenize(self, text):
        return text.split()


    def preprocess(self, X_raw, y):

        max_vocab_size = 10000
        max_len = self.params["max_sequence_length"]
        embedding_dim = self.params["embedding_dim"]

        counter = Counter()
        tokenized_texts = []

        for text in X_raw:
            tokens = self.tokenize(text)
            tokenized_texts.append(tokens)
            counter.update(tokens)

        # Build word-to-index mapping from most frequent tokens in training data
        # <PAD> = 0, <OOV> = 1 for padding and unknown words
        most_common = [word for word, _ in counter.most_common(max_vocab_size)]
        self.word2idx = {word: idx + 2 for idx, word in enumerate(most_common)}
        self.word2idx["<PAD>"] = 0
        self.word2idx["<OOV>"] = 1
        self.vocab_size = len(self.word2idx)

        # Encode each token list into a fixed-length sequence of indices
        def encode(tokens):
            ids = [self.word2idx.get(token, 1) for token in tokens]
            return ids[:max_len] + [0] * max(0, max_len - len(ids))

        X_sequences = [encode(tokens) for tokens in tokenized_texts]
        self.embedding_matrix = self._load_glove_embeddings(glove_path, embedding_dim)

        return np.array(X_sequences, dtype=np.int64), y


    # Tokenizes and encodes new text examples using the already-built vocabulary
    def transform(self, X_raw, y=None):

        def encode(tokens):
            ids = [self.word2idx.get(token, 1) for token in self.tokenize(tokens)]
            return ids[:self.params["max_sequence_length"]] + [0] * max(0, self.params["max_sequence_length"] - len(ids))

        X_sequences = [encode(text) for text in X_raw]

        return (np.array(X_sequences, dtype=np.int64), y) if y is not None else np.array(X_sequences, dtype=np.int64)


    # Load pretrained GloVe vectors and initialize embedding matrix
    # Words not found in GloVe will use random embeddings (OOV - out of vocabulary)
    def _load_glove_embeddings(self, glove_path, embedding_dim):
        embeddings_index = {}

        with open(glove_path, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = vector

        matrix = np.random.uniform(-0.05, 0.05, (self.vocab_size, embedding_dim))

        covered = 0

        for word, idx in self.word2idx.items():
            if idx < self.vocab_size:
                vector = embeddings_index.get(word)
                if vector is not None:
                    matrix[idx] = vector
                    covered += 1

        print(f"[GloVe] Covered {covered}/{self.vocab_size} tokens ({(covered / self.vocab_size) * 100:.2f}%)")

        return torch.tensor(matrix, dtype=torch.float32)


    def build(self, input_shape):

        return LSTMNetwork(
            vocab_size=self.vocab_size,
            embedding_dim=self.params['embedding_dim'],
            lstm_units=self.params['lstm_units'],
            dropout_rate=self.params['dropout_rate'],
            dense_units=self.params['dense_units'],
            dense_activation=self.params['dense_activation'],
            embedding_matrix=self.embedding_matrix
        )


    @property
    def input_dtype(self):
        return torch.long


class LSTMNetwork(nn.Module):

    def __init__(self, vocab_size, embedding_dim, lstm_units, dropout_rate, dense_units, dense_activation, embedding_matrix):
        super().__init__()

        # Embedding layer initialized with pretrained GloVe vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = True

        # Bidirectional LSTM with 2 layers and dropout between layers
        self.lstm = nn.LSTM(
            input_size = embedding_dim,
            hidden_size = lstm_units,
            num_layers = 2,
            batch_first = True,
            bidirectional = True,
            dropout = dropout_rate
        )

        # Classification layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(2 * lstm_units, dense_units)
        self.fc2 = nn.Linear(dense_units, 1)
        self.activation = nn.ReLU() if dense_activation == 'relu' else nn.Tanh()


    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.activation(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x