import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import Counter

from .base_dl_model import BaseDLModel

# Path to GloVe embeddings
base_dir = os.path.dirname(os.path.abspath(__file__))
glove_path = os.path.join(base_dir, "..", "GloVe", "glove.6B.100d.txt")
glove_path = os.path.abspath(glove_path)


class CNNModel(BaseDLModel):

    def __init__(self, params):
        super().__init__(params)
        self.word2idx = {}
        self.embedding_matrix = None
        self.vocab_size = None

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

        most_common = [word for word, _ in counter.most_common(max_vocab_size)]
        self.word2idx = {word: idx + 2 for idx, word in enumerate(most_common)}
        self.word2idx["<PAD>"] = 0
        self.word2idx["<OOV>"] = 1
        self.vocab_size = len(self.word2idx)

        def encode(tokens):
            ids = [self.word2idx.get(token, 1) for token in tokens]
            return ids[:max_len] + [0] * max(0, max_len - len(ids))

        X_sequences = [encode(tokens) for tokens in tokenized_texts]
        self.embedding_matrix = self._load_glove_embeddings(glove_path, embedding_dim)

        return np.array(X_sequences, dtype=np.int64), y

    def transform(self, X_raw, y=None):
        def encode(tokens):
            ids = [self.word2idx.get(token, 1) for token in self.tokenize(tokens)]
            return ids[:self.params["max_sequence_length"]] + [0] * max(0, self.params["max_sequence_length"] - len(ids))

        X_sequences = [encode(text) for text in X_raw]

        return (np.array(X_sequences, dtype=np.int64), y) if y is not None else np.array(X_sequences, dtype=np.int64)

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
        return CNNNetwork(
            vocab_size=self.vocab_size,
            embedding_dim=self.params['embedding_dim'],
            num_filters=self.params['num_filters'],
            kernel_size=self.params['kernel_size'],
            second_conv=self.params.get('second_conv', False),
            dropout_rate=self.params['dropout_rate'],
            dense_units=self.params['dense_units'],
            dense_activation=self.params['dense_activation'],
            max_sequence_length=input_shape,
            embedding_matrix=self.embedding_matrix
        )

    @property
    def input_dtype(self):
        return torch.long


class CNNNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, kernel_size, second_conv,
                 dropout_rate, dense_units, dense_activation, max_sequence_length, embedding_matrix):
        super(CNNNetwork, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = True

        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size)
        self.second_conv = second_conv

        if second_conv:
            self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=3)

        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(num_filters, dense_units)
        self.output = nn.Linear(dense_units, 1)

        self.activation_fn = nn.ReLU() if dense_activation == 'relu' else nn.Tanh()

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        x = F.relu(self.conv1(x))
        if self.second_conv:
            x = F.relu(self.conv2(x))

        x = self.pool(x).squeeze(2)
        x = self.dropout(x)
        x = self.activation_fn(self.dense(x))
        x = torch.sigmoid(self.output(x))

        return x
