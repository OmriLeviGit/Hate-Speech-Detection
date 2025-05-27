import torch
import torch.nn as nn
import torch.nn.functional as F

import spacy
import numpy as np
from collections import Counter

from .base_dl_model import BaseDLModel

class CNNModel(BaseDLModel):

    # Converts raw text into padded sequences using a tokenizer
    # Returns padded X and unchanged y
    def preprocess(self, X_raw, y):

        self.max_len = self.params["max_sequence_length"]
        max_vocab_size = 5000

        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"])

        # Build vocabulary
        counter = Counter()
        for text in X_raw:
            counter.update(text)

        most_common = [word for word, _ in counter.most_common(max_vocab_size)]
        # +2 to reserve 0 and 1
        self.word2idx = {word: idx + 2 for idx, word in enumerate(most_common)}
        self.word2idx["<PAD>"] = 0
        self.word2idx["<OOV>"] = 1

        self.vocab_size = len(self.word2idx)

        def encode(text):
            ids = [self.word2idx.get(token, 1) for token in text]  # 1 = <OOV>
            if len(ids) < self.max_len:
                ids += [0] * (self.max_len - len(ids))  # 0 = <PAD>
            return ids[:self.max_len]

        sequences = [encode(text) for text in X_raw]
        return np.array(sequences, dtype=np.int64), y


    def transform(self, X_raw, y=None):

        def encode(text):
            ids = [self.word2idx.get(token, 1) for token in text]  # 1 = <OOV>
            if len(ids) < self.max_len:
                ids += [0] * (self.max_len - len(ids))  # 0 = <PAD>
            return ids[:self.max_len]

        sequences = [encode(text) for text in X_raw]
        return (np.array(sequences, dtype=np.int64), y) if y is not None else np.array(sequences, dtype=np.int64)


    # Builds and compiles the CNN model
    # input_shape is the sequence length
    def build(self, input_shape):
        model = CNNNetwork(
            vocab_size=self.vocab_size,
            embedding_dim=self.params['embedding_dim'],
            num_filters=self.params['num_filters'],
            kernel_size=self.params['kernel_size'],
            second_conv=self.params.get('second_conv', False),
            dropout_rate=self.params['dropout_rate'],
            dense_units=self.params['dense_units'],
            dense_activation=self.params['dense_activation'],
            # Unused inside model, but passed anyway
            max_sequence_length=input_shape
        )
        return model

    @property
    def input_dtype(self):
        return torch.long


class CNNNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, kernel_size, second_conv,
                 dropout_rate, dense_units, dense_activation, max_sequence_length):
        super(CNNNetwork, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)

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


