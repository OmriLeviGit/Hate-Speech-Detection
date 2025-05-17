import os
import numpy as np

from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Bidirectional

from .base_dl_model import BaseDLModel


# Dynamically resolve the GloVe path relative to this file
base_dir = os.path.dirname(os.path.abspath(__file__))
glove_path = os.path.join(base_dir, "..", "GloVe", "glove.6B.300d.txt")
glove_path = os.path.abspath(glove_path)


class LSTMModel(BaseDLModel):

    def __init__(self, params):
        super().__init__(params)
        self.tokenizer = None
        self.embedding_matrix = None
        self.vocab_size = None


    def preprocess(self, X_raw, y):

        self.tokenizer = Tokenizer(num_words = 10000, oov_token = "<OOV>")
        self.tokenizer.fit_on_texts(X_raw)

        sequences = self.tokenizer.texts_to_sequences(X_raw)

        padded = pad_sequences(sequences, maxlen=self.params['max_sequence_length'], padding="post")

        # Mandatory line as index 0 is reserved by Tokenizer for padding
        self.vocab_size = len(self.tokenizer.word_index) + 1

        self.embedding_matrix = self._load_glove_embeddings(glove_path, self.params['embedding_dim'])

        return padded, y


    def transform(self, X_raw, y=None):
        sequences = self.tokenizer.texts_to_sequences(X_raw)
        padded = pad_sequences(sequences, maxlen=self.params['max_sequence_length'], padding="post")
        return padded, y if y is not None else None


    def _load_glove_embeddings(self, glove_path, embedding_dim):

        embeddings_index = {}

        with open(glove_path, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = vector

        embedding_matrix = np.random.uniform(-0.05, 0.05, (self.vocab_size, embedding_dim))

        for word, i in self.tokenizer.word_index.items():
            if i < self.vocab_size:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def build(self, input_shape):

        model = Sequential()

        model.add(Input(shape=(input_shape,)))
        model.add(Embedding(
            input_dim=self.vocab_size,
            output_dim=self.params['embedding_dim'],
            weights=[self.embedding_matrix],
            trainable=True
        ))

        model.add(Bidirectional(LSTM(
            self.params['lstm_units'],
            dropout=self.params['dropout_rate'],
            recurrent_dropout=0.1
        )))

        model.add(Dense(self.params['dense_units'], activation=self.params['dense_activation']))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=Adam(self.params['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model



