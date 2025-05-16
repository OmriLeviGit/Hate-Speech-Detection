from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .base_dl_model import BaseDLModel


class CNNModel(BaseDLModel):

    # Converts raw text into padded sequences using a tokenizer
    # Returns padded X and unchanged y
    def preprocess(self, X_raw, y):

        self.tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(X_raw)

        sequences = self.tokenizer.texts_to_sequences(X_raw)

        self.padded = pad_sequences(sequences, maxlen=self.params['max_sequence_length'], padding="post")

        # Mandatory line as index 0 is reserved by Tokenizer for padding
        self.vocab_size = len(self.tokenizer.word_index) + 1

        return self.padded, y


    def transform(self, X_raw, y=None):
        sequences = self.tokenizer.texts_to_sequences(X_raw)
        padded = pad_sequences(sequences, maxlen=self.params['max_sequence_length'], padding="post")
        return padded, y if y is not None else None


    # Builds and compiles the CNN model
    # input_shape is the sequence length
    def build(self, input_shape):

        model = Sequential()

        model.add(Input(shape=(input_shape,)))
        model.add(Embedding(input_dim=self.vocab_size, output_dim=self.params['embedding_dim']))

        # First conv layer
        model.add(Conv1D(filters=self.params['num_filters'], kernel_size=self.params['kernel_size'],activation='relu'))

        # Second conv layer (optional layer)
        # model.add(Conv1D(filters=self.params['num_filters'],kernel_size = 3, activation = 'relu'))

        model.add(GlobalMaxPooling1D())
        model.add(Dropout(self.params['dropout_rate']))
        model.add(Dense(64, activation='relu'))  # non-linear layer
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(learning_rate=self.params['learning_rate']),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

