from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from .base_dl_model import BaseDLModel

from classifier.src.SpacySingleton import SpacyModel



class MLPModel(BaseDLModel):

    def __init__(self, params):
        super().__init__(params)
        self.vectorizer = None  # Will hold fitted vectorizer


    def _clean(self, text):

        nlp = SpacyModel.get_instance()

        doc = nlp(text)
        return ' '.join(
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct
        ).strip()


    # Converts raw (but already normalized) text to TF-IDF vectors
    # Returns X as a NumPy array and y unchanged
    def preprocess(self, X_raw, y):
        processed_data = [self._clean(post) for post in X_raw]
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X_processed = self.vectorizer.fit_transform(processed_data).toarray()
        return X_processed, y


    # Uses the already-fitted vectorizer to transform new data
    def transform(self, X_raw, y=None):
        processed_data = [self._clean(post) for post in X_raw]
        X_processed = self.vectorizer.transform(processed_data).toarray()
        return (X_processed, y) if y is not None else X_processed

    # Builds and compiles the MLP model using input shape and hyperparameters
    def build(self, input_shape):

        model = Sequential()

        model.add(Input(shape=(input_shape,)))
        model.add(Dense(self.params['hidden_units'], activation='relu'))
        model.add(Dropout(self.params['dropout_rate']))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=Adam(learning_rate=self.params['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model
