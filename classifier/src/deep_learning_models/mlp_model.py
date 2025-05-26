import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from classifier.src.SpacySingleton import SpacyModel

from .base_dl_model import BaseDLModel

class MLPModel(BaseDLModel):

    def __init__(self, params):
        super().__init__(params)
        self.vectorizer = None

    def _clean(self, text):
        nlp = SpacyModel.get_instance()
        doc = nlp(text)
        return ' '.join(
            token.lemma_ for token in doc
            if not token.is_stop and not token.is_punct
        ).strip()

    def preprocess(self, X_raw, y):
        processed_data = [self._clean(post) for post in X_raw]
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X_processed = self.vectorizer.fit_transform(processed_data).toarray()
        return X_processed, y

    def transform(self, X_raw, y=None):
        processed_data = [self._clean(post) for post in X_raw]
        X_processed = self.vectorizer.transform(processed_data).toarray()
        return (X_processed, y) if y is not None else X_processed

    def build(self, input_shape):
        model = MLPNetwork(
            input_dim=input_shape,
            hidden_units=self.params['hidden_units'],
            dropout_rate=self.params['dropout_rate']
        )
        return model

    @property
    def input_dtype(self):
        return torch.float32


class MLPNetwork(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout_rate):
        super(MLPNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x
