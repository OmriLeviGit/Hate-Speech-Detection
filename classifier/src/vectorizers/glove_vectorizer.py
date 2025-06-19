import os
import numpy as np

class GloVeVectorizer:
    """Creates an average word embedding vector for each text using GloVe."""
    def __init__(self, glove_path, dim=50):
        self.glove_path = glove_path
        self.dim = dim
        self.embeddings = {}
        self.vocab = set()

    def fit(self, X, y=None):
        self.embeddings = self._load_glove_embeddings()
        self.vocab = set(self.embeddings.keys())
        return self

    def transform(self, texts):
        vectors = [self._average_vector(text.split()) for text in texts]
        return np.array(vectors)

    def fit_transform(self, texts, y=None):
        self.fit(texts, y)
        return self.transform(texts)

    def _load_glove_embeddings(self):
        embeddings = {}
        full_path = os.path.join(os.path.dirname(__file__), "../GloVe", self.glove_path)
        with open(full_path, "r", encoding="utf8") as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                vec = np.array(parts[1:], dtype=np.float32)
                embeddings[word] = vec
        return embeddings

    def _average_vector(self, tokens):
        vecs = [self.embeddings[token] for token in tokens if token in self.vocab]
        if not vecs:
            return np.zeros(self.dim)
        return np.mean(vecs, axis=0)
