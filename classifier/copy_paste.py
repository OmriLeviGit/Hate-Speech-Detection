import random
import time
from copy import copy

from spacy.util import minibatch, fix_random_seed
from spacy.symbols import ORTH
from spacy.training import Example

from classifier.BaseTextClassifier import BaseTextClassifier
from classifier.preprocessing.TextNormalizer import TextNormalizer


class SpacyModels_SM_LG(BaseTextClassifier):
    def __init__(self, model: any, preprocessor: TextNormalizer() = None, seed: int = 42):
        super().__init__(model, preprocessor, seed)
        self.training_time = None
        self.training_history = None

    def preprocess_data(self, datasets: any, custom_lemmas: dict[str, str] = None) -> dict[str, list[str]]:
        """Apply preprocessing to datasets"""
        datasets = super().preprocess_data(datasets)  # Custom preprocessing

        special_tokens = self.get_text_normalizer().get_special_tokens()
        self.add_tokens(special_tokens)  # Add special tokens to the tokenizer
        self.add_lemmas(custom_lemmas)  # Add custom lemmas to the lemmatizer
        nlp = self.get_model()

        # Run text through the entire spacy NLP pipeline
        processed_datasets = {}
        for label, posts in datasets.items():
            processed_data = []
            for post in posts:
                doc = nlp(post)

                tokens = [
                    token.lemma_ for token in doc
                    if token.is_alpha and not token.is_stop and not token.is_punct
                ]

                lemmatized_text = ' '.join(tokens)
                processed_data.append(lemmatized_text)

            processed_datasets[label] = processed_data

        return processed_datasets

    def add_tokens(self, special_tokens: set):
        """Register all special tokens with spaCy tokenizer"""
        model = self.get_model()

        for token in special_tokens:
            special_case = [{ORTH: token}]
            model.tokenizer.add_special_case(token, special_case)

    def train(self, processed_datasets: dict[str, list[tuple[str, str]]], learning_rate: float = 0.001,
              l2_regularization: float = 0.001, epochs: int = 100, batch_size: int = 32, dropout: float = 0.2) -> None:
        pass


    def evaluate(self, datasets: dict[str, list[tuple[str, str]]]) -> dict[str, float]:
        """Evaluate the model"""
        pass