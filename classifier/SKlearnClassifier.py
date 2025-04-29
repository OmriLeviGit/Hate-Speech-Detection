import time

import spacy
from sklearn.model_selection import GridSearchCV
from spacy.util import is_package

from classifier.BaseTextClassifier import BaseTextClassifier
from classifier.normalization.TextNormalizer import TextNormalizer


class SKlearnClassifier(BaseTextClassifier):
    def __init__(self, labels: list, normalizer: TextNormalizer(), config: dict, seed: int = 42):
        super().__init__(labels, seed)

        # Sklearn-specific attributes
        self.config = config
        self.normalizer = normalizer
        self.model_name = config.get("model_name")
        self.model_class = config.get("model_class")
        self.vectorizer = config.get("vectorizer")
        self.param_grid = config.get("param_grid")

        self.best_model = None
        self.best_score = None
        self.best_params = None

        self.nlp = self._load_spacy()  # Load Spacy model

    def _load_spacy(self):
        nlp_model_name = "en_core_web_lg"

        if not is_package(nlp_model_name):
            print(f"'{nlp_model_name}' is not installed. Installing...")
            spacy.cli.download(nlp_model_name)

        print(f"Loading: '{nlp_model_name}'...")

        return spacy.load(nlp_model_name)

    def preprocess(self, text_list: list[str]) -> list[str]:
        normalizer = self.normalizer
        normalized_text_list = normalizer.normalize_texts(text_list)

        unrecognized_tokens = 0
        processed_text_list = []
        for post in normalized_text_list:
            doc = self.nlp(post)
            tokens = []

            for token in doc:
                if not token.is_stop and not token.is_punct:
                    tokens.append(token.lemma_)

                    if not token.has_vector:
                        unrecognized_tokens += 1

            lemmatized_text = ' '.join(tokens).strip()
            processed_text_list.append(lemmatized_text)

        if unrecognized_tokens > 0:
            print(f"Undetected tokens found: {unrecognized_tokens} ")

        return processed_text_list

    def train(self, X, y):
        # Validate config
        # if config is None:
        #     raise ValueError("Config with model_name, model_class, vectorizer and param_grid is required")

        required_keys = ['model_name', 'model_class', 'vectorizer', 'param_grid']
        missing_keys = [key for key in required_keys if key not in self.config or self.config[key] is None]

        if missing_keys:
            missing_keys_str = ", ".join(missing_keys)
            raise ValueError(f"Config is missing required keys: {missing_keys_str}")

        # Vectorize and encode
        X_vectorized = self.vectorizer.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)

        start_time = time.time()

        # Run grid search
        grid_search = GridSearchCV(
            estimator=self.model_class,
            param_grid=self.param_grid,
            scoring='f1_weighted',
            cv=5,
            verbose=1,
            n_jobs=-1,
        )

        grid_search.fit(X_vectorized, y_encoded)

        training_duration = time.time() - start_time

        self.best_model = grid_search.best_estimator_
        self.best_score = round(grid_search.best_score_, 2)
        y_pred = self.best_model.predict(X_vectorized)

        self.print_model_results(grid_search, y_encoded, y_pred, training_duration)

        return {
            "model": self.best_model,
            "score": self.best_score,
            "params": grid_search.best_params_
        }

    def predict(self, text, output=False):
        # Handle both single text and list of texts
        single_input = isinstance(text, str)

        # Convert single string to list if needed
        text_list = [text] if single_input else text

        # Process the list
        texts_processed = self.preprocess(text_list)
        texts_vectorized = self.vectorizer.transform(texts_processed)
        y_pred = self.best_model.predict(texts_vectorized)

        if output:
            y_pred_decoded = self.label_encoder.inverse_transform(y_pred).tolist()

            print()
            for pred in zip(y_pred_decoded, text):
                print(pred)

        # Return single item or full list based on input type
        return y_pred[0] if single_input else y_pred
