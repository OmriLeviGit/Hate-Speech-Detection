import time, os, pickle, joblib, copy


from sklearn.model_selection import GridSearchCV

from classifier.BaseTextClassifier import BaseTextClassifier
from classifier.SpacySingleton import SpacyModel
from classifier.normalization.TextNormalizer import TextNormalizer


class SKLearnClassifier(BaseTextClassifier):
    def __init__(self, labels: list, normalizer: TextNormalizer(), vectorizer, config: dict, seed: int = 42):
        super().__init__(labels, seed)

        self.config = config
        self.normalizer = normalizer

        self.model_name = config.get("model_name")
        self.model_class = config.get("model_class")
        self.param_grid = config.get("param_grid")

        self.vectorizer = vectorizer
        self.nlp = SpacyModel.get_instance()

        self.best_model = None
        self.best_score = None
        self.best_params = None

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

    def train(self, X: list[str], y: list[str], param_grid=None):
        """Optimize hyperparameters with GridSearchCV"""
        # Validate hyperparameter config
        if param_grid:
            self.param_grid = param_grid

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
            n_jobs=2,
        )

        grid_search.fit(X_vectorized, y_encoded)

        training_duration = time.time() - start_time

        self.best_model = grid_search.best_estimator_
        self.best_score = round(float(grid_search.best_score_), 2)
        self.best_params = grid_search.best_params_
        y_pred = self.best_model.predict(X_vectorized)

        self.print_best_model_results(self.best_score, self.best_params, y_encoded, y_pred, training_duration)

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

    def save_model(self, path: str):
        sklearn_path = os.path.join(path, "sklearn")
        os.makedirs(sklearn_path, exist_ok=True)

        tmp = copy.deepcopy(self)
        tmp.best_model = None

        with open(os.path.join(sklearn_path, "classifier_class.pkl"), "wb") as f:
            pickle.dump(tmp, f)

        joblib.dump(self.best_model, os.path.join(sklearn_path, "sk_model.pkl"))

    @staticmethod
    def load_model(path: str):
        sklearn_path = os.path.join(path, "sklearn")
        with open(os.path.join(sklearn_path, "classifier_class.pkl"), "rb") as f:
            obj = pickle.load(f)
            obj.best_model = None
            obj.tokenizer = None

        obj.best_model = joblib.load(os.path.join(sklearn_path, "sk_model.pkl"))

        return obj