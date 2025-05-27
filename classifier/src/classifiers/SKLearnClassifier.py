import time, os, pickle, joblib

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from classifier.src.classifiers.BaseTextClassifier import BaseTextClassifier
from classifier.src.SpacySingleton import SpacyModel
from classifier.src.normalization.TextNormalizer import TextNormalizer


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
        self.cv_score = None
        self.best_params = None

    def preprocess(self, text_list: list[str], output=False) -> list[str]:
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

        if output is True and unrecognized_tokens > 0:
            print(f"Undetected tokens found: {unrecognized_tokens} ")

        return processed_text_list

    def train(self, X: list[str], y: list[str], param_grid=None):
        """Optimize hyperparameters with GridSearchCV"""
        print(f"Training {self.model_name}")

        # Validate hyperparameter config
        if param_grid:
            self.param_grid = param_grid

        X_preprocessed = self.preprocess(X)
        X_vectorized = self.vectorizer.fit_transform(X_preprocessed)
        y_encoded = self.label_encoder.fit_transform(y)

        start_time = time.time()

        # scoring=make_scorer(self.compute_custom_f1, greater_is_better=True),

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
        self.cv_score = round(float(grid_search.best_score_), 2)
        self.best_params = grid_search.best_params_

        # Calibrate models after training if they don't support probability estimation
        # This improves decision boundary quality and enables accessing confidence scores if needed
        if not (hasattr(self.best_model, 'predict_proba') and callable(self.best_model.predict_proba)):
            calibrated = CalibratedClassifierCV(FrozenEstimator(self.best_model))
            calibrated.fit(X_vectorized, y_encoded)
            self.best_model = calibrated

        self.print_best_model_results(self.cv_score, self.best_params, training_duration)

    def predict(self, text, threshold=0.5, output=False):
        single_input = isinstance(text, str)
        text_list = [text] if single_input else text

        texts_processed = self.preprocess(text_list)
        texts_vectorized = self.vectorizer.transform(texts_processed)

        if threshold == 0.5:
            y_pred = self.best_model.predict(texts_vectorized)
        else:
            y_pred_proba = self.best_model.predict_proba(texts_vectorized)

            antisemitic_class = 0
            antisemitic_idx = list(self.best_model.classes_).index(antisemitic_class)
            antisemitic_proba = y_pred_proba[:, antisemitic_idx]

            y_pred = np.where(antisemitic_proba >= threshold, antisemitic_class, 1 - antisemitic_class)

        if output:
            y_pred_decoded = self.label_encoder.inverse_transform(y_pred).tolist()

            sorted_results = sorted(zip(y_pred_decoded, text_list), key=lambda x: x[0])
            for pred in sorted_results:
                print(pred)

        return y_pred[0] if single_input else y_pred

    def save_model(self):
        sklearn_path = str(os.path.join(BaseTextClassifier.save_models_path, "sklearn", self.model_name))
        os.makedirs(sklearn_path, exist_ok=True)

        # Save model
        joblib.dump(self.best_model, os.path.join(sklearn_path, "sk_model.pkl"))
        joblib.dump(self.vectorizer, os.path.join(sklearn_path, "vectorizer.pkl"))

        # Save temporary references
        temp_best_model = self.best_model
        temp_vectorizer = self.vectorizer

        # Clear problematic attributes
        self.best_model = None
        self.vectorizer = None

        with open(os.path.join(sklearn_path, "classifier_class.pkl"), "wb") as f:
            pickle.dump(self, f)

        self.best_model = temp_best_model
        self.vectorizer = temp_vectorizer

    @staticmethod
    def load_model(path: str):
        sklearn_path = str(os.path.join(BaseTextClassifier.save_models_path, "sklearn", path))
        with open(os.path.join(sklearn_path, "classifier_class.pkl"), "rb") as f:
            obj = pickle.load(f)

        obj.best_model = joblib.load(os.path.join(sklearn_path, "sk_model.pkl"))
        obj.vectorizer = joblib.load(os.path.join(sklearn_path, "vectorizer.pkl"))

        return obj