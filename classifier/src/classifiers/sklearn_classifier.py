import os
import time
import random
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from classifier.src.vectorizers.glove_vectorizer import GloVeVectorizer
from classifier.src.deep_learning_models.DataHelper import DataHelper
from classifier.src.normalization.TextNormalizer import TextNormalizer
from classifier.src.utils import print_header, format_duration


def prepare_data(helper, data_config, raw_data):
    X_raw, X_test, y_raw, y_test = helper.prepare_dataset(
        raw_data,
        test_size=0.2,
        balance_pct=data_config["balance_pct"],
        augment_ratio=data_config["augment_ratio"],
        irrelevant_ratio=data_config["irrelevant_ratio"]
    )

    y_trainval = helper.label_encoder.transform(y_raw)
    y_test = helper.label_encoder.transform(y_test)

    normalizer = TextNormalizer(emoji="text")
    X_trainval = normalizer.normalize_texts(X_raw)
    X_test = normalizer.normalize_texts(X_test)

    return X_trainval, X_test, y_trainval, y_test


def run_grid_search(pipeline, param_grid, X, y):

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(pipeline, param_grid, cv=skf, scoring='f1_weighted', n_jobs=-1, verbose=1)
    grid.fit(X, y)

    print_header("Grid Search Results")
    print(f"\nBest Params: {grid.best_params_}")
    print(f"\nBest Cross-Validated f1_weighted: {grid.best_score_:.4f}")

    return grid.best_estimator_


def evaluate_on_test(model, X_test, y_test):
    print_header("Final Test Set Evaluation")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=["Not Antisemitic", "Antisemitic"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def main():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)

    helper = DataHelper()
    raw_data = helper.load_data()

    data_configs = [
        # {"balance_pct": 0.5, "augment_ratio": 0.0, "irrelevant_ratio": 0.0},
        {"balance_pct": 0.5, "augment_ratio": 0.33, "irrelevant_ratio": 0.4},
    ]

    # === Define vectorizers ===
    vectorizer_configs = [
        # {
        #     "name": "TF-IDF",
        #     "vectorizer": TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.9)
        # },
        {
            "name": "GloVe",
            "vectorizer": GloVeVectorizer("glove.twitter.27B.200d.txt", dim=200)
        }
    ]

    # === Define models ===
    model_configs = [
        # {
        #     "name": "LogisticRegression",
        #     "model": LogisticRegression(max_iter=1000, random_state=42),
        #     "param_grid": {
        #         "model__C": [1.0]
        #     }
        # },
        #
        # {
        #     "name": "LinearSVC",
        #     "model": LinearSVC(max_iter=1000, random_state=42),
        #     "param_grid": {
        #         "model__C": [1.0]
        #     }
        # },

        # {
        #     "name": "MultinomialNB",
        #     "model": MultinomialNB(),
        #     "param_grid": {
        #         "model__alpha": [0.1, 0.5, 1.0],
        #         "model__fit_prior": [True, False]
        #     }
        # },

        {
            "name": "RandomForest",
            "model": RandomForestClassifier(random_state=42),
            "param_grid": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [10, 20, 40],
                "model__min_samples_split": [5],
            }
        },
        #
        # {
        #     "name": "KNeighborsClassifier",
        #     "model": KNeighborsClassifier(),
        #     "param_grid": {
        #         "model__n_neighbors": [3, 5, 7],
        #         "model__weights": ['uniform', 'distance']
        #     }
        # },

        # {
        #     "name": "GradientBoosting",
        #     "model": GradientBoostingClassifier(random_state=42),
        #     "param_grid": {
        #         "model__learning_rate": [0.01, 0.1],
        #         "model__n_estimators": [100],
        #         "model__max_depth": [3, 5],
        #         "model__subsample": [0.8, 1.0]
        #     }
        # },

        # {
        #     "name": "SGDClassifier",
        #     "model": SGDClassifier(loss='log_loss', random_state=42),
        #     "param_grid": {
        #         "model__alpha": [0.001],
        #         "model__penalty": ['l2'],
        #         "model__max_iter": [1000],
        #         "model__learning_rate": ['optimal']
        #     }
        # }

    ]

    for config in data_configs:

        print(f"\nRunning with data config: {config}")

        X_trainval, X_test, y_trainval, y_test = prepare_data(helper, config, raw_data)

        for vec_cfg in vectorizer_configs:
            print_header(f"Using Vectorizer: {vec_cfg['name']}")

            for model_cfg in model_configs:
                print_header(f"Training model: {model_cfg['name']}")

                pipeline = Pipeline([
                    ("vectorizer", vec_cfg["vectorizer"]),
                    ("model", model_cfg["model"])
                ])

                param_grid = {
                    **model_cfg["param_grid"]
                }

                best_model = run_grid_search(pipeline, param_grid, X_trainval, y_trainval)
                best_model.fit(X_trainval, y_trainval)

                # Training accuracy
                y_train_pred = best_model.predict(X_trainval)
                train_acc = accuracy_score(y_trainval, y_train_pred)
                print(f"Train Accuracy on trainval set: {train_acc:.4f}")

                evaluate_on_test(best_model, X_test, y_test)

    total_duration = time.time() - start_time
    print("\nTotal experiment time:", format_duration(total_duration))


if __name__ == "__main__":
    main()
