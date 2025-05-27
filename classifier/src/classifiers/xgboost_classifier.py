from xgboost import XGBClassifier

import numpy as np
import random
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV

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


def vectorize_texts(X_trainval, X_test):
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_trainval)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test, vectorizer


def run_grid_search(X, y, cv_folds=5):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    model = XGBClassifier(eval_metric='logloss', random_state=42)

    param_grid = {
        'n_estimators': [200],
        'learning_rate': [0.1],
        'max_depth': [4, 6],
        'colsample_bytree': [0.2, 0.5, 1.0],    # use %x of features used per tree, a little like dropout_rate in deep learning
        'reg_alpha': [0, 0.5],                  # L1 regularization
        'reg_lambda': [1, 2],                   # L2 regularization
    }

    grid = GridSearchCV(model,param_grid, cv=skf, scoring='f1', n_jobs=-1, verbose=1, return_train_score=True)
    grid.fit(X, y)

    print_header("Grid Search Results")
    print(f"\nBest Params: {grid.best_params_}")
    print(f"\nCV F1 Score: {grid.best_score_:.4f}")

    return grid.best_estimator_


def evaluate_on_test(model, X_test, y_test):

    print_header("Final Test Set Evaluation")
    print()

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def main():

    start_time = time.time()

    random.seed(42)
    np.random.seed(42)

    helper = DataHelper()
    raw_data = helper.load_data()

    data_config = [
        # {"balance_pct": 0.5, "augment_ratio": 0.0, "irrelevant_ratio": 0.5},
        {"balance_pct": 0.33, "augment_ratio": 0.0, "irrelevant_ratio": 0.5},
        # {"balance_pct": 0.4, "augment_ratio": 0.0, "irrelevant_ratio": 0.5},

    ]

    for config in data_config:

        print(f"\nRunning with data config: {config}")

        X_trainval_texts, X_test_texts, y_trainval, y_test = prepare_data(helper, config, raw_data)
        X_train, X_test, vectorizer = vectorize_texts(X_trainval_texts, X_test_texts)

        best_model = run_grid_search(X_train, y_trainval, cv_folds=5)

        # Re-train on full trainval set
        best_model.fit(X_train, y_trainval)

        evaluate_on_test(best_model, X_test, y_test)

        train_preds = best_model.predict(X_train)
        train_acc = accuracy_score(y_trainval, train_preds)
        print(f"\nTrain Accuracy on full trainval set: {train_acc:.4f}")

        config_duration = time.time() - start_time
        print("\nTotal config time:", format_duration(config_duration))

        print_header("End of config")

    total_duration = time.time() - start_time
    print("\nTotal experiment time:", format_duration(total_duration))

if __name__ == "__main__":
    main()
