import time

import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from spacy.util import is_package

from classifier.normalization.TextNormalizer import TextNormalizer
from classifier.SpacyClassifier import SpacyClassifier



# LogisticRegression
lr_param_grid = {
    'C': [0.1, 1, 10],  # Regularization strength
    'penalty': ['l2'],  # Regularization type
    'solver': ['liblinear', 'lbfgs'],  # Optimization algorithm
    'max_iter': [1000]  # Maximum number of iterations for convergence
}

# RandomForestClassifier
rf_param_grid = {
    'n_estimators': [100, 200],  # Number of trees (more trees generally better)
    'max_depth': [None, 10, 20],  # Depth of the trees, None means no limit
    'min_samples_split': [2, 5],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2],  # Minimum number of samples required at a leaf node
    'max_features': ['sqrt', 'log2', None]  # How many features to consider when looking for the best split
}

# KNeighborsClassifier
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 11],  # Number of neighbors to use in classification
    'weights': ['uniform', 'distance'],  # How to weight neighbors (uniform or by distance)
    'metric': ['euclidean', 'manhattan']  # Distance metric to use (Euclidean or Manhattan)
}

# LinearSVC
svc_param_grid = {
    'C': [0.1, 1, 10],  # Regularization strength
    'max_iter': [1000],  # Maximum number of iterations
    'loss': ['squared_hinge'],  # Loss function to use (hinge loss is common for SVC)
    'dual': [False]  # Whether to use the dual formulation
}

# SGDClassifier
sgd_param_grid = {
    'loss': ['hinge', 'log_loss'],  # Loss function to use (hinge for linear SVM, log_loss for logistic regression)
    'penalty': ['l2', 'elasticnet'],  # Regularization technique to use (L2 or elasticnet)
    'alpha': [1e-4, 1e-3],  # Regularization strength (smaller = stronger regularization)
    'max_iter': [1000]  # Maximum number of iterations
}

models = {
    "LogisticRegression": (LogisticRegression(), lr_param_grid),
    "RandomForestClassifier": (RandomForestClassifier(), rf_param_grid),
    "KNeighborsClassifier": (KNeighborsClassifier(), knn_param_grid),
    "LinearSVC": (LinearSVC(), svc_param_grid),
    "SGDClassifier": (SGDClassifier(), sgd_param_grid),
}


def load_model(model_name):
    if not is_package(model_name):
        print(f"'{model_name}' is not installed. Installing...")
        spacy.cli.download(model_name)

    print(f"Loading nlp framework and model: '{model_name}'...")

    return spacy.load(model_name)


def format_duration(seconds):
    """ Convert duration from seconds to hh:mm:ss or mm:ss format """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{int(hours)}:{int(minutes):02}:{int(seconds):02}"
    return f"{int(minutes)}:{int(seconds):02}"


def run_model_search():
    start_time = time.time()

    nlp = load_model("en_core_web_lg")
    normalizer = TextNormalizer(emoji='text')
    labels = ["antisemistic", "not_antisemistic"]

    print("Setting framework..")
    classifier = SpacyClassifier(nlp, normalizer, labels)
    data = classifier.load_data(set_to_min=True)
    data = classifier.preprocess_data(data)
    X, y = classifier.prepare_dataset(data)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)

    best_model_overall = None
    best_score_overall = -1
    best_model_name = ""

    for name, (model, param_grid) in models.items():
        model_start_time = time.time()

        print(f"\n--- Running model: {name} ---")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='f1_weighted',
            cv=5,
            verbose=1,
            n_jobs=4,
        )
        grid_search.fit(X, y)

        print()
        print("Best parameters:", grid_search.best_params_)
        print("Best cross-validation score:", grid_search.best_score_)
        print()

        best_model = grid_search.best_estimator_
        y_pred = cross_val_predict(best_model, X, y, cv=5)

        print("Classification Report:\n", classification_report(y, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
        print()

        if grid_search.best_score_ > best_score_overall:
            best_score_overall = grid_search.best_score_
            best_model_overall = grid_search.best_estimator_
            best_model_name = name


        model_duration = time.time() - model_start_time
        print(f"Model '{name}' took {format_duration(model_duration)} seconds.")

    total_duration = time.time() - start_time
    print(f"Total runtime: {format_duration(total_duration)} seconds.")

    print(f"\n=== Best overall model: {best_model_name} with score: {best_score_overall} ===")
    print("Full estimator:\n", best_model_overall)



if __name__ == "__main__":
    run_model_search()
