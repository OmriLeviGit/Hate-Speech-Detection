import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

from classifier import utils
from classifier.normalization.TextNormalizer import TextNormalizer
from classifier.SpacyClassifier import SpacyClassifier

# LogisticRegression
lr_param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [1000]
}

# LinearSVC
svc_param_grid = {
    'C': [0.1, 1, 10],
    'max_iter': [1000],
    'loss': ['squared_hinge'],
    'dual': [False]
}

# KNeighborsClassifier
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# RandomForestClassifier
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2', None]
}

# SGDClassifier
sgd_param_grid = {
    'loss': ['hinge', 'log_loss'],
    'penalty': ['l2', 'elasticnet'],
    'alpha': [1e-4, 1e-3],
    'max_iter': [1000]
}

models = {
    "LogisticRegression": (LogisticRegression(), lr_param_grid),
    "LinearSVC": (LinearSVC(), svc_param_grid),
    "KNeighborsClassifier": (KNeighborsClassifier(), knn_param_grid),
    "RandomForestClassifier": (RandomForestClassifier(), rf_param_grid),
    "SGDClassifier": (SGDClassifier(), sgd_param_grid),
}


def run_model_search():
    start_time = time.time()

    # 0 == antisemistic, 1 == not_antisemistic, 2 == irrelevant. Must be an integer
    labels = [0, 1]
    model_name = "en_core_web_lg"
    normalizer = TextNormalizer(emoji='text')

    classifier = SpacyClassifier(model_name, normalizer, labels)
    data = classifier.load_data(set_to_min=True)
    data = classifier.preprocess_data(data)
    X, y = classifier.prepare_dataset(data)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)

    model_results = []

    for name, (model, param_grid) in models.items():
        model_start_time = time.time()

        utils.print_model_header(name)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='f1_weighted',
            cv=5,
            verbose=1,
            n_jobs=2,
        )
        grid_search.fit(X, y)

        best_model = grid_search.best_estimator_
        best_score = round(grid_search.best_score_, 2)

        y_pred = cross_val_predict(best_model, X, y, cv=5)
        model_duration = time.time() - model_start_time
        total_duration = time.time() - start_time

        utils.print_model_results(best_model, best_score, y, y_pred, model_duration, total_duration)

        model_results.append((name, best_model, best_score))

    print("\n=== Models sorted by score ===")
    for name, model, score in sorted(model_results, key=lambda x: x[2], reverse=True):
        print(f"{name}: {score}")

    best_name, best_model, best_score = max(model_results, key=lambda x: x[2])
    print(f"\n=== Best overall model: {best_name} with score: {best_score} ===")
    print("Full estimator:\n", best_model)


if __name__ == "__main__":
    run_model_search()
