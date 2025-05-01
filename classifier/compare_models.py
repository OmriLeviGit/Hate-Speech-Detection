import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from classifier.SKLearnClassifier import SKLearnClassifier
from classifier.normalization.TextNormalizer import TextNormalizer


# LogisticRegression
lr_param_grid = {
    'C': [0.5, 1, 5],
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
    'metric': ['euclidean']
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

configs = [
    {
        "model_name": "LogisticRegression",
        "model_class": LogisticRegression(),
        "param_grid": lr_param_grid,
    },
    {
        "model_name": "LinearSVC",
        "model_class": LinearSVC(),
        "param_grid": svc_param_grid,
    },
    {
        "model_name": "KNeighborsClassifier",
        "model_class": KNeighborsClassifier(),
        "param_grid": knn_param_grid,
    },
    {
        "model_name": "RandomForestClassifier",
        "model_class": RandomForestClassifier(),
        "param_grid": rf_param_grid,
    },
    {
        "model_name": "SGDClassifier",
        "model_class": SGDClassifier(),
        "param_grid": sgd_param_grid,
    }
]

def ini_sklearn_models(labels):
    models = []
    for config in configs:
        normalizer = TextNormalizer(emoji='text')
        classifier = SKLearnClassifier(labels, normalizer, TfidfVectorizer(), config)
        models.append(classifier)

    return models

def main():
    labels = ["antisemitic", "not_antisemitic"]
    normalizer = TextNormalizer(emoji='text')

    classifier = SKLearnClassifier(labels, normalizer, TfidfVectorizer(), configs[0])

    data = classifier.load_data(set_to_min=True, debug='debug')

    X_train, X_test, y_train, y_test = classifier.prepare_dataset(data)

    X_train = classifier.preprocess(X_train)

    classifier.train(X_train, y_train)

    classifier.evaluate(X_test, y_test)

    predictions = classifier.predict(X_test, True)


if __name__ == "__main__":
    main()
