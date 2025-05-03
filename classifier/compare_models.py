from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from classifier.SKLearnClassifier import SKLearnClassifier
from classifier.normalization.TextNormalizer import TextNormalizer


configs = [
    {
        "model_name": "LogisticRegression",
        "model_class": LogisticRegression(),
        "param_grid": {
            'C': [0.5, 1, 5],
            'penalty': ['l2'],
            'solver': ['liblinear', 'lbfgs'],
            'max_iter': [1000]
        },
    },
    {
        "model_name": "LinearSVC",
        "model_class": LinearSVC(),
        "param_grid": {
            'C': [0.1, 1, 10],
            'max_iter': [1000],
            'loss': ['squared_hinge'],
            'dual': [False]
        },
    },
    {
        "model_name": "KNeighborsClassifier",
        "model_class": KNeighborsClassifier(),
        "param_grid": {
            'n_neighbors': [3, 5, 7, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean']
        },
    },
    {
        "model_name": "RandomForestClassifier",
        "model_class": RandomForestClassifier(),
        "param_grid": {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2', None]
        },
    },
    {
        "model_name": "SGDClassifier",
        "model_class": SGDClassifier(),
        "param_grid": {
            'loss': ['hinge', 'log_loss'],
            'penalty': ['l2', 'elasticnet'],
            'alpha': [1e-4, 1e-3],
            'max_iter': [1000]
        }
    }
]

def compare_models(models, X_train, y_train):
    best_model = None
    results = []

    for model in models:

        model.train(X_train, y_train)

        score = model.best_score
        results.append((model.model_name, score, model.best_params))

        if not best_model or score > best_model.best_score:
            best_model = model

    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    print("Models sorted by score:")
    for model_name, score, params in sorted_results:
        print(f"{model_name}: {score:.2f} | Params: {params}")

    return best_model

"""
core changes:
- changed the name from run_model_search to compare models, as theres no need to find the best hyper parameters,
    the grid inside train() does that for us
- the param_grid is just combined it with the "models" list and renamed to "configs"
- the predict it part of the sub class, and the evaluate function is part of the base class (and just calls predict())
- I have a more advanced initialization that supports variants changing labels/normalizers/vectorizers/tokenizers
    or whatever else you wish to add, so feel free to change the arguments your class requires

so essentially, if the word2vec works as-is then all we have to change the vectorizer or the grid and the cnn will
probably require its own classifier

and the bert file here is just a skeleton, very non-functional :^]
"""

def main():
    debug = False
    labels = ["antisemitic", "not_antisemitic"]

    # initialization, a note in the string above^
    models = []
    for config in configs:
        normalizer = TextNormalizer(emoji='text')
        classifier = SKLearnClassifier(labels, normalizer, TfidfVectorizer(), config)   # or any other classifier
        models.append(classifier)

    data = models[0].load_data(set_to_min=True, debug=debug)
    X_train, X_test, y_train, y_test = models[0].prepare_dataset(data)

    X_train = models[0].preprocess(X_train)
    X_test = models[0].preprocess(X_test)

    best_model = compare_models(models, X_train, y_train)

    accuracy, f1 = best_model.evaluate(X_test, y_test)

    print(accuracy, f1)


if __name__ == "__main__":
    main()