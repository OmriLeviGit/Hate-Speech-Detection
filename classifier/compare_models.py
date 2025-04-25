import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from classifier import utils
from classifier.normalization.TextNormalizer import TextNormalizer
from classifier.SpacyClassifier import SpacyClassifier

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

models = {
    "LogisticRegression": (LogisticRegression(), lr_param_grid),
    "LinearSVC": (LinearSVC(), svc_param_grid),
    "KNeighborsClassifier": (KNeighborsClassifier(), knn_param_grid),
    "RandomForestClassifier": (RandomForestClassifier(), rf_param_grid),
    "SGDClassifier": (SGDClassifier(), sgd_param_grid),
}


def run_model_search(X, y):
    start_time = time.time()

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
            n_jobs=1,
        )
        grid_search.fit(X, y)

        y_pred = cross_val_predict(grid_search.best_estimator_, X, y, cv=5)
        classification = classification_report(y, y_pred)
        conf_mat = confusion_matrix(y, y_pred)

        model_duration = time.time() - model_start_time
        total_duration = time.time() - start_time

        utils.print_model_results(grid_search, classification, conf_mat, model_duration, total_duration)

        model_results.append((name, grid_search))

    print("\n\n=== Models sorted by score ===")
    for name, model in sorted(model_results, key=lambda x: x[1].best_score_, reverse=True):
        print(f"{name}: {round(model.best_score_, 2)}")

    best_name, best_model = max(model_results, key=lambda x: x[1].best_score_)
    print(f"\n=== Best overall model: '{best_name}' with score: {round(best_model.best_score_, 2)} ===")
    print("Best params:\n", best_model.best_params_)

    return best_model, vectorizer


def main():
    # parameters
    label_encoder = LabelEncoder()
    nlp_model_name = "en_core_web_lg"
    normalizer = TextNormalizer(emoji='text')
    labels = ["antisemitic", "not_antisemitic"]

    # load, preprocess, prepare
    classifier = SpacyClassifier(nlp_model_name, normalizer, labels)
    data = classifier.load_data(set_to_min=True)
    data = classifier.preprocess_datasets(data)
    X, y = classifier.prepare_dataset(data)

    y_encoded = label_encoder.fit_transform(y)

    # train
    trained_model, vectorizer = run_model_search(X, y_encoded)

    # EXAMPLE:
    posts_to_predict = ["test", "te"]   # tweet to predict

    # predict
    preprocessed_posts = classifier.preprocess_text_list(posts_to_predict)
    vectorized_posts = vectorizer.transform(preprocessed_posts)
    predictions = trained_model.predict(vectorized_posts)

    decoded_predictions = label_encoder.inverse_transform(predictions)
    # print(decoded_predictions)


if __name__ == "__main__":
    main()
