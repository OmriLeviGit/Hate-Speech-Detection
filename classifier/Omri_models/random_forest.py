import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict
from tune_sklearn import TuneGridSearchCV
from spacy.util import is_package

from classifier.normalization.TextNormalizer import TextNormalizer
from classifier.SpacyClassifier import SpacyClassifier



def load_model(model_name):
    if not is_package(model_name):
        print(f"Model '{model_name}' is not installed. Installing...")
        spacy.cli.download(model_name)

    print(f"Loading model: '{model_name}'...")

    return spacy.load(model_name)


def random_forest(classifier, X, y, forest_param_grid):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)

    rf = RandomForestClassifier(random_state=classifier.seed)

    grid_search = TuneGridSearchCV(
        estimator=rf,
        param_grid=forest_param_grid,
        cv=5,
        verbose=1,
        scoring='f1_weighted',
    )

    print("Starting grid search with tune-sklearn...")
    grid_search.fit(X_tfidf, y)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    y_pred = cross_val_predict(best_model, X_tfidf, y, cv=5)

    print("Classification Report:\n", classification_report(y, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))

    return best_model, vectorizer


def main():
    nlp = load_model("en_core_web_lg")
    normalizer = TextNormalizer(emoji='text')
    labels = ["antisemistic", "not_antisemistic"]

    print("Setting framework..")
    classifier = SpacyClassifier(nlp, normalizer, labels)
    data = classifier.load_data(set_to_min=True)
    data = classifier.preprocess_data(data)
    X, y = classifier.prepare_dataset(data)

    forest_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    random_forest(classifier, X, y, forest_param_grid)


if __name__ == "__main__":
    main()
