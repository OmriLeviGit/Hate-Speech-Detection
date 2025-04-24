import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict
from spacy.util import is_package

from classifier.normalization.TextNormalizer import TextNormalizer
from classifier.SpacyClassifier import SpacyClassifier
from sklearn.svm import LinearSVC


def load_model(model_name):
    if not is_package(model_name):
        print(f"Model '{model_name}' is not installed. Installing...")
        spacy.cli.download(model_name)

    print(f"Loading model: '{model_name}'...")

    return spacy.load(model_name)


def svc():
    nlp = load_model("en_core_web_lg")
    normalizer = TextNormalizer(emoji='text')
    labels = ["antisemistic", "not_antisemistic"]
    classifier = SpacyClassifier(nlp, normalizer, labels)
    data = classifier.load_data(set_to_min=True)
    data = classifier.preprocess_data(data)
    X, y = classifier.prepare_dataset(data)

    # from here on its model specific
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)
    model = LinearSVC(C=0.5, class_weight="balanced", random_state=classifier.seed)
    y_pred = cross_val_predict(model, X_tfidf, y, cv=5)

    print("Classification Report:\n", classification_report(y, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))


def main():
    svc()

if __name__ == "__main__":
    main()
