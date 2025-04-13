import spacy
from spacy.util import is_package

from classifier.Omri_model.Spacy3Classes import Spacy3Classes
from classifier.preprocessing.TextPreprocessor import TextPreprocessor

custom_lemmas = {"hamas": "hamas", }  # { original word: change into }, for words that are not lemmatized correctly


def load_model(model_name):
    if not is_package(model_name):
        print(f"Model {model_name} is not installed. Installing...")
        spacy.cli.download(model_name)

    return spacy.load(model_name)


def run_evaluation():
    model_name = "en_core_web_sm"
    nlp = load_model(model_name)

    text_preprocessor = TextPreprocessor(
        emoji=None)  # 'text' to convert to text, 'config' to get description from 'config.json'

    classifier = Spacy3Classes(model=nlp, label_count=3, preprocessor=text_preprocessor)

    data = classifier.load_data(1000, 1000, 1000, debug=True)

    prepared_data = classifier.prepare_datasets(data)  # combine_irrelevant=True to combine irrelevant with not-antisemistic

    processed_data = classifier.preprocess_data(prepared_data, custom_lemmas)
    epochs = 100
    learning_rate = 0.001
    l2_regularization = 1e-6
    batch_size = 8

    classifier.train(processed_data, epochs, learning_rate, l2_regularization, batch_size)


if __name__ == '__main__':
    run_evaluation()
