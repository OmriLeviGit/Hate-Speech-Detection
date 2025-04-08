import spacy
from spacy.util import is_package
from classifier.TestModel import TestModel
from classifier.preprocessing.TextPreprocessor import TextPreprocessor


def load_model(model_name):
    if not is_package(model_name):
        print(f"Model {model_name} is not installed. Installing...")
        spacy.cli.download(model_name)

    return spacy.load(model_name)


def run_evaluation():
    model_name = "en_core_web_sm"
    nlp = load_model(model_name)

    text_preprocessor = TextPreprocessor(emoji='text')  # 'text' to convert to text, 'config' to get description from 'config.json'

    classifier = TestModel(nlp, text_preprocessor)

    data = classifier.load_data(1000, 1000, 1000, debug=True)

    data = classifier.prepare_datasets(data)   # combine_irrelevant=True to combine irrelevant with not-antisemistic

    data = classifier.preprocess_data(data)

    classifier.train(data)


if __name__ == '__main__':
    run_evaluation()
