import spacy
from spacy.util import is_package

from classifier.ExampleModel import ExampleModel
from classifier.preprocessing.TextNormalizer import TextNormalizer

custom_lemmas = ["hamas"]  # if you encounter words that don't get lemmatized correctly, add them here


def load_model(model_name):
    if not is_package(model_name):
        print(f"Model {model_name} is not installed. Installing...")
        spacy.cli.download(model_name)

    return spacy.load(model_name)


def run_evaluation():
    model_name = "en_core_web_sm"
    nlp = load_model(model_name)

    text_normalizer = TextNormalizer(emoji=None)  # 'text' to convert to text, 'config' to get description from 'config.json'

    classifier = ExampleModel(nlp, text_normalizer)

    data = classifier.load_data(1000, 1000, 1000, debug=True)

    data = classifier.prepare_datasets(data)   # combine_irrelevant=True to combine irrelevant with not-antisemistic

    data = classifier.preprocess_data(data, custom_lemmas)

    classifier.train(data)

    evaluation = classifier.evaluate(data)

    print(evaluation)




if __name__ == '__main__':
    run_evaluation()
