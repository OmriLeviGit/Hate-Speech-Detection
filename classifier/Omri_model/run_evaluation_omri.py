from pprint import pprint

import spacy
from spacy.util import is_package

from classifier.Omri_model.Spacy3Classes import Spacy3Classes
from classifier.comparison_visualization import visualize
from classifier.preprocessing.TextPreprocessor import TextPreprocessor

# { original word: change into }, for words that are not lemmatized correctly
custom_lemmas = {"hamas": "hamas"}

models_config = [
    # {
    #     "model_name": "debug 3 classes spacy",
    #     "base_model": "en_core_web_sm",
    #     "emoji_processing": None,  # 'text' to convert to text, 'config' to convert to the meaning in config.json()
    #     "distribution": {"negative": 10, "positive": 10, "irrelevant": 10},   # can remove if no irrelevant
    #     "source": "csv_files",
    #     "combine_irrelevant": False,    # combine_irrelevant=True to combine irrelevant with negative
    #     "hyper_parameters": {
    #         "epochs": 100,
    #         "learning_rate": 0.001,
    #         "l2_regularization": 1e-5,
    #         "batch_size": 32,
    #     }
    # },
    {
        "model_name": "3 classes spacy",
        "base_model": "en_core_web_sm",
        "emoji_processing": None,  # 'text' to convert to text, 'config' to convert to the meaning in config.json()
        "distribution": {"negative": 100, "positive": 100, "irrelevant": 100},   # can remove if no irrelevant
        "source": "csv_files",
        "combine_irrelevant": False,    # combine_irrelevant=True to combine irrelevant with negative
        "hyper_parameters": {
            "epochs": 100,
            "learning_rate": 0.001,
            "l2_regularization": 1e-5,
            "batch_size": 32,
        }
    },
    {
        "model_name": "3 classes spacy negative/irrelevant 50/50",
        "base_model": "en_core_web_sm",
        "emoji_processing": None,  # 'text' to convert to text, 'config' to convert to the meaning in config.json()
        "distribution": {"negative": 1000, "positive": 350, "irrelevant": 350},   # can remove if no irrelevant
        "source": "csv_files",
        "combine_irrelevant": False,    # combine_irrelevant=True to combine irrelevant with negative
        "hyper_parameters": {
            "epochs": 100,
            "learning_rate": 0.001,
            "l2_regularization": 1e-5,
            "batch_size": 32,
        }
    },
    {
        "model_name": "2 classes spacy",
        "base_model": "en_core_web_sm",
        "emoji_processing": None,  # 'text' to convert to text, 'config' to convert to the meaning in config.json()
        "distribution": {"negative": 1000, "positive": 1000},  # can remove if no irrelevant
        "source": "csv_files",
        "combine_irrelevant": False,
        "hyper_parameters": {
            "epochs": 100,
            "learning_rate": 0.001,
            "l2_regularization": 1e-5,
            "batch_size": 32,
        }
    },
]


def print_model_header(model_name, total_length=80):
    name_length = len(model_name)
    side_length = (total_length - name_length - 2) // 2

    extra_dash = 1 if (total_length - name_length - 2) % 2 != 0 else 0

    print("\n", "-" * side_length + f" {model_name} " + "-" * (side_length + extra_dash))


def load_model(model_name):
    if not is_package(model_name):
        print(f"Model {model_name} is not installed. Installing...")
        spacy.cli.download(model_name)

    return spacy.load(model_name)


def run_evaluation(models_config):
    metrics_list = []
    for i, config in enumerate(models_config):

        print_model_header(config['model_name'])

        base_model = config["base_model"]
        nlp = load_model(base_model)
        text_preprocessor = TextPreprocessor(emoji=config["emoji_processing"])

        classifier = Spacy3Classes(model=nlp, preprocessor=text_preprocessor)

        data = classifier.load_data(
            config["distribution"].get("negative"),
            config["distribution"].get("positive"),
            config["distribution"].get("irrelevant"),
            source=config['source']
        )

        prepared_data = classifier.prepare_datasets(data, combine_irrelevant=config["combine_irrelevant"])

        processed_data = classifier.preprocess_data(prepared_data, custom_lemmas)

        classifier.train(processed_data, **config["hyper_parameters"])

        metrics = classifier.evaluate(prepared_data)
        print(f"accuuracy: {metrics['accuracy']:.2f}")

        metrics_list.append((config, metrics))

    visualize(metrics_list)


if __name__ == '__main__':
    run_evaluation(models_config)
