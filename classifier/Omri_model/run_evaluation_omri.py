import random
import time

import numpy as np
import spacy
from spacy.util import is_package

from classifier import utils
from classifier.Omri_model.SpacyModels_SM_LG import SpacyModels_SM_LG
from classifier.Omri_model.SpacyModels_TRF import SpacyModels_TRF
from classifier.preprocessing.TextNormalizer import TextNormalizer
from classifier.utils import print_model_header

# { original word: change into }, for words that are not lemmatized correctly for the small model
custom_lemmas = {"hamas": "hamas"}

"""
Example:

{
    "model_name": "3 classes lg",       # a description of the model
    "base_model": "en_core_web_lg",     # model type
    "emoji_processing": 'text',  # None: no processing, 'text': demojize, 'config': custom meaning in config.json()
    "distribution": {"negative": 700, "positive": 700, "irrelevant": 700},  # irrelevant can be None
    "source": "csv_files",  # possible values: None: db, 'debug': mock data, 'csv_files': predownloaded data
    "combine_irrelevant": False,  # combine_irrelevant=True to treat irrelevant as negative
    "hyper_parameters": {       # all hyper parameters can be inserted here
        "learning_rate": 0.001,
        "l2_regularization": 0.001,
    }
},
"""

# configs = [
#     {
#         "model_name": "3 classes small model",
#         "base_model": "en_core_web_sm",
#         "emoji_processing": 'text',
#         "distribution": {"negative": 700, "positive": 700, "irrelevant": 700},
#         "source": "csv_files",
#         "combine_irrelevant": False,
#         "hyper_parameters": {
#             "learning_rate": 0.001,
#             "l2_regularization": 0.001,
#         }
#     },
#     {
#         "model_name": "2 classes bert",
#         "base_model": "en_core_web_trf",
#         "emoji_processing": 'text',
#         "distribution": {"negative": 700, "positive": 700},
#         "source": "csv_files",
#         "combine_irrelevant": False,
#         "hyper_parameters": {
#             "learning_rate": 0.001,
#             "l2_regularization": 0.001,
#         }
#     },
#     {
#         "model_name": "2 classes bert with 350 irrelevant, combined with positive",
#         "base_model": "en_core_web_trf",
#         "emoji_processing": 'text',
#         "distribution": {"negative": 700, "positive": 700, "irrelevant": 350},
#         "source": "csv_files",
#         "combine_irrelevant": True,
#         "hyper_parameters": {
#             "learning_rate": 0.001,
#             "l2_regularization": 0.001,
#         }
#     },
# ]

configs = [
    {
        "model_name": "2 classes sm",
        "base_model": "en_core_web_sm",
        "emoji_processing": 'text',
        "distribution": {"positive": 700, "negative": 700},
        "source": "csv_files",
        "combine_irrelevant": True,
        "hyper_parameters": {
            "learning_rate": 0.001,
            "l2_regularization": 0.001,
        }
    },    {
        "model_name": "3 classes trf",
        "base_model": "en_core_web_trf",
        "emoji_processing": 'text',
        "distribution": {"positive": 700, "negative": 700},
        "source": "csv_files",
        "combine_irrelevant": True,
        "hyper_parameters": {
            "learning_rate": 0.001,
            "l2_regularization": 0.001,
        }
    },

]


def load_model(model_name):
    if not is_package(model_name):
        print(f"Model '{model_name}' is not installed. Installing...")
        spacy.cli.download(model_name)

    print(f"Loading model: '{model_name}'...")

    return spacy.load(model_name)


def run_evaluation(models_config):
    total_start_time = time.time()
    seed = 42

    metrics_list = []
    accuracy_list = []

    # optimizer = nlp.begin_training(component_cfg={"seed": seed})

    for config in models_config:
        print_model_header(config['model_name'])

        random.seed(seed)
        np.random.seed(seed)

        base_model = config["base_model"]
        nlp = load_model(base_model)
        text_normalizer = TextNormalizer(emoji=config["emoji_processing"])

        if base_model.endswith('trf'):
            classifier = SpacyModels_TRF(model=nlp, preprocessor=text_normalizer, seed=seed)
        else:
            classifier = SpacyModels_SM_LG(model=nlp, preprocessor=text_normalizer, seed=seed)

        print("Loading and preprocessing data...")
        data = classifier.load_data(
            config["distribution"].get("positive"),
            config["distribution"].get("negative"),
            config["distribution"].get("irrelevant", None),
            source=config['source']
        )

        prepared_data = classifier.prepare_datasets(data, combine_irrelevant=config["combine_irrelevant"])

        if base_model.endswith('sm'):
            processed_data = classifier.preprocess_data(prepared_data, custom_lemmas)
        else:
            processed_data = classifier.preprocess_data(prepared_data)

        classifier.train(processed_data, **config["hyper_parameters"])

        metrics = classifier.evaluate(prepared_data)
        print(f"accuracy: {metrics['accuracy']:.2f}")

        metrics_list.append((config, metrics))
        accuracy_list.append((config['model_name'], f"{metrics['accuracy']:.2f}"))

    total_time = time.time() - total_start_time
    print("\n" + "=" * 60)
    print(f"Total execution time: {total_time:.2f}s ({total_time / 60:.2f} minutes)")
    print("=" * 60 + "\n")

    sorted_accuracy = sorted(accuracy_list, key=lambda x: x[1], reverse=True)

    for i in sorted_accuracy:
        print(i)

    utils.visualize(metrics_list)
    utils.create_summary_table(metrics_list)


if __name__ == '__main__':
    learning_rates = [0.005, 0.001, 0.0005]
    l2_values = [0.005, 0.001, 0.0005]
    # configs = utils.generate_model_configs(configs, learning_rates, l2_values)    # check with differnt parameters
    run_evaluation(configs)
