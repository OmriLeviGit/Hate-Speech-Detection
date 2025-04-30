
from classifier.BERTClassifier import BERTClassifier
from classifier.normalization.TextNormalizer import TextNormalizer

"""
{"model_name": "distilbert-base-uncased", "learning_rate": 2e-5, "batch_size": 32, "epochs": 5,
 "weight_decay": 0.01},
{"model_name": "distilbert-base-uncased", "learning_rate": 5e-5, "batch_size": 32, "epochs": 5,
 "weight_decay": 0.01},

 {"model_name": "vinai/bertweet-base", "learning_rate": 2e-5, "batch_size": 32, "epochs": 5,
 "weight_decay": 0.01},
{"model_name": "vinai/bertweet-base", "learning_rate": 5e-5, "batch_size": 32, "epochs": 5,
 "weight_decay": 0.01},
"""

# running with single values
debug_config = {
    "model_name": "distilbert-base-uncased",
    "hyper_parameters": {
        "learning_rate_range": (5e-5, 5e-5),
        "learning_rate_log": True,
        "batch_sizes": [8],
        "epochs_range": (1, 1),
        "weight_decay_range": (0.01, 0.01),
    }
}

def main():
    labels = ["antisemitic", "not_antisemitic"]
    normalizer = TextNormalizer(emoji='text')

    config = {
        "model_name": "distilbert-base-uncased",
        "hyper_parameters": {
            "learning_rate_range": (5e-6, 1e-4),
            "learning_rate_log": True,
            "batch_sizes": [16, 32],
            "epochs_range": (2, 5),
            "weight_decay_range": (0.001, 0.1),
        }
    }

    classifier = BERTClassifier(labels, normalizer, debug_config)

    data = classifier.load_data(set_to_min=True, source='debug')

    X_train, X_test, y_train, y_test = classifier.prepare_dataset(data)

    X_train = classifier.preprocess(X_train)

    classifier.train(X_train, y_train, n_trials=1)

    classifier.evaluate(X_test, y_test)

    predictions = classifier.predict(X_test)


if __name__ == "__main__":
    main()

