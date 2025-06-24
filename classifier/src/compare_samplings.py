import os
import sys
import io

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from transformers import AutoTokenizer

from classifier.src.classifiers.BaseTextClassifier import BaseTextClassifier
from classifier.src.classifiers.BertClassifier import BertClassifier
from classifier.src.classifiers.ClassicalModelClassifier import ClassicalModelClassifier
from classifier.src.model_generation import generate_models
from classifier.src.normalization.TextNormalizer import TextNormalizer
from classifier.src import utils

params = {
    "dropout": 0.0073115464911744925,
    "learning_rate": 6.3597575095971233e-05,
    "batch_size": 16,
    "epochs": 5,
    "weight_decay": 0.02700262172627848
}

# large search ranges
values1 = [
    ('control (0, 0, 0.5)', 0, 0, 0.5),

    # Data Augmentation = 0
    ('(0, 0.15, 0.5)', 0, 0.15, 0.5),
    ('(0, 0.3, 0.5)', 0, 0.3, 0.5),
    ('(0, 0.15, 0.6)', 0, 0.15, 0.6),
    ('(0, 0.3, 0.6)', 0, 0.3, 0.6),
    ('(0, 0.15, 0.7)', 0, 0.15, 0.7),
    ('(0, 0.3, 0.7)', 0, 0.3, 0.7),
    ('(0, 0.3, 0.8)', 0, 0.3, 0.8),

    # Data Augmentation = 0.5
    ('(0.5, 0, 0.5)', 0.5, 0, 0.5),
    ('(0.5, 0.15, 0.5)', 0.5, 0.15, 0.5),
    ('(0.5, 0.3, 0.5)', 0.5, 0.3, 0.5),
    ('(0.5, 0.15, 0.6)', 0.5, 0.15, 0.6),
    ('(0.5, 0.3, 0.6)', 0.5, 0.3, 0.6),
    ('(0.5, 0.15, 0.7)', 0.5, 0.15, 0.7),
    ('(0.5, 0.3, 0.7)', 0.5, 0.3, 0.7),
    ('(0.5, 0.3, 0.8)', 0.5, 0.3, 0.8),

    # Data Augmentation = 1
    ('(1, 0, 0.5)', 1, 0, 0.5),
    ('(1, 0.15, 0.5)', 1, 0.15, 0.5),
    ('(1, 0.3, 0.5)', 1, 0.3, 0.5),
    ('(1, 0.15, 0.6)', 1, 0.15, 0.6),
    ('(1, 0.3, 0.6)', 1, 0.3, 0.6),
    ('(1, 0.15, 0.7)', 1, 0.15, 0.7),
    ('(1, 0.3, 0.7)', 1, 0.3, 0.7),
    ('(1, 0.3, 0.8)', 1, 0.3, 0.8)
]

# minimizing search range
values2 = [
    ('control (0, 0, 0.5)', 0, 0, 0.5),

    # Data Augmentation = 0.3
    ('(0.3, 0.2, 0.5)', 0.3, 0.2, 0.5),
    ('(0.3, 0.3, 0.5)', 0.3, 0.3, 0.5),
    ('(0.3, 0.4, 0.5)', 0.3, 0.4, 0.5),
    ('(0.3, 0.2, 0.55)', 0.3, 0.2, 0.55),
    ('(0.3, 0.3, 0.55)', 0.3, 0.3, 0.55),
    ('(0.3, 0.4, 0.55)', 0.3, 0.4, 0.55),
    ('(0.3, 0.2, 0.6)', 0.3, 0.2, 0.6),
    ('(0.3, 0.3, 0.6)', 0.3, 0.3, 0.6),
    ('(0.3, 0.4, 0.6)', 0.3, 0.4, 0.6),

    # Data Augmentation = 0.5
    ('(0.5, 0.2, 0.5)', 0.5, 0.2, 0.5),
    ('(0.5, 0.3, 0.5)', 0.5, 0.3, 0.5),
    ('(0.5, 0.4, 0.5)', 0.5, 0.4, 0.5),
    ('(0.5, 0.2, 0.55)', 0.5, 0.2, 0.55),
    ('(0.5, 0.3, 0.55)', 0.5, 0.3, 0.55),
    ('(0.5, 0.4, 0.55)', 0.5, 0.4, 0.55),
    ('(0.5, 0.2, 0.6)', 0.5, 0.2, 0.6),
    ('(0.5, 0.3, 0.6)', 0.5, 0.3, 0.6),
    ('(0.5, 0.4, 0.6)', 0.5, 0.4, 0.6),

    # Data Augmentation = 0.7
    ('(0.7, 0.2, 0.5)', 0.7, 0.2, 0.5),
    ('(0.7, 0.3, 0.5)', 0.7, 0.3, 0.5),
    ('(0.7, 0.4, 0.5)', 0.7, 0.4, 0.5),
    ('(0.7, 0.2, 0.55)', 0.7, 0.2, 0.55),
    ('(0.7, 0.3, 0.55)', 0.7, 0.3, 0.55),
    ('(0.7, 0.4, 0.55)', 0.7, 0.4, 0.55),
    ('(0.7, 0.2, 0.6)', 0.7, 0.2, 0.6),
    ('(0.7, 0.3, 0.6)', 0.7, 0.3, 0.6),
    ('(0.7, 0.4, 0.6)', 0.7, 0.4, 0.6)
]

# testing higher ranges of irrelevant, with combination of higher percentage
values3 = [
    # Control point
    ('control (0.3, 0.3, 0.5)', 0.3, 0.3, 0.5),

    # Baseline with 0.4 type B (from previous results)
    ('(0.3, 0.4, 0.5)', 0.3, 0.4, 0.5),
    ('(0.3, 0.4, 0.55)', 0.3, 0.4, 0.55),
    ('(0.3, 0.4, 0.6)', 0.3, 0.4, 0.6),

    # Push type B ratio higher
    ('(0.3, 0.45, 0.5)', 0.3, 0.45, 0.5),
    ('(0.3, 0.5, 0.5)', 0.3, 0.5, 0.5),

    # Increased balance with higher type B
    ('(0.3, 0.45, 0.55)', 0.3, 0.45, 0.55),
    ('(0.3, 0.5, 0.55)', 0.3, 0.5, 0.55),
    ('(0.3, 0.45, 0.6)', 0.3, 0.45, 0.6),
    ('(0.3, 0.5, 0.6)', 0.3, 0.5, 0.6),

    # Try 0.4 augmentation with higher type B
    ('(0.4, 0.4, 0.5)', 0.4, 0.4, 0.5),
    ('(0.4, 0.45, 0.5)', 0.4, 0.45, 0.5),
    ('(0.4, 0.5, 0.5)', 0.4, 0.5, 0.5),
    ('(0.4, 0.4, 0.55)', 0.4, 0.4, 0.55),
    ('(0.4, 0.45, 0.55)', 0.4, 0.45, 0.55),
    ('(0.4, 0.5, 0.55)', 0.4, 0.5, 0.55),
    ('(0.4, 0.4, 0.6)', 0.4, 0.4, 0.6),
    ('(0.4, 0.45, 0.6)', 0.4, 0.45, 0.6),
    ('(0.4, 0.5, 0.6)', 0.4, 0.5, 0.6)
]

validation_values = [
    # Control
    ('(0, 0, 0.5)', 0, 0, 0.5),

    # Top performers
    ('(0.3, 0.4, 0.5)', 0.3, 0.4, 0.5),
    ('(0.3, 0.45, 0.6)', 0.3, 0.45, 0.6),

    # Augmentation effect
    ('(0.2, 0.4, 0.5)', 0, 0.3, 0.5),
    ('(0.4, 0.4, 0.5)', 0.7, 0.3, 0.5),

    # Mix with irrelevant effect
    ('(0.3, 0.5, 0.6)', 0.3, 0.5, 0.5),

    # Balance percent effect
    ('(0.3, 0.3, 0.7)', 0.3, 0.3, 0.7),
]

# large search ranges
values4 = [
    # Data Augmentation = 0
    ('(0.33, 0, None)', 0.33, 0, None),
    ('(0.33, 0.33, None)', 0.33, 0.33, None),
    ('(0.33, 0.5, None)', 0.33, 0.5, None),
]

best_values = [
    ('(0.3, 0.4, 0.5)', 0.3, 0.4, 0.5)
]


file_name = "sampling_results.txt"

def run(model, v, params, debug=None):
    utils.reset_seeds(model.seed)

    data = model.load_data(debug=debug)
    _, ar, ir, pct = v

    X_train, X_test, y_train, y_test = model.prepare_dataset(
        data, augment_ratio=ar, irrelevant_ratio=ir, balance_pct=pct)

    utils.reset_seeds(model.seed)

    # model.train_final_model(X_train, y_train, params)   # bert
    model.train(X_train, y_train) # sklearn

    res = model.evaluate(X_test, y_test)

    return model.model_name, res

def main():
    debug = False

    # utils.check_device()

    models = generate_models(1)
    values = best_values

    results = []

    for model in models:
        original_name = model.model_name
        for v in values:
            for seed in range(10):
                print(f"running {original_name}, seed {seed}, value {v[0]}")

                model.model_name = original_name + f"{seed}"

                name, res = run(model, v, params, debug=debug)

                results.append((name, res))

    sorted_results = sorted(results, key=lambda x: x[1][1], reverse=True)

    output_path = os.path.join(BaseTextClassifier.save_models_path, file_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'a') as f:
        for item in sorted_results:
            f.write(str(item) + '\n')

if __name__ == "__main__":
    main()
