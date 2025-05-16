import os
import sys
import io

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from transformers import AutoTokenizer

from classifier.src.classifiers.BaseTextClassifier import BaseTextClassifier
from classifier.src.classifiers.BertClassifier import BertClassifier
from classifier.src.classifiers.SKLearnClassifier import SKLearnClassifier
from classifier.src.normalization.TextNormalizer import TextNormalizer
from classifier.src import utils

params = {
    "dropout": 0.0078115464911744925,
    "learning_rate": 2.3597575095971233e-05,
    "batch_size": 16,
    "epochs": 2,
    "weight_decay": 0.06700262172627848
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

# chosen parameters to validate the search across models
validation_values = [
    # Baseline/control (for comparison)
    ('control group (0, 0, 0.5)', 0, 0, 0.5),  # Original baseline

    # Top performers
    ('(0.3, 0.4, 0.5)', 0.3, 0.4, 0.5),  # Best overall performance
    ('(0.3, 0.3, 0.5)', 0.3, 0.3, 0.5),  # Best class 0 recall with high accuracy
    ('(0.3, 0.45, 0.6)', 0.3, 0.45, 0.6),  # Best imbalanced configuration

    # Critical data points showing impact of each parameter

    # Augmentation effect (keeping other params constant)
    ('(0, 0.3, 0.5)', 0, 0.3, 0.5),  # No augmentation
    ('(0.7, 0.3, 0.5)', 0.7, 0.3, 0.5),  # Excessive augmentation

    # Type B ratio effect (keeping other params constant)
    ('(0.3, 0.2, 0.5)', 0.3, 0.2, 0.5),  # Lower diversity
    ('(0.3, 0.5, 0.5)', 0.3, 0.5, 0.5),  # Higher diversity

    # Balance effect (keeping other params constant)
    ('(0.3, 0.3, 0.6)', 0.3, 0.3, 0.6),  # Moderate imbalance
    ('(0.3, 0.3, 0.7)', 0.3, 0.3, 0.7),  # Higher imbalance
]

output_path = os.path.join(BaseTextClassifier.save_models_path, "evaluation_results.txt")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

def run(model, v, params, debug=None):
    utils.reset_seeds(model.seed)

    data = model.load_data(debug=debug)
    _, ar, ir, pct = v

    X_train, X_test, y_train, y_test = model.prepare_dataset(
        data, test_size=0.2, augment_ratio=ar, irrelevant_ratio=ir, balance_pct=pct)

    utils.reset_seeds(model.seed)

    model.train_final_model(X_train, y_train, params)
    # model.train(X_train, y_train)

    captured_output = io.StringIO()
    original_stdout = sys.stdout

    try:
        sys.stdout = captured_output
        res = model.evaluate(X_test, y_test)
        output_string = captured_output.getvalue()
    finally:
        sys.stdout = original_stdout


    with open(output_path, 'a') as f:
        f.write(output_string)

    return model.model_name, res

def main():
    debug = True
    # utils.check_device()

    # change seed, check why the control set performs badly
    results = []


    for v in validation_values:
        print("running ", v[0])
        # bert
        config = {
            'model_name': v[0],
            'model_type': "distilbert-base-uncased"
        }

        model = BertClassifier(
            ["antisemitic", "not_antisemitic"],
            TextNormalizer(emoji='text'),
            AutoTokenizer.from_pretrained(config["model_type"]),
            config
        )

        # config = {
        #     "model_name": v[0],
        #     "model_class": SGDClassifier(),
        #     "param_grid": {
        #         'loss': ['hinge', 'log_loss'],
        #         'penalty': ['l2', 'elasticnet'],
        #         'alpha': [1e-4, 1e-3],
        #         'max_iter': [1000]
        #     }
        # }
        # model = SKLearnClassifier(
        #     ["antisemitic", "not_antisemitic"],
        #     TextNormalizer(emoji='text'),
        #     TfidfVectorizer(),
        #     config
        # )

        name, res = run(model, v, params, debug=debug)

        results.append((name, res))

    sorted_results = sorted(results, key=lambda x: x[1][1], reverse=True)

    with open(output_path, 'a') as f:
        for item in sorted_results:
            f.write(str(item) + '\n')

if __name__ == "__main__":
    main()
