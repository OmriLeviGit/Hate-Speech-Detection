import os
import sys
import io

from transformers import AutoTokenizer

from classifier.src.classifiers.BaseTextClassifier import BaseTextClassifier
from classifier.src.classifiers.BertClassifier import BertClassifier
from classifier.src.normalization.TextNormalizer import TextNormalizer
from classifier.src import utils

params = {
    "dropout": 0.0078115464911744925,
    "learning_rate": 2.3597575095971233e-05,
    "batch_size": 16,
    "epochs": 2,
    "weight_decay": 0.06700262172627848
}

values = [
    ('control (0, 0, 0.5)', 0, 0, 0.5),
    ('(0, 0.15, 0.5)', 0, 0.15, 0.5),
    ('(0, 0.3, 0.5)', 0, 0.3, 0.5),
    ('(0, 0.15, 0.6)', 0, 0.15, 0.6),
    ('(0, 0.3, 0.6)', 0, 0.3, 0.6),
    ('(0, 0.15, 0.7)', 0, 0.15, 0.7),
    ('(0, 0.3, 0.7)', 0, 0.3, 0.7),
    ('(0, 0.3, 0.8)', 0, 0.3, 0.8),
    ('(0.5, 0, 0.5)', 0.5, 0, 0.5),
    ('(0.5, 0.15, 0.5)', 0.5, 0.15, 0.5),
    ('(0.5, 0.3, 0.5)', 0.5, 0.3, 0.5),
    ('(0.5, 0.15, 0.6)', 0.5, 0.15, 0.6),
    ('(0.5, 0.3, 0.6)', 0.5, 0.3, 0.6),
    ('(0.5, 0.15, 0.7)', 0.5, 0.15, 0.7),
    ('(0.5, 0.3, 0.7)', 0.5, 0.3, 0.7),
    ('(0.5, 0.3, 0.8)', 0.5, 0.3, 0.8),
    ('(1, 0, 0.5)', 1, 0, 0.5),
    ('(1, 0.15, 0.5)', 1, 0.15, 0.5),
    ('(1, 0.3, 0.5)', 1, 0.3, 0.5),
    ('(1, 0.15, 0.6)', 1, 0.15, 0.6),
    ('(1, 0.3, 0.6)', 1, 0.3, 0.6),
    ('(1, 0.15, 0.7)', 1, 0.15, 0.7),
    ('(1, 0.3, 0.7)', 1, 0.3, 0.7),
    ('(1, 0.3, 0.8)', 1, 0.3, 0.8)
]

output_path = os.path.join(BaseTextClassifier.save_models_path, "evaluation_results.txt")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

def run(model, v, params, debug=None):
    utils.reset_seeds(model.seed)

    data = model.load_data(debug=debug)
    _, ar, ir, pct = v

    X_train, X_test, y_train, y_test = model.prepare_dataset(
        data, test_size=0.2, irrelevant_ratio=ir, augment_ratio=ar, balance_pct=pct)

    utils.reset_seeds(model.seed)

    model.train_final_model(X_train, y_train, params)

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
    debug = False
    # utils.check_device()

    # change seed, check why the control set performs badly
    results = []

    for v in values:
        print("running ", v[0])
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

        name, res = run(model, v, params, debug=debug)

        results.append((name, res))

    sorted_results = sorted(results, key=lambda x: x[1][0], reverse=True)

    with open(output_path, 'a') as f:
        for item in sorted_results:
            f.write(str(item) + '\n')

if __name__ == "__main__":
    main()
