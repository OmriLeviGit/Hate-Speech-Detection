import os
import time
import itertools
import warnings
import json
import random
import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.exceptions import UndefinedMetricWarning

# Import model classes
from classifier.src.deep_learning_models.mlp_model import MLPModel
from classifier.src.deep_learning_models.lstm_model import LSTMModel
from classifier.src.deep_learning_models.cnn_model import CNNModel
from classifier.src.deep_learning_models.DataHelper import DataHelper

from classifier.src.normalization.TextNormalizer import TextNormalizer
from classifier.src.utils import print_header, format_duration


# Set fixed seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.random.set_seed(SEED)

# Prevents printing TensorFlow and sklearn warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# Mapping model type strings to their corresponding classes
model_registry = {
    "MLP": MLPModel,
    "LSTM": LSTMModel,
    "CNN": CNNModel
}

# Hyperparameter search spaces for each model

# mlp_param_grid = {
#     'hidden_units': [64, 128, 256],
#     'dropout_rate': [0.3, 0.5, 0.6],
#     'learning_rate': [0.01, 0.001, 0.0005],
#     'batch_size': [32, 64],
#     'epochs': [5, 10]
# }
#
# lstm_param_grid = {
#     'embedding_dim': [100, 200],
#     'lstm_units': [64, 128],
#     'dropout_rate': [0.3, 0.5],
#     'learning_rate': [0.001, 0.0005],
#     'batch_size': [32],
#     'epochs': [5, 10],
#     'max_sequence_length': [100]
# }
#
# cnn_param_grid = {
#     'embedding_dim': [100, 200],
#     'num_filters': [128, 256],
#     'kernel_size': [3, 5],
#     'dropout_rate': [0.3, 0.5],
#     'learning_rate': [0.001, 0.0005],
#     'batch_size': [32],
#     'epochs': [5, 10],
#     'max_sequence_length': [100]
# }

mlp_param_grid = {
    'hidden_units': [32],
    'dropout_rate': [0.3],
    'learning_rate': [0.001],
    'batch_size': [32],
    'epochs': [1]
}

lstm_param_grid = {
    'embedding_dim': [300],
    'lstm_units': [32],
    'dropout_rate': [0.2],
    'learning_rate': [0.001],
    'batch_size': [64],
    'epochs': [10],
    'max_sequence_length': [60]
}

cnn_param_grid = {
    'embedding_dim': [100],
    'num_filters': [128],
    'kernel_size': [5],
    'dropout_rate': [0.5],
    'learning_rate': [0.0005],
    'batch_size': [32],
    'epochs': [5],
    'max_sequence_length': [120]
}


# Trains the given Keras model and evaluates it on the validation set
# Returns F1 score, classification report, confusion matrix, and training duration
def train_and_evaluate(model, X_train, y_train, X_val, y_val, batch_size, epochs):

    start_time = time.time()

    model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs = epochs,
        batch_size = batch_size,
        verbose = 0  # suppress Keras logs
    )

    duration = time.time() - start_time

    # Predict on validation set — outputs probabilities between 0 and 1
    y_pred = (model.predict(X_val) > 0.5).astype(int)

    f1 = f1_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)
    matrix = confusion_matrix(y_val, y_pred)

    return f1, report, matrix, duration


# Runs k-fold cross-validation over all combinations of parameters for a given model type
# Returns a list of result dictionaries (one per combination)
def run_grid_search(model_type, param_grid, X_raw, y, num_folds = 5):

    print_header(model_type)
    ModelClass = model_registry[model_type]

    results = []
    best_result = None
    total_model_time = 0

    keys, values = zip(*param_grid.items())

    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))

        model_wrapper = ModelClass(params)
        X_proc, y_proc = model_wrapper.preprocess(X_raw, y)

        skf = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = 42)
        fold_scores = []
        fold_times = []
        last_report = None
        last_matrix = None

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_proc, y_proc)):
            X_train, X_val = X_proc[train_idx], X_proc[val_idx]
            y_train, y_val = y_proc[train_idx], y_proc[val_idx]

            model = model_wrapper.build(input_shape=X_train.shape[1])

            f1, report, matrix, duration = train_and_evaluate(
                model, X_train, y_train, X_val, y_val,
                batch_size=params['batch_size'],
                epochs=params['epochs']
            )

            fold_scores.append(f1)
            fold_times.append(duration)
            last_report = report
            last_matrix = matrix

        avg_f1 = round(np.mean(fold_scores), 4)
        total_time = round(sum(fold_times), 2)
        accuracy = round(last_matrix.trace() / last_matrix.sum(), 4)
        total_model_time += total_time

        result = {
            "model_type": model_type,
            "params": params,
            "f1_score": avg_f1,
            "accuracy": accuracy,
            "avg_training_time_sec": total_time,
            "report": last_report,
            "confusion_matrix": last_matrix
        }
        results.append(result)

        if best_result is None or avg_f1 > best_result["f1_score"]:
            best_result = result

    print(f"\n------- Summary for {model_type} Training -------")
    print(f"\nTotal training time: {format_duration(total_model_time)}")
    print(f"\nBest F1 Score: {best_result['f1_score']}")
    print("\nBest Params:")
    for key, value in best_result["params"].items():
        print(f"  {key}: {value}")
    print("\nClassification Report (best config):")
    print(best_result["report"])
    print("Confusion Matrix (best config):")
    print(best_result["confusion_matrix"])

    return results


def main():

    helper = DataHelper()
    raw_data = helper.load_data()

    X_raw, X_test, y_raw, y_test = helper.prepare_dataset(
        raw_data,
        test_size=0.2,
        balance_pct=0.5,
        augment_ratio=0.0,
        irrelevant_ratio=0
    )

    # Convert string labels into numbers
    label_encoder = LabelEncoder()
    y_trainval = label_encoder.fit_transform(y_raw)
    y_test = label_encoder.transform(y_test)

    # Apply common normalization for all models: lowercasing, emoji to text, removing user mention and links
    normalizer = TextNormalizer(emoji="text")
    X_normalized = normalizer.normalize_texts(X_raw)
    X_test_normalized = normalizer.normalize_texts(X_test)

    # Define models and their grids
    model_grids = [
        ("MLP", mlp_param_grid),
        ("CNN", cnn_param_grid),
        ("LSTM", lstm_param_grid)
    ]

    all_results = []

    # Run grid search for each model on train+val only
    for model_type, param_grid in model_grids:
        model_results = run_grid_search(model_type, param_grid, X_normalized, y_trainval)
        all_results.extend(model_results)

    # Convert params, currently in dicts, to compact string format to fit inside the csv we export later
    for result in all_results:
        result['params'] = json.dumps(result['params'], separators=(',', ': '))

    # Remove report before saving (but it keeps accuracy and confusion_matrix)
    df = pd.DataFrame(all_results)
    df.drop(columns=["report"], inplace=True)
    df.to_csv("dl_model_results_summary.csv", index=False)

    # Print summary of all models, showing the best combination of hyper-paramters (config) of each one
    print_header("Final Results", 80)
    summary = df[["model_type", "f1_score", "accuracy"]].sort_values(by="f1_score", ascending=False)
    print(summary.to_string(index=False))

    # Group by model_type and get best result per model
    best_per_model = {}

    for result in all_results:
        model_type = result["model_type"]
        if model_type not in best_per_model or result["f1_score"] > best_per_model[model_type]["f1_score"]:
            best_per_model[model_type] = result

    # Evaluate each model's best config on the same test set
    for model_type, best_result in best_per_model.items():

        print_header(f"Final Evaluation on Test Set: {model_type}")
        print("\nBest Params:", best_result["params"])
        print("\n")

        best_params = json.loads(best_result["params"])
        ModelClass = model_registry[model_type]
        model_wrapper = ModelClass(best_params)

        X_trainval_proc, y_trainval_proc = model_wrapper.preprocess(X_normalized, y_trainval)
        X_test_proc, y_test_proc = model_wrapper.transform(X_test_normalized, y_test)

        model = model_wrapper.build(input_shape=X_trainval_proc.shape[1])
        model.fit(
            X_trainval_proc, y_trainval_proc,
            epochs=best_params["epochs"],
            batch_size=best_params["batch_size"],
            verbose=0
        )

        y_pred = (model.predict(X_test_proc) > 0.5).astype(int)
        test_f1 = f1_score(y_test_proc, y_pred)

        y_pred = y_pred.flatten()
        y_test_proc = np.array(y_test_proc).flatten()

        correct_predictions = (y_pred == y_test_proc).sum()
        test_accuracy = round(correct_predictions / len(y_test_proc), 4)

        test_report = classification_report(y_test_proc, y_pred)
        test_matrix = confusion_matrix(y_test_proc, y_pred)

        print("\nTest Classification Report:")
        print(test_report)

        print(f"\nTest F1 Score: {test_f1}")
        print(f"Test Accuracy: {test_accuracy}")

        print("\nTest Confusion Matrix:")
        print(test_matrix)


if __name__ == "__main__":
    main()