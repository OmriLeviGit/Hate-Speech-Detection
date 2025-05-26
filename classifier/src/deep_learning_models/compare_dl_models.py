import os
import time
import itertools
import json
import random
import numpy as np
import pandas as pd
import sys
from datetime import datetime
import logging


import torch
from torch.utils.data import TensorDataset, DataLoader

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

# Set logging .txt file for the run
base_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(base_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join(log_dir, f"run_log_{timestamp}.txt")

sys.stdout = open(log_file, "w")
sys.stderr = sys.stdout

# Set fixed seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Mapping model type strings to their corresponding classes
model_registry = {
    "MLP": MLPModel,
    "LSTM": LSTMModel,
    "CNN": CNNModel
}

# Hyperparameters grids for the different models
mlp_param_grid = {
    'hidden_units': [64, 128, 256],
    'dropout_rate': [0.5, 0.7],
    'learning_rate': [0.001, 0.001],
    'batch_size': [16, 32],
    'dense_activation': ['relu'],
    'epochs': [10]
}

cnn_param_grid = {
    'embedding_dim': [100],
    'num_filters': [64, 128],
    'kernel_size': [3, 5],
    'dropout_rate': [0.5],
    'learning_rate': [0.001, 0.0001],
    'batch_size': [32],
    'epochs': [10],
    'max_sequence_length': [120],
    'dense_units': [64],
    'dense_activation': ['relu'],
    'second_conv': [False, True]
}

lstm_param_grid = {
    'embedding_dim': [300],
    'lstm_units': [64, 128],
    'dropout_rate': [0.2, 0.5],
    'learning_rate': [0.0001, 0.0005, 0.001],
    'batch_size': [32],
    'epochs': [10],
    'max_sequence_length': [120],
    'dense_units': [64],
    'dense_activation': ['relu']
}



# Trains the given Keras model and evaluates it on the validation set
# Returns F1 score, classification report, confusion matrix, and training duration
def train_and_evaluate(model, X_train, y_train, X_val, y_val, batch_size, epochs, input_dtype, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    # Convert data to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=input_dtype)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    X_val_tensor = torch.tensor(X_val, dtype=input_dtype)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCELoss()

    start_time = time.time()

    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()

    duration = time.time() - start_time

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_val_tensor.to(device)).cpu().numpy()
    y_pred = (preds > 0.5).astype(int)

    f1 = f1_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)
    matrix = confusion_matrix(y_val, y_pred)

    return f1, report, matrix, duration



# Runs k-fold cross-validation for a given model type and its hyperparameters combination
# Returns a list of result dictionaries where each row is a hyperparameter combination
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

        # Stratified - each fold maintains the same class distribution as the full dataset
        skf = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = 42)
        fold_scores = []
        fold_times = []
        last_report = None
        last_matrix = None

        # train_idx: indices of the training subset for of a fold
        # val_idx: indices of the validation subset for of a fold
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_proc, y_proc)):
            X_train, X_val = X_proc[train_idx], X_proc[val_idx]
            y_train, y_val = y_proc[train_idx], y_proc[val_idx]

            model = model_wrapper.build(input_shape=X_train.shape[1])

            f1, report, matrix, duration = train_and_evaluate(
                model, X_train, y_train, X_val, y_val,
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                input_dtype=model_wrapper.input_dtype
            )

            fold_scores.append(f1)
            fold_times.append(duration)
            last_report = report
            last_matrix = matrix

        avg_f1 = round(np.mean(fold_scores), 3)
        total_time = round(sum(fold_times), 2)
        # Trace gives us the correct predictions, as accuracy = (TP + TN) / (TP + TN + FP + FN)
        accuracy = round(last_matrix.trace() / last_matrix.sum(), 3)
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

    print(f"Summary for {model_type} Training")
    print(f"\nTotal training time: {format_duration(total_model_time)}")
    print(f"\nBest average F1 Score: {best_result['f1_score']}")
    print("\nBest Params:")
    for key, value in best_result["params"].items():
        print(f"  {key}: {value}")

    return results


# Takes the raw data, splits it to training/test sets and normalizes it
def prepare_data(helper, data_config, raw_data):
    X_raw, X_test, y_raw, y_test = helper.prepare_dataset(
        raw_data,
        test_size=0.2,
        balance_pct=data_config["balance_pct"],
        augment_ratio=data_config["augment_ratio"],
        irrelevant_ratio=data_config["irrelevant_ratio"]
    )

    label_encoder = LabelEncoder()
    y_trainval = label_encoder.fit_transform(y_raw)
    y_test = label_encoder.transform(y_test)

    normalizer = TextNormalizer(emoji="text")
    X_trainval = normalizer.normalize_texts(X_raw)
    X_test = normalizer.normalize_texts(X_test)

    return X_trainval, X_test, y_trainval, y_test


# Runs every combination in the hyperparameter grid for each model type
def test_model_combinations(model_grids, X_trainval, X_test, y_trainval, y_test, data_config):
    all_results = []

    for model_type, param_grid in model_grids:
        model_results = run_grid_search(model_type, param_grid, X_trainval, y_trainval)
        all_results.extend(model_results)

    for result in all_results:
        result['params'] = json.dumps(result['params'], separators=(',', ': '))

    df = pd.DataFrame(all_results)
    val_summary = df[["model_type", "f1_score", "accuracy", "params"]].sort_values(by="f1_score", ascending=False)

    print_header("Sorted Training Results On Validation set", 80)
    print(val_summary.drop(columns=["params"]).to_string(index=False))
    print("\nSee full params for each results in the CSV file")

    best_per_model = {}
    for result in all_results:
        mtype = result["model_type"]
        if mtype not in best_per_model or result["f1_score"] > best_per_model[mtype]["f1_score"]:
            best_per_model[mtype] = result

    test_results = evaluate_on_test_set(best_per_model, X_trainval, X_test, y_trainval, y_test, data_config)
    return df, val_summary, test_results


# Evaluates the best model config (based on validation F1) on the test set for each model type
def evaluate_on_test_set(best_per_model, X_trainval, X_test, y_trainval, y_test, data_config):
    test_results = []

    for model_type, best_result in best_per_model.items():
        print_header(f"Final Evaluation on Test Set: {model_type}")
        print("\nBest Params:", best_result["params"])

        best_params = json.loads(best_result["params"])
        ModelClass = model_registry[model_type]
        model_wrapper = ModelClass(best_params)

        X_trainval_proc, y_trainval_proc = model_wrapper.preprocess(X_trainval, y_trainval)
        X_test_proc, y_test_proc = model_wrapper.transform(X_test, y_test)

        model = model_wrapper.build(input_shape=X_trainval_proc.shape[1])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.train()

        X_tensor = torch.tensor(X_trainval_proc, dtype=model_wrapper.input_dtype)
        y_tensor = torch.tensor(y_trainval_proc, dtype=torch.float32).unsqueeze(1)
        train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=best_params["batch_size"], shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["learning_rate"])
        loss_fn = torch.nn.BCELoss()

        for _ in range(best_params["epochs"]):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            test_preds = model(torch.tensor(X_test_proc, dtype=model_wrapper.input_dtype).to(device)).cpu().numpy()
        y_pred = (test_preds > 0.5).astype(int)

        test_f1 = f1_score(y_test_proc, y_pred)
        test_accuracy = round((y_pred.flatten() == np.array(y_test_proc).flatten()).mean(), 3)
        test_report = classification_report(y_test_proc, y_pred)
        test_matrix = confusion_matrix(y_test_proc, y_pred)

        print("\nTest Classification Report:")
        print(test_report)
        print(f"\nTest F1 Score: {round(test_f1, 3)}")
        print(f"Test Accuracy: {test_accuracy}")
        print("\nTest Confusion Matrix:")
        print(test_matrix)

        test_results.append({
            "model_type": model_type,
            "data_config": json.dumps(data_config),
            "params": best_result["params"],
            "test_f1": round(test_f1, 3),
            "test_accuracy": test_accuracy,
            "cv_f1": round(best_result["f1_score"], 3),
            "cv_accuracy": round(best_result["accuracy"], 3)
        })

    return test_results


# Exports two .csv files:
# 1. Training set results on all models configs (tested on validation set)
# 2. Test set results on the best config for each model type (tested on the test set)
def export_results(data_config, val_summary, test_results):

    file_name = data_config_to_string(data_config)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    training_results_dir = os.path.join(base_dir, "training_set_metrics")
    os.makedirs(training_results_dir, exist_ok=True)
    val_path = os.path.join(training_results_dir, f"{file_name}.csv")
    val_summary.to_csv(val_path, index=False)
    print("\nValidation results exported to CSV")

    test_results_dir = os.path.join(base_dir, "test_set_metrics")
    os.makedirs(test_results_dir, exist_ok=True)
    df_test = pd.DataFrame(test_results).sort_values(by="test_f1", ascending=False)
    test_path = os.path.join(test_results_dir, f"{file_name}.csv")
    df_test.to_csv(test_path, index=False)
    print("\nTest set results exported to CSV")


# Returns a unique string per data configuration to attach to every csv file exported for results comparison
def data_config_to_string(config):
    return "_".join(f"{key}{str(value).replace('.', '')}" for key, value in config.items())


def main():

    start_time = time.time()

    # Make sure using GPU when avaiable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU only'})")

    # Load data
    helper = DataHelper()
    raw_data = helper.load_data()

    # Set all configurations to be tested
    data_configs = [
        {"balance_pct": 0.5, "augment_ratio": 0.0, "irrelevant_ratio": 0.0},
        {"balance_pct": 0.5, "augment_ratio": 0.0, "irrelevant_ratio": 0.5},
        {"balance_pct": 0.5, "augment_ratio": 0.0, "irrelevant_ratio": 0.7},
        {"balance_pct": 0.5, "augment_ratio": 0.5, "irrelevant_ratio": 0.0},
        {"balance_pct": 0.5, "augment_ratio": 0.5, "irrelevant_ratio": 0.5},
        {"balance_pct": 0.5, "augment_ratio": 0.5, "irrelevant_ratio": 0.7},
    ]

    model_grids = [
        ("MLP", mlp_param_grid),
        ("CNN", cnn_param_grid),
        ("LSTM", lstm_param_grid),
    ]

    # Run experiments
    for config in data_configs:
        print(f"\nRunning with data config: {config}")
        X_trainval, X_test, y_trainval, y_test = prepare_data(helper, config, raw_data)
        df_val, val_summary, test_results = test_model_combinations(model_grids, X_trainval, X_test, y_trainval, y_test, config)
        export_results(config, val_summary, test_results)
        print_header("End of config")

    total_duration = time.time() - start_time
    print("\nTotal experiment time:", format_duration(total_duration))

if __name__ == "__main__":
    main()