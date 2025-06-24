import tempfile
import time, os, pickle

import numpy as np
import optuna
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.model_selection import KFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments, AutoConfig,
)

from classifier.src.classifiers.BaseTextClassifier import BaseTextClassifier
from classifier.src.normalization.TextNormalizer import TextNormalizer
from classifier.src.utils import format_duration


class BertClassifier(BaseTextClassifier):
    def __init__(self, labels: list, normalizer: TextNormalizer(), tokenizer, config, seed: int = 42):
        super().__init__(labels, seed)

        self.config = config
        self.normalizer = normalizer

        self.model_name = config.get("model_name")
        self.model_type = config.get("model_type")
        self.tokenizer = tokenizer
        # torch.set_num_threads(1)

        self.hp_ranges = config.get("hyper_parameters")
        self.n_trials = config.get("n_trials", 5)

        self.best_model = None
        self.cv_score = None
        self.best_params = None

    def preprocess(self, texts: list[str]) -> list[str]:
        """Normalize texts using the provided normalizer"""
        return self.normalizer.normalize_texts(texts)

    def _create_model(self, num_labels, dropout=None):
        """Create model appropriate for this model type"""
        config = AutoConfig.from_pretrained(self.model_type)
        config.num_labels = num_labels

        if dropout is not None:
            if "roberta" in self.model_type.lower():
                config.hidden_dropout_prob = dropout
                config.attention_probs_dropout_prob = dropout

        return AutoModelForSequenceClassification.from_pretrained(
            self.model_type,
            config=config
        )

    def _tokenize(self, X_preprocessed):
        return self.tokenizer(list(X_preprocessed), truncation=True, padding="max_length", max_length=128, return_tensors="pt")

    def train(self, X: list[str], y: list[str]):
        """wrapper function, a better design would be to call each one directly, but I didn't want to make too
        many structural changes all at once"""
        self.optimize_hyperparameters(X, y)
        self.train_final_model(X, y, self.best_params)

    def optimize_hyperparameters(self, X: list[str], y: list[str]):
        """Train model with hyperparameter optimization"""
        print(f"Start optimizing {self.model_name}, Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

        # Preprocess texts and encode labels
        X_preprocessed = self.preprocess(X)
        y_encoded = self.label_encoder.transform(y)

        # Create and run Optuna study
        opt_start_time = time.time()
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=2,
                interval_steps=1
            )
        )
        study.optimize(
            lambda trial: self._objective_function(trial, X_preprocessed, y_encoded),
            n_trials=self.n_trials
        )
        training_duration = time.time() - opt_start_time

        self.cv_score = study.best_value
        self.best_params = study.best_trial.params

        self.print_best_model_results(self.cv_score, self.best_params, training_duration)

    def _objective_function(self, trial, X_preprocessed, y_encoded):
        """Optuna objective function for a single trial"""
        # Set up trial parameters
        trial_config = {
            "learning_rate": trial.suggest_float("learning_rate", *self.hp_ranges["learning_rate_range"],
                                                 log=self.hp_ranges["learning_rate_log"]),
            "batch_size": trial.suggest_categorical("batch_size", self.hp_ranges["batch_sizes"]),
            "epochs": trial.suggest_int("epochs", *self.hp_ranges["epochs_range"]),
            "weight_decay": trial.suggest_float("weight_decay", *self.hp_ranges["weight_decay_range"]),
            "dropout": trial.suggest_float("dropout", *self.hp_ranges["dropout_range"]),
        }

        return self._perform_cross_validation(X_preprocessed, y_encoded, trial_config, trial=trial)

    def _perform_cross_validation(self, X, y_encoded, params, trial=None, n_splits=5) -> float:
        """Perform k-fold cross-validation with given hyperparameters"""
        print("\n=== Start cross validation ===")
        cv_scores = []
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

        start_time = time.time()

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\nTrial {trial.number if trial else 'N/A'} - FOLD {fold_idx + 1}/{n_splits}")


            X_train_fold = [X[i] for i in train_idx]
            X_val_fold = [X[i] for i in val_idx]
            y_train_fold = y_encoded[train_idx]
            y_val_fold = y_encoded[val_idx]

            X_train_tokenized = self._tokenize(X_train_fold)
            X_val_tokenized = self._tokenize(X_val_fold)

            # Train model with best hyperparameters
            val_score = self._train_model(
                X_train_tokenized,
                X_val_tokenized,
                y_train_fold,
                y_val_fold,
                params
            )

            cv_scores.append(val_score)

            if trial is not None:
                trial.report(np.mean(cv_scores), fold_idx)
                if trial.should_prune():
                    print(f"Pruned - fold: {fold_idx}, trial: {trial}")
                    raise optuna.TrialPruned()

        training_duration = time.time() - start_time

        print(f"\nK-cross validation took: {format_duration(training_duration)}")

        return float(np.mean(cv_scores))

    def _train_model(self, X_train, X_val, y_train, y_val, params):
        """Train a model with given parameters and data"""
        # Create datasets
        train_dataset = CustomTextDataset(X_train, list(y_train))
        val_dataset = CustomTextDataset(X_val, list(y_val))

        # Initialize model
        model = self._create_model(num_labels=len(self.LABELS), dropout=params["dropout"])

        # Create temporary directory for checkpoints
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Get training arguments
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                learning_rate=params["learning_rate"],
                per_device_train_batch_size=params["batch_size"],
                per_device_eval_batch_size=params["batch_size"],
                num_train_epochs=params["epochs"],
                weight_decay=params["weight_decay"],
                save_strategy="epoch",
                save_total_limit=1,
                report_to="none",
                metric_for_best_model="f1",
                eval_strategy="epoch",
                load_best_model_at_end=True,    # Best score *per epoch*
            )

            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
                compute_metrics=self._compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )

            trainer.train()
            val_score = trainer.evaluate()["eval_f1"]

        return val_score

    def train_final_model(self, X, y, params=None):
        """Train final model on all data using best parameters"""
        print("Training final model with best parameters...")

        if params is None:
            params = self.best_params

        X_preprocessed = self.preprocess(X)
        y_encoded = self.label_encoder.transform(y)

        X_tokenized = self._tokenize(X_preprocessed)

        train_dataset = CustomTextDataset(X_tokenized, list(y_encoded))

        model = self._create_model(num_labels=len(self.LABELS), dropout=params["dropout"])

        start_time = time.time()

        # Create temporary directory for checkpoints
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                learning_rate=params["learning_rate"],
                per_device_train_batch_size=params["batch_size"],
                per_device_eval_batch_size=params["batch_size"],
                num_train_epochs=params["epochs"],
                weight_decay=params["weight_decay"],
                save_strategy="epoch",
                save_total_limit=1,
                report_to="none",
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
                compute_metrics=self._compute_metrics
            )

            trainer.train()

        training_duration = time.time() - start_time
        print(f"Final model training took: {format_duration(training_duration)}")

        self.best_model = model

    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)

        return self.compute_all_metrics(labels, preds)

    def predict(self, text, output=False):
        self.best_model.eval()  # evaluation mode

        single_input = isinstance(text, str)
        text_list = [text] if single_input else text

        texts_processed = self.preprocess(text_list)
        inputs = self.tokenizer(texts_processed, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

        # Get device and move inputs
        device = next(self.best_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get predictions directly from model
        with torch.no_grad():
            outputs = self.best_model(**inputs)

        # Get prediction and probabilities
        y_pred = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        probabilities = F.softmax(outputs.logits, dim=1)
        max_probs = probabilities[range(len(y_pred)), y_pred].cpu().numpy()

        if output:
            y_pred_decoded = self.label_encoder.inverse_transform(y_pred).tolist()

            print()
            for pred in zip(y_pred_decoded, max_probs, text):
                print(pred)

        if single_input:
            return y_pred[0], max_probs[0]

        return y_pred, max_probs

    def save_model(self, path=None):
        # Create the BERT directory
        if path:
            bert_path = str(os.path.join(os.path.abspath(__file__), self.model_name))
        else:
            bert_path = str(os.path.join(BaseTextClassifier.save_models_path, "bert", self.model_name))

        os.makedirs(bert_path, exist_ok=True)

        # Save model and tokenizer in the BERT directory
        self.best_model.save_pretrained(os.path.join(bert_path, "model"), safe_serialization=False)
        self.tokenizer.save_pretrained(os.path.join(bert_path, "model"), safe_serialization=False)

        # Save temporary references
        temp_best_model = self.best_model
        temp_tokenizer = self.tokenizer

        # Clear problematic attributes
        self.best_model = None
        self.tokenizer = None

        with open(os.path.join(bert_path, "classifier_class.pkl"), "wb") as f:
            pickle.dump(self, f)

        # Restore models and other cleared attributes
        self.best_model = temp_best_model
        self.tokenizer = temp_tokenizer

    @staticmethod
    def load_model(path: str, in_saved_models=False):
        if in_saved_models:
            bert_path = str(os.path.join(BaseTextClassifier.save_models_path, "bert", path))
        else:
            bert_path = path

        with open(os.path.join(bert_path, "classifier_class.pkl"), "rb") as f:
            obj = pickle.load(f)

        obj.best_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(bert_path, "model"))
        obj.tokenizer = AutoTokenizer.from_pretrained(os.path.join(bert_path, "model"))

        return obj


class CustomTextDataset(Dataset):
    """
    A lightweight PyTorch Dataset implementation for text classification tasks.
    Replaces the HuggingFace datasets library dependency to significantly reduce Docker image size.
    """
    def __init__(self, tokenized_inputs, labels):
        self.input_ids = tokenized_inputs['input_ids']
        self.attention_mask = tokenized_inputs['attention_mask']
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }
