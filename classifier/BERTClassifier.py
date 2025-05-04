import tempfile
import time, os, pickle

import numpy as np
import optuna
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from classifier.BaseTextClassifier import BaseTextClassifier
from classifier.normalization.TextNormalizer import TextNormalizer


class BERTClassifier(BaseTextClassifier):
    def __init__(self, labels: list, normalizer: TextNormalizer(), tokenizer, config, seed: int = 42):
        super().__init__(labels, seed)

        self.config = config
        self.normalizer = normalizer

        self.model_name = config.get("model_name")
        self.model_type = config.get("model_type")
        self.tokenizer = tokenizer

        self.hp_ranges = config.get("hyper_parameters")
        self.n_trials = config.get("n_trials", 1)

        self.best_model = None
        self.best_score = None
        self.best_params = None

    def preprocess(self, texts: list[str]) -> list[str]:
        """Normalize texts using the provided normalizer"""
        return self.normalizer.normalize_texts(texts)

    def _create_model(self, num_labels, dropout=None):
        """Create model appropriate for this model type"""
        if dropout is None:
            return AutoModelForSequenceClassification.from_pretrained(
                self.model_type,
                num_labels=num_labels
            )
        else:
            return AutoModelForSequenceClassification.from_pretrained(
                self.model_type,
                num_labels=num_labels,
                dropout=dropout,
                attention_dropout=dropout
            )

    def _tokenize(self, X_preprocessed):
        return self.tokenizer(list(X_preprocessed), truncation=True, padding="max_length", max_length=128, return_tensors="pt")

    def train(self, X: list[str], y: list[str], hp_ranges=None):
        """Optimize hyperparameters with Optuna"""
        print(f"Training {self.model_name}")

        # Validate hyperparameter config
        if hp_ranges:
            self.hp_ranges = hp_ranges

        # Preprocess texts and encode labels
        X_preprocessed = self.preprocess(X)
        y_encoded = self.label_encoder.fit_transform(y)

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        # Create and run Optuna study
        start_time = time.time()
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self._objective_function(trial, X_preprocessed, y_encoded),
            n_trials=self.n_trials
        )
        training_duration = time.time() - start_time

        # Get best trial parameters
        self.best_params = study.best_trial.params
        self.best_score = self._perform_cross_validation(X, y_encoded, self.best_params)

        self.print_best_model_results(self.best_score, self.best_params, training_duration)

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

        X_train, X_val, y_train, y_val = train_test_split(
            X_preprocessed, y_encoded,
            test_size=0.2,
            stratify=y_encoded,
            random_state=self.seed
        )

        X_train_tokenized = self._tokenize(X_train)
        X_val_tokenized = self._tokenize(X_val)

        # Train and evaluate model for this trial
        model, val_score = self._train_model(X_train_tokenized, X_val_tokenized, y_train, y_val, trial_config)

        # Track best model in memory
        if not self.best_score or val_score > self.best_score:
            self.best_score = val_score # best overall
            self.best_model = model

        return val_score

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
                eval_strategy="epoch",
                metric_for_best_model="accuracy",
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

            # Train model
            trainer.train()

            # Now trainer.model contains the best model from this training run
            best_model = trainer.model

            # Get validation score using the best model
            val_score = trainer.evaluate()["eval_accuracy"]

        # Return the best model and its score
        return best_model, val_score

    def _perform_cross_validation(self, X, y_encoded, params, n_splits=5) -> float:
        """Perform k-fold cross-validation with given hyperparameters"""
        cv_scores = []
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

        for train_idx, val_idx in kf.split(X):
            X_train_fold = [X[i] for i in train_idx]
            X_val_fold = [X[i] for i in val_idx]
            y_train_fold = y_encoded[train_idx]
            y_val_fold = y_encoded[val_idx]

            X_train_tokenized = self._tokenize(X_train_fold)
            X_val_tokenized = self._tokenize(X_val_fold)

            # Train model with best hyperparameters
            model, val_score = self._train_model(
                X_train_tokenized,
                X_val_tokenized,
                y_train_fold,
                y_val_fold,
                params
            )
            cv_scores.append(val_score)

        return float(np.mean(cv_scores))

    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        return {"accuracy": acc, "f1": f1}

    def predict(self, text, return_decoded=False, threshold=0.8, output=False):
        """Predict class for a single text with confidence threshold for antisemitic class"""
        single_input = isinstance(text, str)
        text_list = [text] if single_input else text

        texts_processed = self.preprocess(text_list)
        inputs = self._tokenize(texts_processed)

        if not torch.is_tensor(inputs['input_ids']):
            inputs = {k: torch.tensor(v) for k, v in inputs.items()}

        # Get predictions with confidence threshold
        with torch.no_grad():
            outputs = self.best_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            # Get max predictions
            max_probs, max_indices = torch.max(probs, dim=-1)

            # Get number of labels in model
            num_labels = logits.shape[-1]

            # Apply threshold logic
            if num_labels == 2:
                # If antisemitic prediction doesn't meet threshold, change to not_antisemitic
                y_pred = torch.where(
                    (max_indices == 0) & (max_probs < threshold),
                    torch.ones_like(max_indices),  # Change to index 1 (not_antisemitic)
                    max_indices
                )
            else:  # num_labels == 3
                antisemitic = probs[:, 0]
                not_antisemitic = probs[:, 1]

                y_pred = torch.full_like(max_indices, 2)  # default to irrelevant
                y_pred = torch.where(antisemitic > threshold, torch.zeros_like(y_pred), y_pred)
                y_pred = torch.where((not_antisemitic > threshold) & (antisemitic <= threshold),
                                     torch.ones_like(y_pred), y_pred)


            y_pred = y_pred.cpu().numpy()
            probs_np = probs.cpu().numpy()

        if output:
            y_pred_decoded = self.label_encoder.inverse_transform(y_pred).tolist()
            print()
            for i, (pred, txt) in enumerate(zip(y_pred_decoded, text_list)):
                conf = probs_np[i][y_pred[i]]  # confidence of final predicted class
                print(f"Text: {txt}")
                print(f"Prediction: {pred} (confidence: {conf:.3f})")

        return y_pred[0] if single_input else y_pred

    def save_model(self):
        # Create the BERT directory
        bert_path = str(os.path.join(BaseTextClassifier.save_models_path, "bert", self.model_name))
        os.makedirs(bert_path, exist_ok=True)

        # Save model and tokenizer in the BERT directory
        self.best_model.save_pretrained(os.path.join(bert_path, "model"))
        self.tokenizer.save_pretrained(os.path.join(bert_path, "model"))

        # Save temporary references
        temp_best_model = self.best_model
        temp_tokenizer = self.tokenizer

        # Clear potentially problematic attributes
        self.best_model = None
        self.tokenizer = None

        with open(os.path.join(bert_path, "classifier_class.pkl"), "wb") as f:
            pickle.dump(self, f)

        # Restore models and other cleared attributes
        self.best_model = temp_best_model
        self.tokenizer = temp_tokenizer

    @staticmethod
    def load_model(path: str):
        bert_path = os.path.join(BaseTextClassifier.save_models_path, "bert", path)
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
