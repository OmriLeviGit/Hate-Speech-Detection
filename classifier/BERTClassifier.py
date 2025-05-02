import tempfile
import time, os, pickle

import optuna
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
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
        self.n_trials = config.get("n_trials", 10)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.best_model = None
        self.best_score = None
        self.best_params = None

    def preprocess(self, texts: list[str]) -> list[str]:
        """Normalize texts using the provided normalizer"""
        return self.normalizer.normalize_texts(texts)

    def _create_model(self, num_labels):
        """Create model appropriate for this model type"""
        return AutoModelForSequenceClassification.from_pretrained(self.model_type, num_labels=num_labels)

    def train(self, X: list[str], y: list[str], hp_ranges=None):
        """Optimize hyperparameters with Optuna"""
        # Validate hyperparameter config
        if hp_ranges:
            self.hp_ranges = hp_ranges

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        # Reset best model tracking
        self.best_score = None
        self.best_model = None

        # Create and run Optuna study
        start_time = time.time()
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self._objective_function(trial, X, y_encoded),
            n_trials=self.n_trials
        )
        training_duration = time.time() - start_time

        # Get best trial parameters
        self.best_params = study.best_trial.params

        # Get predictions using the best model
        y_pred = self.predict(X)

        self.print_best_model_results(self.best_score, self.best_params, y_encoded, y_pred, training_duration)

    def _objective_function(self, trial, X, y_encoded):
        """Optuna objective function for a single trial"""
        # Set up trial parameters
        trial_config = {
            "learning_rate": trial.suggest_float("learning_rate", *self.hp_ranges["learning_rate_range"],
                                                 log=self.hp_ranges["learning_rate_log"]),
            "batch_size": trial.suggest_categorical("batch_size", self.hp_ranges["batch_sizes"]),
            "epochs": trial.suggest_int("epochs", *self.hp_ranges["epochs_range"]),
            "weight_decay": trial.suggest_float("weight_decay", *self.hp_ranges["weight_decay_range"]),
        }

        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded,
            test_size=0.2,
            stratify=y_encoded,
            random_state=self.seed
        )

        X_train_processed = self.preprocess(X_train) if hasattr(self, 'preprocess') else X_train
        X_val_processed = self.preprocess(X_val) if hasattr(self, 'preprocess') else X_val

        X_train_tokenized = self.tokenizer(list(X_train_processed), truncation=True, padding="max_length", max_length=128)
        X_val_tokenized = self.tokenizer(list(X_val_processed), truncation=True, padding="max_length", max_length=128)

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
        model = self._create_model(num_labels=len(self.LABELS))

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
                load_best_model_at_end=True,    # Best score per epoch
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

    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        return {"accuracy": acc, "f1": f1}

    def predict(self, text, output=False):
        """Predict class for a single text"""
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

        # Get predicted class indices
        y_pred = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        if output:
            y_pred_decoded = self.label_encoder.inverse_transform(y_pred).tolist()

            print()
            for pred in zip(y_pred_decoded, text):
                print(pred)

        return y_pred[0] if single_input else y_pred

    def save_model(self, path: str):
        # Create the BERT directory
        bert_path = os.path.join(path, "bert")
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
        bert_path = os.path.join(path, "BERT")
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
