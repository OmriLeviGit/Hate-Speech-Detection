import time, os, pickle
import numpy as np
import optuna
import torch
from datasets import Dataset
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
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_type, num_labels=num_labels
        )

    def train(self, X: list[str], y: list[str], hp_ranges=None):
        """Optimize hyperparameters with Optuna"""
        # Validate hyperparameter config
        if hp_ranges:
            self.hp_ranges = hp_ranges

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        # Create and run Optuna study
        start_time = time.time()

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self._objective_function(trial, X, y_encoded),
            n_trials=self.n_trials
        )

        training_duration = time.time() - start_time

        # Get best trial and attributes
        best_trial = study.best_trial

        self.best_model = self.best_model_trainer.model
        self.best_score = best_trial.value
        self.best_params = best_trial.params

        X_tokenized = self.tokenizer(list(X), truncation=True, padding="max_length", max_length=128)

        # Create a dataset for prediction
        prediction_dataset = Dataset.from_dict({
            'input_ids': X_tokenized['input_ids'],
            'attention_mask': X_tokenized['attention_mask']
        })

        predictions = self.best_model_trainer.predict(prediction_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)

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

        # Get train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=self.seed
        )

        X_train_tokenized = self.tokenizer(list(X_train), truncation=True, padding="max_length", max_length=128)
        X_val_tokenized = self.tokenizer(list(X_val), truncation=True, padding="max_length", max_length=128)

        # Train and evaluate model for this trial
        trainer, val_score = self._train_and_evaluate_trial(X_train_tokenized, X_val_tokenized, y_train, y_val, trial.number, trial_config)

        if not self.best_score or val_score > self.best_score:
            self.best_score = val_score
            self.best_model_trainer = trainer

        return val_score

    def _train_and_evaluate_trial(self, X_train, X_val, y_train, y_val, trial_number, trial_config):
        """Train and evaluate model for a single Optuna trial"""

        # Create datasets
        train_dataset = Dataset.from_dict({
            'input_ids': X_train['input_ids'],
            'attention_mask': X_train['attention_mask'],
            'labels': list(y_train)
        })

        val_dataset = Dataset.from_dict({
            'input_ids': X_val['input_ids'],
            'attention_mask': X_val['attention_mask'],
            'labels': list(y_val)
        })

        # Initialize model for this trial
        trial_model = self._create_model(num_labels=len(self.LABELS))

        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"./results/{self.model_name.replace('/', '_')}_trial_{trial_number}",
            learning_rate=trial_config["learning_rate"],
            per_device_train_batch_size=trial_config["batch_size"],
            per_device_eval_batch_size=trial_config["batch_size"],
            num_train_epochs=trial_config["epochs"],
            weight_decay=trial_config["weight_decay"],
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model="accuracy",
        )

        # Create and run trainer
        print(f"Start training: {self.model_name}")
        trainer = Trainer(
            model=trial_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        # Train the model
        trainer.train()

        # Get validation score
        val_score = trainer.evaluate()["eval_accuracy"]

        return trainer, val_score

    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        acc = np.mean(preds == labels)
        return {"accuracy": acc}

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

        # prediction_dataset = Dataset.from_dict({
        #     'input_ids': inputs['input_ids'],
        #     'attention_mask': inputs['attention_mask']
        # })
        #
        # predictions = self.best_model_trainer.predict(prediction_dataset)
        # y_pred = np.argmax(predictions.predictions, axis=1)

        if output:
            y_pred_decoded = self.label_encoder.inverse_transform(y_pred).tolist()

            print()
            for pred in zip(y_pred_decoded, text):
                print(pred)

        return y_pred[0] if single_input else y_pred


    # def save_model(self, path: str):
    #     # Create the BERT directory
    #     bert_path = os.path.join(path, "BERT")
    #     os.makedirs(bert_path, exist_ok=True)
    #
    #     # Save temporary references and clear models before pickling
    #     temp_best_model = self.best_model
    #     temp_tokenizer = self.tokenizer
    #     self.best_model = None
    #     self.tokenizer = None
    #
    #     # Save model and tokenizer in the BERT directory
    #     temp_best_model.save_pretrained(os.path.join(bert_path, "model"))
    #     temp_tokenizer.save_pretrained(os.path.join(bert_path, "model"))
    #
    #     with open(os.path.join(bert_path, "classifier_class.pkl"), "wb") as f:
    #         pickle.dump(self, f)
    #
    #     # Restore models
    #     self.best_model = temp_best_model
    #     self.tokenizer = temp_tokenizer

    def save_model(self, path: str):
        # Create the BERT directory
        bert_path = os.path.join(path, "BERT")
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

        # Try to find problematic attributes
        problematic_keys = []
        for key, value in self.__dict__.items():
            try:
                pickle.dumps(value)
            except Exception as e:
                problematic_keys.append(key)
                setattr(self, key, None)
                print(f"Setting {key} to None due to: {str(e)}")

        # Try to pickle the object
        try:
            with open(os.path.join(bert_path, "classifier_class.pkl"), "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            print(f"Still can't pickle: {str(e)}")
            # Fall back to dict-based saving
            raise e

        # Restore models and other cleared attributes
        self.best_model = temp_best_model
        self.tokenizer = temp_tokenizer

        # If we identified problematic keys, warn the user
        if problematic_keys:
            print(f"Warning: The following attributes were set to None before saving: {problematic_keys}")


    # def save_model(self, path: str):
    #     # Create the BERT directory
    #     bert_path = os.path.join(path, "BERT")
    #     os.makedirs(bert_path, exist_ok=True)
    #
    #     # Save model and tokenizer in the BERT directory
    #     self.best_model.save_pretrained(os.path.join(bert_path, "model"))
    #     self.tokenizer.save_pretrained(os.path.join(bert_path, "model"))
    #
    #     self.best_model = None
    #     self.tokenizer = None
    #
    #     with open(os.path.join(bert_path, "classifier_class.pkl"), "wb") as f:
    #         pickle.dump(self, f)

    @staticmethod
    def load_model(path: str):
        bert_path = os.path.join(path, "BERT")
        with open(os.path.join(bert_path, "classifier_class.pkl"), "rb") as f:
            obj = pickle.load(f)

        obj.best_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(bert_path, "model"))
        obj.tokenizer = AutoTokenizer.from_pretrained(os.path.join(bert_path, "model"))

        return obj