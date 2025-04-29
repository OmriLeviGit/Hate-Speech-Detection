import time
from abc import ABC
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


class BaseTransformerClassifier(BaseTextClassifier, ABC):
    def __init__(self, normalizer, labels: list, config: dict, seed: int = 42):
        super().__init__(labels, seed)

        # Transformer-specific attributes
        self.normalizer = normalizer
        self.config = config
        self.model_name = config["model_name"]
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_encoder = None

    def preprocess(self, texts: list[str]) -> list[str]:
        """Normalize texts using the provided normalizer"""
        return self.normalizer.normalize_texts(texts)

    def _create_tokenizer(self):
        """Create tokenizer appropriate for this model type"""
        return AutoTokenizer.from_pretrained(self.model_name)

    def _create_model(self, num_labels):
        """Create model appropriate for this model type"""
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels
        )

    def optimize_hyperparameters(self, X: list[str], y: list[str], n_trials=20) -> dict:
        """Optimize hyperparameters with Optuna"""
        start_time = time.time()

        # Preprocess the texts
        X_normalized = self.preprocess(X)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Create and run Optuna study
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self._objective_function(trial, X_normalized, y_encoded),
            n_trials=n_trials
        )

        # Get best trial
        best_trial = study.best_trial

        # Update config with best parameters
        self.config.update(best_trial.params)

        # Train model with best parameters
        result = self.train(X, y)

        training_duration = time.time() - start_time

        # Return results
        return {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "score": result["score"],
            "params": best_trial.params
        }

    def _objective_function(self, trial, X_normalized, y_encoded):
        """Optuna objective function for a single trial"""
        # Set up trial parameters
        trial_config = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "epochs": trial.suggest_int("epochs", 2, 5),
            "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.1),
        }

        # Store original config
        original_config = self.config.copy()

        # Update config with trial parameters
        self.config.update(trial_config)

        self.tokenizer = self._create_tokenizer()
        self.model = self._create_model(num_labels=len(self.LABELS))

        # Get train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X_normalized, y_encoded, test_size=0.2, stratify=y_encoded, random_state=self.seed
        )

        # Train and evaluate model for this trial
        val_score = self._train_and_evaluate_trial(X_train, X_val, y_train, y_val, trial.number)

        # Restore original config
        self.config = original_config

        return val_score

    def _train_and_evaluate_trial(self, X_train, X_val, y_train, y_val, trial_number):
        """Train and evaluate model for a single Optuna trial"""
        # Tokenize inputs
        train_encodings = self.tokenizer(list(X_train), truncation=True, padding="max_length", max_length=128)
        val_encodings = self.tokenizer(list(X_val), truncation=True, padding="max_length", max_length=128)

        # Create datasets
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': list(y_train)
        })

        val_dataset = Dataset.from_dict({
            'input_ids': val_encodings['input_ids'],
            'attention_mask': val_encodings['attention_mask'],
            'labels': list(y_val)
        })

        # Initialize model for this trial
        trial_model = self._create_model(num_labels=len(self.LABELS))

        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"./results/trial_{trial_number}",
            learning_rate=self.config["learning_rate"],
            per_device_train_batch_size=self.config["batch_size"],
            per_device_eval_batch_size=self.config["batch_size"],
            num_train_epochs=self.config["epochs"],
            weight_decay=self.config["weight_decay"],
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model="accuracy",
        )

        # Create and run trainer
        trainer = Trainer(
            model=trial_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        # Train the model
        trainer.train()

        # Get validation score
        val_score = trainer.evaluate()["eval_accuracy"]

        return val_score

    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        acc = np.mean(preds == labels)
        return {"accuracy": acc}

    def train(self, X: list[str], y: list[str]) -> None:
        """Train the transformer model with the specified configuration"""
        # Preprocess the texts
        X_normalized = self.preprocess(X)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Initialize tokenizer and model
        self.tokenizer = self._create_tokenizer()
        self.model = self._create_model(num_labels=len(self.LABELS))

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_normalized, y_encoded, test_size=0.2, stratify=y_encoded, random_state=self.seed
        )

        # Tokenize inputs
        train_encodings = self.tokenizer(list(X_train), truncation=True, padding="max_length", max_length=128)
        val_encodings = self.tokenizer(list(X_val), truncation=True, padding="max_length", max_length=128)

        # Create datasets
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': list(y_train)
        })

        val_dataset = Dataset.from_dict({
            'input_ids': val_encodings['input_ids'],
            'attention_mask': val_encodings['attention_mask'],
            'labels': list(y_val)
        })

        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"./results/{self.model_name.replace('/', '_')}",
            learning_rate=self.config["learning_rate"],
            per_device_train_batch_size=self.config["batch_size"],
            per_device_eval_batch_size=self.config["batch_size"],
            num_train_epochs=self.config["epochs"],
            weight_decay=self.config["weight_decay"],
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model="accuracy",
        )

        # Define metrics computation
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = logits.argmax(axis=-1)
            acc = np.mean(preds == labels)
            return {"accuracy": acc}

        # Create and run trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        # Train the model
        trainer.train()

        # Get validation score
        val_score = trainer.evaluate()["eval_accuracy"]

        return {"model": self.model, "tokenizer": self.tokenizer, "score": val_score}

    def predict_with_bert(self, texts, batch_size=32):
        """Helper function to predict with BERT models"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not trained yet")

        self.model.eval()  # Set model to evaluation mode
        all_predictions = []

        # Preprocess texts
        texts = self.preprocess(texts)

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize batch
            encoded_batch = self.tokenizer(
                batch_texts,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )

            with torch.no_grad():
                outputs = self.model(
                    encoded_batch['input_ids'],
                    attention_mask=encoded_batch['attention_mask']
                )
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_predictions.extend(preds)

        return np.array(all_predictions)

    def predict(self, text):
        """Predict class for a single text"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not trained yet")

        single_input = isinstance(text, str)
        text_list = [text] if single_input else text

        # Tokenize inputs
        encoded_input = self.tokenizer(
            text_list,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**encoded_input)
            logits = outputs.logits
            predictions = torch.softmax(logits, dim=1)
            numeric_predictions = torch.argmax(predictions, dim=1).cpu().numpy()

        # Return single result or list based on input type
        predicted_labels = self.label_encoder.inverse_transform(numeric_predictions)

        return predicted_labels[0] if single_input else predicted_labels


class StandardBERTClassifier(BaseTransformerClassifier):
    def __init__(self, labels: list, normalizer: TextNormalizer(), config: dict, seed: int = 42):
        super().__init__(normalizer, labels, config, seed)

    @staticmethod
    def get_default_configs():
        """Return default configurations for standard BERT models"""
        return [
            {"model_name": "distilbert-base-uncased", "learning_rate": 2e-5, "batch_size": 32, "epochs": 5,
             "weight_decay": 0.01},
            {"model_name": "distilbert-base-uncased", "learning_rate": 5e-5, "batch_size": 32, "epochs": 5,
             "weight_decay": 0.01},
        ]


class BERTweetClassifier(BaseTransformerClassifier):
    def __init__(self, labels: list, normalizer: TextNormalizer(), config: dict, seed: int = 42):
        super().__init__(normalizer, labels, config, seed)

    def _create_tokenizer(self):
        """Create BERTweet tokenizer with normalization"""
        try:
            return AutoTokenizer.from_pretrained(self.model_name, normalization=True)
        except:
            return AutoTokenizer.from_pretrained(self.model_name)

    @staticmethod
    def get_default_configs():
        """Return default configurations for BERTweet models"""
        return [
            {"model_name": "vinai/bertweet-base", "learning_rate": 2e-5, "batch_size": 32, "epochs": 5,
             "weight_decay": 0.01},
            {"model_name": "vinai/bertweet-base", "learning_rate": 5e-5, "batch_size": 32, "epochs": 5,
             "weight_decay": 0.01},
        ]

#
# def compare_transformer_models(X, y, labels, label_encoder):
#     """Compare different transformer models and configurations"""
#     all_models = []
#
#     # Add StandardBERT configurations
#     for config in StandardBERTClassifier.get_default_configs():
#         model = StandardBERTClassifier(
#             normalizer=TextNormalizer(emoji='text'),
#             labels=labels,
#             config=config
#         )
#         all_models.append(("StandardBERT", model, config))
#
#     # Add BERTweet configurations
#     for config in BERTweetClassifier.get_default_configs():
#         model = BERTweetClassifier(
#             normalizer=TextNormalizerRoBERTa(),
#             labels=labels,
#             config=config
#         )
#         all_models.append(("BERTweet", model, config))
#
#     results = []
#
#     # Train and evaluate each model
#     for model_type, model, config in all_models:
#         print(f"\n=== Training {model_type} with config: {config} ===")
#
#         # Train the model
#         train_result = model.train(X, y)
#
#         # Evaluate on validation set (included in training)
#         score = train_result["score"]
#
#         results.append({
#             'model_type': model_type,
#             'config': config,
#             'score': score,
#             'model': model
#         })
#
#         print(f"Validation score: {score:.4f}")
#
#     # Find best model
#     best_result = max(results, key=lambda x: x['score'])
#
#     print("\n=== Model Comparison Results ===")
#     for result in results:
#         print(
#             f"{result['model_type']} {result['config']['model_name']} (lr={result['config']['learning_rate']}): {result['score']:.4f}")
#
#     print(f"\nBest model: {best_result['model_type']} with {best_result['config']['model_name']} "
#           f"(lr={best_result['config']['learning_rate']}) - Score: {best_result['score']:.4f}")
#
#     return best_result['model']