import time
from copy import copy

import spacy
from spacy.symbols import ORTH
from spacy.training import Example

from classifier.BaseTextClassifier import BaseTextClassifier
from classifier.preprocessing.TextPreprocessor import TextPreprocessor


class Spacy3Classes(BaseTextClassifier):

    def __init__(self, model: any, preprocessor: TextPreprocessor() = None, seed: int = 42):
        super().__init__(model, preprocessor, seed)
        self.training_time = None
        self.training_history = None

    def preprocess_data(self, datasets: any, custom_lemmas: dict[str, str] = None) -> dict[str, list[tuple[str, str]]]:
        """Apply preprocessing to datasets"""
        datasets = super().preprocess_data(datasets)  # Custom preprocessing

        nlp = self.get_model()

        special_tokens = self.get_text_preprocessor().get_special_tokens()
        self.add_tokens(special_tokens)  # Add special tokens to the tokenizer
        self.add_lemmas(custom_lemmas)  # Add custom lemmas to the lemmatizer

        # Run text through the entire spacy NLP pipeline
        processed_datasets = {}
        for dataset_name, data in datasets.items():
            processed_data = []
            for text, label in data:
                doc = nlp(text)
                lemmatized_text = " ".join([token.lemma_ for token in doc])
                processed_data.append((lemmatized_text, label))
            processed_datasets[dataset_name] = processed_data

        return processed_datasets

    def add_lemmas(self, custom_lemmas: dict):
        """
        Add custom lemmatization rules to spaCy's lemmatizer

        Args:
            custom_lemmas: dict mapping words to their desired lemma forms
        """

        nlp = self.get_model()
        # Get the lemmatizer if it exists
        if 'lemmatizer' not in nlp.pipe_names:
            return

        custom_lemmas = {word: word for word in custom_lemmas}

        lemmatizer = nlp.get_pipe('lemmatizer')
        lemma_exc = lemmatizer.lookups.get_table("lemma_exc")

        # Add custom exceptions for each POS tag
        for word, lemma in custom_lemmas.items():
            for pos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']:
                # Check if this POS tag exists in the exceptions
                if pos not in lemma_exc:
                    lemma_exc[pos] = {}

                # Add our exception
                lemma_exc[pos][word.lower()] = [lemma.lower()]

        # Update the lookups table
        lemmatizer.lookups.set_table("lemma_exc", lemma_exc)

    def add_tokens(self, special_tokens: set):
        """Register all special tokens with spaCy tokenizer"""
        model = self.get_model()

        for token in special_tokens:
            special_case = [{ORTH: token}]
            model.tokenizer.add_special_case(token, special_case)

    def train(self, processed_datasets: dict[str, list[tuple[str, str]]], epochs: int, learning_rate: float,
              l2_regularization: float, batch_size: int = 8, dropout: float = 0.2) -> None:
        """Train the model"""
        nlp = self.get_model()

        # Setup text categorization
        textcat = nlp.add_pipe("textcat")
        for category in self.LABELS:
            textcat.add_label(category)

        # Configure optimizer
        optimizer = nlp.begin_training()
        optimizer.learn_rate = learning_rate
        optimizer.L2 = l2_regularization

        # Prepare in spacy format (must be after 'nlp.begin_training()')
        train_examples = self._create_spacy_examples(nlp, processed_datasets['train'], self.LABELS)
        validation_examples = self._create_spacy_examples(nlp, processed_datasets.get('validation'), self.LABELS)

        self._start_training(nlp, train_examples, validation_examples, optimizer, epochs, batch_size,
                             dropout, patience=5)

    def _start_training(self, nlp, train_examples, validation_examples,
                        optimizer, epochs, batch_size, dropout, patience=None):
        """Start training the model"""
        start_time = time.time()
        training_history = []
        best_model = None
        best_score = 0
        no_improvement = 0

        for i in range(epochs):
            epoch_start = time.time()
            self.random_generator.shuffle(train_examples)
            losses = {}

            # Training batches
            batches = list(spacy.util.minibatch(train_examples, size=batch_size))
            for batch in batches:
                nlp.update(batch, drop=dropout, sgd=optimizer, losses=losses)

            # Evaluate on validation set
            train_results = nlp.evaluate(train_examples)
            eval_results = nlp.evaluate(validation_examples)
            epoch_time = time.time() - epoch_start

            # Track metrics
            epoch_metrics = self._record_metrics(i, losses, train_results, eval_results, epoch_time)
            training_history.append(epoch_metrics)

            # Log progress
            self._log_progress(i, epochs, epoch_time, losses, train_results, eval_results)

            # Check for early stopping
            if eval_results["cats_score"] > best_score:
                best_score = eval_results["cats_score"]
                best_model = copy(nlp)
                no_improvement = 0
            else:
                no_improvement += 1

            if patience and no_improvement >= patience:
                print(f"Early stopping at epoch {i} - no improvement for {patience} epochs")
                break

        # Save best model and training history
        if best_model is not None:
            self.set_model(best_model)

        total_time = time.time() - start_time
        print(f"Total training time: {total_time:.2f} seconds")

        self.training_history = training_history
        self.training_time = total_time

    def _record_metrics(self, epoch, losses, train_results, eval_results, epoch_time):
        """Record metrics for an epoch"""
        return {
            "epoch": epoch,
            "train_loss": losses["textcat"],
            "train_accuracy": train_results["cats_score"],
            "val_accuracy": eval_results["cats_score"],
            "accuracy_gap": train_results["cats_score"] - eval_results["cats_score"],
            "precision": eval_results["cats_micro_p"],
            "recall": eval_results["cats_micro_r"],
            "f1": eval_results["cats_micro_f"],
            "time": epoch_time
        }

    def _log_progress(self, epoch, total_epochs, epoch_time, losses, train_results, eval_results):
        """Log training progress"""
        print(f"Epoch {epoch}/{total_epochs}, Time: {epoch_time:.2f}s, "
              f"Train Loss: {losses['textcat']:.4f}, "
              f"Train Accuracy: {train_results['cats_score']:.4f}, "
              f"Val Accuracy: {eval_results['cats_score']:.4f}, "
              f"Gap: {(train_results['cats_score'] - eval_results['cats_score']):.4f}")

    def evaluate(self, datasets: dict[str, list[tuple[str, str]]]) -> dict[str, float]:
        """Evaluate the model"""
        start_time = time.time()

        nlp = self.get_model()

        # Create test examples directly using the new method
        test_examples = self._create_spacy_examples(nlp, datasets.get('test'), self.LABELS)

        # Evaluate model
        results = nlp.evaluate(test_examples)
        eval_time = time.time() - start_time

        # Extract and return metrics
        return self._compile_evaluation_metrics(results, eval_time)

    def _compile_evaluation_metrics(self, results, eval_time):
        """Compile evaluation metrics into a dictionary"""
        # Basic metrics
        metrics = {
            "accuracy": results["cats_score"],
            "precision": results["cats_micro_p"],
            "recall": results["cats_micro_r"],
            "f1": results["cats_micro_f"],
            "time": eval_time
        }

        # Add per-category scores
        for label in self.LABELS:
            if f"cats_{label}_p" in results:
                metrics[f"{label}_precision"] = results[f"cats_{label}_p"]
                metrics[f"{label}_recall"] = results[f"cats_{label}_r"]
                metrics[f"{label}_f1"] = results[f"cats_{label}_f"]

        # Add training history if available
        if hasattr(self, 'training_history'):
            metrics["training_history"] = self.training_history

        if hasattr(self, 'training_time'):
            metrics["total_training_time"] = self.training_time

        # Add learning curves data for plotting
        if hasattr(self, 'training_history'):
            epochs = [entry["epoch"] for entry in self.training_history]
            train_losses = [entry["train_loss"] for entry in self.training_history]
            train_accuracies = [entry.get("train_accuracy", 0) for entry in self.training_history]
            val_accuracies = [entry.get("val_accuracy", 0) for entry in self.training_history]
            accuracy_gaps = [entry.get("accuracy_gap", 0) for entry in self.training_history]

            metrics["learning_curves"] = {
                "epochs": epochs,
                "train_losses": train_losses,
                "train_accuracies": train_accuracies,
                "val_accuracies": val_accuracies,
                "accuracy_gaps": accuracy_gaps
            }

        return metrics

    def _create_spacy_examples(self, nlp, data, labels):
        """
        Convert dataset from (text, label) tuples directly to spaCy Examples

        Args:
            nlp: The spaCy model
            data: A list of (text, label) tuples
            labels: List of all possible category labels

        Returns:
            List of spaCy Example objects
        """
        examples = []

        for text, label in data:
            # Create the cats dictionary with 1.0 for the correct label, 0.0 for others
            cats = {cat: 1.0 if cat == label else 0.0 for cat in labels}

            # Create Example object directly
            examples.append(Example.from_dict(nlp(text), {"cats": cats}))

        return examples

    def predict(self, text: str) -> dict[str, float]:
        """Make prediction on a single text"""
        pass

    def save_model(self, path: str) -> None:
        """Save the model"""
        pass

    def load_model(self, path: str) -> None:
        """Load a saved model"""
        pass
