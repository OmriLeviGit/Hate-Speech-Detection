import random
import time
from copy import copy

from spacy.util import minibatch, fix_random_seed
from spacy.symbols import ORTH
from spacy.training import Example

from classifier.deprecated.BaseTextClassifierOld import BaseTextClassifier
from classifier.normalization.TextNormalizer import TextNormalizer


class SpacyModels_SM_LG(BaseTextClassifier):
    def __init__(self, nlp_pipeline: any, text_normalizer: TextNormalizer() = None, seed: int = 42):
        super().__init__(nlp_pipeline, text_normalizer, seed)
        self.training_time = None
        self.training_history = None

    def preprocess_data(self, datasets: any, custom_lemmas: dict[str, str] = None) -> dict[str, list[tuple[str, str]]]:
        """Apply preprocessing to datasets"""
        datasets = super().preprocess_data(datasets)  # Custom preprocessing

        nlp = self.get_nlp()

        special_tokens = self.get_text_normalizer().get_special_tokens()
        self.add_tokens(special_tokens)  # Add special tokens to the tokenizer
        self.add_lemmas(custom_lemmas)  # Add custom lemmas to the lemmatizer

        # Run text through the entire spacy NLP pipeline
        processed_datasets = {}
        for dataset_name, data in datasets.items():
            processed_data = []
            for text, label in data:
                doc = nlp(text)

                tokens = [
                    token.lemma_ for token in doc
                    if token.is_alpha and not token.is_stop and not token.is_punct
                ]

                lemmatized_text = ' '.join(tokens)

                vector = nlp(lemmatized_text).post
                processed_data.append((vector, label))
            processed_datasets[dataset_name] = processed_data

        return processed_datasets

    def add_lemmas(self, custom_lemmas: dict):
        """
        Add custom lemmatization rules to spaCy's lemmatizer

        Args:
            custom_lemmas: dict mapping words to their desired lemma forms
        """

        nlp = self.get_nlp()
        # Get the lemmatizer if it exists
        if not custom_lemmas or 'lemmatizer' not in nlp.pipe_names:
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
        model = self.get_nlp()

        for token in special_tokens:
            special_case = [{ORTH: token}]
            model.tokenizer.add_special_case(token, special_case)

    def train(self, processed_datasets: dict[str, list[tuple[str, str]]], learning_rate: float,
              l2_regularization: float, epochs: int = 100, batch_size: int = 32, dropout: float = 0.2) -> None:
        """
        Train the model

        For binary classification (2 labels), we configure the model to optimize for F1 score
        by setting a specific class as the 'positive_label'. This focuses evaluation on the
        F1 score of this target class instead of overall accuracy, which is especially helpful
        for imbalanced datasets where correctly detecting the positive class is more important.

        For multiclass classification (3+ labels), we use the default configuration which will
        automatically use macro-averaged F1 (averaging F1 scores across all classes).
        """
        nlp = self.get_nlp()

        textcat = nlp.add_pipe("textcat", last=True)
        for category in self.LABELS:
            textcat.add_label(category)

        # Configure optimizer
        optimizer = nlp.begin_training()
        optimizer.learn_rate = learning_rate
        optimizer.L2 = l2_regularization

        # Prepare for spacy format (for cnn based models, must come after 'nlp.begin_training()')
        train_examples = self._create_spacy_examples(nlp, processed_datasets['train'])
        validation_examples = self._create_spacy_examples(nlp, processed_datasets.get('validation'))

        pipe_exceptions = ["textcat"]
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

        with nlp.disable_pipes(*other_pipes):
            self._start_training(nlp, train_examples, validation_examples, optimizer, epochs, batch_size,
                                 dropout, patience=5)

    def _start_training(self, nlp, train_examples, validation_examples,
                        optimizer, epochs, batch_size, dropout, patience=None):
        """Start training the model"""
        print("Start training...")
        fix_random_seed(self.seed)

        start_time = time.time()
        training_history = []
        best_model = None
        best_score = 0
        no_improvement = 0

        for epoch in range(epochs):
            epoch_start = time.time()
            self.random_generator.shuffle(train_examples)
            random.shuffle(train_examples)
            losses = {}

            # Training batches
            batches = list(minibatch(train_examples, size=batch_size))
            for batch in batches:
                nlp.update(batch, drop=dropout, sgd=optimizer, losses=losses)

            # Evaluate on validation set
            train_results = nlp.evaluate(train_examples)
            eval_results = nlp.evaluate(validation_examples)
            epoch_time = time.time() - epoch_start

            # Track metrics
            epoch_metrics = self._record_metrics(epoch, losses, train_results, eval_results, epoch_time)
            training_history.append(epoch_metrics)

            # Log progress
            self._log_progress(epoch, epochs, epoch_time, losses, train_results, eval_results)

            # Check for early stopping
            if eval_results["cats_score"] > best_score:
                best_score = eval_results["cats_score"]
                best_model = copy(nlp)
                no_improvement = 0
            elif epoch > 2:
                no_improvement += 1

            if patience and no_improvement >= patience:
                print(f"Early stopping at epoch {epoch} - no improvement for {patience} epochs")
                break

        # Save best model and training history
        if best_model is not None:
            self.set_model(best_model)

        total_time = time.time() - start_time
        print(f"Total training time: {total_time:.2f} seconds")

        self.training_history = training_history
        self.training_time = total_time

    def evaluate(self, datasets: dict[str, list[tuple[str, str]]]) -> dict[str, float]:
        """Evaluate the model"""
        nlp = self.get_nlp()

        test_examples = self._create_spacy_examples(nlp, datasets.get('test'))
        results = nlp.evaluate(test_examples)

        return self._compile_evaluation_metrics(results)

    def _create_spacy_examples(self, nlp, data):
        """
        Converts raw data into spaCy Example objects formatted for text classification.

        Handles both binary and ternary classification scenarios:
        - For binary classification: Creates examples with two mutually exclusive labels
        - For ternary classification: Creates examples with three mutually exclusive labels

        Args:
            nlp (spacy.language.Language): spaCy Language object for tokenization
            data (list): List of (text, label) tuples where label is a string matching one in self.LABELS

        Returns:
            list: List of spaCy Example objects ready for training or evaluation

        Raises:
            ValueError: If self.LABELS doesn't contain exactly 2 or 3 elements
        """
        examples = []
        for text, label in data:
            doc = nlp.make_doc(text)

            if len(self.LABELS) == 2:
                # Binary classification with exclusive labels
                cats = {}
                for l in self.LABELS:
                    cats[l] = 1.0 if l == label else 0.0
                examples.append(Example.from_dict(doc, {"cats": cats}))

            elif len(self.LABELS) == 3:
                # Ternary classification with exclusive labels
                cats = {l: 1.0 if l == label else 0.0 for l in self.LABELS}
                examples.append(Example.from_dict(doc, {"cats": cats}))

            else:
                raise ValueError("Only 2 or 3 labels supported")

        return examples
