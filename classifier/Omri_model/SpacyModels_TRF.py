import time
from copy import copy

from spacy.util import minibatch, fix_random_seed
from spacy.symbols import ORTH
from spacy.training import Example

from classifier.BaseTextClassifier import BaseTextClassifier
from classifier.preprocessing.TextPreprocessor import TextPreprocessor


class SpacyModels_TRF(BaseTextClassifier):

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
        model = self.get_model()

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
        nlp = self.get_model()

        if len(self.LABELS) == 2:
            textcat = nlp.add_pipe("textcat", last=True,
                                   config={"exclusive_classes": True,
                                           "positive_label": self.LABELS[0]})
        else:
            textcat = nlp.add_pipe("textcat", last=True)

        for category in self.LABELS:
            textcat.add_label(category)

        # Prepare for spacy format
        train_examples = self._create_spacy_examples(nlp, processed_datasets['train'], self.LABELS)
        validation_examples = self._create_spacy_examples(nlp, processed_datasets.get('validation'), self.LABELS)

        pipe_exceptions = ["textcat"]
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
        with nlp.disable_pipes(*other_pipes):  # Only train textcat
            # Initialize with a sample batch to set dimensions properly
            sample_batch = list(minibatch(train_examples, size=batch_size))[0]
            nlp.initialize(lambda: sample_batch)

            # Configure optimizer after initialization
            optimizer = nlp.create_optimizer()
            optimizer.learn_rate = learning_rate
            optimizer.L2 = l2_regularization

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
        nlp = self.get_model()

        test_examples = self._create_spacy_examples(nlp, datasets.get('test'), self.LABELS)
        results = nlp.evaluate(test_examples)

        return self._compile_evaluation_metrics(results)

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
            cats = {cat: 1.0 if cat == label else 0.0 for cat in labels}

            doc = nlp.make_doc(text)    # for transformer-based models

            examples.append(Example.from_dict(doc, {"cats": cats}))

        return examples
