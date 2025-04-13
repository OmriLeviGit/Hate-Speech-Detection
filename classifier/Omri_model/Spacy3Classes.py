import spacy
from spacy.symbols import ORTH
from spacy.training import Example

from classifier.BaseTextClassifier import BaseTextClassifier
from classifier.preprocessing.TextPreprocessor import TextPreprocessor


class Spacy3Classes(BaseTextClassifier):

    def __init__(self, model: any, label_count: int, preprocessor: TextPreprocessor() = None, seed: int = 42):
        super().__init__(model, label_count, preprocessor, seed)

    def preprocess_data(self, datasets: any, custom_lemmas: dict[str, str] = None) -> dict[str, list[tuple[str, str]]]:
        """Apply preprocessing to datasets"""
        datasets = super().preprocess_data(datasets)    # Default preprocessing

        nlp = self.get_model()

        special_tokens = self.get_text_preprocessor().get_special_tokens()
        self.add_tokens(special_tokens)     # Add special tokens to the tokenizer
        self.add_lemmas(custom_lemmas)      # Add custom lemmas to the lemmatizer

        # Run through the spacy pipeline
        processed_datasets = {}
        for dataset_name, data in datasets.items():
            processed_data = []
            for text, label in data:
                doc = nlp(text)
                lemmatized_text = " ".join([token.lemma_ for token in doc])
                processed_data.append((lemmatized_text, label))
            processed_datasets[dataset_name] = processed_data

        return processed_datasets

    def add_tokens(self, special_tokens: set):
        """Register all special tokens with spaCy tokenizer"""
        model = self.get_model()

        for token in special_tokens:
            special_case = [{ORTH: token}]
            model.tokenizer.add_special_case(token, special_case)

    def train(self, processed_datasets: dict[str, list[tuple[str, str]]], epochs: int, lr: float, l2: float, batch_size: int = 8, dropout: float = 0.2) -> None:
        """Train the model"""
        nlp = self.get_model()

        textcat = nlp.add_pipe("textcat")
        for category in self.LABELS:
            textcat.add_label(category)

        optimizer = nlp.begin_training()
        optimizer.learn_rate = lr   # Set a custom learning rate
        optimizer.L2 = l2           # Set L2 regularization

        processed_datasets = self._convert_to_spacy_format(processed_datasets, self.LABELS)

        train_data = processed_datasets['train']
        validation_data = processed_datasets.get('validation')

        # Create training examples directly from the already-formatted data
        train_examples = []
        for item in train_data:
            text = item['text']
            cats = item['cats']
            train_examples.append(Example.from_dict(nlp(text), {"cats": cats}))

        # Create validation examples
        validation_examples = []
        for item in validation_data:
            text = item['text']
            cats = item['cats']
            validation_examples.append(Example.from_dict(nlp(text), {"cats": cats}))

        # Train the model
        for i in range(epochs):
            self.random_generator.shuffle(train_examples)
            losses = {}

            # Training batches
            batches = list(spacy.util.minibatch(train_examples, size=batch_size))
            for batch in batches:
                nlp.update(batch, drop=dropout, sgd=optimizer, losses=losses)

            # Evaluate on validation set
            eval_results = nlp.evaluate(validation_examples)
            print(f"Iteration {i}, Train Loss: {losses}, Validation Accuracy: {eval_results['cats_score']}")

    def _convert_to_spacy_format(self, datasets, labels):
        """
        Convert dataset from (text, label) tuples to spaCy's format with 'cats' dictionaries

        Args:
            datasets: Dictionary with 'train', 'validation', and 'test' keys, each containing
                     a list of (text, label) tuples
            labels: List of all possible category labels

        Returns:
            Dictionary with the same keys, but data converted to spaCy's format:
            { 'text': text, 'cats': {label1: 1.0, label2: 0.0, ...} }
        """
        spacy_formatted = {}

        for split, data in datasets.items():
            formatted_data = []

            for text, label in data:
                # Create the cats dictionary with 1.0 for the correct label, 0.0 for others
                cats = {cat: 1.0 if cat == label else 0.0 for cat in labels}

                formatted_example = {
                    'text': text,
                    'cats': cats
                }

                formatted_data.append(formatted_example)

            spacy_formatted[split] = formatted_data

        return spacy_formatted

    def evaluate(self, test_dataset: list[tuple[str, str]]) -> dict[str, float]:
        """Evaluate the model"""
        pass

    def predict(self, text: str) -> dict[str, float]:
        """Make prediction on a single text"""
        pass

    def save_model(self, path: str) -> None:
        """Save the model"""
        pass

    def load_model(self, path: str) -> None:
        """Load a saved model"""
        pass
