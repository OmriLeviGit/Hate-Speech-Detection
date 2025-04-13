from spacy.symbols import ORTH

from classifier.BaseTextClassifier import BaseTextClassifier


class Spacy3Classes(BaseTextClassifier):

    def preprocess_data(self, datasets: any, exclude_from_lemma: list[str] = None) -> dict[str, list[tuple[str, str]]]:
        """Apply preprocessing to datasets"""

        datasets = super().preprocess_data(datasets)

        # Add special tokens to the tokenizer
        special_tokens = self.get_text_preprocessor().get_special_tokens()
        self._handle_special_tokens(special_tokens)
        nlp = self.get_model()

        processed_datasets = {}
        for dataset_name, data in datasets.items():
            processed_data = []
            for text, label in data:
                doc = nlp(text)

                # ensures words that are to be excluded, are not lemmatizied
                if exclude_from_lemma:
                    lemmatized_text = " ".join(
                        [token.lemma_ if token.text not in exclude_from_lemma else token.text for token in doc])
                else:
                    lemmatized_text = " ".join([token.lemma_ for token in doc])

                processed_data.append((lemmatized_text, label))

            processed_datasets[dataset_name] = processed_data

        return processed_datasets

    def _handle_special_tokens(self, special_tokens):
        """Register all special tokens with spaCy tokenizer"""
        model = self.get_model()

        for token in special_tokens:
            special_case = [{ORTH: token}]
            model.tokenizer.add_special_case(token, special_case)

    def train(self, processed_datasets: any, **kwargs) -> None:
        """Train the model"""

        pass

    def evaluate(self, test_dataset: any) -> dict[str, float]:
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
