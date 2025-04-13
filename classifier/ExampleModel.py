from classifier.BaseTextClassifier import BaseTextClassifier


class ExampleModel(BaseTextClassifier):

    def preprocess_data(self, datasets: any, exclude_from_lemma: list[str] = None) -> dict[str, list[tuple[str, str]]]:
        """Apply preprocessing to datasets"""
        datasets = super().preprocess_data(datasets)

        # additional preprocessing such as tokenization

        return datasets

    def add_tokens(self, special_tokens):
        pass

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
