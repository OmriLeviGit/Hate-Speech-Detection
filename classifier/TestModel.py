from BaseTextClassifier import BaseTextClassifier


class TestModel(BaseTextClassifier):

    def preprocess_data(self, datasets: any) -> any:
        """Apply preprocessing to datasets."""
        pass

    def train(self, processed_datasets: any, **kwargs) -> None:
        """Train the model."""
        pass

    def evaluate(self, test_dataset: any) -> dict[str, float]:
        """Evaluate the model."""
        pass

    def predict(self, text: str) -> dict[str, float]:
        """Make prediction on a single text."""
        pass

    def save_model(self, path: str) -> None:
        """Save the model."""
        pass

    def load_model(self, path: str) -> None:
        """Load a saved model."""
        pass