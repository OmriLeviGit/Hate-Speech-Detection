from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datasets import DatasetDict


class BaseTextClassifier(ABC):
    """Abstract base class for text classifiers."""

    def __init__(self):
        # Initialize default hyperparameters
        self.hyperparameters = self._get_default_hyperparameters()

    @abstractmethod
    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """
        Return default hyperparameters for this classifier.
        Each subclass should override this method.
        """
        return {}

    @abstractmethod
    def set_hyperparameters(self, **kwargs) -> None:
        """Set hyperparameters for the classifier."""
        for param, value in kwargs.items():
            if param in self.hyperparameters:
                self.hyperparameters[param] = value
            else:
                raise ValueError(f"Unknown hyperparameter '{param}' for {self.__class__.__name__}")

    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        return self.hyperparameters.copy()

    @abstractmethod
    def load_data(self, file_path: Optional[str] = None) -> Dict[str, List]:
        """Load data from file or use sample data."""
        pass

    @abstractmethod
    def prepare_datasets(self, data: Dict[str, List], test_size: float = 0.2, validation_size: float = 0.1) -> Any:
        """Prepare train, validation and test datasets."""
        pass

    @abstractmethod
    def preprocess_data(self, datasets: Any) -> Any:
        """Apply preprocessing to datasets."""
        pass

    @abstractmethod
    def train(self, processed_datasets: Any, **kwargs) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def evaluate(self, test_dataset: Any) -> Dict[str, float]:
        """Evaluate the model."""
        pass

    @abstractmethod
    def predict(self, text: str) -> Dict[str, float]:
        """Make prediction on a single text."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a saved model."""
        pass

# Hugging Face implementation
class HuggingFaceClassifier(BaseTextClassifier):
    """Text classifier using Hugging Face transformers."""

    def __init__(self, model_name: str = "vinai/bertweet-base", num_labels: int = 3):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.model_name = model_name
        self.num_labels = num_labels
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.label_names = ["positive", "negative", "neutral"]

    def load_data(self, file_path: Optional[str] = None) -> Dict[str, List]:
        # Implementation as before
        pass

    def prepare_datasets(self, data: Dict[str, List], test_size: float = 0.2,
                         validation_size: float = 0.1) -> DatasetDict:
        # Implementation as before
        pass

    def preprocess_data(self, datasets: DatasetDict) -> DatasetDict:
        # Custom preprocessing + tokenization
        pass

    def train(self, processed_datasets: DatasetDict, **kwargs) -> None:
        # Implementation as before
        pass

    # Other methods as before


# spaCy implementation
class SpacyClassifier(BaseTextClassifier):
    """Text classifier using spaCy."""

    def __init__(self, model_name: str = "en_core_web_md", num_labels: int = 3):
        import spacy

        self.model_name = model_name
        self.num_labels = num_labels
        self.nlp = spacy.load(model_name)
        self.textcat = None
        self.label_names = ["positive", "negative", "neutral"]

    def load_data(self, file_path: Optional[str] = None) -> Dict[str, List]:
        # Similar implementation as HuggingFaceClassifier
        pass

    def prepare_datasets(self, data: Dict[str, List], test_size: float = 0.2,
                         validation_size: float = 0.1) -> Dict[str, List]:
        # spaCy-specific implementation
        pass

    def preprocess_data(self, datasets: Dict[str, List]) -> List:
        # spaCy-specific preprocessing and conversion to spaCy format
        pass

    def train(self, processed_datasets: List, **kwargs) -> None:
        # spaCy-specific training implementation
        pass

    # Other methods specific to spaCy

