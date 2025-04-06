from abc import ABC, abstractmethod
import sys
import os
import random

from preprocessing.TextPreprocessor import TextPreprocessor
from tagging_website.serverside.db_service import get_db_instance

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BaseTextClassifier(ABC):
    """Abstract base class for text classifiers."""

    def __init__(self):
        self._preprocessor = TextPreprocessor()
        self._random_generator = random.Random(42)  # should be used intead of random.random() to keep results consistent

        self._CLASS_0 = "antisemistic"
        self._CLASS_1 = "not_antisemistic"
        self._CLASS_2 = "irrelevant"


    def load_data(self, class_0_count, class_1_count, class_2_count=None, debug=False) -> dict[str, list]:
        """Load data from file or use sample data.

        This function loads text data for classification either from a database (when debug=False)
        or generates mock data (when debug=True). The data is organized by class labels.

        Args:
            class_0_count: Number of samples to load for class 0 (antisemistic)
            class_1_count: Number of samples to load for class 1 (not_antisemistic)
            class_2_count: Number of samples to load for class 2 (irrelevant), optional

        Returns:
            Dictionary mapping class labels to lists of text samples.
            In debug mode, the lists contain placeholder values (zeros).
        """

        if not debug:
            db = get_db_instance()

            data = {
                self._CLASS_0: db.get_result_posts(label=self._CLASS_0, count=class_0_count),
                self._CLASS_1: db.get_result_posts(label=self._CLASS_1, count=class_1_count)
            }

            if class_2_count is not None:
                data[self._CLASS_2] = db.get_result_posts(label=self._CLASS_2, count=class_2_count)
        else:
            data = {
                self._CLASS_0: [f"{self._CLASS_0}_post_{i}" for i in range(class_0_count)],
                self._CLASS_1: [f"{self._CLASS_1}_post_{i}" for i in range(class_1_count)]
            }

            if class_2_count is not None:
                data[self._CLASS_2] = [f"{self._CLASS_2}_post_{i}" for i in range(class_2_count)]

        return data

    def prepare_datasets(self, data: dict[str, list], test_size: float = 0.2, validation_size: float = 0.1, combine_irrelevant=False) -> any:
        """Prepare train, validation and test datasets.

        This function splits the input data into training, validation, and test sets
        while maintaining the class distribution in each set. The data is shuffled to ensure randomness.

        Args:
            data: Dictionary mapping class labels to lists of posts
            test_size: Proportion of data to use for testing (default: 0.2)
            validation_size: Proportion of data to use for validation (default: 0.1)
            combine_irrelevant: If True, combines irrelevant class with not_antisemistic class (default: False)

        Returns:
            Dictionary containing 'train', 'validation', and 'test' keys, each mapping to
            a list of (post, label) tuples
        """

        if test_size + validation_size > 1:
            print("training, test, and validation sizes must sum up to 1")
            return

        # Handle combining irrelevant with not-antisemistic if specified
        if combine_irrelevant and self._CLASS_1 in data and self._CLASS_2 in data:
            data[self._CLASS_1] = data[self._CLASS_1] + data.pop(self._CLASS_2)

        train_data = {'posts': [], 'labels': []}
        validation_data = {'posts': [], 'labels': []}
        test_data = {'posts': [], 'labels': []}

        # Process each class separately to maintain class distribution
        for label, posts in data.items():
            total_posts = len(posts)
            test_count = int(total_posts * test_size)
            validation_count = int(total_posts * validation_size)
            train_count = total_posts - test_count - validation_count

            train_posts = posts[:train_count]
            validation_posts = posts[train_count:train_count + validation_count]
            test_posts = posts[train_count + validation_count:]

            train_data['posts'].extend(train_posts)
            train_data['labels'].extend([label] * len(train_posts))

            validation_data['posts'].extend(validation_posts)
            validation_data['labels'].extend([label] * len(validation_posts))

            test_data['posts'].extend(test_posts)
            test_data['labels'].extend([label] * len(test_posts))

        train_combined = list(zip(train_data['posts'], train_data['labels']))
        validation_combined = list(zip(validation_data['posts'], validation_data['labels']))
        test_combined = list(zip(test_data['posts'], test_data['labels']))

        self._random_generator.shuffle(train_combined)
        self._random_generator.shuffle(validation_combined)
        self._random_generator.shuffle(test_combined)

        return {
            'train': train_combined,
            'validation': validation_combined,
            'test': test_combined
        }

    @abstractmethod
    def preprocess_data(self, datasets: any) -> any:
        """Apply preprocessing to datasets."""
        pass

    @abstractmethod
    def train(self, processed_datasets: any, **kwargs) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def evaluate(self, test_dataset: any) -> dict[str, float]:
        """Evaluate the model."""
        pass

    @abstractmethod
    def predict(self, text: str) -> dict[str, float]:
        """Make prediction on a single text."""
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save the model."""
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load a saved model."""
        pass

    # @abstractmethod
    # def _get_default_hyperparameters(self) -> Dict[str, Any]:
    #     """
    #     Return default hyperparameters for this classifier.
    #     Each subclass should override this method.
    #     """
    #     return {}
    #
    # @abstractmethod
    # def set_hyperparameters(self, **kwargs) -> None:
    #     """Set hyperparameters for the classifier."""
    #     for param, value in kwargs.items():
    #         if param in self.hyperparameters:
    #             self.hyperparameters[param] = value
    #         else:
    #             raise ValueError(f"Unknown hyperparameter '{param}' for {self.__class__.__name__}")
    #
    # @abstractmethod
    # def get_hyperparameters(self) -> Dict[str, Any]:
    #     """Get current hyperparameters."""
    #     return self.hyperparameters.copy()

# # Hugging Face implementation
# class HuggingFaceClassifier(BaseTextClassifier):
#     """Text classifier using Hugging Face transformers."""
#
#     def __init__(self, model_name: str = "vinai/bertweet-base", num_labels: int = 3):
#         import torch
#         from transformers import AutoTokenizer, AutoModelForSequenceClassification
#
#         self.model_name = model_name
#         self.num_labels = num_labels
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = None
#         self.label_names = ["positive", "negative", "neutral"]
#
#     def load_data(self, file_path: Optional[str] = None) -> Dict[str, List]:
#         # Implementation as before
#         pass
#
#     def prepare_datasets(self, data: Dict[str, List], test_size: float = 0.2,
#                          validation_size: float = 0.1) -> DatasetDict:
#         # Implementation as before
#         pass
#
#     def preprocess_data(self, datasets: DatasetDict) -> DatasetDict:
#         # Custom preprocessing + tokenization
#         pass
#
#     def train(self, processed_datasets: DatasetDict, **kwargs) -> None:
#         # Implementation as before
#         pass
#
#     # Other methods as before
#
#
# # spaCy implementation
# class SpacyClassifier(BaseTextClassifier):
#     """Text classifier using spaCy."""
#
#     def __init__(self, model_name: str = "en_core_web_md", num_labels: int = 3):
#         import spacy
#
#         self.model_name = model_name
#         self.num_labels = num_labels
#         self.nlp = spacy.load(model_name)
#         self.textcat = None
#         self.label_names = ["positive", "negative", "neutral"]
#
#     def load_data(self, file_path: Optional[str] = None) -> Dict[str, List]:
#         # Similar implementation as HuggingFaceClassifier
#         pass
#
#     def prepare_datasets(self, data: Dict[str, List], test_size: float = 0.2,
#                          validation_size: float = 0.1) -> Dict[str, List]:
#         # spaCy-specific implementation
#         pass
#
#     def preprocess_data(self, datasets: Dict[str, List]) -> List:
#         # spaCy-specific preprocessing and conversion to spaCy format
#         pass
#
#     def train(self, processed_datasets: List, **kwargs) -> None:
#         # spaCy-specific training implementation
#         pass
#
#     # Other methods specific to spaCy
#
