from abc import ABC, abstractmethod


class BaseDLModel(ABC):

    # Store hyperparameters needed to build and train the model
    # params is a dictionary of hyperparameters
    def __init__(self, params):
        self.params = params

    # Convert raw text (X_raw) and labels (y) into numerical tensors suitable for training
    # This method should return preprocessed X and y (TF-IDF, tokenized sequences)
    @abstractmethod
    def preprocess(self, X_raw, y):
        pass

    # Build and return a compiled Keras model
    @abstractmethod
    def build(self, input_shape):
        pass
