from classifier.src.classifiers.BaseTextClassifier import BaseTextClassifier

class DataHelper(BaseTextClassifier):
    def __init__(self):
        super().__init__(labels= ['not_antisemitic', 'antisemitic'], seed = 42)

    def preprocess(self, datasets):
        pass

    def train(self, *args):
        pass

    def predict(self, text):
        pass

    def save_model(self):
        pass

    @staticmethod
    def load_model(path):
        pass
