import spacy
from spacy.util import is_package

from classifier.BaseTextClassifier import BaseTextClassifier
from classifier.normalization.TextNormalizer import TextNormalizer


class SpacyClassifier(BaseTextClassifier):
    def __init__(self, nlp_model_name: str, text_normalizer: TextNormalizer(), labels: list, seed: int = 42):
        nlp_pipeline = self._load_nlp(nlp_model_name)
        super().__init__(nlp_pipeline, text_normalizer, labels, seed)

    def _load_nlp(self, model_name: str):
        if not is_package(model_name):
            print(f"'{model_name}' is not installed. Installing...")
            spacy.cli.download(model_name)

        print(f"Loading: '{model_name}'...")

        return spacy.load(model_name)

    def preprocess_data(self, datasets: any) -> dict[str, list[str]]:
        datasets = super().normalize(datasets)
        nlp = self.get_nlp()

        # Run text through the entire spacy NLP pipeline
        processed_datasets = {}
        count = 0
        for label, posts in datasets.items():
            processed_data = []
            for post in posts:
                doc = nlp(post)

                tokens = []

                for token in doc:
                    if not token.is_alpha or (not token.is_stop and not token.is_punct):
                        tokens.append(token.lemma_)

                        if not token.has_vector:
                            # print(token)
                            count += 1

                lemmatized_text = ' '.join(tokens)
                processed_data.append(lemmatized_text)

            processed_datasets[label] = processed_data

        if count > 0:
            print(f"Undetected tokens found: {count} ")

        return processed_datasets

    def train(self, processed_datasets: dict[str, list[tuple[str, str]]]) -> None:
        pass

    def evaluate(self, test_dataset: any) -> dict[str, float]:
        pass

    def predict(self, text: str) -> dict[str, float]:
        pass

