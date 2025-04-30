import spacy
from spacy.util import is_package


class SpacyModel:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls._load_spacy()
        return cls._instance

    @staticmethod
    def _load_spacy():
        nlp_model_name = "en_core_web_lg"

        if not is_package(nlp_model_name):
            print(f"'{nlp_model_name}' is not installed. Installing...")
            spacy.cli.download(nlp_model_name)

        print(f"Loading: '{nlp_model_name}'...")

        return spacy.load(nlp_model_name)