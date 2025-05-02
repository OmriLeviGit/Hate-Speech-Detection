import spacy


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

        try:
            print(f"Attempting to load spacy: '{nlp_model_name}'...")
            return spacy.load(nlp_model_name)
        except OSError:
            # If loading fails, then we need to download
            print(f"'{nlp_model_name}' is not installed. Installing...")
            spacy.cli.download(nlp_model_name)
            print(f"Loading spacy: '{nlp_model_name}'...")
            return spacy.load(nlp_model_name)