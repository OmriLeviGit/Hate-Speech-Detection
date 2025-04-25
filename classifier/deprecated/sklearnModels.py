# from spacy.tokenizer import ORTH
#
# from classifier.BaseTextClassifier import BaseTextClassifier
# from classifier.normalization.TextNormalizer import TextNormalizer
#
#
# class SKLearnModels(BaseTextClassifier):
#     def __init__(self, nlp_pipeline: any, text_normalizer: TextNormalizer(), labels: list, seed: int = 42):
#         super().__init__(nlp_pipeline, text_normalizer, labels, seed)
#
#     def preprocess_datasets(self, datasets: any) -> dict[str, list[str]]:
#         datasets = super().normalize(datasets)
#
#         # Get spacy's NLP pipeline
#         nlp = self.get_nlp()
#
#         # Add special tokens to the tokenizer
#         special_tokens = self.get_text_normalizer().get_special_tokens()
#         self.add_tokens(special_tokens)
#
#         # Run text through the entire spacy NLP pipeline
#         processed_datasets = {}
#         count = 0
#         for label, posts in datasets.items():
#             processed_data = []
#
#             for post in posts:
#                 doc = nlp(post)
#
#                 tokens = []
#
#                 for token in doc:
#                     if not token.is_alpha or (not token.is_stop and not token.is_punct):
#                         tokens.append(token.lemma_)
#
#                         if not token.has_vector:
#                             # print(token)
#                             count += 1
#
#                 lemmatized_text = ' '.join(tokens)
#                 processed_data.append(lemmatized_text)
#
#             processed_datasets[label] = processed_data
#         print("Num of undetected tokens: ", count, "\n")
#         return processed_datasets
#
#     def preprocess_text_list(self, text_list: list[str]) -> list[str]:
#         nlp = self.get_nlp()
#         processed_data = []
#         count = 0
#
#         for post in text_list:
#             doc = nlp(post)
#             tokens = []
#
#             for token in doc:
#                 if not token.is_alpha or (not token.is_stop and not token.is_punct):
#                     tokens.append(token.lemma_)
#
#                     if not token.has_vector:
#                         count += 1
#
#             lemmatized_text = ' '.join(tokens)
#             processed_data.append(lemmatized_text)
#
#         if count > 0:
#             print(f"Undetected tokens found: {count} ")
#
#         return processed_data
#
#     def add_tokens(self, special_tokens):
#         """Add special tokens detected in the normalization to the tokenizer"""
#         nlp = self.get_nlp()
#
#         for token in special_tokens:
#             special_case = [{ORTH: token}]
#             nlp.tokenizer.add_special_case(token, special_case)
#
#     def train(self, processed_datasets: dict[str, list[tuple[str, str]]]) -> None:
#         pass
#
#     def evaluate(self, test_dataset: any) -> dict[str, float]:
#         pass
#
#     def predict(self, text: str) -> dict[str, float]:
#         pass
