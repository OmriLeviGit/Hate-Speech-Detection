# from datasets import Dataset
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
#
# from classifier.BaseTextClassifier import BaseTextClassifier
# from classifier.normalization.TextNormalizer import TextNormalizer
# from classifier.normalization.TextNormalizerRoBERTa import TextNormalizerRoBERTa
#
#
# class BertClassifier(BaseTextClassifier):
#     def __init__(self, model_name="distilbert-base-uncased", learning_rate=2e-5, batch_size=32, epochs=5):
#         super().__init__()
#         self.model_name = model_name
#         self.config = {
#             "model_name": model_name,
#             "learning_rate": learning_rate,
#             "batch_size": batch_size,
#             "epochs": epochs
#         }
#         self.normalizer = TextNormalizer(emoji='text')
#         self.tokenizer = None
#         self.model = None
#         self.label_encoder = None
#         self.trainer = None
#
#     def preprocess(self, text_list: list[str]) -> list[str]:
#         return self.normalizer.normalize_texts(text_list)
#
#     def _initialize_tokenizer_and_model(self, num_labels):
#         """Initialize tokenizer and model with appropriate parameters."""
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         self.model = AutoModelForSequenceClassification.from_pretrained(
#             self.model_name, num_labels=num_labels
#         )
#
#     def _prepare_datasets(self, X_train, y_train_encoded, X_val, y_val_encoded):
#         """Tokenize inputs and create datasets."""
#         train_encodings = self.tokenizer(X_train, truncation=True, padding=True)
#         val_encodings = self.tokenizer(X_val, truncation=True, padding=True)
#
#         train_dataset = Dataset.from_dict({
#             'input_ids': train_encodings['input_ids'],
#             'attention_mask': train_encodings['attention_mask'],
#             'labels': y_train_encoded
#         })
#
#         val_dataset = Dataset.from_dict({
#             'input_ids': val_encodings['input_ids'],
#             'attention_mask': val_encodings['attention_mask'],
#             'labels': y_val_encoded
#         })
#
#         return train_dataset, val_dataset
#
#     def _setup_trainer(self, train_dataset, val_dataset):
#         """Configure and return a trainer object."""
#         training_args = TrainingArguments(
#             output_dir=f"./results/{self.model_name.replace('/', '_')}",
#             learning_rate=self.config["learning_rate"],
#             per_device_train_batch_size=self.config["batch_size"],
#             per_device_eval_batch_size=self.config["batch_size"],
#             num_train_epochs=self.config["epochs"],
#             weight_decay=0.01,
#             evaluation_strategy="epoch",
#             save_strategy="epoch",
#             load_best_model_at_end=True,
#             save_total_limit=1,
#         )
#
#         return Trainer(
#             model=self.model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=val_dataset,
#         )
#
#     def train(self, X, y) -> None:
#         X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
#
#         # Encode labels
#         self.label_encoder = LabelEncoder()
#         y_train_encoded = self.label_encoder.fit_transform(y_train)
#         y_val_encoded = self.label_encoder.transform(y_val)
#
#         # Initialize tokenizer and model
#         self._initialize_tokenizer_and_model(len(self.label_encoder.classes_))
#
#         # Prepare datasets
#         train_dataset, val_dataset = self._prepare_datasets(X_train, y_train_encoded, X_val, y_val_encoded)
#
#         # Setup and run trainer
#         self.trainer = self._setup_trainer(train_dataset, val_dataset)
#         self.trainer.train()
#
#     def evaluate(self, test_dataset):
#         X_test, y_test = test_dataset
#
#         # Encode labels
#         y_test_encoded = self.label_encoder.transform(y_test)
#
#         # Tokenize data
#         test_encodings = self.tokenizer(X_test, truncation=True, padding=True)
#
#         # Create dataset
#         test_dataset = Dataset.from_dict({
#             'input_ids': test_encodings['input_ids'],
#             'attention_mask': test_encodings['attention_mask'],
#             'labels': y_test_encoded
#         })
#
#         # Evaluate
#         metrics = self.trainer.evaluate(test_dataset)
#         return metrics
#
#     def predict(self, text: str):
#         pass
#
#
#
# class BertweetClassifier(BertClassifier):
#     def __init__(self, model_name="vinai/bertweet-base", learning_rate=2e-5, batch_size=32, epochs=5):
#         super().__init__(model_name, learning_rate, batch_size, epochs)
#         self.normalizer = TextNormalizerRoBERTa()
#
#     def _initialize_tokenizer_and_model(self, num_labels):
#         """Override to use BERTweet specific parameters."""
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, normalization=True)
#         self.model = AutoModelForSequenceClassification.from_pretrained(
#             self.model_name, num_labels=num_labels
#         )