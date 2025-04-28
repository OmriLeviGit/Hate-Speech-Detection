import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from classifier.SpacyClassifier import SpacyClassifier
from classifier.normalization.TextNormalizer import TextNormalizer

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    DataCollatorWithPadding, TextDataset
from datasets import Dataset
from sklearn.model_selection import train_test_split

from classifier.normalization.TextNormalizerRoBERTa import TextNormalizerRoBERTa


def train_standard_bert(X, y, label_encoder, configs=None):
    """Train standard BERT models using regular text normalization."""
    if configs is None:
        configs = [
            {"model_name": "distilbert-base-uncased", "learning_rate": 2e-5, "batch_size": 32, "epochs": 5},
            # {"model_name": "distilbert-base-uncased", "learning_rate": 5e-5, "batch_size": 32, "epochs": 5},
        ]

    # Initialize normalizer
    normalizer = TextNormalizer(emoji='text')

    # Normalize texts with standard normalizer
    X_normalized = normalizer.normalize_texts(X)

    # Initialize tokenizers and models
    tokenizers = {}
    models = {}
    for config in configs:
        model_name = config["model_name"]
        tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
        models[model_name] = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(label_encoder.classes_)
        )

    # Run models with shared training function
    return run_models(X_normalized, y, configs, tokenizers, models)


def train_bertweet(X, y, label_encoder, configs=None):
    """Train BERTweet models using RoBERTa-specific text normalization."""
    if configs is None:
        configs = [
            # {"model_name": "vinai/bertweet-base", "learning_rate": 2e-5, "batch_size": 32, "epochs": 5},
            {"model_name": "vinai/bertweet-base", "learning_rate": 5e-5, "batch_size": 32, "epochs": 5},
        ]

    # Initialize RoBERTa normalizer
    normalizer = TextNormalizerRoBERTa()

    # Normalize texts with RoBERTa normalizer
    X_normalized = normalizer.normalize_texts(X)

    # Initialize tokenizers and models
    tokenizers = {}
    models = {}
    for config in configs:
        model_name = config["model_name"]
        tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name, normalization=True)
        models[model_name] = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(label_encoder.classes_)
        )

    # Run models with shared training function
    return run_models(X_normalized, y, configs, tokenizers, models)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}


def run_models(X, y, configs, tokenizers, models):
    """Shared function to run model training and evaluation with pre-initialized components."""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    best_score = 0
    best_model = None
    best_tokenizer = None
    best_config = None

    for config in configs:
        print(f"\n=== Training with config: {config} ===")
        model_name = config["model_name"]

        # Use pre-initialized tokenizer and model
        tokenizer = tokenizers[model_name]
        model = models[model_name]

        # Tokenize inputs
        train_encodings = tokenizer(list(X_train), truncation=True, padding="max_length", max_length=128)
        val_encodings = tokenizer(list(X_val), truncation=True, padding="max_length", max_length=128)

        # Create datasets
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': list(y_train)
        })

        val_dataset = Dataset.from_dict({
            'input_ids': val_encodings['input_ids'],
            'attention_mask': val_encodings['attention_mask'],
            'labels': list(y_val)
        })

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=f"./results/{model_name.replace('/', '_')}",
            learning_rate=config["learning_rate"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            num_train_epochs=config["epochs"],
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            dataloader_num_workers=0,
            load_best_model_at_end=True,
            logging_steps=50,
            save_total_limit=1,
        )

        # Train model
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        val_score = trainer.evaluate()["eval_accuracy"]

        if val_score > best_score:
            best_score = val_score
            best_model = model
            best_tokenizer = tokenizer
            best_config = config

    return best_model, best_tokenizer, best_config, best_score


def compare_models(X, y, label_encoder):
    """Run and compare both model types."""
    # Train standard BERT models
    standard_results = train_standard_bert(X, y, label_encoder)
    standard_model, standard_tokenizer, standard_config, standard_score = standard_results

    # Train BERTweet models
    bertweet_results = train_bertweet(X, y, label_encoder)
    bertweet_model, bertweet_tokenizer, bertweet_config, bertweet_score = bertweet_results

    # Compare results
    print("\n=== Model Comparison ===")
    print(f"Best standard model: {standard_config['model_name']} - Accuracy: {standard_score:.4f}")
    print(f"Best BERTweet model: {bertweet_config['model_name']} - Accuracy: {bertweet_score:.4f}")

    best_config = standard_config if standard_score > bertweet_score else bertweet_config
    best_overall = standard_results if standard_score > bertweet_score else bertweet_results

    print(f"\nThe best model is {best_config['model_name']}")

    return best_overall, best_config


def normalize_temporary(config, texts):
    if config["model_name"] == "vinai/bertweet-base":
        normalizer = TextNormalizerRoBERTa()
    else:
        normalizer = TextNormalizer(emoji='text')

    return normalizer.normalize_texts(texts)


def predict_with_bert(model, tokenizer, texts, batch_size=16):
    """Helper function to predict with BERT models"""
    model.eval()  # Set model to evaluation mode
    all_predictions = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize batch
        encoded_batch = tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        with torch.no_grad():
            outputs = model(
                encoded_batch['input_ids'],
                attention_mask=encoded_batch['attention_mask']
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_predictions.extend(preds)

    return np.array(all_predictions)


def evaluate_bert_model(model_results, config, X_test, y_test):
    """Evaluate a BERT model on test data"""
    model, tokenizer, _, _ = model_results

    X_test = normalize_temporary(config, X_test)
    y_pred = predict_with_bert(model, tokenizer, X_test)

    print("\n=== Final Test Set Evaluation ===")
    print(f"Model: {config['model_name']}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return y_pred


def predict_bert_text(model_results, config, texts):
    """Predict class for new texts using BERT model"""
    model, tokenizer, _, _ = model_results
    texts = normalize_temporary(config, texts)
    return predict_with_bert(model, tokenizer, texts)

def main():
    nlp_model_name = "en_core_web_lg"
    labels = ["antisemitic", "not_antisemitic"]

    # load, prepare
    classifier = SpacyClassifier(nlp_model_name, None, labels)
    data = classifier.load_data(set_to_min=True, source='debug')
    X_train, X_test, y_train, y_test = classifier.prepare_dataset(data)

    # encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # train and compare
    best_model_results, best_config = compare_models(X_train, y_train_encoded, label_encoder)

    # evaluate
    evaluate_bert_model(best_model_results, best_config, X_test, y_test_encoded)

    # prediction example with X_test as an input
    predictions = predict_bert_text(best_model_results, best_config, X_test)
    test_predictions = label_encoder.inverse_transform(predictions)

    for pred in zip(X_test, test_predictions):
        print(pred)


if __name__ == "__main__":
    main()
