from sklearn.preprocessing import LabelEncoder
from classifier.SpacyClassifier import SpacyClassifier
from classifier.normalization.TextNormalizer import TextNormalizer

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
from sklearn.model_selection import train_test_split

from classifier.normalization.TextNormalizerRoBERTa import TextNormalizerRoBERTa


def train_standard_bert(X, y, label_encoder, configs=None):
    """Train standard BERT models using regular text normalization."""
    if configs is None:
        configs = [
            {"model_name": "distilbert-base-uncased", "learning_rate": 2e-5, "batch_size": 32, "epochs": 5},
            {"model_name": "distilbert-base-uncased", "learning_rate": 5e-5, "batch_size": 32, "epochs": 5},
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
            {"model_name": "vinai/bertweet-base", "learning_rate": 2e-5, "batch_size": 32, "epochs": 5},
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

    return {
        'standard': standard_results,
        'bertweet': bertweet_results
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}


def main():
    nlp_model_name = "en_core_web_lg"
    labels = ["antisemitic", "not_antisemitic"]

    # load, preprocess, prepare
    classifier = SpacyClassifier(nlp_model_name, None, labels)
    data = classifier.load_data(set_to_min=True)
    X, y = classifier.prepare_dataset(data)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    results = compare_models(X, y_encoded, label_encoder)

if __name__ == "__main__":
    main()
