from src.utils import prep_data
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from src.dataset import TweetDataset
from src.model import get_model
from src.train import train_model
from src.evaluate import compute_metrics, evaluate_model
from config import *

def main():
    texts, labels = prep_data(data_path)
    test_prep_data(texts, labels)

    # Tokenization
    # TODO: change on retrain
    # tokenizer = BertTokenizer.from_pretrained("asafaya/bert-base-arabic")
    tokenizer = BertTokenizer.from_pretrained(output_dir)

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    max_length = 128  # Based on analysis
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length)

    # Create Datasets
    train_dataset = TweetDataset(train_encodings, train_labels)
    val_dataset = TweetDataset(val_encodings, val_labels)

    # Model for init training
    # model = get_model()

    # TODO: change on retrain
    # Model for continuing training
    model = get_model(output_dir)

    # Train
    trainer = train_model(model, tokenizer, train_dataset, val_dataset, output_dir)

    # Evaluate
    trainer.compute_metrics = compute_metrics
    evaluation_results = evaluate_model(trainer)
    print(evaluation_results)

def test_prep_data(texts, labels):
    # Check data types
    print(f"Type of texts: {type(texts)}")
    print(f"Type of labels: {type(labels)}")

    # Check lengths
    print(f"Length of texts: {len(texts)}")
    print(f"Length of labels: {len(labels)}")

    # Inspect a few samples
    print("Sample texts:", texts[:5])
    print("Sample labels:", labels[:5])

    # Check unique labels
    unique_labels = set(labels)
    print(f"Unique labels: {unique_labels}")

    # Additional checks
    if isinstance(texts, list) and isinstance(labels, list):
        if all(isinstance(text, str) for text in texts):
            print("All texts are strings.")
        else:
            print("Error: Not all texts are strings.")

        if all(isinstance(label, int) for label in labels):
            print("All labels are integers.")
        else:
            print("Error: Not all labels are integers.")

        if len(texts) == len(labels):
            print("Texts and labels have the same length.")
        else:
            print("Error: Texts and labels do not have the same length.")
    else:
        print("Error: Texts or labels are not lists.")

if __name__ == "__main__":
    main()