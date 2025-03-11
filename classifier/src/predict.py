import os
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

def load_model_and_tokenizer(output_dir):
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', output_dir))
    if not os.path.isdir(output_dir):
        raise ValueError(f"The output directory {output_dir} does not exist or is not a directory.")

    tokenizer = BertTokenizer.from_pretrained(output_dir)
    model = BertForSequenceClassification.from_pretrained(output_dir)
    return tokenizer, model

def predict_tweets(model, tokenizer, tweets):
    encodings = tokenizer(tweets, truncation=True, padding=True, max_length=128, return_tensors='pt')
    model.eval()
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).tolist()
    return predictions

def map_predictions_to_labels(predictions):
    label_mapping = {0: 'Positive', 1: 'Irrelevant'}
    return [label_mapping[pred] for pred in predictions]

def load_test_data(file_path):
    data = pd.read_csv(file_path)
    return data['text'].tolist(), data

def save_predictions_to_csv(data, predictions, output_file_path):
    data['predicted_label'] = predictions
    output_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', output_file_path))
    data.to_csv(output_file_path, index=False, encoding='utf-8-sig')
    print("Predictions saved to", output_file_path)