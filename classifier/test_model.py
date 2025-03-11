from config import *
from src.predict import load_model_and_tokenizer, predict_tweets, map_predictions_to_labels, load_test_data, save_predictions_to_csv

def test_model():
    # Load the model and tokenizer
    tokenizer, model = load_model_and_tokenizer(output_dir)

    # Load test data
    new_texts, test_data = load_test_data(test_file_path)

    # Predict new tweets
    predictions = predict_tweets(model, tokenizer, new_texts)
    predicted_labels = map_predictions_to_labels(predictions)

    # Save predictions to CSV
    save_predictions_to_csv(test_data, predicted_labels, output_file_path)

if __name__ == "__main__":
    test_model()