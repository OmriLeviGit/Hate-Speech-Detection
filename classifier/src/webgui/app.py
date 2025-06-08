import os
from pathlib import Path
import tempfile

import gradio as gr
import csv
import pandas as pd

from classifier.src.classifiers.BertClassifier import BertClassifier

# consts
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "saved_models"))
HISTORY_PATH = os.path.join(BASE_DIR, "tweet_history.csv")

model = BertClassifier.load_model("distilbert uncased", in_saved_models=True)


# Loads csv file with the history of tweets users wanted to predict
def load_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        data = list(reader)[-10:]
        return list(reversed(data))


# Saves to csv file a predicted tweet
def save_history(data):
    with open(HISTORY_PATH, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)


def read_file_to_dataframe(file_path):
    """Read CSV file and return as pandas DataFrame, only using first column"""
    try:
        # Try UTF-8 first, then fall back to latin-1
        try:
            df = pd.read_csv(
                file_path,
                encoding='utf-8',
                usecols=[0],  # Only read first column
                header=None,  # Don't assume headers
                on_bad_lines='skip',  # Skip malformed lines
                engine='python'
            )
        except UnicodeDecodeError:
            df = pd.read_csv(
                file_path,
                encoding='latin-1',
                usecols=[0],  # Only read first column
                header=None,  # Don't assume headers
                on_bad_lines='skip',
                engine='python'
            )

        return df
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")


def process_file_upload(file):
    if file is None:
        return "No file uploaded", gr.File(visible=False)

    try:
        # Read the file using pandas
        df = read_file_to_dataframe(file.name)

        if df.empty:
            return "Empty file", gr.File(visible=False)

        # Get tweets from first column
        first_column = df.iloc[:, 0]
        tweets = first_column.dropna().astype(str).tolist()

        if not tweets:
            return "No valid tweets found in first column", gr.File(visible=False)

        # Process each tweet and make predictions
        processed_data = []
        for tweet in tweets:
            tweet = tweet.strip()
            if not tweet:  # Skip empty tweets
                continue

            # Make prediction
            prediction, prob = model.predict(tweet)
            percentage = f"{prob * 100:.2f}%"
            model_label = "Antisemitic" if prediction == 1 else "Not Antisemitic"

            processed_data.append([tweet, model_label, percentage])

        if not processed_data:
            return "No valid tweets found to process", gr.File(visible=False)

        # Create output filename based on original file
        original_filename = Path(file.name).stem  # Get filename without extension
        output_filename = f"{original_filename}_processed.csv"

        # Use temp directory to avoid permission issues
        temp_dir = tempfile.gettempdir()
        output_file = os.path.join(temp_dir, output_filename)

        with open(output_file, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Add headers
            writer.writerow(["Tweet", "Model Prediction", "Certainty"])
            writer.writerows(processed_data)

        message = f"✅ Successfully processed {len(processed_data)} tweets!"
        return message, gr.File(value=output_file, visible=True)

    except Exception as e:
        return f"❌ Error processing file: {str(e)}", gr.File(visible=False)


def predict_and_store(tweet):
    """Make prediction on single tweet without requiring user label"""
    Path(HISTORY_PATH).touch(exist_ok=True)
    with open(HISTORY_PATH, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        full_history = list(reader)

    if not tweet.strip():
        return "⚠️ Please enter a tweet", list(reversed(full_history[-10:]))

    # Make prediction automatically
    prediction, prob = model.predict(tweet)
    percentage = f"{prob * 100:.2f}%"
    model_label = "Antisemitic" if prediction == 1 else "Not Antisemitic"

    # Add to history with same format as file processing
    full_history.append([tweet, model_label, percentage])
    save_history(full_history)

    result = f"{model_label} ({percentage})"

    return result, list(reversed(full_history[-10:]))


def create_app():
    """Create and return the Gradio app interface"""
    with gr.Blocks() as demo:
        gr.Markdown("# Tweet Classifier", elem_id="title")

        # Single Tweet Prediction and Batch Processing (in same centered row)
        with gr.Row(elem_id="centered-row"):
            with gr.Column(min_width=600, scale=0):
                gr.Markdown("## Single Tweet Prediction")
                tweet_input = gr.Textbox(
                    label="Tweet",
                    lines=3,
                    placeholder="Paste your tweet here",
                    elem_id="tweet-input"
                )
                predict_btn = gr.Button("Predict", elem_id="predict-btn")
                model_output = gr.Label(label="Model Prediction", elem_id="model-output")

            # Batch Processing (limited width, right column)
            with gr.Column(scale=0, min_width=400):  # Fixed width column
                gr.Markdown("## Batch Process from CSV")
                csv_upload = gr.File(
                    label="Upload a CSV file with tweets in the first column",
                    file_types=[".csv"],
                    type="filepath"
                )
                process_btn = gr.Button("Process Batch")
                upload_status = gr.Textbox(label="Processing Status", interactive=False)

                # File output for download
                download_file = gr.File(label="Download Processed Results", visible=False)

        # History Table (full width below the row)
        gr.Markdown("## Last 10 Processed Tweets")
        history_table = gr.Dataframe(
            headers=["Tweet", "Model Prediction", "Certainty"],
            row_count=10,
            wrap=True,
            type="array",
            elem_id="history-table"
        )

        # Call history loader on app launch
        demo.load(fn=load_history, outputs=history_table)

        # Apply custom style for the app
        with open(os.path.join(BASE_DIR, "style.css")) as f:
            gr.HTML(f"<style>{f.read()}</style>")

        # Event bindings
        predict_btn.click(
            fn=predict_and_store,
            inputs=[tweet_input],
            outputs=[model_output, history_table]
        )

        # Simplified process button event
        process_btn.click(
            fn=process_file_upload,
            inputs=[csv_upload],
            outputs=[upload_status, download_file]
        )
    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch(show_api=False, server_port=80, server_name="0.0.0.0")