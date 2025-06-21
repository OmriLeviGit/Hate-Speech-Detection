import os
import shutil, psutil
from datetime import datetime
from pathlib import Path
import tempfile

import gradio as gr
import csv
import pandas as pd

from classifier.src.classifiers.BertClassifier import BertClassifier
from classifier.src.classifiers.ClassicalModelClassifier import ClassicalModelClassifier

# consts
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "saved_models"))
HISTORY_PATH = os.path.join(BASE_DIR, "tweet_history.csv")
DATASETS_PATH = os.path.join(BASE_DIR, "datasets")

# model = SKLearnClassifier.load_model("RandomForestClassifier", in_saved_models=True)
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


def save_to_dataset(files):
    print(f"Function called with files: {files}")  # Debug line

    if not files or len(files) == 0:
        gr.Warning("No files selected")
        return "No files selected", gr.update(value=None)

    # Use the DATASETS_PATH you defined
    os.makedirs(DATASETS_PATH, exist_ok=True)

    # Filter to only CSV files
    csv_files = [file for file in files if file.name.lower().endswith('.csv')]

    if not csv_files:
        gr.Warning("No CSV files found. Only CSV files are allowed.")
        return "❌ No CSV files found. Only CSV files are allowed.", gr.update(value=None)

    results = []
    success_count = 0

    for file in csv_files:
        original_filename = "Unknown file"
        try:
            # Extract filename and extension
            original_filename = os.path.basename(file.name)

            # Get current date and time in YYYY-MM-DD_HH-MM-SS format
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            name, ext = os.path.splitext(original_filename)

            # Create new filename with date and time
            new_filename = f"{name}_{current_datetime}{ext}"

            # Full path for the new file
            new_filepath = os.path.join(DATASETS_PATH, new_filename)

            # Handle duplicate filenames by adding a counter
            counter = 1
            while os.path.exists(new_filepath):
                new_filename = f"{name}_{current_datetime}_{counter}{ext}"
                new_filepath = os.path.join(DATASETS_PATH, new_filename)
                counter += 1

            # Copy the uploaded file to datasets directory
            shutil.copy2(file.name, new_filepath)
            results.append(f"✅ {original_filename} → {new_filename}")
            success_count += 1

        except Exception as e:
            results.append(f"❌ {original_filename}: {str(e)}")

    # Create summary message
    summary = f"Processed {len(csv_files)} CSV files. {success_count} successful."
    status_message = summary + "\n\n" + "\n".join(results)

    # Show success notification
    if success_count > 0:
        gr.Info(f"✅ Successfully added {success_count} file(s) to dataset!")

    if success_count < len(csv_files):
        gr.Warning(f"⚠️ {len(csv_files) - success_count} file(s) failed to save")

    # Return status and clear the file upload
    return status_message, gr.update(value=None)


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


def load_history_with_visibility():
    history_data = load_history()  # Your existing function

    # Check if history has any data
    if history_data and len(history_data) > 0:
        # Has data - show the section and return the data
        return history_data, gr.update(visible=True)
    else:
        # No data - keep section hidden
        return [], gr.update(visible=False)


def predict_and_store_with_visibility(tweet):
    # Your existing prediction logic
    prediction, updated_history = predict_and_store(tweet)

    # Check if history has data to determine visibility
    if updated_history and len(updated_history) > 0:
        return prediction, updated_history, gr.update(visible=True)
    else:
        return prediction, [], gr.update(visible=False)


def retrain_model():
    global model

    print(f"Memory at start: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
    print("Training started!")

    try:
        data = model.load_data()
        print(f"Memory at start: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
        X_train, X_test, y_train, y_test = model.prepare_dataset(data, augment_ratio=0.33, irrelevant_ratio=0.4)
        print(f"Memory at start: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")

        model.train(X_train, y_train)   # sklearn - random forest
        # model.train_final_model(X_train, y_train) # bert
        print(f"Memory at start: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")

        import gc
        gc.collect()
        model.save_model()
        model.model_name = "test"
        print(f"Memory at test: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")

        print(f"Memory at after save: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")

        gr.Info("🎉 Training completed! Model has been updated.")
        return "✅ Training completed! Model updated and ready to use."

    except Exception as e:
        gr.Error(f"Training failed: {str(e)}")
        return f"❌ Training failed: {str(e)}"

def confirm_training():
    return gr.update(visible=True)

def cancel_training():
    return gr.update(visible=False)


def create_training_section():
    """Create the training and dataset management section with event handlers, requires machine with more ram"""
    gr.Markdown("---")

    # Dataset Management and Training (single column block)
    with gr.Column():
        gr.Markdown("# Add data & Retrain")
        gr.Markdown("## Dataset Management")
        gr.Markdown("""
        Upload one or more CSV files with the following format:
        - **Column 1:** 'content' - the text data
        - **Column 2:** 'sentiment' - labeled as one of the following:
          - 'Positive' for antisemitic content
          - 'Negative' for political/religious content  
          - 'Irrelevant' for other content
        """)

        # Compact file upload in a row with the button
        file_upload = gr.File(
            label="Add CSV files",
            file_count="multiple",
            file_types=[".csv"],  # Only allow CSV files in the file picker
            height=150
        )
        add_button = gr.Button("Add to Dataset", variant="primary", scale=1)

        # Hidden status for file upload (to avoid warnings)
        dataset_status = gr.Textbox(label="Status", interactive=False, visible=False)

        # Training section (below dataset)
        gr.Markdown("## Model Training")
        train_button = gr.Button("Train Model", variant="secondary")

        # Confirmation dialog (initially hidden)
        with gr.Group(visible=False) as confirmation_group:
            gr.Markdown("⚠️ **Are you sure you want to start training?** This process may take a while and will temporarily freeze the interface.")
            with gr.Row():
                confirm_btn = gr.Button("Yes, Start Training", variant="stop")
                cancel_btn = gr.Button("Cancel", variant="secondary")

        training_status = gr.Textbox(label="Train Model", interactive=False)

        # Event handlers for training section
        add_button.click(
            fn=save_to_dataset,
            inputs=file_upload,
            outputs=[dataset_status, file_upload]  # Clear the file upload after saving
        )

        # Training confirmation flow
        train_button.click(
            fn=lambda: gr.update(visible=True),
            outputs=confirmation_group
        )

        confirm_btn.click(
            fn=lambda: ["🔄 Training in progress...", gr.update(visible=False)],
            outputs=[training_status, confirmation_group]
        ).then(
            fn=retrain_model,  # This runs after UI updates
            outputs=[training_status]
        )

        cancel_btn.click(
            fn=lambda: gr.update(visible=False),
            outputs=confirmation_group
        )

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
            with gr.Column(scale=0, min_width=400):
                gr.Markdown("## Batch Process from CSV")
                csv_upload = gr.File(
                    label="Upload a CSV file with tweets in the first column",
                    file_types=[".csv"],
                    type="filepath"
                )
                process_btn = gr.Button("Process Batch")
                upload_status = gr.Textbox(label="Processing Status", interactive=False)
                download_file = gr.File(label="Download Processed Results", visible=False)

        # History section - initially hidden
        with gr.Group(visible=False) as history_section:
            gr.Markdown("## Last 10 Processed Tweets")
            history_table = gr.Dataframe(
                headers=["Tweet", "Model Prediction", "Certainty"],
                row_count=10,
                wrap=True,
                type="array",
                elem_id="history-table"
            )

        create_training_section()
        
        # Call history loader on app launch
        demo.load(
            fn=load_history_with_visibility,
            outputs=[history_table, history_section]
        )

        # Apply custom style for the app
        with open(os.path.join(BASE_DIR, "style.css")) as f:
            gr.HTML(f"<style>{f.read()}</style>")

        # Only the main prediction/batch processing events remain here
        predict_btn.click(
            fn=predict_and_store_with_visibility,
            inputs=[tweet_input],
            outputs=[model_output, history_table, history_section]
        )

        process_btn.click(
            fn=process_file_upload,
            inputs=[csv_upload],
            outputs=[upload_status, download_file]
        )

    return demo

if __name__ == "__main__":
    app = create_app()
    app.launch(show_api=False, server_port=80, server_name="0.0.0.0") # port change requires updating the docker-compose

