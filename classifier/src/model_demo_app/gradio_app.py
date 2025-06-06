import os
from pathlib import Path

import gradio as gr
import csv
import joblib

from classifier.src.classifiers.BertClassifier import BertClassifier
from classifier.src.normalization.TextNormalizer import TextNormalizer

# Load model and vectorizer
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "saved_models"))
HISTORY_PATH = os.path.join(BASE_DIR, "tweet_history.csv")

# model = BertClassifier.load_model(os.path.join(MODEL_DIR, "bert", "distilbert uncased"))
model = BertClassifier.load_model("distilbert uncased", in_saved_models=True)

# model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
# vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
# normalizer = TextNormalizer(emoji="text")

# Loads csv file with the history of tweets users wanted to predict
def load_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        # Make sure every entry has exactly 4 elements
        data = [row[:4] for row in reader if len(row) >= 4]
        return list(reversed(data[-10:]))

# Saves to csv file a predicted tweet
def save_history(data):
    with open(HISTORY_PATH, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)

def predict_and_store(tweet, user_label):
    Path(HISTORY_PATH).touch(exist_ok=True) # create file if it doesnt exist
    with open(HISTORY_PATH, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        full_history = [row[:4] for row in reader if len(row) >= 4]

    # Checks the user entered a text input and its classification
    if not tweet.strip():
        return "⚠️ Please enter a tweet", list(reversed(full_history[-10:]))
    if user_label is None:
        return "⚠️ Please select your label", list(reversed(full_history[-10:]))

    # Preprocess and predict
    # norm_text = normalizer.normalize(tweet)
    # vectorized = vectorizer.transform([norm_text])
    # prediction = model.predict(vectorized)[0]
    prediction = model.predict(tweet)

    model_label = "Antisemitic" if prediction == 1 else "Not Antisemitic"
    correct = "✔️" if user_label == model_label else "✖️"

    # Appends new row to full history
    full_history.append([tweet, user_label, model_label, correct])

    # Saves full history (don’t truncate here)
    save_history(full_history)

    # Returns last 10 rows in "most recent" to "oldest" order
    return model_label, list(reversed(full_history[-10:]))


# App layout
with gr.Blocks() as demo:
    gr.Markdown("# Tweet Classifier", elem_id="title")

    # First main row: Input + Prediction
    with gr.Row(elem_id="centered-row"):
        with gr.Column(min_width=600, scale=0, elem_id="left-panel"):
            tweet_input = gr.Textbox(
                label="Make a Prediction",
                lines=3,
                placeholder="Paste your tweet here",
                elem_id="tweet-input"
            )

            user_label_input = gr.Radio(
                elem_id="centered-row",
                label="",
                choices=["Antisemitic", "Not Antisemitic"],
            )

            predict_btn = gr.Button("Predict", elem_id="predict-btn")
            model_output = gr.Textbox(label="Model Prediction", elem_id="model-output", interactive=False)

    # Second main row: History Table (full-width)
    gr.Markdown("## Last 10 Tweets")
    history_table = gr.Dataframe(
        headers=["Tweet", "User Label", "Model Prediction", "Correct?"],
        row_count=10,
        wrap=True,
        type="array"
    )

    # Call history loader on app launch
    demo.load(fn=load_history, outputs=history_table)

    # Apply custom style for the app
    with open(os.path.join(BASE_DIR, "style.css")) as f:
        gr.HTML(f"<style>{f.read()}</style>")

    # Event binding
    predict_btn.click(
        fn=predict_and_store,
        inputs=[tweet_input, user_label_input],
        outputs=[model_output, history_table]
    )

demo.launch(show_api=False, server_port=7860, server_name="0.0.0.0")