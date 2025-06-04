import os
import gradio as gr
import joblib
from classifier.src.normalization.TextNormalizer import TextNormalizer

# Load model and vectorizer
BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "best_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))
normalizer = TextNormalizer(emoji="text")

history = []

def predict_and_store(tweet, user_label):
    if not tweet.strip():
        return "⚠️ Please enter a tweet", history
    if user_label is None:
        return "⚠️ Please select your label", history

    norm_text = normalizer.normalize(tweet)
    vectorized = vectorizer.transform([norm_text])
    prediction = model.predict(vectorized)[0]
    model_label = "Antisemitic" if prediction == 1 else "Not Antisemitic"

    correct = "v️" if user_label == model_label else "x"

    if len(history) >= 10:
        history.pop(0)
    history.append((tweet, user_label, model_label, correct))

    return model_label, history


with gr.Blocks(theme=gr.themes.Default()) as demo:

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

    # Custom style
    gr.HTML("""
    <style>
      #title {
        text-align: center;
      }

      #centered-row {
        display: flex;
        justify-content: center;
        }

      #left-panel {
        margin: 20px;
        padding: 10px;
        border-radius: 8px;
      }
      
      #predict-btn {
        background-color: #ff9800 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 6px;
        padding: 10px 20px;
        font-size: 1rem;
        border: none !important;
        }
        
        #predict-btn:hover {
        background-color: #e57c00 !important;
      }

      #user-label {
        margin-left: 10px;
      }

      .gr-dataframe {
        font-family: inherit !important;
        font-size: 0.95rem;
        max-height: 300px;
        overflow: auto;
        border-radius: 8px;
      }

      .gr-dataframe table {
        table-layout: fixed;
        width: 100%;
      }

      .gr-dataframe table thead tr th:first-child,
      .gr-dataframe table tbody tr td:first-child {
        max-width: 300px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      #tweet-input {
        flex: 1;
      }
      
      #model-output {
        font-size: 1.1rem;
        font-weight: bold;
      }
      
    </style>
    """)


    # Event binding
    predict_btn.click(
        fn=predict_and_store,
        inputs=[tweet_input, user_label_input],
        outputs=[model_output, history_table]
    )

demo.launch(show_api=False)