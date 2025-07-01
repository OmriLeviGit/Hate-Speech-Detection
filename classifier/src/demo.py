from classifier.src.classifiers.BertClassifier import BertClassifier
from classifier.src.classifiers.ClassicalModelClassifier import ClassicalModelClassifier
from classifier.src.model_generation import generate_models


def train_model():
    global model
    print("Training started!")

    try:
        data = model.load_data(folder_name='datasets', debug=True)
        X_train, X_test, y_train, y_test = model.prepare_dataset(
            data, augment_ratio=0.33, irrelevant_ratio=0.4, balance_pct=0.5)

        model.train(X_train, y_train)

        model.save_model()

        print("Training completed! Model updated and ready to use.")

    except Exception as e:
        print(f"Training failed: {str(e)}")

# Load
use_bert = False    # True for the better but more demanding model
try:
    if use_bert:
        name = "distilbert uncased"
        print(f"Loading {name}...")
        model = BertClassifier.load_model(name, in_saved_models=True)
    else:
        name = "XGBoost"
        print(f"Loading {name}...")
        model = ClassicalModelClassifier.load_model("XGBoost", in_saved_models=True)

except Exception as e:
    """
    expected to happen when:
    1. no model is found
    2. model was trained inside docker and is loaded outside and vice versa
    3. the model was corrupted during the last save (happens due to insufficient RAM)
    """

    print(f"Error: {e}")
    model = generate_models(seed=1, name=name)[0]
    train_model()

# Examples
tweet = "tweet"
prediction, prob = model.predict(tweet)
print(f"prediction: {prediction}, prob: {prob}")

tweets = ["tweet 1", "tweet 2"]
prediction, prob = model.predict(tweets)
print(f"predictions: {prediction}, probs: {prob}")
