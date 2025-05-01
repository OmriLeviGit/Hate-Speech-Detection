import os
import time

from classifier import utils
from classifier.BERTClassifier import BERTClassifier
from classifier.BaseTextClassifier import BaseTextClassifier
from classifier.SKLearnClassifier import SKLearnClassifier
from classifier.model_generation import generate_models


def compare_models(models, dataset):
    X_train, X_test, y_train, y_test = dataset

    best_model = None
    results = []

    start_time = time.time()
    for model in models:

        model.train(X_train, y_train)

        score = model.best_score
        results.append((model.model_name, score))

        if not best_model or score > best_model.best_score:
            best_model = model

    end_time = time.time()
    utils.print_header(f"Done comparing | Time: {end_time - start_time} | Best model: {best_model.model_name}")

    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    print("Models sorted by score:")
    for model_name, score in sorted_results:
        print(f"{model_name}: {score:.2f}")

    best_model.evaluate(X_test, y_test)

    best_model.save_model("model")

    return best_model


def main():
    models = generate_models()
    models = [models[5]]


    data = models[0].load_data(set_to_min=True, source='debug')
    dataset = models[0].prepare_dataset(data)

    compare_models(models, dataset)

    # loaded_classifier = BERTClassifier.load_model("model")
    # loaded_classifier = BaseTextClassifier.load_best_model("model")
    # print(loaded_classifier)




if __name__ == "__main__":
    main()
