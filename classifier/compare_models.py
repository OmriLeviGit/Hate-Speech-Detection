import os
import time

from classifier import utils
from classifier.SKlearnClassifier import SKlearnClassifier
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

    path = ""
    best_model.save_model(os.path.join(path, "model"))

    return best_model


def main():
    # increase trials before training. add to hp
    models = generate_models()

    data = models[0].load_data(set_to_min=True)
    dataset = models[0].prepare_dataset(data)

    compare_models(models, dataset)

    # loaded_classifier = SKlearnClassifier.load_model("model")   # make it part of the base
    # print(loaded_classifier)


if __name__ == "__main__":
    main()
