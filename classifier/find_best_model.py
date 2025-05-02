import os
import time

import pandas as pd

from classifier import utils
from classifier.BaseTextClassifier import BaseTextClassifier
from classifier.model_generation import generate_models


save_models_path = "/app/classifier/saved_models"

def compare_models(models, dataset):
    model_names = [model.model_name for model in models]
    utils.print_header(f"\nComparing models: {model_names}\n")

    X_train, X_test, y_train, y_test = dataset

    best_model = None
    results = []

    start_time = time.time()
    for model in models:

        model.train(X_train, y_train)

        score = model.best_score
        results.append((model.model_name, score, model.best_params))

        if not best_model or score > best_model.best_score:
            best_model = model

    end_time = time.time()
    utils.print_header(f"Done comparing | Time: {end_time - start_time} | Best model for the type: {best_model.model_name}")

    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    best_model.evaluate(X_test, y_test)
    best_model.save_model(save_models_path)

    return sorted_results


def main():
    debug = True

    sklearn_models, bert_models = generate_models(debug)

    data = sklearn_models[0].load_data(set_to_min=True, debug=debug)
    dataset = sklearn_models[0].prepare_dataset(data)

    sklearn_results = compare_models(sklearn_models, dataset)
    bert_results = compare_models(bert_models, dataset)

    model_results = sklearn_results + bert_results

    sorted_results = sorted(model_results, key=lambda x: x[1], reverse=True)

    print(f"\n\nBest model overall: {sorted_results[0]}\n")

    print("Models sorted by score:")
    for model_name, score, params in sorted_results:
        print(f"{model_name}: {score:.2f} | Params: {params}")

    df = pd.DataFrame(sorted_results, columns=["Model name", "Score", "Params"])
    output_path = os.path.join(save_models_path, "final_model_results.csv")
    df.to_csv(output_path, index=False)

    # loaded_classifier = BaseTextClassifier.load_best_model("saved_models")


if __name__ == "__main__":
    main()
