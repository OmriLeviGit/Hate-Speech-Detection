import os
import time

import pandas as pd
import torch

from classifier import utils
from classifier.BaseTextClassifier import BaseTextClassifier
from classifier.model_generation import generate_models



def compare_models(models, debug=False):
    # load and prepare once
    data = models[0].load_data(set_to_min=True, debug=debug)
    X_train, X_test, y_train, y_test = models[0].prepare_dataset(data)

    results = []
    start_time = time.time()

    for model in models:
        model.train(X_train, y_train)

        cv_score = model.best_score
        evaluation = model.evaluate(X_test, y_test)
        results.append((model.model_name, cv_score, evaluation, model.best_params))

        model.save_model()

    end_time = time.time()
    utils.print_header(f"Done comparing | Time: {end_time - start_time}")

    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)  # sort by cv_score

    return sorted_results


def main():
    debug = True
    print("GPU available" if torch.cuda.isavailable() else "GPU not available")

    models = generate_models(debug=debug)
    model_results = compare_models(models, debug=debug)

    print(f"\n\nBest model overall: {model_results[0]}\n\nModels sorted by score:")
    for model_name, cv_score, evaluation, params in model_results:
        print(f"{model_name}: CV Score = {cv_score:.2f} | Evaluation = {evaluation} | Params: {params}")

    df = pd.DataFrame(model_results, columns=["Model name", "CV Score", "Evaluation", "Params"])
    output_path = os.path.join(BaseTextClassifier.save_models_path, "comparison_result.csv")
    df.to_csv(output_path, index=False)

    loaded_classifier = BaseTextClassifier.load_best_model(save_models_path)


if __name__ == "__main__":
    main()
