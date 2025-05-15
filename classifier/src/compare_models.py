import os
import time

import pandas as pd

from classifier.src import utils
from classifier.src.classifiers.BaseTextClassifier import BaseTextClassifier
from classifier.src.model_generation import generate_models
from classifier.src.utils import reset_seeds


def compare_models(models, debug=False):
    reset_seeds(models[0].seed)

    # load and prepare once
    data = models[0].load_data(debug=debug)

    X_train, X_test, y_train, y_test = models[0].prepare_dataset(data, balance_pct=0.5)
    # X_train, X_test, y_train, y_test = models[0].prepare_dataset_old(data)

    results = []
    start_time = time.time()

    for model in models:
        reset_seeds(model.seed)

        model.train(X_train, y_train)
        evaluation = model.evaluate(X_test, y_test)
        cv_score = model.best_score

        results.append((model.model_name, evaluation, cv_score, model.best_params))

        model.save_model()

    total_time = utils.format_duration(time.time() - start_time)
    utils.print_header(f"Done comparing | Time: {total_time}")

    sorted_results = sorted(results, key=lambda x: x[1][1], reverse=True)  # sort by f1 score

    return sorted_results, total_time

def main():
    debug = False
    seed = 1
    print("@@@@@@@@@@@@@@@@@@@@@@@@@")
    # utils.check_device()

    models = generate_models(seed=1, debug=debug)
    model_results, total_time = compare_models(models, debug=debug)

    # Save results
    df = pd.DataFrame(model_results, columns=["Model name", "CV Score", "Evaluation", "Params"])
    df.loc[len(df)] = ['Total Time: ', '', total_time, '']

    output_path = os.path.join(BaseTextClassifier.save_models_path, "comparison_result.csv")
    df.to_csv(output_path, index=False)

    print(f"Finished running in {total_time}, best model: {model_results[0]}")


if __name__ == "__main__":
    main()
