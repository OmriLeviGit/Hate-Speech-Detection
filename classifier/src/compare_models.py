import os
import time

import pandas as pd

from classifier.src import utils
from classifier.src.classifiers.BaseTextClassifier import BaseTextClassifier
from classifier.src.model_generation import generate_models
from classifier.src.utils import reset_seeds


"""
(augment_ratio, irrelevant_ratio, balance_pct)
complex models can benefit from higher ir and b_pct, but high b_pct usually performs much worse.

Best sampling parameters:
    default_sampling = (0, 0, 0.5) # default
    sklearn_sampling = (0.33, 0.33, 0.5) # best performance for sklearn (ar, ir, b_pct)
    bert_sampling = (0.33, 0.45, 0.6) # models best performance for bert (ar, ir, b_pct)
"""
def compare_models(models, debug=False):
    data = models[0].load_data(debug=debug)

    reset_seeds(models[0].seed)
    ar, ir, b_pct = (0.33, 0.45, 0.6)   # bert sampling

    X_train, X_test, y_train, y_test = models[0].prepare_dataset(data,
            augment_ratio=ar, irrelevant_ratio=ir, balance_pct=b_pct)

    results = []
    start_time = time.time()

    for model in models:
        reset_seeds(model.seed)

        model.train(X_train, y_train)
        evaluation = model.evaluate(X_test, y_test, output_file="evaluation_results.txt")

        results.append((model.model_name, evaluation, model.cv_score, model.best_params))

        model.save_model()

    total_time = utils.format_duration(time.time() - start_time)
    utils.print_header(f"Done comparing | Time: {total_time}")

    sorted_results = sorted(results, key=lambda x: x[1][1], reverse=True)  # sort by f1 score

    return sorted_results, total_time

def save_results(model_results, total_time):
    df = pd.DataFrame(model_results, columns=["Model name", "Evaluation", "CV Score", "Params"])
    df.loc[len(df)] = ['Total Time: ', total_time, '', '']

    output_path = os.path.join(BaseTextClassifier.save_models_path, "comparison_result.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)


def main():
    debug = False
    seed = 1
    # utils.check_device()

    models = generate_models(seed=seed, debug=debug)
    model_results, total_time = compare_models(models, debug=debug)
    save_results(model_results, total_time)

    print(f"Finished running in {total_time}, best model: {model_results[0]}")


if __name__ == "__main__":
    main()
