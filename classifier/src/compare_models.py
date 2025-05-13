import os
import time

import pandas as pd

from classifier.src import utils
from classifier.src.classifiers.BaseTextClassifier import BaseTextClassifier
from classifier.src.model_generation import generate_models



def compare_models(models, debug=False):
    # load and prepare once
    data = models[0].load_data(irrelevant_ratio=0.33, debug=debug)

    X_train, X_test, y_train, y_test = models[0].prepare_dataset(
        data, test_size=0.2, augment_ratio=1.0, balance_classes=True)

    results = []
    start_time = time.time()

    for model in models:
        model.train(X_train, y_train)

        cv_score = model.best_score
        evaluation = model.evaluate(X_test, y_test)
        results.append((model.model_name, cv_score, evaluation, model.best_params))

        model.save_model()

    total_time = utils.format_duration(time.time() - start_time)
    utils.print_header(f"Done comparing | Time: {total_time}")

    sorted_results = sorted(results, key=lambda x: x[2][0], reverse=True)  # sort by evaluation score

    return sorted_results, total_time

def main():
    debug = True
    # utils.check_device()

    models = generate_models(debug=debug)
    model_results, total_time = compare_models(models, debug=debug)

    # Save results
    df = pd.DataFrame(model_results, columns=["Model name", "CV Score", "Evaluation", "Params"])
    df.loc[len(df)] = ['Total Time: ', '', total_time, '']

    output_path = os.path.join(BaseTextClassifier.save_models_path, "comparison_result.csv")
    df.to_csv(output_path, index=False)

    print(f"Finished running in {total_time}, best model: {model_results[0]}")


if __name__ == "__main__":
    main()
