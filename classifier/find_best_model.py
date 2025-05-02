import time

from classifier import utils
from classifier.BaseTextClassifier import BaseTextClassifier
from classifier.model_generation import generate_models


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
    best_model.save_model("saved_models")

    return sorted_results[0]


def main():
    debug = False

    sklearn_models, bert_models = generate_models(debug)

    data = sklearn_models[0].load_data(set_to_min=True, debug=debug)
    dataset = sklearn_models[0].prepare_dataset(data)

    best_sklearn = compare_models(sklearn_models, dataset)
    best_bert = compare_models(bert_models, dataset)

    print(f"\n\nFinal performance: Best sklearn model: {best_sklearn}, Best bert model: {best_bert}\n\n")

    # loaded_classifier = BaseTextClassifier.load_best_model("saved_models")


if __name__ == "__main__":
    main()
