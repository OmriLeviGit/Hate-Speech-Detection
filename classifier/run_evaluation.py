def run_evaluation():
    # Create instances of different classifiers

    classifier_list = [HuggingFaceClassifier(), SpacyClassifier()]

    # Evaluate all classifiers
    results = evaluate_classifiers(classifier_list)

    # You could also save the best model
    best_classifier = max(results.columns, key=lambda x: results.loc["accuracy", x])
    print(f"Best classifier: {best_classifier}")


if __name__ == '__main__':
    run_evaluation()
