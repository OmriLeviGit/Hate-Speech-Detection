from classifier.TestModel import TestModel


def run_evaluation():

    classifier = TestModel()

    data = classifier.load_data(1000, 1000, 1000, debug=True)

    prepared = classifier.prepare_datasets(data, combine_irrelevant=True)



if __name__ == '__main__':
    run_evaluation()
