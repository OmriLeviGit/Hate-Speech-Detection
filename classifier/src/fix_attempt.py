from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from transformers import AutoTokenizer

from classifier.src.classifiers.BertClassifier import BertClassifier
from classifier.src.classifiers.SKLearnClassifier import SKLearnClassifier
from classifier.src.normalization.TextNormalizer import TextNormalizer

params = {
    "dropout": 0.0078115464911744925,
    "learning_rate": 2.3597575095971233e-05,
    "batch_size": 16,
    "epochs": 2,
    "weight_decay": 0.06700262172627848
}

def main():
    debug = False
    # utils.check_device()

    config = {
        'model_name': "new",
        'model_type': "distilbert-base-uncased"
    }
    
    model = BertClassifier(
        ["antisemitic", "not_antisemitic"],
        TextNormalizer(emoji='text'),
        AutoTokenizer.from_pretrained(config["model_type"]),
        config
    )

    # config =  {
    #     "model_name": "SGDClassifier",
    #     "model_class": SGDClassifier(),
    #     "param_grid": {
    #         'loss': ['hinge', 'log_loss'],
    #         'penalty': ['l2', 'elasticnet'],
    #         'alpha': [1e-4, 1e-3],
    #         'max_iter': [1000]
    #     }
    # }

    # model = SKLearnClassifier(
    #     ["antisemitic", "not_antisemitic"],
    #     TextNormalizer(emoji='text'),
    #     TfidfVectorizer(),
    #     config
    # )

    data = model.load_data(debug=debug)

    X_train, X_test, y_train, y_test = model.prepare_dataset(data)
    # X_train, X_test, y_train, y_test = model.prepare_dataset_old(data)

    # model.train_final_model(X_train, y_train, params)
    # model.train(X_train, y_train)


    res = model.evaluate(X_test, y_test)
    print(res)



if __name__ == "__main__":
    main()
