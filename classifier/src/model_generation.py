from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from transformers import AutoTokenizer

from classifier.src.classifiers.BertClassifier import BertClassifier
from classifier.src.classifiers.ClassicalModelClassifier import ClassicalModelClassifier
from classifier.src.normalization.TextNormalizer import TextNormalizer
from classifier.src.normalization.TextNormalizerRoBERTa import TextNormalizerRoBERTa

debug_classical_configs = [
    {
        "model_name": "DEBUG CLASSICAL",
        "model_class": LogisticRegression(),
        "param_grid": {
            'C': [1],
            'penalty': ['l2'],
            'solver': ['liblinear'],
            'max_iter': [100]
        },
    },
]

debug_bert_configs = [
    {
        "model_name": "DEBUG BERT",
        "model_type": "distilbert-base-uncased",
        "hyper_parameters": {
            "learning_rate_range": (5e-6, 5e-6),
            "learning_rate_log": True,
            "batch_sizes": [8],
            "epochs_range": (1, 1),
            "weight_decay_range": (0.01, 0.01),
            "dropout_range": (0.1, 0.1),
        },
    }
]

classical_configs = [
    # {
    #     "model_name": "LinearSVC",
    #     "model_class": LinearSVC(),
    #     "param_grid": {
    #         'C': [0.1, 1, 10],
    #         'max_iter': [1000],
    #         'loss': ['squared_hinge'],
    #         'dual': [False]
    #     },
    # },
    # {
    #     "model_name": "LogisticRegression",
    #     "model_class": LogisticRegression(),
    #     "param_grid": {
    #         'C': [0.5, 1, 5],
    #         'penalty': ['l2'],
    #         'solver': ['liblinear', 'lbfgs'],
    #         'max_iter': [1000]
    #     },
    # },
    # {
    #     "model_name": "KNeighborsClassifier",
    #     "model_class": KNeighborsClassifier(),
    #     "param_grid": {
    #         'n_neighbors': [3, 5, 7, 11],
    #         'weights': ['uniform', 'distance'],
    #         'metric': ['euclidean']
    #     },
    # },
    # {
    #     "model_name": "SGDClassifier",
    #     "model_class": SGDClassifier(),
    #     "param_grid": {
    #         'loss': ['hinge', 'log_loss'],
    #         'penalty': ['l2', 'elasticnet'],
    #         'alpha': [1e-4, 1e-3],
    #         'max_iter': [1000]
    #     }
    # },
    {
        "model_name": "RandomForestClassifier",
        "model_class": RandomForestClassifier(),
        "param_grid": {
            'n_estimators': [100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [2],
            'max_features': ['sqrt', 'log2'],
            'class_weight': [None, 'balanced', {0: 2, 1: 1}, {0: 3, 1: 1}]
        },
    },
    {
    'model_class': XGBClassifier(eval_metric='logloss', random_state=42),
    'model_name': 'XGBoost',
    'param_grid': {
        'n_estimators': [200],
        'learning_rate': [0.005, 0.1],
        'max_depth': [4],
        'colsample_bytree': [0.3, 0.5], # use %x of features used per tree, a little like dropout_rate in deep learning
        'reg_alpha': [0, 0.2],          # L1 regularization
        'reg_lambda': [1, 2],           # L2 regularization.
        },
    }
]

bert_hyperparameters = {
            "learning_rate_range": (5e-6, 1e-4),
            "learning_rate_log": True,
            "batch_sizes": [16],
            "epochs_range": (2, 5),
            "weight_decay_range": (0.001, 0.1),
            "dropout_range": (0, 0.5),
        }

bert_configs = [
    {
        "model_name": "distilbert uncased",
        "model_type": "distilbert-base-uncased",
        "hyper_parameters": bert_hyperparameters
    },
    {
        "model_name": "vinai bertweet",
        "model_type": "vinai/bertweet-base",
        "hyper_parameters": bert_hyperparameters,
        "variants": [
            {
                "variant_name": "RoBERTa normalizer, tokenizer_normalization=True",
                "normalizer": TextNormalizerRoBERTa(),
                "tokenizer_normalization": True,
            },
            {
                "variant_name": "RoBERTa normalizer, tokenizer_normalization=False",
                "normalizer": TextNormalizerRoBERTa(),
                "tokenizer_normalization": False,
            },
        ]
    },
]

def ini_classical_models(configs, default_labels, seed, debug=False):
    if debug:
        configs = debug_classical_configs
        default_labels.append("irrelevant")

    sklearn_models = []

    default_normalizer = TextNormalizer(emoji='text')
    default_vectorizer = TfidfVectorizer()

    for config in configs:
        base_config = config.copy()
        variants = base_config.pop("variants", [])

        # Create the standard model
        classifier = ClassicalModelClassifier(
            default_labels,
            default_normalizer,
            default_vectorizer,
            base_config,
            seed
        )
        sklearn_models.append(classifier)

        # Create any additional variants
        for variant in variants:
            model_config = base_config.copy()

            normalizer = variant.get("normalizer", default_normalizer)
            vectorizer = variant.get("vectorizer", default_vectorizer)
            default_labels = variant.get("labels", default_labels)

            model_config["model_name"] = f"{base_config['model_name']}_{variant.get('variant_name')}"

            classifier = ClassicalModelClassifier(default_labels, normalizer, vectorizer, model_config, seed=seed)
            sklearn_models.append(classifier)

    return sklearn_models


def ini_bert_models(configs, default_labels, seed, debug=False):
    if debug:
        configs = debug_bert_configs

    bert_models = []

    default_normalizer = TextNormalizer(emoji='text')

    for config in configs:
        base_config = config.copy()
        variants = base_config.pop("variants", [])

        default_tokenizer = AutoTokenizer.from_pretrained(base_config["model_type"])

        classifier = BertClassifier(
            default_labels,
            default_normalizer,
            default_tokenizer,
            base_config,
            seed=seed
        )
        bert_models.append(classifier)

        # Create any additional variants
        for variant in variants:
            model_config = base_config.copy()

            normalizer = variant.get("normalizer", default_normalizer)
            tokenizer = default_tokenizer

            # Create tokenizer with normalization parameter if specified
            normalization = variant.get("tokenizer_normalization")
            if normalization is not None:
                tokenizer = AutoTokenizer.from_pretrained(model_config["model_type"], normalization=normalization)

            labels = variant.get("labels", default_labels)

            model_config["model_name"] = f"{base_config['model_name']}_{variant.get('variant_name')}"

            classifier = BertClassifier(labels, normalizer, tokenizer, model_config, seed=seed)
            bert_models.append(classifier)

    return bert_models


def generate_models(seed, debug=False):
    labels = ["antisemitic", "not_antisemitic"]

    models = []
    models.extend(ini_classical_models(classical_configs, labels, seed=seed, debug=debug))
    models.extend(ini_bert_models(bert_configs, labels, seed=seed, debug=debug))

    model_names = [model.model_name for model in models]

    print(f"Generated {len(models)} model objects: \n{model_names}")

    return models
