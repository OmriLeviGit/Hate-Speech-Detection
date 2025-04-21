# The following code is based on:
# https://github.com/codebasics/nlp-tutorials/blob/main/14_word_vectors_spacy_text_classification/word_vectors_spacy_text_classification.ipynb
# https://www.youtube.com/watch?v=ibi5hvw6f3g&list=PLeo1K3hjS3uuvuAXhYjV2lMEShq2UYSwX&index=21&ab_channel=codebasics
# The code then got refactored by ChatGPT

import pandas as pd
import spacy
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC


def load_and_prepare_data(path):
    # Read the dataset and store it in a variable df
    df = pd.read_csv(path)
    df = df.dropna(subset=['cleaned_content'])

    # Keep only positive and negative
    df = df[df['sentiment'].isin(['Positive', 'Negative'])]

    # Balance classes
    min_count = df['sentiment'].value_counts().min()
    positive = df[df['sentiment'] == 'Positive'].sample(n=min_count, random_state=42)
    negative = df[df['sentiment'] == 'Negative'].sample(n=min_count, random_state=42)

    df = pd.concat([positive, negative]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Check the distribution of labels
    print("Class distribution:\n", df['sentiment'].value_counts())
    return df


def compute_spacy_embeddings(df, column='cleaned_content'):
    nlp = spacy.load("en_core_web_lg")
    # Convert text into vectors (This might take some time)
    df['vector'] = df[column].apply(lambda text: nlp(text).vector)
    return df


def prepare_features_and_labels(df):
    # This model doesn't know how to deal with arrays that each cell is an array, so we use numpy stack
    X = np.stack(df['vector'].values)
    y = df['sentiment'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, label="Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n--- {label} Results ---")
    print(classification_report(y_test, y_pred))


def tune_linear_svc(X_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'loss': ['squared_hinge'],
        'dual': [False],
        'max_iter': [5000]
    }

    svc = LinearSVC()
    grid = GridSearchCV(svc, param_grid, cv=5, scoring='f1_weighted', verbose=1)
    grid.fit(X_train, y_train)

    print("\nBest LinearSVC parameters found:", grid.best_params_)
    print("Best cross-validated F1 score:", grid.best_score_)
    return grid.best_estimator_


# This model doesn't know how to deal with negative numbers so we use MinMaxScalar
# scaler = MinMaxScaler()
# scaled_train_embed = scaler.fit_transform(X_train_2d)
# scaled_test_embed = scaler.transform(X_test_2d)

if __name__ == "__main__":
    # Step 1: Load + prepare
    df = load_and_prepare_data("cleaned_results.csv")

    # Step 2: Get embeddings
    df = compute_spacy_embeddings(df)

    # Step 3: Features + labels
    X_train, X_test, y_train, y_test = prepare_features_and_labels(df)

    # Step 4: Baseline with KNN
    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    train_and_evaluate_model(knn, X_train, X_test, y_train, y_test, label="KNN")

    # Step 5: Tune and evaluate LinearSVC
    best_svc = tune_linear_svc(X_train, y_train)
    train_and_evaluate_model(best_svc, X_train, X_test, y_train, y_test, label="Tuned LinearSVC")