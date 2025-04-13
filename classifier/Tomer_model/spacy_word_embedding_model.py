# The following code is based on:
# https://github.com/codebasics/nlp-tutorials/blob/main/14_word_vectors_spacy_text_classification/word_vectors_spacy_text_classification.ipynb
# https://www.youtube.com/watch?v=ibi5hvw6f3g&list=PLeo1K3hjS3uuvuAXhYjV2lMEShq2UYSwX&index=21&ab_channel=codebasics


import pandas as pd
import spacy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix


# Read the dataset and store it in a variable df
df = pd.read_csv("cleaned_results.csv")
df = df.dropna(subset=['cleaned_content'])

# Add the new column which gives a unique number to each of these labels
df['label_num'] = df['sentiment'].map({'Negative' : 0, 'Positive': 1, 'Irrelevant' : -1})

# Check the distribution of labels
print(df['sentiment'].value_counts())

nlp = spacy.load("en_core_web_lg")

# Convert text into vectors (This might take some time)
df['vector'] = df['cleaned_content'].apply(lambda text: nlp(text).vector)

X_train, X_test, y_train, y_test = train_test_split(
    df.vector.values,
    df.label_num,
    test_size = 0.2,
    random_state = 42
)

# This model doesn't know how to deal with arrays that each cell is an array, so we use numpy stack
X_train_2d = np.stack(X_train)
X_test_2d = np.stack(X_test)

# This model doesn't know how to deal with negative numbers so we use MinMaxScalar
scaler = MinMaxScaler()
scaled_train_embed = scaler.fit_transform(X_train_2d)
scaled_test_embed = scaler.transform(X_test_2d)


#1. creating a KNN model object
clf = KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean')

#2. fit with all_train_embeddings and y_train
clf.fit(X_train_2d, y_train)

#3. get the predictions for all_test_embeddings and store it in y_pred
y_pred = clf.predict(X_test_2d)

#4. print the classification report
print(classification_report(y_test, y_pred))
