
import spacy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from classifier.copy_paste import SpacyModels_SM_LG
from classifier.preprocessing.TextNormalizer import TextNormalizer


nlp = spacy.load("en_core_web_lg")
text_normalizer = TextNormalizer()
classifier = SpacyModels_SM_LG(nlp, text_normalizer)

data = classifier.load_data(700, 700, 350, "csv_files")
processed_data = classifier.preprocess_data(data)

# **** dugri, from here and below (until mentioned) should be in the "prepare", and prepare should be called after preprocessed

vectors = []
labels = []

for label_name, items in processed_data.items():
    for vector, _ in items:
        vectors.append(vector)
        labels.append(classifier.LABELS.index(label_name))

X = np.array(vectors)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# **** "prepare" ends here, these 4 lists should be returned by prepare

# **** whatever is below here should be in train\evaluate\predict

# This model doesn't know how to deal with arrays that each cell is an array, so we use numpy stack
X_train_2d = np.stack(X_train)
X_test_2d = np.stack(X_test)

# This model doesn't know how to deal with negative numbers so we use MinMaxScalar
scaler = MinMaxScaler()
scaled_train_embed = scaler.fit_transform(X_train_2d)
scaled_test_embed = scaler.transform(X_test_2d)

#1. creating a KNN model object
clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

#2. fit with all_train_embeddings and y_train
clf.fit(X_train_2d, y_train)

#3. get the predictions for all_test_embeddings and store it in y_pred
y_pred = clf.predict(X_test_2d)

#4. print the classification report
print(classification_report(y_test, y_pred))
