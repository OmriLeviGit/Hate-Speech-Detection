
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC

from classifier.copy_paste import SpacyModels_SM_LG
from classifier.preprocessing.TextNormalizer import TextNormalizer

# **** dugri, see the comments
#
#
# **** this belongs to the preprocessing
# this belongs to "prepare"from here and below (until mentioned) should be in the "prepare", and prepare should be called after preprocessed


nlp = spacy.load("en_core_web_lg")
text_normalizer = TextNormalizer()
classifier = SpacyModels_SM_LG(nlp, text_normalizer)

data = classifier.load_data(700, 700, 350, "csv_files")
processed_data = classifier.preprocess_data(data)


posts = []
labels = []

label_list = list(processed_data.keys())
for label_name, post_list in processed_data.items():
    label_index = label_list.index(label_name)

    for post in post_list:
        posts.append(post)
        labels.append(f"{label_index} - {label_name}")

X = np.array(posts)
y = np.array(labels)


# 3. Split into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Train Logistic Regression model
# model = LogisticRegression(max_iter=1000)
# model = MultinomialNB()
# model = RandomForestClassifier(n_estimators=100, random_state=42)
model = LinearSVC(C=0.05, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# 6. Evaluate
y_pred = model.predict(X_test_tfidf)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
