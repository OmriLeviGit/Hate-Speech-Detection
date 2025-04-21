import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Load and filter dataset
df = pd.read_csv("cleaned_results.csv")
df = df[df["sentiment"].isin(["Positive", "Negative"])]

X_text = df["cleaned_content"].fillna("")
y = LabelEncoder().fit_transform(df["sentiment"])  # 0 = Negative, 1 = Positive

# Print original distribution
print("Class distribution BEFORE oversampling:")
print(df["sentiment"].value_counts())

# Oversample
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_resample(X_text.values.reshape(-1, 1), y)
X_resampled = pd.Series(X_resampled.flatten()).fillna("")
y_resampled = pd.Series(y_resampled)

# Print new distribution
print("\nClass distribution AFTER oversampling:")
print(Counter(y_resampled.map({0: "Negative", 1: "Positive"})))

# Train/test split (from original, unbalanced data)
X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.3, random_state=24)


# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_resampled_tfidf = tfidf.fit_transform(X_resampled)
X_test_tfidf = tfidf.transform(X_test.fillna(""))

# MLP model
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)
mlp.fit(X_resampled_tfidf, y_resampled)

# Check if any test sample exists in the training set
duplicates = set(X_train).intersection(set(X_test))
print(f"Overlapping entries between train and test: {len(duplicates)}")

# Predict and evaluate
preds = mlp.predict(X_test_tfidf)
print("\nClassification Report:\n", classification_report(y_test, preds))

# Confusion matrix
cm = confusion_matrix(y_test, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Get predictions
preds = mlp.predict(X_test_tfidf)

# See correct predictions
print("\n✅ Correct predictions:")
for i in range(len(preds)):
    if preds[i] == y_test[i]:
        print(f"Text: {X_test.iloc[i]}\nTrue: {y_test[i]}, Predicted: {preds[i]}\n")
        break

# See incorrect predictions
print("\n❌ Incorrect predictions:")
for i in range(len(preds)):
    if preds[i] != y_test[i]:
        print(f"Text: {X_test.iloc[i]}\nTrue: {y_test[i]}, Predicted: {preds[i]}\n")
        break

from sklearn.model_selection import cross_val_score

scores = cross_val_score(mlp, tfidf.transform(X_text.fillna("")), y, cv=5)
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())
