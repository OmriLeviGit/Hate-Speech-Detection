import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


# 1. Load cleaned data
df = pd.read_csv('cleaned_results.csv')
# Drops rows where cleaned_content is NaN
df = df.dropna(subset=['cleaned_content'])
# If we want to treat 'Irrelevant' as 'Negative' (and get a binary classification of Positive/Negative)
df['binary_sentiment'] = df['sentiment'].apply(lambda x: 'Positive' if x == 'Positive' else 'Negative')


# 2. Extract features and labels
X = df['cleaned_content']
# Use the next line for binary sentiment
y = df['binary_sentiment']
# Use the next line for {Positive, Negative, Irrelevant} sentiment
# y = df['sentiment']

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
