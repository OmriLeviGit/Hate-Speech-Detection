import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

# 1. Load cleaned data
df = pd.read_csv('results.csv')
# df = df.dropna(subset=['cleaned_content'])

# Keep only Positive and Negative
df = df[df['sentiment'].isin(['Positive', 'Negative'])]

# Balance the two classes
min_count = min(df['sentiment'].value_counts().values)
positive = df[df['sentiment'] == 'Positive'].sample(n=min_count, random_state=42)
negative = df[df['sentiment'] == 'Negative'].sample(n=min_count, random_state=42)

print("len: ", len(positive), len(negative))
df = pd.concat([positive, negative]).sample(frac=1, random_state=42).reset_index(drop=True)

print("Class distribution:\n", df['sentiment'].value_counts())

# 2. Extract features and labels
X = df['content']
y = df['sentiment']

# 3. TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# 4. Choose your model
# model = LogisticRegression(max_iter=1000)
# model = MultinomialNB()
model = RandomForestClassifier(n_estimators=100, random_state=42)
# model = LinearSVC(C=0.05, class_weight='balanced')

# 5. Perform k-fold cross-validation
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

# Use cross_val_predict to get predictions across folds
y_pred = cross_val_predict(model, X_tfidf, y, cv=5)

# 6. Evaluate
print("Classification Report:\n", classification_report(y, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
