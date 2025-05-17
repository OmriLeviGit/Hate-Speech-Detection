# Load and pre-process data
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('cleaned_results.csv')
df = df[df['sentiment'].isin(['Positive', 'Negative'])]  # binary classification

print("Class distribution BEFORE oversampling:")
print(df['sentiment'].value_counts())


X_text = df['cleaned_content']
y = LabelEncoder().fit_transform(df['sentiment'])

# Oversampling
# Note that Random OverSampling creates balanced classes
ros = RandomOverSampler()
X_text_resampled, y_resampled = ros.fit_resample(X_text.values.reshape(-1, 1), y)
X_text_resampled = X_text_resampled.flatten()
# Drop or fill NaNs
X_text_resampled = pd.Series(X_text_resampled).fillna("").astype(str)
y_resampled = pd.Series(y_resampled)


# Prepare TF-IDF + MLP Pipeline
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X_text_resampled)

mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=10, random_state=42)
mlp.fit(X_tfidf, y_resampled)


# Prepare Word2Vec + LSTM Pipeline
import gensim
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Tokenization
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_text_resampled)
sequences = tokenizer.texts_to_sequences(X_text_resampled)
X_seq = pad_sequences(sequences, maxlen=100)
vocab_size = len(tokenizer.word_index) + 1

# Word2Vec (train from scratch)
sentences = [text.split() for text in X_text_resampled]
w2v_model = gensim.models.Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# Create embedding matrix
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]

# LSTM model
lstm_model = Sequential([
    Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=100, trainable=False),
    LSTM(64),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(X_seq, y_resampled, epochs=3, batch_size=32, validation_split=0.2)


# Combine predictions
from sklearn.metrics import classification_report

# Create test set from original (unbalanced) data
X_train, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)

# Clean test text to avoid NaNs
X_test_text = pd.Series(X_test_text).fillna("").astype(str)

# MLP prediction
X_test_tfidf = tfidf.transform(X_test_text)
mlp_preds = mlp.predict_proba(X_test_tfidf)[:, 1]

# LSTM prediction
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test_text), maxlen=100)
lstm_preds = lstm_model.predict(X_test_seq).flatten()

# Combine predictions
final_preds = (mlp_preds + lstm_preds) / 2
binary_preds = (final_preds >= 0.5).astype(int)

print(classification_report(y_test, binary_preds))

from sklearn.metrics import accuracy_score

print("MLP only accuracy:", accuracy_score(y_test, mlp_preds >= 0.5))
print("LSTM only accuracy:", accuracy_score(y_test, lstm_preds >= 0.5))


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, binary_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
