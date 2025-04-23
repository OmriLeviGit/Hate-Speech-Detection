import pandas as pd
import re
import spacy
from sklearn.utils import resample

# Load model
# (first time, run: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

# Load the data
df = pd.read_csv('results.csv')

# Balance classes so each class has the same amount of examples for the model
neg = df[df['sentiment'] == 'Negative']
pos = df[df['sentiment'] == 'Positive']
irr = df[df['sentiment'] == 'Irrelevant']

min_len = min(len(neg), len(pos), len(irr))

df_balanced = pd.concat([
    resample(neg, replace=False, n_samples=min_len, random_state=42),
    resample(pos, replace=False, n_samples=min_len, random_state=42),
    resample(irr, replace=False, n_samples=min_len, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

# Cleaning function using spaCy
def clean_text(text):
    # Replace @UserMention with <user>
    text = re.sub(r'@\w+', '<user>', text)
    # Remove the # sign from hashtags
    text = re.sub(r'#(\w+)', r'hashtag_\1', text)
    # Remove URLS
    text = re.sub(r'http\S+|www\S+', '', text)
    # Convert to lowercase letters
    text = text.lower()

    doc = nlp(text)

    tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and not token.is_stop and not token.is_punct
    ]
    return ' '.join(tokens)

# Start text pre-processing
df_balanced['cleaned_content'] = df_balanced['content'].apply(clean_text)
df_balanced.to_csv('cleaned_results.csv', index=False)

# Preview result
print(df_balanced[['cleaned_content', 'sentiment']].head())