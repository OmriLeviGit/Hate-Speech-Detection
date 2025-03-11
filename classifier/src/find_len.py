import pandas as pd
from transformers import BertTokenizer

# Load your dataset
data = pd.read_csv('../data/pos_irr_tweets_train.csv')
texts = data['text'].tolist()

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained("asafaya/bert-base-arabic")

# Tokenize the texts
tokenized_texts = [tokenizer(text, truncation=False, padding=False) for text in texts]

# Calculate the length of each tokenized text
lengths = [len(tokenized_text['input_ids']) for tokenized_text in tokenized_texts]

# Analyze the lengths
length_series = pd.Series(lengths)
print(length_series.describe())

# Plot the distribution of text lengths
import matplotlib.pyplot as plt

plt.hist(lengths, bins=50)
plt.xlabel('Number of Tokens')
plt.ylabel('Frequency')
plt.title('Distribution of Text Lengths')
plt.show()
