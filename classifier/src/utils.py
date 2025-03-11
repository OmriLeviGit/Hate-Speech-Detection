import pandas as pd
from sklearn.utils import shuffle

def prep_data(data_path):
    # Load the data
    data = pd.read_csv(data_path)
    # Shuffle the data
    data = shuffle(data, random_state=42)

    # Map the labels to numerical values.
    label_map = {'Positive': 0, 'Irrelevant': 1}
    data['label'] = data['label'].map(label_map)

    return data['text'].tolist(), data['label'].tolist()