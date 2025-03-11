from transformers import BertForSequenceClassification

def get_model(output_dir=None):
    if output_dir:
        # Load the trained model from the output directory
        model = BertForSequenceClassification.from_pretrained(output_dir)
    else:
        # Load the pre-trained BERT model for sequence classification
        model = BertForSequenceClassification.from_pretrained("asafaya/bert-base-arabic", num_labels=2)
    return model
