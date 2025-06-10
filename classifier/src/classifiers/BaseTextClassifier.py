import os
import pickle
import random
from abc import ABC, abstractmethod
import nlpaug.augmenter.word as naw
from collections import Counter

import numpy as np
import pandas as pd
from huggingface_hub.errors import RepositoryNotFoundError
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score, precision_score, \
    recall_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import snapshot_download

from classifier.src.utils import format_duration, capture_output


class BaseTextClassifier(ABC):
    """Abstract base class for text classifiers"""

    save_models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_models")
    final_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "final_model")

    def __init__(self, labels: list = None, seed = None):
        self.LABELS = labels
        self.seed = seed

        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(self.LABELS)

        self.best_model = None
        self.model_name = None

    def load_data(self, folder_name='datasets', debug=False) -> dict[str, list]:
        if debug:
            print("loading with 'debug' dataset")
            return self._initialize_test_dataset()

        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        folder_path = os.path.join(script_dir, folder_name)

        # Get all CSV files in the folder
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        if not csv_files:
            raise ValueError(f"No CSV files found in {folder_path}")

        # Load and concatenate all CSV files
        dfs = []
        for file in csv_files:
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            dfs.append(df)

        # Concatenate all dataframes and remove duplicates
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['content'], keep='first')

        data = {
            'antisemitic': combined_df[combined_df['sentiment'].str.lower() == 'positive']['content'].tolist(),
            'not_antisemitic': combined_df[combined_df['sentiment'].str.lower() == 'negative']['content'].tolist(),
            'irrelevant': combined_df[combined_df['sentiment'].str.lower() == 'irrelevant']['content'].tolist()
        }

        return data

    def prepare_dataset(self, raw_data: dict[str, list[str]], test_size=0.2, augment_ratio=0, irrelevant_ratio=0,
                        balance_pct=None, balance_classes=False) -> tuple[
        list[str], list[str], list[str], list[str]]:
        """
        Prepare and split into train and test sets based on the number of labels

        Args:
            raw_data: Dictionary of label -> list of texts
            test_size: Proportion of data for testing
            augment_ratio: How much to augment (1.0 = double the size)
            balance_pct: Target percentage for not_antisemitic samples (0.5 = equal split) in 2 label classifications
            balance_classes: Balance the size of the classes to match the size of the minority class in multi-label classification
            irrelevant_ratio: Ratio of irrelevant samples to mix into not_antisemitic for 2-label case
        """

        if len(self.LABELS) == 2:
            return self._prepare_binary_dataset(
                raw_data,
                test_size=test_size,
                augment_ratio=augment_ratio,
                balance_pct=balance_pct,
                irrelevant_ratio=irrelevant_ratio
            )
        else:
            return self._prepare_multiclass_dataset(
                raw_data,
                test_size=test_size,
                augment_ratio=augment_ratio,
                balance_classes=balance_classes
            )

    def _prepare_binary_dataset(self, raw_data: dict[str, list[str]], test_size=0.2, irrelevant_ratio=0,
                                augment_ratio=0, balance_pct=None) -> tuple[
        list[str], list[str], list[str], list[str]]:
        """
        Prepare binary classification dataset (antisemitic vs not_antisemitic)

        Args:
            raw_data: Dictionary of label -> list of texts
            test_size: Proportion of data for testing
            augment_ratio: How much to augment antisemitic samples (1.0 = double)
            balance_pct: Target percentage for not_antisemitic samples (0.5 = equal split)
            irrelevant_ratio: Target ratio of irrelevant samples within not_antisemitic
        """
        # Extract raw data for each category
        antisemitic_texts = raw_data['antisemitic']
        not_antisemitic_texts = raw_data['not_antisemitic']
        irrelevant_texts = raw_data['irrelevant']

        # First, split each category into train and test sets
        # This ensures we maintain the ratios in both train and test sets
        antisemitic_train, antisemitic_test = train_test_split(
            antisemitic_texts,
            test_size=test_size,
            random_state=self.seed
        )

        not_antisemitic_train, not_antisemitic_test = train_test_split(
            not_antisemitic_texts,
            test_size=test_size,
            random_state=self.seed
        )

        irrelevant_train, irrelevant_test = train_test_split(
            irrelevant_texts,
            test_size=test_size,
            random_state=self.seed
        )

        # Calculate how many irrelevant samples to include in train and test sets
        # to achieve the desired ratio
        not_antisemitic_train_count = len(not_antisemitic_train)
        not_antisemitic_test_count = len(not_antisemitic_test)

        # Calculate irrelevant samples to add for train set
        train_irrelevant_to_add = int(not_antisemitic_train_count * irrelevant_ratio / (1 - irrelevant_ratio))
        train_irrelevant_to_add = min(train_irrelevant_to_add, len(irrelevant_train))

        # Calculate irrelevant samples to add for test set
        test_irrelevant_to_add = int(not_antisemitic_test_count * irrelevant_ratio / (1 - irrelevant_ratio))
        test_irrelevant_to_add = min(test_irrelevant_to_add, len(irrelevant_test))

        # Create binary train set
        X_train = antisemitic_train + not_antisemitic_train + irrelevant_train[:train_irrelevant_to_add]
        y_train = ['antisemitic'] * len(antisemitic_train) + \
                  ['not_antisemitic'] * (len(not_antisemitic_train) + train_irrelevant_to_add)

        # Keep track of original categories for balancing
        orig_train = ['antisemitic'] * len(antisemitic_train) + \
                     ['not_antisemitic'] * len(not_antisemitic_train) + \
                     ['irrelevant'] * train_irrelevant_to_add

        # Create binary test set
        X_test = antisemitic_test + not_antisemitic_test + irrelevant_test[:test_irrelevant_to_add]
        y_test = ['antisemitic'] * len(antisemitic_test) + \
                 ['not_antisemitic'] * (len(not_antisemitic_test) + test_irrelevant_to_add)

        augmented_count = 0
        if augment_ratio > 0:
            augmented_texts, augmented_labels = self._augment_specific_class(
                X_train,
                y_train,
                target_class='antisemitic',
                augment_ratio=augment_ratio
            )

            # Add augmented texts to training set
            X_train.extend(augmented_texts)
            y_train.extend(augmented_labels)
            orig_train.extend(['antisemitic'] * len(augmented_texts))
            augmented_count = len(augmented_texts)

        # Balance classes if requested
        if balance_pct:
            orig_test = ['antisemitic'] * len(antisemitic_test) + \
                        ['not_antisemitic'] * len(not_antisemitic_test) + \
                        ['irrelevant'] * test_irrelevant_to_add

            X_train, y_train = self._balance_binary_dataset(X_train, y_train, orig_train, balance_pct)
            X_test, y_test = self._balance_binary_dataset(X_test, y_test, orig_test, balance_pct)

        train_counts = Counter(y_train)
        test_counts = Counter(y_test)

        print(f"\nFinal *train* set: {train_counts['antisemitic']} antisemitic "
              f"({augmented_count} augmented), {train_counts['not_antisemitic']} not_antisemitic")

        print(f"Final *test* set: {test_counts['antisemitic']} antisemitic, "
              f"{test_counts['not_antisemitic']} not_antisemitic\n")

        X_train, y_train = shuffle(X_train, y_train, random_state=self.seed)
        X_test, y_test = shuffle(X_test, y_test, random_state=self.seed)

        return X_train, X_test, y_train, y_test

    def _balance_binary_dataset(self, texts: list[str], labels: list[str], orig_categories: list[str],
                                balance_pct) -> tuple[list[str], list[str]]:
        """
        Balance binary classification dataset to achieve desired class distribution
        while preserving the ratio of original categories within not_antisemitic

        Args:
            texts: List of text samples
            labels: List of corresponding labels
            orig_categories: Original categories before merging
            balance_pct: Target percentage for not_antisemitic samples (0.5 = equal split)

        Returns:
            Tuple of (balanced_texts, balanced_labels)
        """
        # Get indices for each class and original category
        antisemitic_indices = [i for i, label in enumerate(labels) if label == 'antisemitic']

        true_not_antisemitic_indices = [
            i for i, (label, orig) in enumerate(zip(labels, orig_categories))
            if label == 'not_antisemitic' and orig == 'not_antisemitic'
        ]

        irrelevant_indices = [
            i for i, (label, orig) in enumerate(zip(labels, orig_categories))
            if label == 'not_antisemitic' and orig == 'irrelevant'
        ]

        # Count of samples
        antisemitic_count = len(antisemitic_indices)
        true_not_antisemitic_count = len(true_not_antisemitic_indices)
        irrelevant_count = len(irrelevant_indices)

        # If there are no antisemitic samples or no not_antisemitic samples, return original data
        if antisemitic_count == 0 or (true_not_antisemitic_count + irrelevant_count) == 0:
            print("Cannot balance dataset - missing samples from one of the classes")
            return texts, labels

        # Calculate target size for not_antisemitic group based on balance_pct
        if balance_pct >= 1.0:
            # Edge case: if balance_pct is 1.0, we would keep all not_antisemitic samples
            not_antisemitic_target = true_not_antisemitic_count + irrelevant_count
        else:
            not_antisemitic_target = int(antisemitic_count * balance_pct / (1 - balance_pct))
            # Ensure we don't exceed the available samples
            not_antisemitic_target = min(not_antisemitic_target, true_not_antisemitic_count + irrelevant_count)

        # Calculate original ratio of irrelevant to true not_antisemitic
        total_not_antisemitic = true_not_antisemitic_count + irrelevant_count
        irrelevant_ratio = irrelevant_count / total_not_antisemitic if total_not_antisemitic > 0 else 0

        # Calculate how many of each to keep to maintain ratio
        irrelevant_to_keep = int(not_antisemitic_target * irrelevant_ratio)
        true_not_antisemitic_to_keep = not_antisemitic_target - irrelevant_to_keep

        # Ensure we don't request more than available
        true_not_antisemitic_to_keep = min(true_not_antisemitic_to_keep, true_not_antisemitic_count)
        irrelevant_to_keep = min(irrelevant_to_keep, irrelevant_count)

        # If we couldn't meet the target exactly because of available samples, adjust the other category
        if true_not_antisemitic_to_keep < int(not_antisemitic_target * (1 - irrelevant_ratio)):
            # Try to add more irrelevant to compensate
            additional_irrelevant = min(
                not_antisemitic_target - true_not_antisemitic_to_keep - irrelevant_to_keep,
                irrelevant_count - irrelevant_to_keep
            )
            irrelevant_to_keep += additional_irrelevant
            print(f"WARNING: Not enough true not_antisemitic samples to maintain ratio. "
                  f"Added {additional_irrelevant} additional irrelevant samples to compensate.")

        elif irrelevant_to_keep < int(not_antisemitic_target * irrelevant_ratio):
            # Try to add more true_not_antisemitic to compensate
            additional_true = min(
                not_antisemitic_target - true_not_antisemitic_to_keep - irrelevant_to_keep,
                true_not_antisemitic_count - true_not_antisemitic_to_keep
            )
            true_not_antisemitic_to_keep += additional_true
            print(f"WARNING: Not enough irrelevant samples to maintain ratio. "
                  f"Added {additional_true} additional true not_antisemitic samples to compensate.")

        # Sample from each original category
        sampled_true_indices = random.sample(true_not_antisemitic_indices, k=true_not_antisemitic_to_keep)\
            if true_not_antisemitic_to_keep > 0 else []

        sampled_irrelevant_indices = random.sample(irrelevant_indices, k=irrelevant_to_keep
        ) if irrelevant_to_keep > 0 else []

        # Combine all indices
        final_indices = antisemitic_indices + sampled_true_indices + sampled_irrelevant_indices

        # Create balanced dataset
        balanced_texts = [texts[i] for i in final_indices]
        balanced_labels = [labels[i] for i in final_indices]

        # Calculate actual achieved balance
        actual_balance = (true_not_antisemitic_to_keep + irrelevant_to_keep) / len(final_indices)

        # print(f"\nBalanced dataset: {antisemitic_count} antisemitic, "
        #       f"{true_not_antisemitic_to_keep + irrelevant_to_keep} not_antisemitic ({true_not_antisemitic_to_keep} are true not_antisemitic, {irrelevant_to_keep} are irrelevant)")
        # print(f"Target balance: {balance_pct:.2f}, Achieved balance: {actual_balance:.2f}")

        return balanced_texts, balanced_labels

    def _prepare_multiclass_dataset(self, raw_data: dict[str, list[str]], test_size=0.2, augment_ratio=0,
                                    balance_classes=True) -> tuple[list[str], list[str], list[str], list[str]]:
        """
        Prepare multi-class classification dataset

        Args:
            raw_data: Dictionary of label -> list of texts
            test_size: Proportion of data for testing
            augment_ratio: How much to augment antisemitic samples (1.0 = double)
            balance_classes: Whether to balance classes to minority class size
        """
        # Filter to only use categories in LABELS
        datasets = {label: raw_data[label] for label in self.LABELS if label in raw_data}

        # First split each category separately to maintain stratification
        train_sets = {}
        test_sets = {}

        for label, texts in datasets.items():
            if not texts:  # Skip empty categories
                print(f"Warning: No samples for class '{label}'")
                continue

            # Split this category
            train_data, test_data = train_test_split(
                texts,
                test_size=test_size,
                random_state=self.seed
            )

            train_sets[label] = train_data
            test_sets[label] = test_data

            print(f"Class '{label}': {len(train_data)} train, {len(test_data)} test samples")

        # Combine into training and test sets
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        for label, texts in train_sets.items():
            X_train.extend(texts)
            y_train.extend([label] * len(texts))

        for label, texts in test_sets.items():
            X_test.extend(texts)
            y_test.extend([label] * len(texts))

        # Shuffle the combined datasets
        train_indices = list(range(len(X_train)))
        np.random.shuffle(train_indices)
        X_train = [X_train[i] for i in train_indices]
        y_train = [y_train[i] for i in train_indices]

        test_indices = list(range(len(X_test)))
        np.random.shuffle(test_indices)
        X_test = [X_test[i] for i in test_indices]
        y_test = [y_test[i] for i in test_indices]

        # Log dataset composition
        train_counts = Counter(y_train)
        test_counts = Counter(y_test)
        print(f"Multi-class train set composition: {dict(train_counts)}")
        print(f"Multi-class test set composition: {dict(test_counts)}")

        # Track current length to identify augmented samples later
        original_length = len(X_train)
        augmented_indices = []

        # Augment antisemitic samples if requested
        if augment_ratio > 0 and 'antisemitic' in datasets:
            augmented_texts, augmented_labels = self._augment_specific_class(
                X_train, y_train,
                target_class='antisemitic',
                augment_ratio=augment_ratio
            )

            # Track the indices of augmented samples
            augmented_indices = list(range(original_length, original_length + len(augmented_texts)))

            X_train.extend(augmented_texts)
            y_train.extend(augmented_labels)

            print(f"After augmentation - antisemitic samples: {y_train.count('antisemitic')}")
            print(f"Added {len(augmented_texts)} augmented samples at indices {augmented_indices}")

        # Balance classes if requested
        if balance_classes:
            X_train, y_train = self._balance_multiclass_dataset(X_train, y_train, augmented_indices)

        return X_train, X_test, y_train, y_test

    def _balance_multiclass_dataset(self, texts: list[str], labels: list[str],
                                    augmented_indices=None) -> tuple[list[str], list[str]]:
        """
        Balance multi-class dataset by downsampling to the minority class,
        prioritizing original samples over augmented ones

        Args:
            texts: List of text samples
            labels: List of corresponding labels
            augmented_indices: Indices of augmented samples to prioritize removing

        Returns:
            Tuple of (balanced_texts, balanced_labels)
        """
        # Count samples per class
        label_counts = Counter(labels)
        print(f"Original class distribution: {dict(label_counts)}")

        # Find the minimum class size (minority class)
        min_count = min(label_counts.values())
        print(f"Balancing to minority class size: {min_count}")

        # Sample from each class to match the minority class size
        final_indices = []

        for label in self.LABELS:
            if label in label_counts:
                # Get indices for this class
                class_indices = [i for i, l in enumerate(labels) if l == label]

                # Special handling for antisemitic class if we have augmented indices
                if label == 'antisemitic' and augmented_indices is not None and len(class_indices) > min_count:
                    # Identify which antisemitic samples are augmented
                    augmented_antisemitic = [i for i in class_indices if i in augmented_indices]
                    original_antisemitic = [i for i in class_indices if i not in augmented_indices]

                    # First take all original samples we can
                    original_to_keep = min(len(original_antisemitic), min_count)
                    original_sampled = original_antisemitic[:original_to_keep]

                    # If we need more, sample from augmented
                    augmented_needed = min_count - original_to_keep
                    augmented_sampled = []
                    if augmented_needed > 0:
                        augmented_sampled = np.random.choice(
                            augmented_antisemitic,
                            size=min(augmented_needed, len(augmented_antisemitic)),
                            replace=False
                        ).tolist()

                    sampled_indices = original_sampled + augmented_sampled
                    print(
                        f"Class '{label}': kept {len(original_sampled)} original and {len(augmented_sampled)} augmented samples")
                else:
                    # Standard sampling for other classes
                    sampled_indices = np.random.choice(
                        class_indices,
                        size=min(min_count, len(class_indices)),
                        replace=False
                    ).tolist()
                    print(f"Class '{label}': sampled {len(sampled_indices)} from {len(class_indices)}")

                final_indices.extend(sampled_indices)

        # Create balanced dataset
        balanced_texts = [texts[i] for i in final_indices]
        balanced_labels = [labels[i] for i in final_indices]

        # Final check on balance
        final_counts = Counter(balanced_labels)
        print(f"Final class distribution: {dict(final_counts)}")

        return balanced_texts, balanced_labels

    def _augment_specific_class(self, texts: list[str], labels: list[str],
                                target_class: str, augment_ratio: float = 1.0) -> tuple[list[str], list[str]]:
        """
        Augment only the specified class

        Args:
            texts: List of text samples
            labels: List of corresponding labels
            target_class: Class to augment
            augment_ratio: How much to augment (1.0 = double the size)
        """
        # Find target class samples
        target_indices = [i for i, label in enumerate(labels) if label == target_class]
        original_count = len(target_indices)
        samples_needed = int(original_count * augment_ratio)

        if samples_needed <= 0:
            return [], []

        primary_aug = naw.SynonymAug(aug_src='wordnet', aug_min=1, aug_max=2)
        fallback_aug = naw.RandomWordAug(action="swap", aug_min=1, aug_max=2)

        augmented_texts = []
        augmented_labels = []

        # Generate augmentations - remove failed entries
        remaining_indices = target_indices.copy()

        while len(augmented_texts) < samples_needed and remaining_indices:
            indices_to_remove = []

            for idx in remaining_indices:
                if len(augmented_texts) >= samples_needed:
                    break

                text = texts[idx]

                # Try primary augmentation
                aug_text_list = primary_aug.augment(text)
                if isinstance(aug_text_list, list) and len(aug_text_list) > 0:
                    aug_text = aug_text_list[0]
                else:
                    aug_text = aug_text_list

                if aug_text == text:  # Fallback if no change
                    fallback_text_list = fallback_aug.augment(text)
                    if isinstance(fallback_text_list, list) and len(fallback_text_list) > 0:
                        aug_text = fallback_text_list[0]
                    else:
                        aug_text = fallback_text_list

                # Check if augmentation was successful
                if aug_text != text and aug_text not in augmented_texts:
                    augmented_texts.append(aug_text)
                    augmented_labels.append(target_class)
                else:
                    # Mark this index for removal since it failed
                    indices_to_remove.append(idx)

            # Remove failed indices
            for idx in indices_to_remove:
                remaining_indices.remove(idx)

        print(f"Class '{target_class}': {original_count} samples, generated {len(augmented_texts)} augmented samples")
        print(f"Augment ratio: {augment_ratio} (target: {samples_needed} new samples)")

        return augmented_texts, augmented_labels

    @abstractmethod
    def preprocess(self, datasets: list[str]) -> list[str]:
        """Apply preprocessing to datasets."""
        pass

    def compute_all_metrics(self, y_true, y_pred):
        """Compute all standard metrics"""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average='weighted', zero_division=0),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0)
        }

    def compute_custom_f1(self, y_true, y_pred):
        """Core custom F1 logic - shared by all subclasses"""
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        if len(f1_per_class) >= 2:
            f1_class_0, f1_class_1 = f1_per_class[0], f1_per_class[1]
            difference = abs(f1_class_0 - f1_class_1)
            soft_threshold = 0.01
            excess_difference = max(0, difference - soft_threshold)
            penalty = excess_difference ** 2
            alpha = 50
            custom_weighted_f1 = f1 - alpha * penalty
        else:
            custom_weighted_f1 = f1

        return custom_weighted_f1

    @abstractmethod
    def train(self, *args) -> None:
        """Train the model"""
        pass

    def evaluate(self, X_test: list[str], y_test: list[str], output_file=None, print_evaluation=False) -> tuple[float, float, float, float]:
        if not self.best_model:
            raise ValueError("Model not trained yet")

        # Preprocess test data
        X_processed = self.preprocess(X_test)
        y_encoded = self.label_encoder.transform(y_test)
        y_pred, _ = self.predict(X_processed)

        # Calculate metrics
        accuracy = accuracy_score(y_encoded, y_pred)
        f1 = f1_score(y_encoded, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_encoded, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_encoded, y_pred, average='weighted', zero_division=0)

        if print_evaluation:
            import pandas as pd

            y_pred_decoded = self.label_encoder.inverse_transform(y_pred).tolist()

            # Clean tweets by replacing newlines with spaces
            cleaned_tweets = [text.replace('\n', ' ').replace('\r', ' ') for text in X_test]

            comparison_results = list(zip(y_test, y_pred_decoded, cleaned_tweets))
            sorted_results = sorted(comparison_results, key=lambda x: (x[0], x[1]))

            # Create DataFrame
            df = pd.DataFrame(sorted_results, columns=['Expected (user tagged)', 'Actual (model predicted)', 'Tweet'])

            # Copy to clipboard for Excel
            df.to_clipboard(index=False, sep='\t')
            print("Data copied to clipboard! You can now paste it directly into Excel.")

            # Optionally also display the DataFrame
            print(df)

        if output_file:
            output = capture_output(self.print_evaluation, y_encoded, y_pred, accuracy, f1)
            path = os.path.join(BaseTextClassifier.save_models_path, output_file)
            os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, 'a') as f:
                f.write(output)
        else:
            self.print_evaluation(y_encoded, y_pred, accuracy, f1)

        return accuracy, f1, precision, recall

    @abstractmethod
    def predict(self, text):
        """Make prediction on a single text"""
        pass

    def print_best_model_results(self, cv_score, best_param, training_duration=None):
        print(f"\n=== Training result - Model: {self.model_name} ===")
        print("Best Cross-validated Score:", round(cv_score, 2))
        print("Best Params:", best_param)
        if training_duration > 30:
            print(f"\nActual Training time: {format_duration(training_duration)}")

        print()

    def print_evaluation(self, y_true, y_pred, accuracy, f1):
        print(f"\n=== Model evaluation - {self.model_name} ===")
        print(f"Accuracy score: {round(accuracy, 2)} | f1 score: {round(f1, 2)}", )
        print("\nClassification Report:\n", classification_report(y_true, y_pred, zero_division=0))
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred), "\n")

    @abstractmethod
    def save_model(self):
        """Save the model"""
        pass

    @staticmethod
    @abstractmethod
    def load_model(path: str):
        """Load a saved model"""

    @staticmethod
    def load_best_model():
        """Load the best model inside the local folder saved_models"""
        base_path = BaseTextClassifier.save_models_path
        cv_score = float('-inf')
        best_model_type = None
        best_subfolder = None

        try:
            snapshot_download(
                repo_id="olmervii/hatespeech-bert-uncased",
                local_dir=base_path,
            )
        except RepositoryNotFoundError:
            print("Model not found on Hugging Face Hub")
        except Exception as e:
            print(f"Error downloading model: {e}")

        # Check BERT models
        bert_path = os.path.join(base_path, "bert")
        if os.path.exists(bert_path):
            bert_subfolders = [d for d in os.listdir(bert_path) if os.path.isdir(os.path.join(bert_path, d))]

            for subfolder in bert_subfolders:
                classifier_path = os.path.join(bert_path, subfolder, "classifier_class.pkl")
                if os.path.exists(classifier_path):
                    try:
                        with open(classifier_path, "rb") as f:
                            classifier = pickle.load(f)

                        if hasattr(classifier, 'cv_score') and classifier.cv_score > cv_score:
                            cv_score = classifier.cv_score
                            best_model_type = "bert"
                            best_subfolder = subfolder
                    except:
                        continue

        # Check SKlearn models
        sklearn_path = os.path.join(base_path, "sklearn")
        if os.path.exists(sklearn_path):
            sklearn_subfolders = [d for d in os.listdir(sklearn_path) if os.path.isdir(os.path.join(sklearn_path, d))]

            for subfolder in sklearn_subfolders:
                classifier_path = os.path.join(sklearn_path, subfolder, "classifier_class.pkl")
                if os.path.exists(classifier_path):
                    try:
                        with open(classifier_path, "rb") as f:
                            classifier = pickle.load(f)

                        if hasattr(classifier, 'cv_score') and classifier.cv_score > cv_score:
                            cv_score = classifier.cv_score
                            best_model_type = "sklearn"
                            best_subfolder = subfolder
                    except:
                        continue

        # Load the best model found
        if best_model_type == "bert":
            from .BertClassifier import BertClassifier
            model_path = os.path.join(base_path, "bert", best_subfolder)
            return BertClassifier.load_model(model_path)
        elif best_model_type == "sklearn":
            from .SKLearnClassifier import SKLearnClassifier
            model_path = os.path.join(base_path, "sklearn", best_subfolder)
            return SKLearnClassifier.load_model(model_path)
        else:
            raise ValueError(f"No valid classifier model found at path: {base_path}")

    def _initialize_test_dataset(self):
        class_0 = [
            """Hitler is inevitable.""",
            """Before the world was blaming Hamas for the latest attempt of genocide, israel was not only executing 
            Palestinians but attacking funeral processions. \n\nFuck you with your accusations of antisemitism.""",
            """@peterkmurphy @QldPolice Yes we do. Hail Victory""",
            """On Senator Fatima Payman quitting the Labor party. A short 🧵\n\n1. The Labor government has invoked 
            "social cohesion", "social harmony", "avoiding division", "respect", "diversity", being part of a "team" 
            to discipline, punish and silence @SenatorPayman. It's the same language that has been weaponized against 
            students protesting at universities, vast cross-sections of the community joining weekly mass rallies 
            around the country, public servants, artists, teachers, doctors & health care workers, lawyers, 
            journalists, food & hospitality workers who have been vocal against Israel's genocide on Palestinians.""",
            """Are these all JEWS serving as "right wing influencers" and "gatekeepers"? \n\nH/T @observer4obs""",
            """Yes this actually happens. Israelis steal Palestinian homes.""",
            """@DrLoupis Is this the beginning of the end for the absurd Israeli and Jewish dominance of the west and
            their aggression with the Middle East?\n\nIs it time to deny Israel endless allowances for their genocide
            of the Palestinians?""",
            """@jewsagainstfash Jews chestpuffing after the fact, yet I didn't see you down there?\n\nPrefer to have
            your pigs, brown sludge and communist golems do the heavy lifting so you can play twitter toughguy a day
            later, shlomo?""",
            """@Pohl_MdB Sieg heil.""",
            """Where is the Australian government condemnation.  Where are the sanctions?
            #IsraelIsAnApartheidState\nIsrael's occupation of Palestinian territories is illegal, UN court rules -""",
        ]
        class_1 = [
            """@KeithWoodsYT I agree that we shouldn't try and copy paste a movement. It's not even possible but what
            about those of us who identify with these things for their economic ideas? Or rather how their economic
            policies helped to supplement their race?""",
            """We have the right to protest, despise, call for the end of an ideology and state that can only exist
            by pursuing our non-existence. We have the right to say we cannot co-exist with our genocidal oppressors.
            We have the right to demand an end to settler colonialism, apartheid and occupation without factoring in
            how our oppressors "feel". We have the right to imagine that another world is possible. We have the
            right to refuse to be held hostage to confected feelings by people who support our annihilation. (4)""",
            """The Fascist stadium""",
            """@SamParkerSenate @Lucas_Gage_ The 4 be 2's are gonna keep the war going till 2025 . A lot of people
            are gonna start noticing .""",
            """If a certain group were another color, would more people notice their concentration of power?H/T
            @tonyrigatonee 👈 Give him a follow""",
            """Friendly reminder this is not an argument and merely an assertion. Nowhere in Christianity does it
            imply that there aren't superior & inferior people on some level. Equality only exists insofar as souls
            are of equal value in the eyes of God & are likewise loved and judged equally""",
            """@PeterDutton_MP "Properly managed migration" lolSounds like certain someone who shall remain nameless
            but has Peter as a first name will still import record amounts of immigrants like his Liberal and Labor
            predecessors 🤨The people want calls for mass deportation, end of story""",
            """When I say 'Pan-Europeanism' I don't mean the dissolution of specific European identities I mean to
            cooperation of all European ethnicities inside and outside Europe.""",
            """Getting dwarfed by the Mussolini Obelisk""",
            """🎶🎵 Fuck off we're full, fuck off we're full. 🎵🎶🎶🎵 Fuck off we're full Australia for the White 
            man, fuck off we're full! 🎵🎶""",
        ]

        class_2 = [
            """'Sex work' is just the ultimate commodification of self. Truly a weird form of self exploitation.
            Nothing more disrespectful to yourself to offer yourself as nothing but a vessel for the pleasure of
            others. Those who do this do not understand they're a human with a soul.""",
            """It's really fucking simple!""",
            """I AM MOVING ON THE KING'S HIGHWAY""",
            """One year from now?""",
            """@MarkJesser @bordermail Problem?""",
            """@DrewPavlou Obvious being satirical, this conversation would not make sense unironically""",
            """📍Grampians national park @WhiteAusVic""",
            """@DrewPavlou I'm amazed that you didn't pop the tyres when you sat on the scooter""",
            """Seems like a film about Fiume is coming out""",
            """Not a finer group of lads in the country. Shrug off your inaction and join today"""
        ]

        data = {
            'antisemitic': class_0,
            'not_antisemitic': class_1,
            'irrelevant': class_2
        }

        return data
