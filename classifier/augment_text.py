from collections import Counter

import nlpaug.augmenter.word as naw

from classifier.BERTClassifier import BERTClassifier


def augment_minority_class(texts: list[str], labels: list[str], n_aug: int = None) -> tuple[list[str], list[str]]:
    # Find minority class
    label_counts = Counter(labels)
    minority_class = min(label_counts, key=label_counts.get)
    majority_count = max(label_counts.values())
    minority_count = label_counts[minority_class]

    # Calculate augmentations needed for full balance
    samples_needed = majority_count - minority_count
    if samples_needed <= 0:
        return texts, labels  # Already balanced

    # If n_aug not specified, calculate to achieve balance
    minority_indices = [i for i, label in enumerate(labels) if label == minority_class]
    if n_aug is None:
        n_aug = (samples_needed + len(minority_indices) - 1) // len(minority_indices)  # Ceiling division

    n_aug = float('inf')

    # Initialize augmenters
    primary_aug = naw.SynonymAug(aug_src='wordnet', aug_min=1, aug_max=2)
    fallback_aug = naw.RandomWordAug(action="insert", aug_min=1, aug_max=2)

    augmented_texts = texts.copy()
    augmented_labels = labels.copy()

    # Augment minority samples
    augmented_count = 0

    for idx in minority_indices:
        if augmented_count >= samples_needed:
            break

        text = texts[idx]
        current_augmentations = 0

        # Generate up to n_aug augmentations per sample
        while current_augmentations < n_aug and augmented_count < samples_needed:
            # Try primary augmentation
            aug_text = primary_aug.augment(text)
            if aug_text == text:  # Fallback if no change
                aug_text = fallback_aug.augment(text)

            if aug_text != text and aug_text not in augmented_texts:  # Only add if changed and unique
                augmented_texts.append(aug_text)
                augmented_labels.append(minority_class)
                augmented_count += 1
                current_augmentations += 1
            else:
                # If we can't generate more variations, move to next sample
                break

    print(f"Original minority class '{minority_class}': {minority_count} samples")
    print(f"Added {augmented_count} augmented samples")
    print(f"New minority class count: {minority_count + augmented_count}")
    print(f"Majority class count: {majority_count}")

    return augmented_texts, augmented_labels


if __name__ == "__main__":

    loaded_classifier = BERTClassifier.load_model("distilbert uncased")
    data = loaded_classifier.load_data()
    X_train, X_test, y_train, y_test = loaded_classifier.prepare_dataset(data)

    X_train = [loaded_classifier.normalizer.normalize(x) for x in X_train]

    augmented_texts, augmented_labels = augment_minority_class(texts=X_train,labels=y_train)
