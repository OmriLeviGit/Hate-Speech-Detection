from collections import Counter

import nlpaug.augmenter.word as naw

from classifier.BERTClassifier import BERTClassifier


def augment_minority(texts: list[str], labels: list[str], augment_ratio: float = 1.0, n_aug: int = None) -> tuple[
    list[str], list[str]]:
    # Find minority class
    label_counts = Counter(labels)
    minority_class = min(label_counts, key=label_counts.get)
    minority_count = label_counts[minority_class]

    samples_needed = int(minority_count * augment_ratio)

    if samples_needed <= 0:
        return [], []  # No augmentation needed

    # If n_aug not specified, calculate to achieve target augmentation
    minority_indices = [i for i, label in enumerate(labels) if label == minority_class]
    if n_aug is None:
        n_aug = (samples_needed + len(minority_indices) - 1) // len(minority_indices)  # Ceiling division

    primary_aug = naw.SynonymAug(aug_src='wordnet', aug_min=1, aug_max=2)
    fallback_aug = naw.RandomWordAug(action="insert", aug_min=1, aug_max=2)

    augmented_texts = []
    augmented_labels = []

    # Augment minority samples
    for idx in minority_indices:
        if len(augmented_texts) >= samples_needed:
            break

        text = texts[idx]
        current_augmentations = 0

        # Generate up to n_aug augmentations per sample
        while current_augmentations < n_aug and len(augmented_texts) < samples_needed:
            # Try primary augmentation
            aug_text = primary_aug.augment(text)
            if aug_text == text:  # Fallback if no change
                aug_text = fallback_aug.augment(text)

            if aug_text != text and aug_text not in augmented_texts:  # Only add if changed and unique
                augmented_texts.append(aug_text)
                augmented_labels.append(minority_class)
                current_augmentations += 1
            else:
                # If we can't generate more variations, move to next sample
                break

    print(f"Original minority class '{minority_class}': {minority_count} samples")
    print(f"Augment ratio: {augment_ratio} (target: {samples_needed} new samples)")
    print(f"Generated {len(augmented_texts)} augmented samples")

    return augmented_texts, augmented_labels


if __name__ == "__main__":

    loaded_classifier = BERTClassifier.load_model("distilbert uncased")
    data = loaded_classifier.load_data()
    X_train, X_test, y_train, y_test = loaded_classifier.prepare_dataset(data)

    X_train = [loaded_classifier.normalizer.normalize(x) for x in X_train]

    augmented_texts, augmented_labels = augment_minority(texts=X_train,labels=y_train)
