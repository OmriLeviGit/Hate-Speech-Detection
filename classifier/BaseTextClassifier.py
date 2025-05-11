import os
import pickle
from abc import ABC, abstractmethod
import nlpaug.augmenter.word as naw
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from classifier.utils import format_duration


class BaseTextClassifier(ABC):
    """Abstract base class for text classifiers"""

    save_models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models") # static attribute

    def __init__(self, labels: list = None, seed = None):
        self.LABELS = labels
        self.seed = seed
        self.label_encoder = LabelEncoder()

        self.best_model = None
        self.model_name = None

    def load_data(self, irrelevant_ratio=0.33, debug=False) -> dict[str, list]:
        """
        Load data based on self.LABELS configuration

        Args:
            irrelevant_ratio: Ratio of irrelevant samples to mix into not_antisemitic for 2-label case
            debug: Use test dataset if True
        """
        if debug:
            print("loading with 'debug' dataset")
            return self._initialize_test_dataset()

        # Load data from CSV
        script_dir = os.path.dirname(os.path.abspath(__file__))
        df = pd.read_csv(os.path.join(script_dir, 'results.csv'))

        # Extract raw data
        antisemitic_data = df[df['sentiment'] == 'Positive']['content'].tolist()
        not_antisemitic_data = df[df['sentiment'] == 'Negative']['content'].tolist()
        irrelevant_data = df[df['sentiment'] == 'Irrelevant']['content'].tolist()

        data = {}

        if len(self.LABELS) == 2:
            # Binary classification: merge irrelevant into not_antisemitic
            # Calculate how many irrelevant samples to include for desired ratio
            num_irrelevant_to_add = int(len(not_antisemitic_data) * irrelevant_ratio / (1 - irrelevant_ratio))
            num_irrelevant_to_add = min(num_irrelevant_to_add, len(irrelevant_data))

            # Combine not_antisemitic with selected irrelevant samples
            combined_not_antisemitic = not_antisemitic_data + irrelevant_data[:num_irrelevant_to_add]
            np.random.shuffle(combined_not_antisemitic)

            data['antisemitic'] = antisemitic_data
            data['not_antisemitic'] = combined_not_antisemitic

        elif len(self.LABELS) == 3:
            # Three-way classification
            data['antisemitic'] = antisemitic_data
            data['not_antisemitic'] = not_antisemitic_data
            data['irrelevant'] = irrelevant_data

        return data

    def prepare_dataset(self, datasets: dict[str, list[str]], test_size=0.2, augment_ratio=0, balance_classes=True) \
            -> tuple[list[str], list[str], list[str], list[str]]:
        """
        Prepare and split into train and test sets

        Args:
            datasets: Dictionary of label -> list of texts
            test_size: Proportion of data for testing
            augment_antisemitic: Whether to augment the antisemitic class
            augment_ratio: How much to augment (1.0 = double the size)
            balance_classes: Whether to balance classes in training set
        """
        posts = []
        labels = []

        # Combine all data
        for label_name, post_list in datasets.items():
            for post in post_list:
                posts.append(post)
                labels.append(label_name)

        X = np.array(posts)
        y = np.array(labels)

        # Shuffle and split
        X_shuffled, y_shuffled = shuffle(X, y, random_state=self.seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X_shuffled, y_shuffled,
            test_size=test_size,
            stratify=y_shuffled,
            random_state=self.seed
        )

        # Convert to lists
        X_train_list = X_train.tolist()
        y_train_list = y_train.tolist()

        # Augment antisemitic samples if requested
        if augment_ratio > 0 and 'antisemitic' in datasets:
            augmented_texts, augmented_labels = self.augment_specific_class(
                X_train_list, y_train_list,
                target_class='antisemitic',
                augment_ratio=augment_ratio
            )
            X_train_list.extend(augmented_texts)
            y_train_list.extend(augmented_labels)

        # Balance classes if requested (for 2-label case)
        if balance_classes and len(self.LABELS) == 2:
            X_train_list, y_train_list = self.balance_training_set(X_train_list, y_train_list)

        return X_train_list, X_test.tolist(), y_train_list, y_test.tolist()

    def augment_specific_class(self, texts: list[str], labels: list[str],
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

        # Calculate augmentations per sample
        n_aug = (samples_needed + len(target_indices) - 1) // len(target_indices)

        primary_aug = naw.SynonymAug(aug_src='wordnet', aug_min=1, aug_max=2)
        fallback_aug = naw.RandomWordAug(action="insert", aug_min=1, aug_max=2)

        augmented_texts = []
        augmented_labels = []

        # Generate augmentations
        for idx in target_indices:
            if len(augmented_texts) >= samples_needed:
                break

            text = texts[idx]
            current_augmentations = 0

            while current_augmentations < n_aug and len(augmented_texts) < samples_needed:
                # Try primary augmentation
                aug_text_list = primary_aug.augment(text)
                if isinstance(aug_text_list, list) and len(aug_text_list) > 0:
                    aug_text = aug_text_list[0]  # Take the first augmentation
                else:
                    aug_text = aug_text_list  # Fallback if not a list

                if aug_text == text:  # Fallback if no change
                    fallback_text_list = fallback_aug.augment(text)
                    if isinstance(fallback_text_list, list) and len(fallback_text_list) > 0:
                        aug_text = fallback_text_list[0]
                    else:
                        aug_text = fallback_text_list

                if aug_text != text and aug_text not in augmented_texts:
                    augmented_texts.append(aug_text)
                    augmented_labels.append(target_class)
                    current_augmentations += 1
                else:
                    break

        print(f"Original '{target_class}': {original_count} samples")
        print(f"Augment ratio: {augment_ratio} (target: {samples_needed} new samples)")
        print(f"Generated {len(augmented_texts)} augmented samples")

        return augmented_texts, augmented_labels

    def balance_training_set(self, texts: list[str], labels: list[str]) -> tuple[list[str], list[str]]:
        """
        Balance the training set so non_antisemitic matches the size of augmented antisemitic

        Args:
            texts: List of text samples
            labels: List of corresponding labels
        """
        # Count samples per class
        label_counts = Counter(labels)
        antisemitic_count = label_counts.get('antisemitic', 0)
        not_antisemitic_count = label_counts.get('not_antisemitic', 0)

        # If already balanced or no antisemitic samples, return as is
        if antisemitic_count == 0 or antisemitic_count == not_antisemitic_count:
            return texts, labels

        # If non_antisemitic has more samples, downsample to match antisemitic
        if not_antisemitic_count > antisemitic_count:
            # Get indices for each class
            antisemitic_indices = [i for i, label in enumerate(labels) if label == 'antisemitic']
            not_antisemitic_indices = [i for i, label in enumerate(labels) if label == 'not_antisemitic']

            # Randomly sample from not_antisemitic to match antisemitic count
            sampled_not_antisemitic_indices = np.random.choice(
                not_antisemitic_indices,
                size=antisemitic_count,
                replace=False
            )

            # Combine indices
            final_indices = antisemitic_indices + sampled_not_antisemitic_indices.tolist()

            # Create balanced dataset
            balanced_texts = [texts[i] for i in final_indices]
            balanced_labels = [labels[i] for i in final_indices]

            print(f"Balanced training set: {antisemitic_count} antisemitic, {antisemitic_count} not_antisemitic")

            return balanced_texts, balanced_labels

        # If antisemitic has more samples, we don't downsample
        return texts, labels

    @abstractmethod
    def preprocess(self, datasets: list[str]) -> list[str]:
        """Apply preprocessing to datasets."""
        pass

    @abstractmethod
    def train(self, *args) -> None:
        """Train the model"""
        pass

    def evaluate(self, X_test: list[str], y_test: list[str]) -> tuple[float, float]:
        if not self.best_model:
            raise ValueError("Model not trained yet")

        # Preprocess test data
        X_processed = self.preprocess(X_test)
        y_encoded = self.label_encoder.transform(y_test)

        y_pred = self.predict(X_processed)

        # Calculate metrics
        accuracy = accuracy_score(y_encoded, y_pred)
        f1 = f1_score(y_encoded, y_pred, average='weighted', zero_division=0)

        self.print_evaluation(y_encoded, y_pred, accuracy, f1)

        return accuracy, f1

    @abstractmethod
    def predict(self, text):
        """Make prediction on a single text"""
        pass

    def print_best_model_results(self, best_score, best_param, training_duration=None):
        print(f"\n=== Training result - Model: {self.model_name} ===")
        print("Best Cross-validated Score:", round(best_score, 2))
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
        """Load the best model from all subfolders under BERT and SKlearn directories"""
        base_path = BaseTextClassifier.save_models_path
        best_score = float('-inf')
        best_model_type = None
        best_subfolder = None

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

                        if hasattr(classifier, 'best_score') and classifier.best_score > best_score:
                            best_score = classifier.best_score
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

                        if hasattr(classifier, 'best_score') and classifier.best_score > best_score:
                            best_score = classifier.best_score
                            best_model_type = "sklearn"
                            best_subfolder = subfolder
                    except:
                        continue

        # Load the best model found
        if best_model_type == "bert":
            from .BERTClassifier import BERTClassifier
            model_path = os.path.join(base_path, "bert", best_subfolder)
            return BERTClassifier.load_model(model_path)
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
            self.LABELS[0]: class_0,
            self.LABELS[1]: class_1,
            self.LABELS[1]: class_2
        }

        return data
