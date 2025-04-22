from abc import ABC, abstractmethod
import sys
import os
import random
import copy
import pandas as pd


from classifier.normalization.TextNormalizer import TextNormalizer
from tagging_website.serverside.db_service import get_db_instance

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BaseTextClassifier(ABC):
    """Abstract base class for text classifiers"""

    def __init__(self, model: any, preprocessor: TextNormalizer() = None, seed: int = 42):
        self._model = model
        self._preprocessor = preprocessor

        self.seed = seed
        self.random_generator = random.Random(self.seed)  # use this instead of random.random() to keep results consistent
        self.LABELS = ["antisemistic", "not_antisemistic"]

    def load_data(self, class_0_count, class_1_count, class_2_count=None, source=None) -> dict[str, list]:
        """Load data from file or use sample data.

        This function loads text data for classification either from the database, csv files, or mock data.
        The data is organized by class labels.

        Args:
            class_0_count: Number of samples to load for class 0 (antisemistic)
            class_1_count: Number of samples to load for class 1 (not_antisemistic)
            class_2_count: Number of samples to load for class 2 (irrelevant), optional
            source: 'debug' works with generated data, 'csv_files' with local data, else data from the database

        Returns:
            Dictionary mapping class labels to lists of text samples.
        """
        # Initialize data dictionary
        data = {}

        if source == 'csv_files':
            # Load data from CSV files
            data[self.LABELS[0]] = pd.read_csv('positive_results.csv', header=None, nrows=class_0_count)[0].tolist()
            data[self.LABELS[1]] = pd.read_csv('negative_results.csv', header=None, nrows=class_1_count)[0].tolist()

            if class_2_count is not None:
                self.LABELS.append("irrelevant")

                data[self.LABELS[2]] = pd.read_csv('irrelevant_results.csv', header=None, nrows=class_2_count)[
                    0].tolist()

        elif source == 'debug':
            if class_2_count is not None:
                self.LABELS.append("irrelevant")

            data = self._initialize_test_dataset(class_2_count)


        else:
            # Get data from database
            db = get_db_instance()

            data[self.LABELS[0]] = db.get_result_posts(label=self.LABELS[0], count=class_0_count)
            data[self.LABELS[1]] = db.get_result_posts(label=self.LABELS[1], count=class_1_count)

            if class_2_count is not None:
                self.LABELS.append("irrelevant")

                data[self.LABELS[2]] = db.get_result_posts(label=self.LABELS[2], count=class_2_count)

        return data


    def prepare_datasets(self, data: dict[str, list], test_size: float = 0.15, validation_size: float = 0.1, combine_irrelevant: bool = False) -> any:
        """Prepare train, validation and test datasets.

        This function splits the input data into training, validation, and test sets
        while maintaining the class distribution in each set. The data is shuffled to ensure randomness.

        Args:
            data: Dictionary mapping class labels to lists of posts
            test_size: Proportion of data to use for testing (default: 0.2)
            validation_size: Proportion of data to use for validation (default: 0.1)
            combine_irrelevant: If True, combines irrelevant class with not_antisemistic class (default: False)

        Returns:
            Dictionary containing 'train', 'validation', and 'test' keys, each mapping to
            a list of (post, label) tuples
        """

        if test_size + validation_size > 1:
            print("training, test, and validation sizes must sum up to 1")
            return

        # Handle combining irrelevant with not-antisemistic if specified
        if combine_irrelevant and len(self.LABELS) > 2:
            data[self.LABELS[1]] = data[self.LABELS[1]] + data.pop(self.LABELS[2])
            self.LABELS.pop(2)

        # Find the size of the smallest class to determine validation set size. Validation size should be consistent between classes regardless of their size.
        min_class_size = min(len(posts) for posts in data.values())
        validation_count_per_class = int(min_class_size * validation_size)

        train_data = {'posts': [], 'labels': []}
        validation_data = {'posts': [], 'labels': []}
        test_data = {'posts': [], 'labels': []}

        # Process each class separately to maintain class distribution
        for label, posts in data.items():
            validation_posts = posts[:validation_count_per_class]   # Take equal validation samples from each class

            remaining_posts = posts[validation_count_per_class:]
            test_count = int(len(remaining_posts) * test_size / (1 - validation_size))

            test_posts = remaining_posts[:test_count]
            train_posts = remaining_posts[test_count:]

            train_data['posts'].extend(train_posts)
            train_data['labels'].extend([label] * len(train_posts))

            validation_data['posts'].extend(validation_posts)
            validation_data['labels'].extend([label] * len(validation_posts))

            test_data['posts'].extend(test_posts)
            test_data['labels'].extend([label] * len(test_posts))

        train_combined = list(zip(train_data['posts'], train_data['labels']))
        validation_combined = list(zip(validation_data['posts'], validation_data['labels']))
        test_combined = list(zip(test_data['posts'], test_data['labels']))

        return {
            'train': train_combined,
            'validation': validation_combined,
            'test': test_combined
        }

    @abstractmethod
    def preprocess_data(self, datasets: any, custom_lemmas: list[str] = None) -> any:
        """Apply preprocessing to datasets.

        Can be called using super() as preliminary step before additional preprocessing.
        """
        datasets = copy.copy(datasets)
        preprocessor = self.get_text_normalizer()

        if preprocessor:
            for label, posts in datasets.items():
                processed_posts = []
                for post in posts:
                    processed_post = preprocessor.process(post)
                    processed_posts.append(processed_post)

                datasets[label] = processed_posts

        return datasets

    def add_lemmas(self, custom_lemmas: dict):
        """
        Add custom lemmatization rules

        Args:
            custom_lemmas: dict mapping words to their desired lemma forms
        """
        pass

    @abstractmethod
    def add_tokens(self, special_tokens):
        """Hook method for subclasses to handle special tokens"""
        pass

    @abstractmethod
    def train(self, processed_datasets: dict[str, list[tuple[str, str]]], learning_rate: float = 0.001,
              l2_regularization: float = 0.001, epochs: int = 100, batch_size: int = 32, dropout: float = 0.2) -> None:
        """Train the model"""
        pass

    @abstractmethod
    def evaluate(self, test_dataset: any) -> dict[str, float]:
        """Evaluate the model"""
        pass

    def _record_metrics(self, epoch, losses, train_results, eval_results, epoch_time):
        """Record metrics for an epoch"""
        return {
            "epoch": epoch,
            "train_loss": losses["textcat"],
            "train_accuracy": train_results["cats_score"],
            "val_accuracy": eval_results["cats_score"],
            "accuracy_gap": train_results["cats_score"] - eval_results["cats_score"],
            "precision": eval_results["cats_micro_p"],
            "recall": eval_results["cats_micro_r"],
            "f1": eval_results["cats_micro_f"],
            "time": epoch_time
        }

    def _log_progress(self, epoch, total_epochs, epoch_time, losses, train_results, eval_results):
        """Log training progress"""
        print(f"Epoch {epoch}/{total_epochs}, Time: {epoch_time:.2f}s, "
              f"Train Loss: {losses['textcat']:.4f}, "
              f"Train Accuracy: {train_results['cats_score']:.4f}, "
              f"Val Accuracy: {eval_results['cats_score']:.4f}, "
              f"F1 Score: {eval_results['cats_micro_f']:.4f}, "
              f"Gap: {(train_results['cats_score'] - eval_results['cats_score']):.4f}")

    def _compile_evaluation_metrics(self, results):
        """Compile evaluation metrics into a dictionary"""
        metrics = {
            "accuracy": results["cats_score"],
            "precision": results["cats_micro_p"],
            "recall": results["cats_micro_r"],
            "f1": results["cats_micro_f"],
        }

        # Add per-category scores
        for label in self.LABELS:
            if f"cats_{label}_p" in results:
                metrics[f"{label}_precision"] = results[f"cats_{label}_p"]
                metrics[f"{label}_recall"] = results[f"cats_{label}_r"]
                metrics[f"{label}_f1"] = results[f"cats_{label}_f"]

        # Add training history if available
        if hasattr(self, 'training_history'):
            metrics["training_history"] = self.training_history

        if hasattr(self, 'training_time'):
            metrics["total_training_time"] = self.training_time

        # Add learning curves data for plotting
        if hasattr(self, 'training_history'):
            epochs = [entry["epoch"] for entry in self.training_history]
            train_losses = [entry["train_loss"] for entry in self.training_history]
            train_accuracies = [entry.get("train_accuracy", 0) for entry in self.training_history]
            val_accuracies = [entry.get("val_accuracy", 0) for entry in self.training_history]
            accuracy_gaps = [entry.get("accuracy_gap", 0) for entry in self.training_history]

            metrics["learning_curves"] = {
                "epochs": epochs,
                "train_losses": train_losses,
                "train_accuracies": train_accuracies,
                "val_accuracies": val_accuracies,
                "accuracy_gaps": accuracy_gaps
            }

        return metrics

    def predict(self, text: str) -> dict[str, float]:
        """Make prediction on a single text"""
        pass

    def save_model(self, path: str) -> None:
        """Save the model"""
        pass

    def load_model(self, path: str) -> None:
        """Load a saved model"""
        pass

    def get_text_normalizer(self):
        """Set text preprocessor"""
        return self._preprocessor

    def set_text_normalizer(self, preprocessor):
        """Set text preprocessor"""
        self._preprocessor = preprocessor

    def get_model(self):
        """Get text preprocessor"""
        return self._model

    def set_model(self, model):
        """Set text preprocessor"""
        self._model = model

    def _initialize_test_dataset(self, class_2_exists):
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
            self.LABELS[1]: class_1
        }

        if class_2_exists is not None:
            data[self.LABELS[2]] = class_2

        return data
