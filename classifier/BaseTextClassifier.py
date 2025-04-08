from abc import ABC, abstractmethod
import sys
import os
import random
import copy

from classifier.preprocessing.TextPreprocessor import TextPreprocessor
from tagging_website.serverside.db_service import get_db_instance

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BaseTextClassifier(ABC):
    """Abstract base class for text classifiers"""

    def __init__(self, model: any = None, preprocessor: TextPreprocessor() = None, seed: int = 42):
        self._model = model
        self._preprocessor = preprocessor
        self._random_generator = random.Random(seed)  # use this instead of random.random() to keep results consistent

        self._CLASS_0 = "antisemistic"
        self._CLASS_1 = "not_antisemistic"
        self._CLASS_2 = "irrelevant"


    def load_data(self, class_0_count, class_1_count, class_2_count=None, debug=False) -> dict[str, list]:
        """Load data from file or use sample data.

        This function loads text data for classification either from a database (when debug=False)
        or generates mock data (when debug=True). The data is organized by class labels.

        Args:
            class_0_count: Number of samples to load for class 0 (antisemistic)
            class_1_count: Number of samples to load for class 1 (not_antisemistic)
            class_2_count: Number of samples to load for class 2 (irrelevant), optional
            debug: True works with local data, False with data from the database

        Returns:
            Dictionary mapping class labels to lists of text samples.
            In debug mode, the lists contain placeholder values (zeros).
        """

        if debug:
            return self._initialize_test_dataset(class_2_count)

        db = get_db_instance()

        data = {
            self._CLASS_0: db.get_result_posts(label=self._CLASS_0, count=class_0_count),
            self._CLASS_1: db.get_result_posts(label=self._CLASS_1, count=class_1_count)
        }

        if class_2_count is not None:
            data[self._CLASS_2] = db.get_result_posts(label=self._CLASS_2, count=class_2_count)

        return data

    def prepare_datasets(self, data: dict[str, list], test_size: float = 0.2, validation_size: float = 0.1, combine_irrelevant: bool = False) -> any:
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
        if combine_irrelevant and self._CLASS_1 in data and self._CLASS_2 in data:
            data[self._CLASS_1] = data[self._CLASS_1] + data.pop(self._CLASS_2)

        train_data = {'posts': [], 'labels': []}
        validation_data = {'posts': [], 'labels': []}
        test_data = {'posts': [], 'labels': []}

        # Process each class separately to maintain class distribution
        for label, posts in data.items():
            total_posts = len(posts)
            test_count = int(total_posts * test_size)
            validation_count = int(total_posts * validation_size)
            train_count = total_posts - test_count - validation_count

            train_posts = posts[:train_count]
            validation_posts = posts[train_count:train_count + validation_count]
            test_posts = posts[train_count + validation_count:]

            train_data['posts'].extend(train_posts)
            train_data['labels'].extend([label] * len(train_posts))

            validation_data['posts'].extend(validation_posts)
            validation_data['labels'].extend([label] * len(validation_posts))

            test_data['posts'].extend(test_posts)
            test_data['labels'].extend([label] * len(test_posts))

        train_combined = list(zip(train_data['posts'], train_data['labels']))
        validation_combined = list(zip(validation_data['posts'], validation_data['labels']))
        test_combined = list(zip(test_data['posts'], test_data['labels']))

        self._random_generator.shuffle(train_combined)
        self._random_generator.shuffle(validation_combined)
        self._random_generator.shuffle(test_combined)

        return {
            'train': train_combined,
            'validation': validation_combined,
            'test': test_combined
        }

    @abstractmethod
    def preprocess_data(self, datasets: any) -> any:
        """Apply preprocessing to datasets.

        Can be called using super() as preliminary step before additional preprocessing.
        """
        datasets = copy.copy(datasets)
        preprocessor = self.get_text_preprocessor()

        if preprocessor:
            for dataset_name, data in datasets.items():
                processed_data = []
                for post, label in data:
                    processed_post = preprocessor.process(post)
                    processed_data.append((processed_post, label))

                datasets[dataset_name] = processed_data

        return datasets

    @abstractmethod
    def _handle_special_tokens(self, special_tokens):
        """Hook method for subclasses to handle special tokens"""
        pass

    @abstractmethod
    def train(self, processed_datasets: any, **kwargs) -> None:
        """Train the model"""
        pass

    @abstractmethod
    def evaluate(self, test_dataset: any) -> dict[str, float]:
        """Evaluate the model"""
        pass

    @abstractmethod
    def predict(self, text: str) -> dict[str, float]:
        """Make prediction on a single text"""
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save the model"""
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load a saved model"""
        pass

    def get_text_preprocessor(self):
        """Set text preprocessor"""
        return self._preprocessor

    def set_text_preprocessor(self, preprocessor):
        """Set text preprocessor"""
        self._preprocessor = preprocessor

    def get_model(self):
        """Set text preprocessor"""
        return self._model


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
            # """Are these all JEWS serving as "right wing influencers" and "gatekeepers"? \n\nH/T @observer4obs""",
            # """Yes this actually happens. Israelis steal Palestinian homes.""",
            # """@DrLoupis Is this the beginning of the end for the absurd Israeli and Jewish dominance of the west and
            # their aggression with the Middle East?\n\nIs it time to deny Israel endless allowances for their genocide
            # of the Palestinians?""",
            # """@jewsagainstfash Jews chestpuffing after the fact, yet I didn't see you down there?\n\nPrefer to have
            # your pigs, brown sludge and communist golems do the heavy lifting so you can play twitter toughguy a day
            # later, shlomo?""",
            # """@Pohl_MdB Sieg heil.""",
            # """Where is the Australian government condemnation.  Where are the sanctions?
            # #IsraelIsAnApartheidState\nIsrael's occupation of Palestinian territories is illegal, UN court rules -""",
        ]
        class_1 = [
            # """@KeithWoodsYT I agree that we shouldn't try and copy paste a movement. It's not even possible but what
            # about those of us who identify with these things for their economic ideas? Or rather how their economic
            # policies helped to supplement their race?""",
            # """We have the right to protest, despise, call for the end of an ideology and state that can only exist
            # by pursuing our non-existence. We have the right to say we cannot co-exist with our genocidal oppressors.
            # We have the right to demand an end to settler colonialism, apartheid and occupation without factoring in
            # how our oppressors "feel". We have the right to imagine that another world is possible. We have the
            # right to refuse to be held hostage to confected feelings by people who support our annihilation. (4)""",
            # """The Fascist stadium""",
            # """@SamParkerSenate @Lucas_Gage_ The 4 be 2's are gonna keep the war going till 2025 . A lot of people
            # are gonna start noticing .""",
            # """If a certain group were another color, would more people notice their concentration of power?H/T
            # @tonyrigatonee 👈 Give him a follow""",
            # """Friendly reminder this is not an argument and merely an assertion. Nowhere in Christianity does it
            # imply that there aren't superior & inferior people on some level. Equality only exists insofar as souls
            # are of equal value in the eyes of God & are likewise loved and judged equally""",
            # """@PeterDutton_MP "Properly managed migration" lolSounds like certain someone who shall remain nameless
            # but has Peter as a first name will still import record amounts of immigrants like his Liberal and Labor
            # predecessors 🤨The people want calls for mass deportation, end of story""",
            # """When I say 'Pan-Europeanism' I don't mean the dissolution of specific European identities I mean to
            # cooperation of all European ethnicities inside and outside Europe.""",
            # """Getting dwarfed by the Mussolini Obelisk""",
            """🎶🎵 Fuck off we're full, fuck off we're full. 🎵🎶🎶🎵 Fuck off we're full Australia for the White 
            man, fuck off we're full! 🎵🎶""",
            ]

        class_2 = [
            # """'Sex work' is just the ultimate commodification of self. Truly a weird form of self exploitation.
            # Nothing more disrespectful to yourself to offer yourself as nothing but a vessel for the pleasure of
            # others. Those who do this do not understand they're a human with a soul.""",
            # """It's really fucking simple!""",
            # """I AM MOVING ON THE KING'S HIGHWAY""",
            # """One year from now?""",
            # """@MarkJesser @bordermail Problem?""",
            """@DrewPavlou Obvious being satirical, this conversation would not make sense unironically""",
            """📍Grampians national park @WhiteAusVic""",
            # """@DrewPavlou I'm amazed that you didn't pop the tyres when you sat on the scooter""",
            # """Seems like a film about Fiume is coming out""",
            # """Not a finer group of lads in the country. Shrug off your inaction and join today"""
        ]

        data = {
            self._CLASS_0: class_0,
            self._CLASS_1: class_1
        }

        if class_2_exists is not None:
            data[self._CLASS_2] = class_2

        return data
