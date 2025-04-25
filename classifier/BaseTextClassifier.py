import os
from abc import ABC, abstractmethod
import copy

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from classifier.normalization.TextNormalizer import TextNormalizer


class BaseTextClassifier(ABC):
    """Abstract base class for text classifiers"""

    def __init__(self, nlp_pipeline: any, text_normalizer: TextNormalizer(), labels: list, seed: int = 42):
        self._nlp = nlp_pipeline
        self._normalizer = text_normalizer
        self.LABELS = labels
        self.seed = seed

    def load_data(self, class_0_count=None, class_1_count=None, class_2_count=None, source=None,
                  set_to_min=False) -> dict[str, list]:
        """Load data from file or use sample data.

        This function loads text data for classification either from a csv file, or mock data.
        The data is organized by class labels.

        Args:
            class_0_count: Number of samples to load for class 0 (antisemistic)
            class_1_count: Number of samples to load for class 1 (not_antisemistic)
            class_2_count: Number of samples to load for class 2 (irrelevant), optional
            source: 'debug' to work with generated data, else to import data from a csv file
            set_to_min: If True, sets all class counts to the minimum available across classes for balanced dataset

        Returns:
            Dictionary mapping class labels to lists of text samples.
        """
        data = {}

        if source == 'debug':
            print("loading with debug information")
            return self._initialize_test_dataset()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        df = pd.read_csv(os.path.join(script_dir, 'results.csv'))

        sentiment_mapping = {
            'Positive': "antisemitic",
            'Negative': "not_antisemitic",
            'Irrelevant': "irrelevant"
        }

        class_0_data = df[df['sentiment'] == 'Positive']['content'].tolist() if sentiment_mapping[
                                                                                    'Positive'] in self.LABELS else []
        class_1_data = df[df['sentiment'] == 'Negative']['content'].tolist() if sentiment_mapping[
                                                                                    'Negative'] in self.LABELS else []
        class_2_data = df[df['sentiment'] == 'Irrelevant']['content'].tolist() if sentiment_mapping[
                                                                                      'Irrelevant'] in self.LABELS else []

        if set_to_min:
            available_counts = []
            if class_0_data:
                available_counts.append(len(class_0_data))
            if class_1_data:
                available_counts.append(len(class_1_data))
            if class_2_data:
                available_counts.append(len(class_2_data))

            min_count = min(available_counts) if available_counts else 0

            class_0_count = min_count if class_0_data else None
            class_1_count = min_count if class_1_data else None
            class_2_count = min_count if class_2_data else None

        if sentiment_mapping['Positive'] in self.LABELS:
            count = class_0_count if class_0_count is not None else len(class_0_data)
            data[sentiment_mapping['Positive']] = class_0_data[:count]

        if sentiment_mapping['Negative'] in self.LABELS:
            count = class_1_count if class_1_count is not None else len(class_1_data)
            data[sentiment_mapping['Negative']] = class_1_data[:count]

        if sentiment_mapping['Irrelevant'] in self.LABELS:
            count = class_2_count if class_2_count is not None else len(class_2_data)
            data[sentiment_mapping['Irrelevant']] = class_2_data[:count]

        return data

    @abstractmethod
    def preprocess_datasets(self, datasets: any) -> any:
        """Apply preprocessing to datasets."""
        pass

    def normalize(self, datasets: any):
        datasets = copy.copy(datasets)
        normalizer = self.get_text_normalizer()

        if normalizer:
            for label, posts in datasets.items():
                processed_posts = []
                for post in posts:
                    processed_post = normalizer.normalize(post)
                    processed_posts.append(processed_post)

                datasets[label] = processed_posts

        return datasets

    def prepare_dataset(self, datasets: dict[str, list[str]]) -> tuple[np.ndarray, np.ndarray]:
        """Prepare and split into train and test sets"""
        posts = []
        labels = []

        # label_list = list(datasets.keys())
        for label_name, post_list in datasets.items():
            # label_index = label_list.index(label_name)

            for post in post_list:
                posts.append(post)
                labels.append(label_name)

        X = np.array(posts)
        y = np.array(labels)

        return shuffle(X, y, random_state=self.seed)

    @abstractmethod
    def train(self, processed_datasets: dict[str, list[tuple[str, str]]]) -> None:
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

    def save_model(self, path: str) -> None:
        """Save the model"""
        pass

    def load_model(self, path: str) -> None:
        """Load a saved model"""
        pass

    def get_text_normalizer(self):
        """Set text preprocessor"""
        return self._normalizer

    def set_text_normalizer(self, preprocessor):
        """Set text preprocessor"""
        self._normalizer = preprocessor

    def get_nlp(self):
        """Get text preprocessor"""
        return self._nlp

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
