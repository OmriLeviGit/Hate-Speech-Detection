import json
import re
import os

from emoji import demojize
import codecs
import html
import ftfy
import unicodedata
import wordninja


from .ObfuscationMapGenerator import ObfuscationMapGenerator


class TextNormalizer:
    def __init__(self, config_path='config.json', emoji: str = None):
        self._config_path = os.path.join(os.path.dirname(__file__), config_path)
        self._load_config()

        self._obfuscation_map_generator = ObfuscationMapGenerator(obfuscation_config=self._config.get('obfuscation', {}))
        self._obfuscation_map = {}

        if self._obfuscation_map_generator:
            self._obfuscation_map = self._obfuscation_map_generator.get_map()

        self._slang_map = self._config.get('slang', {})
        self._emoji_meaning_map = self._config.get('emoji', {})

        self._processing_pipeline = []
        self._setup_pipeline(emoji)
        self._special_tokens = set()

    def _load_config(self):
        """Load configuration from file"""
        try:
            with open(self._config_path, 'r') as f:

                self._config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading config: {e}")
            self._config = {}

    def _setup_pipeline(self, emoji):
        """Set up the processing pipeline with functions in the correct order"""
        self._processing_pipeline = [
            self._replace_url,
            self._replace_mentions,
            self._replace_hashtags,
            self._deobfuscate,
        ]

        if emoji == "text":
            self._processing_pipeline.append(self._emoji_to_text)

    def normalize_texts(self, *texts):
        # Determine if we got a single list or multiple lists
        if len(texts) == 1 and isinstance(texts[0], list) and all(isinstance(item, str) for item in texts[0]):
            # Process the single list
            return [self.normalize(text) for text in texts[0]]
        elif all(isinstance(arg, list) and all(isinstance(item, str) for item in arg) for arg in texts):
            # Process multiple lists and return as tuple for unpacking
            return tuple([self.normalize(text) for text in arg] for arg in texts)
        else:
            raise TypeError("Arguments must be either a single list of strings or multiple lists of strings")

    def normalize(self, text):
        """Process text by applying all functions to each word before moving to next word"""
        text = self._fix_corruptions(text)
        words = text.lower().split()
        processed_words = []

        for word in words:
            original_word = word

            for function in self._processing_pipeline:
                result = function(word)

                if result != original_word:
                    word = result
                    break

            word = self._ninja(word)

            if word:
                processed_words.append(word)

        # join and split again to split into words after 'ninja'
        processed_words = ' '.join(processed_words).split()

        processed_words = [self._expand_slang(word) for word in processed_words]

        return ' '.join(processed_words)

    def _fix_corruptions(self, text):
        text = html.unescape(text)
        text = codecs.decode(text, 'unicode_escape', errors='replace')
        text = ftfy.fix_text(text)
        text = unicodedata.normalize('NFKD', text)

        text = text.replace('†', '"')  # common character replacement
        is_text_or_emoji = demojize(text).isascii()

        if not is_text_or_emoji:
            print(f"Entry contains non-ASCII or emoji characters: {text[:20]}")

        return text

    def _replace_url(self, word):
        """Remove URLs - returns None if word is a URL, otherwise returns the word"""
        url_pattern = r'^(https?://)?(www\.)?([a-zA-Z0-9-]+\.[a-zA-Z]{2,})(/.*)?$'
        return None if re.match(url_pattern, word) else word

    def _replace_mentions(self, word):
        """Replace @mentions with <USER> token"""
        pattern = r"^@[\w]+$"

        if re.match(pattern, word):
            return "user"

        return word

    def _replace_hashtags(self, word):
        """Remove the pound symbol from words. Words starting with more than one hashtag remain unchanged."""

        if word == '#':
            return word

        return word[1:] if word.startswith('#') and not word[1:].startswith('#') else word

    def _deobfuscate(self, word):
        """Replace obfuscated terms with their normal forms"""
        return self._obfuscation_map.get(word, word)

    def _emoji_to_text(self, word):
        """Convert emojis to their hidden contextual meanings"""
        return demojize(word)

    def _expand_slang(self, word):
        """Expand slang abbreviations to their full forms"""
        return self._slang_map.get(word, word)

    def _ninja(self, word):
        if word:
            return ' '.join(wordninja.split(word))
        return None

    def get_special_tokens(self):
        """Return set of special tokens identified during processing"""
        return self._special_tokens

def test_run():
    example_texts = [
        "The z1oni$ts lobby controls the media",
        "Those j00$ are everywhere",
        "88 brother, 14 words to live by",
        "1488 is my favorite number",
        "h\\t a reliable source",
        "look up at link: https://www.google.com/",
        "my friend @friend has a dog",
        "tell #hashtag"
    ]
    processor = TextNormalizer('config.json')

    print("Example")
    print("\n", "-" * 20 + " Before conversion " + "-" * 20)
    for text in example_texts:
        print(text)

    print("\n", "-" * 20 + " After conversion " + "-" * 20)
    for text in example_texts:
        print(processor.normalize(text))


if __name__ == "__main__":
    test_run()
