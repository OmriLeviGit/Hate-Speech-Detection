import json
import re
import os

from .ObfuscationMapGenerator import ObfuscationMapGenerator


class TextPreprocessor:
    def __init__(self, config_path='config.json'):
        self._config_path = os.path.join(os.path.dirname(__file__), config_path)
        self._load_config()

        self._obfuscation_map_generator = ObfuscationMapGenerator(obfuscation_config=self._config.get('obfuscation', {}))
        self._obfuscation_map = {}

        if self._obfuscation_map_generator:
            self._obfuscation_map = self._obfuscation_map_generator.get_map()

        self._slang_map = self._config.get('slang', {})
        self._emoji_meaning_map = self._config.get('emoji', {})

        self._processing_pipeline = []
        self._setup_pipeline()

    def _load_config(self):
        """Load configuration from file"""
        try:
            with open(self._config_path, 'r') as f:

                self._config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading config: {e}")
            self._config = {}

    def _setup_pipeline(self):
        """Set up the processing pipeline with functions in the correct order"""
        self._processing_pipeline = [
            self._remove_url,
            self._replace_mentions,
            self._remove_hashtags,
            self._deobfuscate,
            self._expand_slang,
            # self._process_emojis,
        ]

    def process(self, text):
        """Process text by applying all functions to each word before moving to next word"""
        text = text.lower()
        words = text.split()
        processed_words = []

        for word in words:
            original_word = word

            for function in self._processing_pipeline:
                result = function(word)

                if result != original_word:
                    word = result
                    break

            if word:
                processed_words.append(word)

        return ' '.join(processed_words)

    def _remove_url(self, word):
        """Remove URLs - returns None if word is a URL, otherwise returns the word"""
        url_pattern = r'^(https?://)?(www\.)?([a-zA-Z0-9-]+\.[a-zA-Z]{2,})(/.*)?$'
        return None if re.match(url_pattern, word) else word

    def _replace_mentions(self, word):
        """Replace @mentions with <USER> token"""
        pattern = r"^@[\w]+$"
        return "<USER>" if re.match(pattern, word) else word

    def _remove_hashtags(self, word):
        """Remove the pound symbol from words. Words starting with more than one hashtag remain unchanged."""

        if word == '#':
            return word

        return word[1:] if word.startswith('#') and not word[1:].startswith('#') else word

    def _deobfuscate(self, word):
        """Replace obfuscated terms with their normal forms"""
        return self._obfuscation_map.get(word, word)

    def _expand_slang(self, word):
        """Expand slang abbreviations to their full forms"""
        return self._slang_map.get(word, word)

    def _process_emojis(self, word):
        """Convert emojis to their hidden contextual meanings"""
        result = word
        for emoji in self._emoji_meaning_map:
            if emoji in result:
                result = result.replace(emoji, self._emoji_meaning_map[emoji])
        return result

    def save_maps(self, file_path):
        """Save all maps to a file for later use"""
        pass


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
    processor = TextPreprocessor('config.json')

    print("Example")
    print("\n", "-" * 20 + " Before conversion " + "-" * 20)
    for text in example_texts:
        print(text)

    print("\n", "-" * 20 + " After conversion " + "-" * 20)
    for text in example_texts:
        print(processor.process(text))


if __name__ == "__main__":
    test_run()
