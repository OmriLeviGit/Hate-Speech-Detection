import json
import re
import spacy

from ObfuscationMapGenerator import ObfuscationMapGenerator


class TextPreprocessor:
    def __init__(self, config_path='config.json'):
        self._config_path = config_path
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
            self._process_emojis,
        ]

    def process_text(self, text):
        """Process text through all replacement functions in the pipeline"""
        processed_text = text.lower()
        for process_func in self._processing_pipeline:
            processed_text = process_func(processed_text)
        return processed_text

    def _process_words(self, text, processor_func):
        """Process each word in text using the provided function"""
        words = text.split()
        result_words = []

        for word in words:
            processed = processor_func(word)
            if processed is not None:
                result_words.append(processed)

        return ' '.join(result_words)

    def _remove_url(self, text):
        """Remove URLs"""
        url_pattern = r'^(https?://)?(www\.)?([a-zA-Z0-9-]+\.[a-zA-Z]{2,})(/.*)?$'

        def process(word):
            return None if re.match(url_pattern, word) else word

        return self._process_words(text, process)

    def _replace_mentions(self, text):
        """Replace @mentions with <USER> token"""
        pattern = r"^@[\w]+$"

        def process(word):
            return "<USER>" if re.match(pattern, word) else word

        return self._process_words(text, process)

    def _remove_hashtags(self, text):
        """Remove the pound symbol from words. Words starting with more than one hashtag remain unchanged."""

        def process(word):
            # if starts with a single hashtag, remove it
            return word[1:] if word.startswith('#') and not word[1:].startswith('#') else word

        return self._process_words(text, process)

    def _deobfuscate(self, text):
        """Replace obfuscated terms with their normal forms"""

        def process(word):
            return self._obfuscation_map.get(word, word)

        return self._process_words(text, process)

    def _expand_slang(self, text):
        """Expand slang abbreviations to their full forms"""

        def process(word):
            return self._slang_map.get(word, word)

        return self._process_words(text, process)

    def _process_emojis(self, text):
        """Convert emojis to their hidden contextual meanings"""

        def process(word):
            # Check if the word contains any emoji from our custom mapping
            for emoji in self._emoji_meaning_map:
                if emoji in word:
                    # Replace the emoji with its contextual meaning
                    word = word.replace(emoji, self._emoji_meaning_map[emoji])
            return word

        return self._process_words(text, process)

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
        "look up at https://www.google.com/",
        "my friend @friend has a dog",
        "tell #him"
    ]
    processor = TextPreprocessor('config.json')

    print("Example")
    print("\n", "-" * 20 + " Before conversion " + "-" * 20)
    for text in example_texts:
        print(text)

    print("\n", "-" * 20 + " After conversion " + "-" * 20)
    for text in example_texts:
        print(processor.process_text(text))


if __name__ == "__main__":
    test_run()
