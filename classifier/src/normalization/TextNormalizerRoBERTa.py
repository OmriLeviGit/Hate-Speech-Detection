import re
from typing_extensions import override
from classifier.src.normalization.TextNormalizer import TextNormalizer


class TextNormalizerRoBERTa(TextNormalizer):
    def __init__(self, config_path='config.json', emoji: str = 'text'):
        super().__init__(config_path, emoji)

    @override
    def _setup_pipeline(self, emoji):
        """Set up the processing pipeline with functions in the correct order"""
        self._processing_pipeline = [
            self._deobfuscate,
        ]

        if emoji == "text":
            self._processing_pipeline.append(self._emoji_to_text)

    @override
    def _replace_url(self, word):
        """Remove URLs - returns None if word is a URL, otherwise returns the word"""
        url_pattern = r'^(https?://)?(www\.)?([a-zA-Z0-9-]+\.[a-zA-Z]{2,})(/.*)?$'
        return "http" if re.match(url_pattern, word) else word
