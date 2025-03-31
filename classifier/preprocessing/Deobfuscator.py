import json
import itertools


class Deobfuscator:
    def __init__(self, config_path=None):
        self._obfuscation_map = {}
        self._letter_map = {
            "a": ["4", "@"],
            "b": ["8"],
            "e": ["3"],
            "i": ["1", "!"],
            "l": ["1"],
            "o": ["0"],
            "s": ["$", "5"],
            "t": ["7"]
        }

        if not config_path:
            config_path = 'obfuscation_config.json'

        self.load_config(config_path)

    def load_config(self, config_path, debug=None):
        with open(config_path, 'r') as f:
            config = json.load(f)

            if "obfuscation" in config:
                self._obfuscation_map = self._generate_obfuscation_map(config["obfuscation"])

                if debug:
                    print(f"Generated {len(self._obfuscation_map)} obfuscated terms")

    def save_obfuscation_map(self):
        with open('obfuscation_map.json', 'w') as f:
            json.dump(self._obfuscation_map, f, indent=2)
        print("Obfuscation map saved to obfuscation_map.json")

    def process(self, text, debug=False):
        # Split text into words for processing
        words = text.lower().split()
        result_words = []

        for word in words:
            # Check if this word is in our obfuscation map
            if word in self._obfuscation_map:
                result_words.append(self._obfuscation_map[word])
            else:
                result_words.append(word)

        # Rejoin into a single string
        deobfuscated_text = ' '.join(result_words)

        if debug:
            print(f"Original: {text}")
            print(f"Deobfuscated: {deobfuscated_text}")
            print("-" * 50)

        return deobfuscated_text

    def _generate_obfuscation_map(self, obfuscation):
        obfuscation_to_base = {}

        for term_obj in obfuscation:
            base_term = term_obj["baseTerm"]

            # Add static replacements
            for static in term_obj.get("staticReplacement", []):
                obfuscation_to_base[static] = base_term

            for term in term_obj.get("substitutablePlural", []):
                singular_variants = self._generate_substitutions(term)

                plural_variants = []

                # Get the plural forms of all variants
                for variant in singular_variants:
                    plural_variants.append(variant + "s")
                    # Special plural with $ if 's' has a $ mapping
                    if 's' in self._letter_map and '$' in self._letter_map['s']:
                        plural_variants.append(variant + "$")

                for variant in singular_variants:
                    if variant != base_term:
                        obfuscation_to_base[variant] = base_term

                for variant in plural_variants:
                    if variant != base_term:
                        obfuscation_to_base[variant] = base_term + 's'

            # Process substitutable
            for term in term_obj.get("substitutable", []):
                term_variants = self._generate_substitutions(term)

                for variant in term_variants:
                    if variant != base_term:
                        obfuscation_to_base[variant] = base_term

        return obfuscation_to_base

    def _generate_substitutions(self, term):
        results = {term}  # Start with the original term

        # Find all substitutable positions in the term
        positions_map = {}
        for i, char in enumerate(term):
            if char in self._letter_map:
                if char not in positions_map:
                    positions_map[char] = []
                positions_map[char].append(i)

        # For each character in the letter map that appears in the term
        for char, positions in positions_map.items():
            replacements = self._letter_map[char]
            current_results = set(results)  # Make a copy of current results

            # For each position where this character appears
            for num_positions in range(1, len(positions) + 1):
                for pos_combo in itertools.combinations(positions, num_positions):
                    # For each variant we already have
                    for variant in current_results:
                        variant_chars = list(variant)

                        # For each replacement option for this character
                        for replacement in replacements:
                            new_variant = variant_chars.copy()
                            # Replace the character at each position in this combination
                            for pos in pos_combo:
                                new_variant[pos] = replacement
                            results.add(''.join(new_variant))

        return results


def test_run():
    example_texts = [
        "The z1oni$ts lobby controls the media",
        "Those j00$ are everywhere",
        "88 brother, 14 words to live by",
        "1488 is my favorite number",
        "h\\t a reliable source"
    ]

    tp = Deobfuscator()

    print("\nDeobfuscation examples:")
    for text in example_texts:
        tp.process(text, debug=True)


if __name__ == "__main__":
    test_run()
