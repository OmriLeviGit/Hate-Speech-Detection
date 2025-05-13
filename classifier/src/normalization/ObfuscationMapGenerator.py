import json
import itertools
import copy


class ObfuscationMapGenerator:
    def __init__(self, obfuscation_config=None, config_path=None, letter_map=None, save=False):
        self._letter_map = letter_map
        self._obfuscation_terms = []
        self._obfuscation_map = {}

        if obfuscation_config is None and config_path is not None:
            obfuscation_config = self._load_config(config_path)

        if obfuscation_config is not None:
            self._process_config(obfuscation_config)

        self._obfuscation_map = self._generate_obfuscation_map()

        if save:
            self.save_obfuscation_map()

    def _process_config(self, obfuscation_config):
        if isinstance(obfuscation_config, dict):
            if self._letter_map is None and "letter_map" in obfuscation_config:
                self._letter_map = obfuscation_config["letter_map"]
            self._obfuscation_terms = obfuscation_config.get("terms", [])
        else:
            self._obfuscation_terms = obfuscation_config

    def _load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config.get("obfuscation", {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading obfuscation config: {e}")
            return {}

    def get_map(self):
        return copy.copy(self._obfuscation_map)

    def _generate_obfuscation_map(self, debug=None):

        obfuscation_to_base = {}

        for term_obj in self._obfuscation_terms:
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

        if debug:
            print(f"Generated {len(obfuscation_to_base)} obfuscated terms")
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


        for char, positions in positions_map.items():
            replacements = self._letter_map[char]
            current_results = set(results)

            for num_positions in range(1, len(positions) + 1):
                for pos_combo in itertools.combinations(positions, num_positions):
                    for variant in current_results:
                        variant_chars = list(variant)
                        for replacement in replacements:
                            new_variant = variant_chars.copy()

                            for pos in pos_combo:
                                new_variant[pos] = replacement
                            results.add(''.join(new_variant))

        return results

    def save_obfuscation_map(self):
        with open('obfuscation_map.json', 'w') as f:
            json.dump(self._obfuscation_map, f, indent=2)
        print("Obfuscation map saved to obfuscation_map.json")




if __name__ == "__main__":
    obf = ObfuscationMapGenerator(config_path='config.json')
    obf.save_obfuscation_map()
