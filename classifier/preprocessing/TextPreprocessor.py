from classifier.preprocessing.Deobfuscator import Deobfuscator


class Preprocessor:

    def __init__(self, config_path=None):
        self._deobfuscator = Deobfuscator('obfuscation_config.json')
        pass

    def process(self, text):

        text = self._deobfuscator.process(text)
        # move the "heard through" and add https://www.noslang.com/twitterslang.php to the social media step
