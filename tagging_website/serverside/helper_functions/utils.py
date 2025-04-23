import html
import codecs
import ftfy
import unicodedata
import emoji


def fix_corrupted_text(text):
    if not text:
        return None

    text = html.unescape(text)
    text = codecs.decode(text, 'unicode_escape', errors='replace')
    text = ftfy.fix_text(text)
    text = unicodedata.normalize('NFKD', text)

    text = text.replace('†', '"')  # Common character replacement

    is_text_or_emoji = emoji.demojize(text).isascii()

    if not is_text_or_emoji:
        return None

    return text
