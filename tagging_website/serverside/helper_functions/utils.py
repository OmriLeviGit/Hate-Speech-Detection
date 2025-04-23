import codecs
import html
import ftfy
import emoji
import unicodedata
import re
import pycountry


def replace_flags_with_country_names(text):
    """
    Country flags don't work like regular emojis, and are represented by special 2-unicode characters.
    Require special processing.
    """
    if not text:
        return None

    flag_pattern = r'[\U0001F1E6-\U0001F1FF]{2}'  # Look for flag emoji sequences
    flags = re.findall(flag_pattern, text)

    if not flags:
        return text

    result = text

    for flag in flags:
        try:
            code_points = [ord(c) for c in flag]
            country_code = ''.join(chr(cp - 0x1F1E6 + ord('A')) for cp in code_points)

            country = pycountry.countries.get(alpha_2=country_code)

            if not country:
                return None

            country_name = country.name
            if hasattr(country, 'common_name'):
                country_name = country.common_name
            elif ',' in country.name:
                country_name = country.name.split(',')[0]

            result = result.replace(flag, country_name)

        except:
            return None

    return result


def fix_corrupted_text(text):
    if not text:
        return None

    text = html.unescape(text)
    text = codecs.decode(text, 'unicode_escape', errors='replace')
    text = ftfy.fix_text(text)
    text = unicodedata.normalize('NFKD', text)
    text = replace_flags_with_country_names(text)

    text = text.replace('†', '"')  # Common character replacement

    is_text_or_emoji = emoji.demojize(text).isascii()

    if not is_text_or_emoji:
        return None

    return text


def main():
    text = "ðŸ‡µðŸ‡¸ðŸ‡®ðŸ‡± some text"
    fixed = fix_corrupted_text(text)
    print(fixed)


if __name__ == "__main__":
    main()
