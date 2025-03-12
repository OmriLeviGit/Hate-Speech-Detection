import codecs
import re
import html
import ftfy

def fix_corrupted_text(text):
    # First handle HTML entities
    try:
        text = html.unescape(text)
    except:
        pass

    # Use ftfy for general text encoding fixes
    text = ftfy.fix_text(text)

    # Handle specific Unicode escape sequences if ftfy didn't catch them
    try:
        if '\\u' in text or '\\U' in text:
            text = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), text)
            text = re.sub(r'\\U([0-9a-fA-F]{8})', lambda m: chr(int(m.group(1), 16)), text)

            # If there are still escapes, try codecs as a fallback
            if '\\u' in text or '\\U' in text:
                text = codecs.decode(text, 'unicode_escape', errors='replace')
    except:
        pass

    # Specific character replacement
    text = text.replace('†', '"')

    return text
