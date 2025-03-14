import codecs
import re
import html
import ftfy
import emoji

def fix_corrupted_text(text):

    if not text:
        return None
    
    # handle HTML entities
    try:
        text = html.unescape(text)
    except:
        pass

    # general text encoding fixes
    text = ftfy.fix_text(text)

    # specific Unicode escape sequences if ftfy didn't catch them
    try:
        if '\\u' in text or '\\U' in text:
            text = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), text)
            text = re.sub(r'\\U([0-9a-fA-F]{8})', lambda m: chr(int(m.group(1), 16)), text)

            # If there are still escapes, try codecs as a fallback
            if '\\u' in text or '\\U' in text:
                text = codecs.decode(text, 'unicode_escape', errors='replace')
    except:
        pass
    
    text = text.replace('†', '"')   # Common character replacement

    # Common character replacement
    text = text.replace('†', '"')

    is_text_or_emoji = emoji.demojize(text).isascii()
    
    if not is_text_or_emoji:
        return None
    
    return text
