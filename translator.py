from langdetect import detect
from deep_translator import GoogleTranslator
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from db_helper import get_cached_translation, set_cached_translation

def detect_language(text):
    """
    Detects whether the input text is English ('en'), Kannada ('kn'), or Hindi ('hi').
    Uses Unicode range inspection as a high-fidelity check, falling back to langdetect.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return 'en'
    
    # Check for Kannada character range
    if any('\u0c80' <= char <= '\u0cff' for char in text):
        return 'kn'
        
    # Check for Devanagari (Hindi) character range
    if any('\u0900' <= char <= '\u097f' for char in text):
        return 'hi'
        
    try:
        lang = detect(text)
        if lang in ['kn', 'hi', 'en']:
            return lang
        return 'en'
    except Exception:
        return 'en'

def translate_text(text, source_lang='auto', target_lang='en'):
    """
    Translates text from source_lang to target_lang using Google Translate, checking SQLite cache first.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return ""
        
    if source_lang == 'auto':
        source_lang = detect_language(text)
        
    if source_lang == target_lang:
        return text
        
    # Check persistent cache
    cached = get_cached_translation(text, source_lang, target_lang)
    if cached and cached.get("translated_text"):
        return cached["translated_text"]
        
    try:
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated = translator.translate(text)
        
        # Calculate transliteration if Kannada/Hindi and store both in cache
        transliterated = ""
        if source_lang in ['kn', 'hi']:
            transliterated = transliterate_text(text, source_lang)
            
        set_cached_translation(text, source_lang, target_lang, translated, transliterated)
        return translated
    except Exception as e:
        print(f"Translation error ({source_lang} -> {target_lang}): {e}")
        return text

def transliterate_text(text, lang):
    """
    Transliterates Kannada or Hindi native script into English letters.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return ""
        
    # Check cache by looking up translations from this language to English
    cached = get_cached_translation(text, lang, 'en')
    if cached and cached.get("transliterated_text"):
        return cached["transliterated_text"]
        
    try:
        if lang == 'kn':
            # Transliterate Kannada to ITRANS (Latin equivalent)
            output = transliterate(text, sanscript.KANNADA, sanscript.ITRANS)
            return output.strip().lower()
        elif lang == 'hi':
            # Transliterate Devanagari to ITRANS
            output = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
            return output.strip().lower()
        return text
    except Exception as e:
        print(f"Transliteration error for language {lang}: {e}")
        return text

