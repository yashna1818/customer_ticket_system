from gtts import gTTS
import tempfile

def synthesize_speech(text, lang='en'):
    """
    Converts a text string directly into spoken human-like audio using Google TTS.
    Returns the absolute path to the generated .mp3 file.
    """
    # Map supported languages to gTTS language codes
    tts_lang = 'en'
    if lang in ['hi', 'kn', 'en']:
        tts_lang = lang
        
    tts = gTTS(text=text, lang=tts_lang, slow=False)
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio_file.name)
    return temp_audio_file.name
