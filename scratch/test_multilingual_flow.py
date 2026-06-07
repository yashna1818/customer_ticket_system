import os
import sys

# Add root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from translator import detect_language, translate_text, transliterate_text
from preprocessing import get_sentiment_and_priority
from model import TicketClassifier
from db_helper import init_db, add_ticket, get_all_tickets, delete_ticket
from text_to_speech import synthesize_speech

def test_language_detection_and_translation():
    print("=== 1. Testing Translation & Transliteration ===")
    
    # Hindi Text
    hi_text = "पासवर्ड रीसेट काम नहीं कर रहा है"
    detected_hi = detect_language(hi_text)
    print(f"Hindi Text: '{hi_text}' -> Detected Lang: {detected_hi}")
    assert detected_hi == 'hi', f"Expected 'hi', got {detected_hi}"
    
    hi_trans = translate_text(hi_text, source_lang=detected_hi, target_lang='en')
    print(f"Translated: '{hi_trans}'")
    assert 'password' in hi_trans.lower() or 'reset' in hi_trans.lower(), "Hindi translation failed to capture core terms"
    
    hi_lat = transliterate_text(hi_text, 'hi')
    print(f"Transliterated: '{hi_lat}'")
    assert len(hi_lat) > 0, "Transliteration returned empty string"
    
    # Kannada Text
    kn_text = "ನನ್ನ ಕ್ರೆಡಿಟ್ ಕಾರ್ಡ್ ಗೆ ಎರಡು ಬಾರಿ ಶುಲ್ಕ ವಿಧಿಸಲಾಗಿದೆ, ದಯವಿಟ್ಟು ಮರುಪಾವತಿ ಮಾಡಿ"
    detected_kn = detect_language(kn_text)
    print(f"Kannada Text: '{kn_text}' -> Detected Lang: {detected_kn}")
    assert detected_kn == 'kn', f"Expected 'kn', got {detected_kn}"
    
    kn_trans = translate_text(kn_text, source_lang=detected_kn, target_lang='en')
    print(f"Translated: '{kn_trans}'")
    assert 'refund' in kn_trans.lower() or 'charge' in kn_trans.lower() or 'card' in kn_trans.lower(), "Kannada translation failed to capture core terms"
    
    kn_lat = transliterate_text(kn_text, 'kn')
    print(f"Transliterated: '{kn_lat}'")
    assert len(kn_lat) > 0, "Transliteration returned empty string"
    
    print("✓ Translation and transliteration tests passed!\n")
    return kn_trans, detected_kn

def test_classification_and_db_logging(translated_text, original_lang):
    print("=== 2. Testing ML Classification & DB Log ===")
    
    clf = TicketClassifier()
    if not clf.load():
        print("Model weights not found. Skipping classification verification.")
        return
        
    # Classify the English translation instead of native text
    all_preds = clf.predict_all(translated_text)
    category = all_preds['logistic']
    print(f"English translation: '{translated_text}' -> Predicted Category: {category}")
    
    # Predict sentiment and priority
    sentiment, priority = get_sentiment_and_priority(translated_text, category=category)
    print(f"Sentiment: {sentiment}, Priority: {priority}")
    
    # Ensure classification runs and returns a valid category
    valid_categories = ['Account Access', 'Billing Issue', 'General Inquiry', 'Refund Request', 'Technical Issue']
    assert category in valid_categories, f"Expected a valid category, got '{category}'"
    
    # Save to SQLite
    init_db()
    ticket_id = add_ticket(
        transcript="ನನ್ನ ಕ್ರೆಡಿಟ್ ಕಾರ್ಡ್ ಗೆ ಎರಡು ಬಾರಿ ಶುಲ್ಕ ವಿಧಿಸಲಾಗಿದೆ, ದಯವಿಟ್ಟು ಮರುಪಾವತಿ ಮಾಡಿ",
        category=category,
        sentiment=sentiment,
        priority=priority,
        model_used='logistic',
        language=original_lang,
        translation=translated_text
    )
    print(f"Ticket #{ticket_id} logged into SQLite successfully.")
    
    # Retrieve and verify
    tickets = get_all_tickets()
    logged_t = next((t for t in tickets if t['id'] == ticket_id), None)
    assert logged_t is not None, "Failed to retrieve logged ticket from DB"
    assert logged_t['language'] == 'kn', f"Expected language 'kn', got {logged_t['language']}"
    assert logged_t['translation'] == translated_text, "Logged translation does not match"
    
    # Clean up
    delete_ticket(ticket_id)
    print("✓ Classification and DB Logging tests passed!\n")

def test_multilingual_tts():
    print("=== 3. Testing Localized Text-to-Speech Synthesis ===")
    
    # Hindi Synthesis
    hi_speech = "यह एक स्वचालित अधिसूचना है"
    file_hi = synthesize_speech(hi_speech, lang='hi')
    print(f"Hindi TTS path: {file_hi}")
    assert os.path.exists(file_hi), "Hindi TTS file not found"
    assert os.path.getsize(file_hi) > 0, "Hindi TTS file is empty"
    os.remove(file_hi)
    
    # Kannada Synthesis
    kn_speech = "ನಿಮ್ಮ ಟಿಕೆಟ್ ಪರಿಹರಿಸಲಾಗಿದೆ"
    file_kn = synthesize_speech(kn_speech, lang='kn')
    print(f"Kannada TTS path: {file_kn}")
    assert os.path.exists(file_kn), "Kannada TTS file not found"
    assert os.path.getsize(file_kn) > 0, "Kannada TTS file is empty"
    os.remove(file_kn)
    
    print("✓ Localized TTS speech synthesis tests passed!\n")

if __name__ == "__main__":
    print("--- Starting Local Multilingual Support System Tests ---\n")
    try:
        kn_trans, kn_lang = test_language_detection_and_translation()
        test_classification_and_db_logging(kn_trans, kn_lang)
        test_multilingual_tts()
        print("All multilingual integration tests passed successfully!")
    except Exception as e:
        print("Test failed with error:", e)
        sys.exit(1)
