import requests
import json

BASE_URL = "http://localhost:8000"

def test_translate():
    print("=== Testing /api/translate ===")
    
    # 1. Hindi test
    payload = {
        "text": "पासवर्ड रीसेट काम नहीं कर रहा है",
        "target_lang": "en"
    }
    res = requests.post(f"{BASE_URL}/api/translate", json=payload)
    print("Hindi translation response status:", res.status_code)
    data = res.json()
    print("Hindi translation response:", json.dumps(data, indent=2, ensure_ascii=False))
    assert data["detected_lang"] == "hi"
    assert "password" in data["translated_text"].lower() or "reset" in data["translated_text"].lower()
    assert data["transliterated_text"] != ""
    
    # 2. Kannada test
    payload = {
        "text": "ನನ್ನ ಖಾತೆಗೆ ಲಾಗ್ ಇನ್ ಮಾಡಲು ಸಾಧ್ಯವಾಗುತ್ತಿಲ್ಲ",
        "target_lang": "en"
    }
    res = requests.post(f"{BASE_URL}/api/translate", json=payload)
    print("\nKannada translation response status:", res.status_code)
    data = res.json()
    print("Kannada translation response:", json.dumps(data, indent=2, ensure_ascii=False))
    assert data["detected_lang"] == "kn"
    assert "log" in data["translated_text"].lower() or "account" in data["translated_text"].lower()
    assert data["transliterated_text"] != ""

def test_analyze():
    print("\n=== Testing /api/analyze ===")
    payload = {
        "text": "ನನ್ನ ಕ್ರೆಡಿಟ್ ಕಾರ್ಡ್ ಗೆ ಎರಡು ಬಾರಿ ಶುಲ್ಕ ವಿಧಿಸಲಾಗಿದೆ, ದಯವಿಟ್ಟು ಮರುಪಾವತಿ ಮಾಡಿ",
        "model": "logistic"
    }
    res = requests.post(f"{BASE_URL}/api/analyze", json=payload)
    print("Analyze response status:", res.status_code)
    data = res.json()
    print("Analyze response:", json.dumps(data, indent=2, ensure_ascii=False))
    
    ticket = data["ticket"]
    assert ticket["language"] == "kn"
    assert ticket["translation"] != ""
    print("Ticket added to database with language:", ticket["language"])
    print("Ticket translation stored:", ticket["translation"])

def test_tickets():
    print("\n=== Testing /api/tickets ===")
    res = requests.get(f"{BASE_URL}/api/tickets")
    print("Get tickets response status:", res.status_code)
    tickets = res.json()
    print(f"Total tickets retrieved: {len(tickets)}")
    for t in tickets[:2]:
        print(f"ID: {t['id']}, Lang: {t['language']}, Translation: {t['translation']}, Original: {t['transcript']}")

def test_tts():
    print("\n=== Testing /api/tts ===")
    payload = {
        "text": "ನಮಸ್ಕಾರ, ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ?",
        "lang": "kn"
    }
    res = requests.post(f"{BASE_URL}/api/tts", json=payload)
    print("TTS response status:", res.status_code)
    print("TTS response headers:", res.headers)
    assert res.headers["content-type"] == "audio/mpeg"
    print("TTS bytes length:", len(res.content))

if __name__ == "__main__":
    try:
        test_translate()
        test_analyze()
        test_tickets()
        test_tts()
        print("\nAll API endpoints tests passed successfully!")
    except Exception as e:
        print("\nTest failed with error:", e)
