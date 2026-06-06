import urllib.request

try:
    response = urllib.request.urlopen("http://localhost:8501")
    html = response.read().decode('utf-8')
    print("Streamlit Server Response status:", response.status)
    if "Streamlit" in html:
        print("Success: Streamlit application is serving requests successfully!")
    else:
        print("Warning: Response content did not contain expected Streamlit elements.")
except Exception as e:
    print("Error connecting to Streamlit server:", e)
