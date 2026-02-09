import requests

ngrok_url = "https://platypus-outgoing-annually.ngrok-free.app/get_auth_code"  # Replace with your actual ngrok URL

try:
    response = requests.get(ngrok_url)
    if response.status_code == 200:
        print("[SUCCESS] Lightning AI can reach your local machine!")
        print("Response:", response.json())
    else:
        print("[ERROR] Lightning AI cannot reach your local machine.")
        print("HTTP Status:", response.status_code)
except requests.exceptions.RequestException as e:
    print(f"[ERROR] Lightning AI request failed: {e}")