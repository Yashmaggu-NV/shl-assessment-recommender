import requests

BASE_URL = "http://127.0.0.1:8000"

r = requests.get(f"{BASE_URL}/health")
print("HEALTH:", r.status_code, r.text)

payload = {
    "messages": [
        {
            "role": "user",
            "content": "Hiring mid-level Java backend engineer with AWS experience"
        }
    ]
}

r = requests.post(f"{BASE_URL}/chat", json=payload)

print("CHAT STATUS:", r.status_code)
print(r.json())