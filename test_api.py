import requests
import json
import time
import os

API_URL = "http://localhost:8000/chat"

def test_chat():
    payload = {
        "username": "api_test_user",
        "question": "How many total events happened?"
    }
    
    print(f"Sending POST to {API_URL}...")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    start_time = time.time()
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        elapsed = time.time() - start_time
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Time: {elapsed:.2f} seconds")
        
        if response.status_code == 200:
            data = response.json()
            print("\n--- AI Answer ---")
            print(data.get("answer"))
            print("-" * 20)
        else:
            print("\n--- Error Response ---")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print(f"\n[!] Connection refused. Is the FastAPI server running on {API_URL}?")
        print("Run it with: uvicorn api:app --reload")

if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY is not set. The server might fail.")
    test_chat()
