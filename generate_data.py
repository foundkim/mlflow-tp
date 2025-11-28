import argparse
import requests
import random
import json
import time

# Configuration
URL = 'http://127.0.0.1:8001/inference/predict'
HEADERS = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}

def generate_iris_data():
    """
    Generates random float values within typical ranges 
    for Iris flower measurements.
    """
    return {
        # Using the specific spelling 'lenght' to match your curl command
        "sepal_lenght": round(random.uniform(4.3, 7.9), 1),
        "sepal_width": round(random.uniform(2.0, 4.4), 1),
        "petal_lenght": round(random.uniform(1.0, 6.9), 1),
        "petal_width": round(random.uniform(0.1, 2.5), 1)
    }

def send_request():
    payload = generate_iris_data()
    
    try:
        print(f"Sending payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(
            URL, 
            headers=HEADERS, 
            json=payload,
            timeout=5
        )
        
        print(f"Status Code: {response.status_code}")
        try:
            print(f"Response: {response.json()}")
        except json.JSONDecodeError:
            print(f"Response Text: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to {URL}. Is the server running?")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    print("-" * 30)

if __name__ == "__main__":
    # Send 5 random requests
    parser = argparse.ArgumentParser(description="Send random Iris data requests to the inference server.")
    parser.add_argument("--num_requests", type=int, default=5, help="Number of requests to send")
    args = parser.parse_args()

    print(f"Starting inference test ({args.num_requests} requests)...\n")
    for _ in range(args.num_requests):
        send_request()
        # Small delay between requests
        time.sleep(0.5)