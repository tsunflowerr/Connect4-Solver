import requests
import json

# Endpoint API của bạn
url = "https://connect4-solver-1-87vz.onrender.com/api/connect4-move"

# Request body cho board Connect 4 chuẩn
payload = {
    "board": [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ],
    "current_player": 1,
    "valid_moves": [0, 1, 2, 3, 4, 5, 6],
    "is_new_game": True
}

# Headers
headers = {
    "Content-Type": "application/json",
    "Accept": "application/json"
}

try:
    print("Status code:", response.status_code)
    response_json = response.json()
    print("Response:", response_json)
except json.JSONDecodeError:
    print("❌ Server không trả về JSON hợp lệ!")
    print("Response text:", response.text)
except requests.exceptions.RequestException as e:
    print("❌ Error occurred:", e)
