import base64
import json
import requests

# Path to your test image
image_path = "data/coco/images/train2017/000000000009.jpg"  # <-- change this to your image

# Read and encode image as Base64
with open(image_path, "rb") as f:
    image_bytes = f.read()
image_b64 = base64.b64encode(image_bytes).decode("utf-8")

# Prepare payload
payload = {
    "image_bytes_b64": "not_a_real_base64_string",
    "image_npy_b64": None  # leave None if not using .npy array
}

# FastAPI inference endpoint
url = "http://127.0.0.1:8000/api/infer"

# Send POST request
response = requests.post(url, json=payload)

# Print response
print("Status Code:", response.status_code)
print("Response JSON:", json.dumps(response.json(), indent=4))
