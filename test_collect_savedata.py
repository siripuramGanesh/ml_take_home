import base64
import json
import requests
from pathlib import Path

# URL for your collect endpoint
url = "http://127.0.0.1:8000/api/collect"

# Folder containing multiple images
data_folder = Path("data/coco/images/train2017")  # adjust as needed

# Example: send multiple images
payloads = []
for img_path in data_folder.glob("*.jpg"):  # loop through jpgs
    with open(img_path, "rb") as f:
        image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    
    payloads.append({
        "image_bytes_b64": image_b64,
        "image_npy_b64": None,
        "bounding_boxes": {
            "bread": [[0, 112, 230, 303]]  # example box
        },
        "message": f"Sample from {img_path.name}"
    })

# If your endpoint only accepts one image per request, loop through payloads
for payload in payloads:
    response = requests.post(url, json=payload)
    print(f"{payload['message']}: {response.status_code} -> {json.dumps(response.json())}")
