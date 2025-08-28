import base64
import json
import requests
from pathlib import Path

# -----------------------------
# Path to your test image
# -----------------------------
image_path = Path("data/coco/images/train2017/000000000009.jpg")  # change if needed

# Read image and encode in Base64
with open(image_path, "rb") as f:
    image_bytes = f.read()
image_b64 = base64.b64encode(image_bytes).decode("utf-8")

# -----------------------------
# FastAPI /collect endpoint URL
# -----------------------------
url = "http://127.0.0.1:8000/api/collect"

# -----------------------------
# Prepare payload
# -----------------------------
payload = {
    "image_bytes_b64": None,
    "image_npy_b64": None,
    "bounding_boxes": {
        "bread": [[0, 112, 303]]  # label -> list of boxes
    },
    "message": "Test sample"
}

# -----------------------------
# Send POST request
# -----------------------------
response = requests.post(url, json=payload)

# -----------------------------
# Print response
# -----------------------------
print("Status Code:", response.status_code)
print("Response JSON:", json.dumps(response.json(), indent=4))

# -----------------------------
# Verify saved files (optional)
# -----------------------------
save_folder = Path("data/new/images")
if save_folder.exists():
    saved_files = [f.name for f in save_folder.iterdir() if f.is_file()]
    print("Saved images:", saved_files)
else:
    print("Save folder does not exist yet.")
