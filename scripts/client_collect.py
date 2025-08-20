import httpx
import json

# FastAPI running locally
url = "http://127.0.0.1:8000/collect"

# Example bounding boxes
annotations = [
    {"bbox": [100, 150, 200, 250], "label": "person", "confidence": 1.0},
    {"bbox": [50, 60, 120, 180], "label": "dog", "confidence": 1.0},
]

# Upload the image with form data
files = {
    "image": ("sample.jpg", open("data/coco/sample.jpg", "rb")),
}
data = {
    "annotations": json.dumps(annotations),
    "message": "new training sample",
}

response = httpx.post(url, files=files, data=data)

print("Status code:", response.status_code)
print("Response:", response.json())
