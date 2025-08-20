import httpx

# FastAPI running locally
url = "http://127.0.0.1:8000/infer"

# Send an image file to the API
files = {"image": ("sample.jpg", open("data/coco/sample.jpg", "rb"))}

response = httpx.post(url, files=files)

print("Status code:", response.status_code)
print("Response:", response.json())
