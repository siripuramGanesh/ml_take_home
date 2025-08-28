# test_image_pipeline.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# --------------------------
# 1️⃣ Create test images of different sizes
# --------------------------
def create_test_image(width, height, color="green"):
    """Generate a solid-color image of specified size"""
    return Image.new("RGB", (width, height), color=color)

# Sizes to test: tiny, normal, large, extreme aspect ratios
test_sizes = [
    (10, 10),       # tiny
    (224, 224),     # normal square
    (3000, 3000),   # large
    (500, 1000),    # tall rectangle
    (1000, 500),    # wide rectangle
    (1, 1000),      # extreme tall
    (1000, 1)       # extreme wide
]

# --------------------------
# 2️⃣ Define augmentations
# --------------------------
augment = transforms.Compose([
    transforms.Resize((224, 224)),   # resize to model input
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(),
    transforms.ToTensor()
])

# --------------------------
# 3️⃣ Dataset and DataLoader
# --------------------------
class DummyDataset(Dataset):
    def __init__(self, sizes, transform=None):
        self.sizes = sizes
        self.transform = transform

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, idx):
        w, h = self.sizes[idx]
        img = create_test_image(w, h)
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(0)  # dummy label
        return img, label

dataset = DummyDataset(test_sizes, transform=augment)
loader = DataLoader(dataset, batch_size=2, shuffle=False)

# --------------------------
# 4️⃣ Test augmentations
# --------------------------
print("=== Testing augmentations ===")
for size in test_sizes:
    try:
        img = create_test_image(*size)
        tensor_img = augment(img)
        print(f"Success: {size} -> Tensor shape: {tensor_img.shape}")
    except Exception as e:
        print(f"Failed on size {size}: {e}")

# --------------------------
# 5️⃣ Test dummy training loop
# --------------------------
print("\n=== Testing dummy training loop ===")
for batch_imgs, batch_labels in loader:
    try:
        # Example forward pass with random weights
        dummy_model = torch.nn.Linear(224*224*3, 10)  # simple model
        inputs = batch_imgs.view(batch_imgs.size(0), -1)  # flatten
        outputs = dummy_model(inputs)
        loss = torch.nn.CrossEntropyLoss()(outputs, torch.zeros(outputs.size(0), dtype=torch.long))
        loss.backward()
        print(f"Batch processed successfully: {batch_imgs.shape}")
    except Exception as e:
        print(f"Failed on batch: {e}")
