# test_mixed_sampling.py
import random
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch

# --------------------------
# Helper: create dummy datasets
# --------------------------
def create_dummy_dataset(num, label):
    class Dummy(Dataset):
        def __len__(self):
            return num
        def __getitem__(self, idx):
            # Create a simple 224x224 image
            img = Image.new("RGB", (224, 224), color="red" if label=="new" else "blue")
            return transforms.ToTensor()(img), label
    return Dummy()

dataset_new = create_dummy_dataset(10, "new")    # 10 new images
dataset_coco = create_dummy_dataset(100, "coco") # 100 COCO images

# --------------------------
# MixedDataset class
# --------------------------
class MixedDataset(Dataset):
    def __init__(self, dataset_new, dataset_coco, new_ratio=0.1):
        self.dataset_new = dataset_new
        self.dataset_coco = dataset_coco
        self.new_ratio = new_ratio
        self.len_new = len(dataset_new)
        self.len_coco = len(dataset_coco)
        self.total_len = self.len_new + self.len_coco

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # Randomly sample from new or coco according to ratio
        if random.random() < self.new_ratio:
            return self.dataset_new[random.randint(0, self.len_new-1)]
        else:
            return self.dataset_coco[random.randint(0, self.len_coco-1)]

# --------------------------
# Create DataLoader
# --------------------------
mixed_dataset = MixedDataset(dataset_new, dataset_coco, new_ratio=0.1)
loader = DataLoader(mixed_dataset, batch_size=10, shuffle=False)

# --------------------------
# Test mixed sampling
# --------------------------
counter = Counter()
for batch_imgs, batch_labels in loader:
    counter.update(batch_labels)

total_samples = sum(counter.values())
print("=== Mixed Dataset Sampling Test ===")
for k, v in counter.items():
    print(f"{k}: {v} samples ({v/total_samples:.2%})")
