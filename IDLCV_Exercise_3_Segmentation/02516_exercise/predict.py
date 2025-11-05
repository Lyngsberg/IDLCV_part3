import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# --- import your architecture and dataset ---
from model import MyModel            # Replace with your model class name
from train import DatasetLoader      # Must be defined in train.py

# --- function to save masks ---
def save_mask(array, path):
    # array should be a 2D numpy array with 0s and 1s
    im_arr = (array * 255)
    Image.fromarray(np.uint8(im_arr)).save(path)

# --- configuration ---
MODEL_PATH = "model.pth"
RESULTS_DIR = "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- load model ---
print("Loading model...")
model = MyModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- load test dataset ---
print("Loading test dataset...")
test_dataset = DatasetLoader(split='test', transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# --- create results directory ---
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- inference and save results ---
print("Generating segmentation masks...")
with torch.no_grad():
    for idx, (image, _) in enumerate(test_loader):
        image = image.to(DEVICE)
        output = model(image)

        # Assuming binary segmentation (1 channel)
        pred = torch.sigmoid(output).cpu().numpy()[0, 0]
        mask = (pred > 0.5).astype(np.uint8)

        save_path = os.path.join(RESULTS_DIR, f"mask_{idx:04d}.png")
        save_mask(mask, save_path)

        print(f"Saved mask: {save_path}")

print("✓ All segmentation masks saved in 'results/'")
