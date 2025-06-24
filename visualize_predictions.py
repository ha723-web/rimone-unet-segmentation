"""
This script visualizes the output of the trained model by overlaying predicted
optic disc and optic cup masks on the original fundus images.

Example: Like tracing shapes (disc/cup) on top of a scanned retina photo
to see how accurate your predictions are.
"""

import torch  # Deep learning framework
import os  # File and folder operations
from torchvision import transforms  # For resizing and tensor conversion
from torch.utils.data import DataLoader  # To load images in batches
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualization
from model import UNet  # Your U-Net model
import importlib.util  # To load dataset class dynamically
from skimage import measure, morphology  # For mask post-processing

# Dynamically load the class FundusSegmentationDataset from dataset.py
spec = importlib.util.spec_from_file_location("dataset", "dataset.py")
dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_module)
FundusSegmentationDataset = dataset_module.FundusSegmentationDataset

# Paths
image_dir = "partitioned_randomly/images/test"
disc_mask_dir = "partitioned_randomly/masks_disc/test"
cup_mask_dir = "partitioned_randomly/masks_cup/test"

output_dir = "overlays"
os.makedirs(output_dir, exist_ok=True)

# Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)), # Resize to match model input
    transforms.ToTensor() # Convert to PyTorch tensor
])

# Dataset and Loader
dataset = FundusSegmentationDataset(image_dir, disc_mask_dir, cup_mask_dir, transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Loading the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load("segmentation_model.pth"))
model.eval()

def save_overlay(image, disc_pred, cup_pred, filename):
    # Converting the tensor to NumPy image
    image_np = image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    disc_np = torch.sigmoid(disc_pred).detach().cpu().numpy().squeeze(0)
    cup_np = torch.sigmoid(cup_pred).detach().cpu().numpy().squeeze(0)
    
    # Taking the first channel if needed
    if disc_np.ndim == 3:
        disc_np = disc_np[0]
    if cup_np.ndim == 3:
        cup_np = cup_np[0]
    
    # Threshold predictions to create binary masks
    disc_mask = disc_np > 0.5
    cup_mask = cup_np > 0.5

    # Cleaning masks to remove tiny noisy regions
    disc_mask = clean_mask(disc_mask, min_size=500)
    cup_mask = clean_mask(cup_mask, min_size=300)

    # Plotting overlays and save
    plt.figure(figsize=(5, 5))
    plt.imshow(image_np)
    plt.contour(disc_mask, colors='r', linewidths=1.5, zorder=2)
    plt.contour(cup_mask, colors='b', linewidths=1.5, zorder=3)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0)
    plt.close()

from skimage.morphology import remove_small_objects
from skimage.measure import label

def clean_mask(mask, min_size=100):
    label_mask = label(mask)
    cleaned = remove_small_objects(label_mask, min_size=min_size)
    # Keeping only the largest component
    if cleaned.max() > 0:
        largest_label = np.argmax(np.bincount(cleaned.flat)[1:]) + 1
        cleaned = cleaned == largest_label
    return cleaned

# Save overlays for first 10 samples
print(f"Saving overlays to '{output_dir}' ...")
        
for idx, (image, _, _) in enumerate(loader):

    image = image.to(device)
    pred = model(image)

    # Separating predictions for disc and cup
    disc_pred = pred[:, 0:1, :, :]
    cup_pred = pred[:, 1:2, :, :]

    # Converting to NumPy and threshold
    disc_np = torch.sigmoid(disc_pred).detach().cpu().numpy().squeeze()
    cup_np = torch.sigmoid(cup_pred).detach().cpu().numpy().squeeze()
    
    # Threshold and clean
    disc_mask = clean_mask(disc_np > 0.5)
    cup_mask = clean_mask(cup_np > 0.5)
    
    # Print pixel count for masks (for debugging)
    print(f"[{idx+1}] Disc mask pixels: {int(disc_mask.sum())}")
    print(f"[{idx+1}] Cup mask  pixels: {int(cup_mask.sum())}")

    # Saving overlay image
    save_overlay(image, disc_pred, cup_pred, f"overlay_{idx+1}.png")

print("Overlay images saved!")
