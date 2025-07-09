# train.py

"""
This file uses the U-Net model from model.py and trains it using real data
(images + masks) so it learns to segment optic disc and cup accurately.
Think of this like assembling a self-driving car (model) and teaching it to recognize road boundaries (optic regions) by driving it around with labeled maps (masks).
"""

# Importing required libraries
import torch                       # Core PyTorch module
import torch.nn as nn              # Tools to build neural network layers
import torch.optim as optim        # Optimization algorithms like Adam
from torch.utils.data import DataLoader  # Efficient loading of data in batches
from torchvision import transforms        # Preprocessing transforms
from dataset import FundusSegmentationDataset  # Our custom dataset class
from model import UNet             # Our U-Net model architecture
import os                          # For directory checks

# Define training data paths
image_dir = "partitioned_randomly/images/train"            # Input retina images
disc_mask_dir = "partitioned_randomly/masks_disc/train"    # Ground truth disc masks
cup_mask_dir = "partitioned_randomly/masks_cup/train"      # Ground truth cup masks

# Image transformation: Resize + convert to tensor
transform = transforms.Compose([
    transforms.Resize((256, 256)),    # Resizes all images to 256x256 (model input size)
    transforms.ToTensor()             # Converts images to PyTorch tensors (0–1 scale)
])

# Load training dataset using our custom class
dataset = FundusSegmentationDataset(image_dir, disc_mask_dir, cup_mask_dir, transform)

# Debugging: Print sample size and few example image names
print("Number of training samples found:", len(dataset))
print("Sample base names:", dataset.samples[:5])

# Handle edge case: if no valid data found
if len(dataset) == 0:
    raise RuntimeError("No training samples found. Check folder paths and matching filenames.")

# Use DataLoader to load images in batches (batch_size=4)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Set up GPU or CPU for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model to the device (GPU if available)
model = UNet().to(device)

# Define loss function (binary cross-entropy with logits)
criterion = nn.BCEWithLogitsLoss()

# Optimizer to update model weights
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop for 10 epochs
epochs = 10
for epoch in range(epochs):
    model.train()         # Set model to training mode
    epoch_loss = 0        # Track total loss for this epoch
    
    for images, disc_masks, cup_masks in loader:
        
        # Move inputs and masks to the correct device (GPU/CPU)
        images = images.to(device)
        disc_masks = disc_masks.to(device)
        cup_masks = cup_masks.to(device)

        optimizer.zero_grad()       # Clear gradients from last batch
        output = model(images)      # Make predictions with U-Net

        # Output has 2 channels → [0]=disc, [1]=cup
        disc_pred = output[:, 0:1, :, :]  # First channel: optic disc
        cup_pred = output[:, 1:2, :, :]   # Second channel: optic cup

        # Calculate loss for both masks
        loss_disc = criterion(disc_pred, disc_masks)
        loss_cup = criterion(cup_pred, cup_masks)
        loss = loss_disc + loss_cup     # Total loss = sum of both

        loss.backward()        # Backpropagation
        optimizer.step()       # Update weights
        epoch_loss += loss.item()  # Track loss

    # Print epoch summary
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

# Save the trained model weights
torch.save(model.state_dict(), "segmentation_model.pth")
print("Model training complete and saved to segmentation_model.pth")
