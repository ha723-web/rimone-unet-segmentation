# train.py

"""  
This file uses Uses the U-Net model from model.py and trains it using real data 
(images + masks) so it learns to segment optic disc and cup accurately.
example- Like actually building the car and teaching it how to drive by taking it on the road.
"""

#Importing libraries

import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader
from torchvision import transforms 
from dataset import FundusSegmentationDataset 
from model import UNet 
import os

# To 
image_dir = "partitioned_randomly/images/train"
disc_mask_dir = "partitioned_randomly/masks_disc/train"
cup_mask_dir = "partitioned_randomly/masks_cup/train"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = FundusSegmentationDataset(image_dir, disc_mask_dir, cup_mask_dir, transform)

print("🔍 Number of training samples found:", len(dataset))
print("📂 Sample base names:", dataset.samples[:5])

if len(dataset) == 0:
    raise RuntimeError("No training samples found. Check folder paths and matching filenames.")

loader = DataLoader(dataset, batch_size=4, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 10
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for images, disc_masks, cup_masks in loader:
        images = images.to(device)
        disc_masks = disc_masks.to(device)
        cup_masks = cup_masks.to(device)

        optimizer.zero_grad()
        output = model(images)
        disc_pred = output[:, 0:1, :, :]
        cup_pred = output[:, 1:2, :, :]
        loss_disc = criterion(disc_pred, disc_masks)
        loss_cup = criterion(cup_pred, cup_masks)
        loss = loss_disc + loss_cup
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"📉 Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

torch.save(model.state_dict(), "segmentation_model.pth")
print("Model training complete and saved to segmentation_model.pth")
