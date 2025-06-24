# dataset.py

"""
dataset.py - It is a tool that reads eye images and their corresponding disc and cup masks for deep learning. 

What this file does?
- picks only the images that have both disc and cup masks.
- applies the right size and format for training.
- returns them in a format the model understands.

"""

#Importing libraries

import os   # used for navigating through the folders (paths and directories)
from PIL import Image #Python Imaging Library - helps to open and process image
from torch.utils.data import Dataset  # let's us define a custom dataset class
import torchvision.transforms as T  # for resize images, covert images to PyTorch tensors..etc


# Custom Class to tell Pytorch how to load images and their corresponding disc and cup masks from the organized folders.

class FundusSegmentationDataset(Dataset):
    def __init__(self, image_dir, disc_mask_dir, cup_mask_dir, transform=None):
        self.image_dir = image_dir
        self.disc_mask_dir = disc_mask_dir
        self.cup_mask_dir = cup_mask_dir
        self.transform = transform
        
       # This part scans all .png images in the image folder, extracts the base name (like r2_Im004), builds the expected 
       # disc and cup mask filenames (like r2_Im004-1-Disc-T.png and r2_Im004-1-Cup-T.png), and saves the base name only if 
       # both masks are present. This helps ensure we use only complete and valid samples during training or evaluation.

        self.samples = []
        for f in os.listdir(self.image_dir):
            if f.endswith(".png"):
                base = f.replace(".png", "")
                disc_mask = base + "-1-Disc-T.png"
                cup_mask = base + "-1-Cup-T.png"

                disc_path = os.path.join(self.disc_mask_dir, disc_mask)
                cup_path = os.path.join(self.cup_mask_dir, cup_mask)

                if os.path.exists(disc_path) and os.path.exists(cup_path):
                    self.samples.append(base)

        print(f"Matched image + disc + cup samples: {len(self.samples)}")
        print(f"Example base names: {self.samples[:5]}")
    
    # returns how many images with masks are present
    def __len__(self):
        return len(self.samples)

    # loads and returns one sample 
    def __getitem__(self, idx):
        base = self.samples[idx]
        image = Image.open(os.path.join(self.image_dir, base + ".png")).convert("RGB")
        disc_mask = Image.open(os.path.join(self.disc_mask_dir, base + "-1-Disc-T.png")).convert("L")
        cup_mask = Image.open(os.path.join(self.cup_mask_dir, base + "-1-Cup-T.png")).convert("L")

        if self.transform:
            image = self.transform(image)
            disc_mask = self.transform(disc_mask)
            cup_mask = self.transform(cup_mask)

        return image, disc_mask, cup_mask