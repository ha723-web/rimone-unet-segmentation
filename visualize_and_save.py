import os
import cv2
import torch
import matplotlib.pyplot as plt
from dataset import RIMOneDataset

def overlay_predictions(model, dataset, save_dir='results', mask_type='disc'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i in range(5):  # Save first 5 test images
            sample = dataset[i]
            img = sample['image'].unsqueeze(0)
            orig = img.squeeze().permute(1, 2, 0).numpy()
            gt_mask = sample[f'{mask_type}_mask'].squeeze().numpy()

            pred = model(img).squeeze().numpy()
            pred_bin = (pred > 0.5).astype(float)

            overlay = orig.copy()
            overlay[..., 1] += pred_bin * 0.5  # Green
            overlay[..., 0] += gt_mask * 0.5   # Blue

            overlay = (overlay * 255).astype('uint8')
            plt.imshow(overlay)
            plt.axis('off')
            plt.title(f'{mask_type.capitalize()} Segmentation')
            plt.savefig(f'{save_dir}/{mask_type}_overlay_{i}.png', bbox_inches='tight')
