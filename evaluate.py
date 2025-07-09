# evaluate.py

"""
This file evaluates the trained segmentation model by:
- Running it on the test set (images + ground truth masks)
- Computing the Dice Coefficient between predicted and actual masks
- Reporting average Dice scores for both optic disc and optic cup
"""

# Main function to compute Dice Coefficients on test data
def evaluate_model():
    
    # Importing necessary libraries inside the function
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import importlib.util
    from model import UNet

    # Dynamically load the custom dataset class from dataset.py
    spec = importlib.util.spec_from_file_location("dataset", "dataset.py")
    dataset_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset_module)
    FundusSegmentationDataset = dataset_module.FundusSegmentationDataset

    # Dice Coefficient Function
    def dice_coeff(pred, target, epsilon=1e-6):
        
        # Applying sigmoid to predictions (range 0–1)
        pred = torch.sigmoid(pred)
        
        # Converting to binary mask using threshold of 0.5
        pred = (pred > 0.5).float()
        
        # Calculate intersection and union
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        
        # Returning average Dice score
        dice = (2. * intersection + epsilon) / (union + epsilon)
        return dice.mean()

    # Paths to test data
    image_dir = "partitioned_randomly/images/test"
    disc_mask_dir = "partitioned_randomly/masks_disc/test"
    cup_mask_dir = "partitioned_randomly/masks_cup/test"

    # Resizing images to match training and convert to tensors
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Loading test dataset
    dataset = FundusSegmentationDataset(image_dir, disc_mask_dir, cup_mask_dir, transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Setup device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained U-Net model and switch to eval mode
    model = UNet().to(device)
    model.load_state_dict(torch.load("segmentation_model.pth"))
    model.eval()

    # Initialize total dice scores
    dice_disc_total = 0.0
    dice_cup_total = 0.0

    # Disable gradient tracking for evaluation
    with torch.no_grad():
        for images, disc_masks, cup_masks in loader:
            images = images.to(device)
            disc_masks = disc_masks.to(device)
            cup_masks = cup_masks.to(device)

            # Getting model predictions (2 channels: [disc, cup])
            pred = model(images)
            disc_pred = pred[:, 0:1, :, :]  # First channel: disc
            cup_pred = pred[:, 1:2, :, :]   # Second channel: cup

            # Adding Dice scores to totals
            dice_disc_total += dice_coeff(disc_pred, disc_masks)
            dice_cup_total += dice_coeff(cup_pred, cup_masks)

    # Average Dice scores
    num_samples = len(loader)
    avg_dice_disc = dice_disc_total / num_samples
    avg_dice_cup = dice_cup_total / num_samples

    # Return final values
    return float(avg_dice_disc), float(avg_dice_cup)


# Run directly as a script to print results
if __name__ == "__main__":
    disc, cup = evaluate_model()
    print(f"Avg Dice Coefficient (Disc): {disc:.4f}")
    print(f"Avg Dice Coefficient (Cup):  {cup:.4f}")
