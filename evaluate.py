def evaluate_model():
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import importlib.util
    from model import UNet

    # Loading dataset.py dynamically
    spec = importlib.util.spec_from_file_location("dataset", "dataset.py")
    dataset_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset_module)
    FundusSegmentationDataset = dataset_module.FundusSegmentationDataset

    # Dice function
    def dice_coeff(pred, target, epsilon=1e-6):
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2. * intersection + epsilon) / (union + epsilon)
        return dice.mean()

    image_dir = "partitioned_randomly/images/test"
    disc_mask_dir = "partitioned_randomly/masks_disc/test"
    cup_mask_dir = "partitioned_randomly/masks_cup/test"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = FundusSegmentationDataset(image_dir, disc_mask_dir, cup_mask_dir, transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load("segmentation_model.pth"))
    model.eval()

    dice_disc_total = 0.0
    dice_cup_total = 0.0

    with torch.no_grad():
        for images, disc_masks, cup_masks in loader:
            images = images.to(device)
            disc_masks = disc_masks.to(device)
            cup_masks = cup_masks.to(device)

            pred = model(images)
            disc_pred = pred[:, 0:1, :, :]
            cup_pred = pred[:, 1:2, :, :]

            dice_disc_total += dice_coeff(disc_pred, disc_masks)
            dice_cup_total += dice_coeff(cup_pred, cup_masks)

    num_samples = len(loader)
    avg_dice_disc = dice_disc_total / num_samples
    avg_dice_cup = dice_cup_total / num_samples

    return float(avg_dice_disc), float(avg_dice_cup)


#Uncomment this if you want to run from terminal too
if __name__ == "__main__":
    disc, cup = evaluate_model()
    print(f"Avg Dice Coefficient (Disc): {disc:.4f}")
    print(f"Avg Dice Coefficient (Cup):  {cup:.4f}")
