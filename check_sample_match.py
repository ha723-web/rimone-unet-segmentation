import os

image_dir = "partitioned_randomly/images/train"
disc_dir = "partitioned_randomly/masks_disc/train"
cup_dir = "partitioned_randomly/masks_cup/train"

test_base = "r1_Im001"
image_path = os.path.join(image_dir, test_base + ".png")
disc_path = os.path.join(disc_dir, test_base + "-1-Disc-T.png")
cup_path = os.path.join(cup_dir, test_base + "-1-Cup-T.png")

print("🔍 Checking paths for one sample:")
print(f"Image path: {image_path} → Exists: {os.path.exists(image_path)}")
print(f"Disc path:  {disc_path} → Exists: {os.path.exists(disc_path)}")
print(f"Cup path:   {cup_path} → Exists: {os.path.exists(cup_path)}")
