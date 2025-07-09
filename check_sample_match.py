# ✅ Import the os module to work with file paths
import os

# 📂 Define the paths where training images and their corresponding masks are stored
image_dir = "partitioned_randomly/images/train"        # Folder containing the original fundus images
disc_dir = "partitioned_randomly/masks_disc/train"     # Folder containing optic disc masks
cup_dir = "partitioned_randomly/masks_cup/train"       # Folder containing optic cup masks

# 🧪 Select a sample image base name (without .png or -1-Disc-T.png suffix)
test_base = "r1_Im001"  # This is the core filename used to locate the image and its two masks

# 🛠️ Construct the full path for the image and corresponding disc & cup masks using the expected naming pattern
image_path = os.path.join(image_dir, test_base + ".png")
disc_path = os.path.join(disc_dir, test_base + "-1-Disc-T.png")
cup_path = os.path.join(cup_dir, test_base + "-1-Cup-T.png")

# 🧾 Print out the constructed file paths and whether each file exists on disk
print("🔍 Checking paths for one sample:")
print(f"Image path: {image_path} → Exists: {os.path.exists(image_path)}")
print(f"Disc path:  {disc_path} → Exists: {os.path.exists(disc_path)}")
print(f"Cup path:   {cup_path} → Exists: {os.path.exists(cup_path)}")
