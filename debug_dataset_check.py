# debug_dataset_check.py
import os

img_dir = "partitioned_randomly/images/train"
disc_dir = "partitioned_randomly/masks_disc/train"
cup_dir = "partitioned_randomly/masks_cup/train"

valid = []
for fname in os.listdir(img_dir):
    if fname.endswith(".png"):
        base = fname.replace(".png", "")
        disc_mask = base + "-Disc-T.png"
        cup_mask = base + "-Cup-T.png"
        if os.path.exists(os.path.join(disc_dir, disc_mask)) and os.path.exists(os.path.join(cup_dir, cup_mask)):
            valid.append(base)

print("Matched image + disc + cup samples:", len(valid))
print("Example base names:", valid[:5])
