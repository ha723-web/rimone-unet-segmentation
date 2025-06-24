import os
import shutil

root = "partitioned_randomly"
img_dirs = {
    "train": os.path.join(root, "images/train"),
    "test": os.path.join(root, "images/test")
}

ref_dirs = [
    os.path.join(root, "Data/normal"),
    os.path.join(root, "Data/glaucoma")
]

dest_dirs = {
    "train": {
        "disc": os.path.join(root, "masks_disc/train"),
        "cup": os.path.join(root, "masks_cup/train")
    },
    "test": {
        "disc": os.path.join(root, "masks_disc/test"),
        "cup": os.path.join(root, "masks_cup/test")
    }
}

for split in dest_dirs:
    for mtype in dest_dirs[split]:
        os.makedirs(dest_dirs[split][mtype], exist_ok=True)

def get_basenames(image_dir):
    return {f.split(".")[0] for f in os.listdir(image_dir) if f.endswith(".png")}

train_bases = get_basenames(img_dirs["train"])
test_bases  = get_basenames(img_dirs["test"])


copied_train, copied_test = 0, 0

for ref in ref_dirs:
    for fname in os.listdir(ref):
        if not fname.endswith(".png"):
            continue

        if "-Disc-T.png" in fname:
            mtype = "disc"
        elif "-Cup-T.png" in fname:
            mtype = "cup"
        else:
            continue

        base = fname.split("-")[0]  
        src_path = os.path.join(ref, fname)

        if base in train_bases:
            shutil.copy(src_path, dest_dirs["train"][mtype])
            copied_train += 1
        elif base in test_bases:
            shutil.copy(src_path, dest_dirs["test"][mtype])
            copied_test += 1

print(f"Copied {copied_train} masks to train folders.")
print(f"Copied {copied_test} masks to test folders.")
