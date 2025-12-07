import os
import shutil
import random
from glob import glob

# Base project folder (after cloning)
base = "CAP6415_F25_project-New-dataset-creation"

# Output YOLO dataset folder
output_root = "dataset"

# YOLO folders
train_images = os.path.join(output_root, "train/images")
train_labels = os.path.join(output_root, "train/labels")
val_images   = os.path.join(output_root, "val/images")
val_labels   = os.path.join(output_root, "val/labels")
test_images  = os.path.join(output_root, "test/images")
test_labels  = os.path.join(output_root, "test/labels")

for d in [train_images, train_labels, val_images, val_labels, test_images, test_labels]:
    os.makedirs(d, exist_ok=True)

# Load pairs of image + label
def load_pairs(img_dir, lbl_dir):
    img_paths = glob(os.path.join(img_dir, "*.*"))
    pairs = []

    for img_path in img_paths:
        name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, name + ".txt")

        if os.path.exists(lbl_path):
            pairs.append((img_path, lbl_path))
        else:
            print(f"⚠ Missing label for: {img_path}")

    return pairs

# Collect all lion + tiger samples
pairs = []
pairs += load_pairs(os.path.join(base, "dataset_images/lion/images"),
                    os.path.join(base, "dataset_labels/lion/labels"))

pairs += load_pairs(os.path.join(base, "dataset_images/tiger/images"),
                    os.path.join(base, "dataset_labels/tiger/labels"))

print(f"\nTotal samples: {len(pairs)}")

# 70/15/15 split
random.shuffle(pairs)
n = len(pairs)

train_end = int(n * 0.70)
val_end   = train_end + int(n * 0.15)

train_pairs = pairs[:train_end]
val_pairs   = pairs[train_end:val_end]
test_pairs  = pairs[val_end:]

print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}  Test: {len(test_pairs)}")

# Copy files
def copy_pairs(pairs, img_dst, lbl_dst):
    for img_path, lbl_path in pairs:
        fname = os.path.basename(img_path)
        name = os.path.splitext(fname)[0]

        shutil.copy(img_path, os.path.join(img_dst, fname))
        shutil.copy(lbl_path, os.path.join(lbl_dst, name + ".txt"))

copy_pairs(train_pairs, train_images, train_labels)
copy_pairs(val_pairs,   val_images,   val_labels)
copy_pairs(test_pairs,  test_images,  test_labels)

print("\n Dataset ready! Saved in 'dataset/'")
