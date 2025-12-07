import os
import zipfile
import shutil
import random
from glob import glob

# -------------------------------------------------------------
# Helper: extract zip to a folder
# -------------------------------------------------------------
def extract_zip(zip_path, extract_to):
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path} into {extract_to} ...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_to)
    else:
        print(f" ZIP not found: {zip_path}")


# -------------------------------------------------------------
# Helper: change class IDs inside label files
# -------------------------------------------------------------
def change_class_ids(label_dir, old_id=0, new_id=1):
    print(f"Converting class {old_id} → {new_id} inside {label_dir}")
    
    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue
        
        path = os.path.join(label_dir, file)
        new_lines = []
        
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    if cls == old_id:
                        parts[0] = str(new_id)
                new_lines.append(" ".join(parts))
        
        with open(path, "w") as f:
            f.write("\n".join(new_lines))


# -------------------------------------------------------------
# 1. Extract lion.zip and tiger.zip into separate folders
# -------------------------------------------------------------
extract_zip("lion.zip", "lion_extracted")
extract_zip("tiger.zip", "tiger_extracted")

# Paths inside extracted folders (Roboflow format)
LION_IMG = "lion_extracted/train/images"
LION_LBL = "lion_extracted/train/labels"

TIGER_IMG = "tiger_extracted/train/images"
TIGER_LBL = "tiger_extracted/train/labels"

# Convert tiger labels from class 0 → 1
if os.path.exists(TIGER_LBL):
    change_class_ids(TIGER_LBL, old_id=0, new_id=1)


# -------------------------------------------------------------
# 2. Output YOLO dataset folder
# -------------------------------------------------------------
output_root = "dataset"

train_images = os.path.join(output_root, "train/images")
train_labels = os.path.join(output_root, "train/labels")
val_images   = os.path.join(output_root, "val/images")
val_labels   = os.path.join(output_root, "val/labels")
test_images  = os.path.join(output_root, "test/images")
test_labels  = os.path.join(output_root, "test/labels")

for d in [train_images, train_labels, val_images, val_labels, test_images, test_labels]:
    os.makedirs(d, exist_ok=True)


# -------------------------------------------------------------
# 3. Load pairs of image + label
# -------------------------------------------------------------
def load_pairs(img_dir, lbl_dir):
    img_paths = glob(os.path.join(img_dir, "*.*"))
    pairs = []
    
    for img_path in img_paths:
        name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, name + ".txt")
        
        if os.path.exists(lbl_path):
            pairs.append((img_path, lbl_path))
        else:
            print(f" Missing label for: {img_path}")
    
    return pairs


# Load lion (class 0) and tiger (class 1)
pairs = []
pairs += load_pairs(LION_IMG, LION_LBL)
pairs += load_pairs(TIGER_IMG, TIGER_LBL)

print(f"\nTotal samples found: {len(pairs)}")


# -------------------------------------------------------------
# 4. 70/15/15 split
# -------------------------------------------------------------
random.shuffle(pairs)
n = len(pairs)

train_end = int(n * 0.70)
val_end   = train_end + int(n * 0.15)

train_pairs = pairs[:train_end]
val_pairs   = pairs[train_end:val_end]
test_pairs  = pairs[val_end:]

print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}  Test: {len(test_pairs)}")


# -------------------------------------------------------------
# 5. Copy into YOLO dataset folders
# -------------------------------------------------------------
def copy_pairs(pairs, img_dst, lbl_dst):
    for img_path, lbl_path in pairs:
        fname = os.path.basename(img_path)
        name  = os.path.splitext(fname)[0]
        
        shutil.copy(img_path, os.path.join(img_dst, fname))
        shutil.copy(lbl_path, os.path.join(lbl_dst, name + ".txt"))


copy_pairs(train_pairs, train_images, train_labels)
copy_pairs(val_pairs, val_images, val_labels)
copy_pairs(test_pairs, test_images, test_labels)

print("\n Dataset processed successfully! Saved in 'dataset/'")
