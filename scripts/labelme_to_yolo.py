import json
import random
from pathlib import Path
from shutil import copy2
from PIL import Image

# ================= CONFIG =================
SRC_IMG_DIR = Path('images_raw')     # your folder with images
SRC_LABEL_DIR = Path('labels_raw')   # your folder with JSONs
DST_ROOT = Path('data')              # YOLO dataset root

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
RANDOM_SEED = 42

# ======== OUTPUT FOLDER STRUCTURE =========
IMG_TRAIN = DST_ROOT / 'images' / 'train'
IMG_VAL   = DST_ROOT / 'images' / 'val'
IMG_TEST  = DST_ROOT / 'images' / 'test'

LBL_TRAIN = DST_ROOT / 'labels' / 'train'
LBL_VAL   = DST_ROOT / 'labels' / 'val'
LBL_TEST  = DST_ROOT / 'labels' / 'test'

# Create folders
for d in [IMG_TRAIN, IMG_VAL, IMG_TEST, LBL_TRAIN, LBL_VAL, LBL_TEST]:
    d.mkdir(parents=True, exist_ok=True)


# ================= CLASS FINDER =================
def find_classes(label_dir):
    classes = set()
    for f in label_dir.glob("*.json"):
        with open(f, "r", encoding="utf-8") as jf:
            data = json.load(jf)
            for s in data.get("shapes", []):
                if "label" in s:
                    classes.add(s["label"])
    return sorted(classes)


# ================= JSON → YOLO CONVERSION =================
def convert_shape_to_yolo(shape, img_w, img_h, class_map):
    if shape.get("shape_type") != "rectangle":
        return None

    label = shape["label"]
    pts = shape["points"]
    x1, y1 = pts[0]
    x2, y2 = pts[1]

    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    x_c = (x_min + x_max) / 2.0 / img_w
    y_c = (y_min + y_max) / 2.0 / img_h
    w   = (x_max - x_min) / img_w
    h   = (y_max - y_min) / img_h

    cls = class_map[label]

    return f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"


# ================= PROCESS ONE FILE =================
def process_one(json_path, class_map):
    with open(json_path, "r", encoding="utf-8") as jf:
        data = json.load(jf)

    img_path = SRC_IMG_DIR / Path(data["imagePath"]).name

    if not img_path.exists():
        return None, None, f"Image not found: {img_path}"

    try:
        with Image.open(img_path) as im:
            img_w, img_h = im.size
    except Exception as e:
        return None, None, f"Cannot read image {img_path}: {e}"

    yolo_lines = []
    for shape in data.get("shapes", []):
        y = convert_shape_to_yolo(shape, img_w, img_h, class_map)
        if y:
            yolo_lines.append(y)

    return img_path, yolo_lines, None


# ================= MAIN =================
def main():
    random.seed(RANDOM_SEED)

    json_files = sorted(SRC_LABEL_DIR.glob("*.json"))
    if not json_files:
        print("No JSON files found in", SRC_LABEL_DIR)
        return

    # Find classes
    classes = find_classes(SRC_LABEL_DIR)
    class_map = {name: i for i, name in enumerate(classes)}
    print("Classes:", class_map)

    # Split dataset
    indices = list(range(len(json_files)))
    random.shuffle(indices)

    n = len(indices)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_idx = set(indices[:n_train])
    val_idx   = set(indices[n_train:n_train+n_val])
    test_idx  = set(indices[n_train+n_val:])

    stats = {"train": 0, "val": 0, "test": 0, "skipped": 0, "errors": []}

    for i, json_path in enumerate(json_files):
        img_path, yolo_lines, err = process_one(json_path, class_map)

        if err:
            stats["skipped"] += 1
            stats["errors"].append(f"{json_path.name}: {err}")
            continue

        base = json_path.stem

        # Assign split
        if i in train_idx:
            img_dst = IMG_TRAIN / img_path.name
            lbl_dst = LBL_TRAIN / f"{base}.txt"
            stats["train"] += 1
        elif i in val_idx:
            img_dst = IMG_VAL / img_path.name
            lbl_dst = LBL_VAL / f"{base}.txt"
            stats["val"] += 1
        else:
            img_dst = IMG_TEST / img_path.name
            lbl_dst = LBL_TEST / f"{base}.txt"
            stats["test"] += 1

        # Copy image + write label
        copy2(img_path, img_dst)
        with open(lbl_dst, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

    # Write data.yaml for YOLO
    yaml_text = (
        f"train: {IMG_TRAIN.as_posix()}\n"
        f"val:   {IMG_VAL.as_posix()}\n"
        f"test:  {IMG_TEST.as_posix()}\n"
        f"nc: {len(classes)}\n"
        f"names: {classes}\n"
    )

    with open(DST_ROOT / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_text)

    print(f"Done.\nTrain: {stats['train']}, Val: {stats['val']}, Test: {stats['test']}, Skipped: {stats['skipped']}")
    print("Generated data.yaml:\n", yaml_text)


if __name__ == "__main__":
    main()
