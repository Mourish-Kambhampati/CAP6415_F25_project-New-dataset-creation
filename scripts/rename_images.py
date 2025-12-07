import os
from pathlib import Path

def rename_split(split_path_images, split_path_labels, padding=4):
    img_dir = Path(split_path_images)
    lbl_dir = Path(split_path_labels)

    images = sorted([p for p in img_dir.iterdir() if p.is_file()])
    print(f"Found {len(images)} images in {img_dir}")

    for idx, img_path in enumerate(images):
        new_name = f"{idx:0{padding}d}{img_path.suffix.lower()}"
        new_label_name = f"{idx:0{padding}d}.txt"

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            print(f"⚠ Missing label for: {img_path.name}")
            continue

        # Temp names to avoid overwrite issues
        temp_img = img_dir / f"tmp_{idx}{img_path.suffix}"
        temp_lbl = lbl_dir / f"tmp_{idx}.txt"

        img_path.rename(temp_img)
        lbl_path.rename(temp_lbl)

        temp_img.rename(img_dir / new_name)
        temp_lbl.rename(lbl_dir / new_label_name)

    print(f"✅ Renamed {len(images)} files in {img_dir}")


def main():
    base = "dataset"   # already inside project folder after cd

    for split in ["train", "val", "test"]:
        img_dir = f"{base}/{split}/images"
        lbl_dir = f"{base}/{split}/labels"
        rename_split(img_dir, lbl_dir, padding=4)

    print("\n🎉 All dataset splits renamed successfully!")


if __name__ == "__main__":
    main()
