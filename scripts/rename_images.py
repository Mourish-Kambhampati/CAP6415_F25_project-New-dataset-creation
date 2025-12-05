from pathlib import Path
from uuid import uuid4

def main():
    # Your folder to rename
    FOLDER = Path("dataset_images/lion")    # change this
    START = 0
    PADDING = 4
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

    # Collect files
    files = [p for p in FOLDER.iterdir() if p.suffix.lower() in EXTS]
    files.sort(key=lambda p: p.name.lower())

    if not files:
        print("No image files found.")
        return

    # Check capacity
    total_needed = START + len(files) - 1
    max_allowed = 10**PADDING - 1
    if total_needed > max_allowed:
        print("ERROR: Not enough padding for renaming.")
        return

    # Build mapping
    mapping = {}
    for idx, p in enumerate(files, start=START):
        new_name = f"{idx:0{PADDING}d}{p.suffix.lower()}"
        mapping[p] = FOLDER / new_name

    # Temp rename to avoid collisions
    uid = uuid4().hex[:8]
    temp_map = {}

    # Step 1 — rename everything to temp names
    for i, (old, final) in enumerate(mapping.items()):
        tmp = FOLDER / f".tmp_{uid}_{i}{old.suffix.lower()}"
        old.rename(tmp)
        temp_map[tmp] = final

    # Step 2 — rename temps to final names
    for tmp, final in temp_map.items():
        tmp.rename(final)

    print(f"Renamed {len(files)} files in {FOLDER} successfully.")


if __name__ == "__main__":
    main()
