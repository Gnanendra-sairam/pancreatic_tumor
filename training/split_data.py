import os
import shutil
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- RAW IMAGE FOLDERS (your dataset) ----
RAW_IMAGES_DIR = os.path.join(BASE_DIR, "raw_data", "images")
SOURCE_NORMAL = os.path.join(RAW_IMAGES_DIR, "negative")  # no tumor
SOURCE_TUMOR  = os.path.join(RAW_IMAGES_DIR, "positive")  # tumor present

# ---- TARGET FOLDERS (used by train_vgg_pancreas.py) ----
TARGET_BASE = os.path.join(BASE_DIR, "data")

SPLITS = ["train", "val", "test"]
CLASSES = ["normal", "tumor"]

ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def ensure_dirs():
    for split in SPLITS:
        for cls in CLASSES:
            path = os.path.join(TARGET_BASE, split, cls)
            os.makedirs(path, exist_ok=True)


def split_and_copy(source_dir, class_name, train_ratio=0.7, val_ratio=0.15):
    print(f"\n[INFO] Reading from: {source_dir}")
    if not os.path.exists(source_dir):
        print(f"[ERROR] Folder does NOT exist: {source_dir}")
        return

    files = [
        f for f in os.listdir(source_dir)
        if f.lower().endswith(ALLOWED_EXT)
    ]

    print(f"[INFO] Found {len(files)} images in {source_dir}")
    if not files:
        print(f"[WARN] No images with extensions {ALLOWED_EXT} in: {source_dir}")
        return

    random.shuffle(files)

    n = len(files)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    n_test = n - n_train - n_val

    datasets = [
        ("train", files[:n_train]),
        ("val",   files[n_train:n_train + n_val]),
        ("test",  files[n_train + n_val:])
    ]

    for split_name, file_list in datasets:
        for fname in file_list:
            src = os.path.join(source_dir, fname)
            dst = os.path.join(TARGET_BASE, split_name, class_name, fname)
            shutil.copy2(src, dst)

    print(f"[DONE] {class_name}: {n_train} train, {n_val} val, {n_test} test (total {n})")


if __name__ == "__main__":
    random.seed(42)
    ensure_dirs()

    print("Processing NORMAL (negative) images...")
    split_and_copy(SOURCE_NORMAL, "normal")

    print("\nProcessing TUMOR (positive) images...")
    split_and_copy(SOURCE_TUMOR, "tumor")

    print("\nAll done! Check the 'data' folder.")
