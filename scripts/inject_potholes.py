import os
import shutil
import zipfile
from pathlib import Path

WORKSPACE = Path(r"c:\Users\soumy\OneDrive\Desktop\SIH-21624")
MASTER_DIR = WORKSPACE / "SIH26124_MASTER"
ZIP_PATH = WORKSPACE / "archive (7).zip"
TEMP_DIR = WORKSPACE / "temp_pothole_inject"

print("Starting Pothole Injection Process...")

if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
TEMP_DIR.mkdir()

print(f"Extracting {ZIP_PATH.name}...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(TEMP_DIR)

# In the SIH26124_MASTER ontology, Pothole is exactly class 3
TARGET_CLASS = 3

# Mapping Roboflow splits to our master dataset splits
splits = {"train": "train", "valid": "val", "test": "test"}

img_count = 0
lbl_count = 0

print(f"Relabeling and Merging into {MASTER_DIR.name}...")

for src_split, dst_split in splits.items():
    img_dir = TEMP_DIR / src_split / "images"
    lbl_dir = TEMP_DIR / src_split / "labels"
    
    if not img_dir.exists() or not lbl_dir.exists():
        continue
        
    for img_path in img_dir.glob("*.*"):
        if not img_path.name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
            
        # Add 'xtrapothole_' prefix to prevent overwriting existing dataset images
        new_name = f"xtrapothole_{img_path.name}"
        new_lbl_name = f"xtrapothole_{img_path.stem}.txt"
        
        valid_lines = []
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    c_id = int(parts[0])
                    # In the source dataset, class 0 is pothole. 
                    if c_id == 0:
                        parts[0] = str(TARGET_CLASS)
                        valid_lines.append(" ".join(parts))
                        lbl_count += 1
        
        # Only copy if there is actually a valid pothole label in the image
        if valid_lines:
            dst_lbl_path = MASTER_DIR / "labels" / dst_split / new_lbl_name
            with open(dst_lbl_path, 'w') as f:
                f.write("\n".join(valid_lines) + "\n")
                
            dst_img_path = MASTER_DIR / "images" / dst_split / new_name
            shutil.copy(img_path, dst_img_path)
            
            img_count += 1

print("\n" + "="*50)
print("POTHOLE INJECTION COMPLETE")
print("="*50)
print(f"Successfully injected {img_count} new images.")
print(f"Successfully relabeled {lbl_count} individual pothole annotations to Class 3.")

# Cleanup
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
print("Temporary files cleaned up.")
