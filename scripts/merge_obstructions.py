import os
import shutil
import zipfile
import yaml
import cv2
import pandas as pd
from pathlib import Path

# =====================================================================
# Configuration
# =====================================================================
WORKSPACE = Path(r"c:\Users\soumy\OneDrive\Desktop\SIH-21624")
OBSTRUCTION_MASTER = WORKSPACE / "OBSTRUCTION_MASTER"
TEMP_DIR = WORKSPACE / "temp_obstruction_data"

ZIPS = {
    "trafficcone": WORKSPACE / "cone-traffic.v1i.yolov11.zip",
    "rock": WORKSPACE / "rocks.v1i.yolov11.zip",
    "debris": WORKSPACE / "Road Debris.v2i.yolov11.zip",
    "fallentree": WORKSPACE / "Fallen Tree.v3i.yolov11.zip"
}

FINAL_CLASSES = {
    0: "TrafficCone",
    1: "Rock",
    2: "RoadDebris",
    3: "FallenTree"
}

# =====================================================================
# Directory Setup
# =====================================================================
if OBSTRUCTION_MASTER.exists():
    print(f"WARNING: {OBSTRUCTION_MASTER} already exists. Cleaning up...")
    shutil.rmtree(OBSTRUCTION_MASTER)

if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)

for split in ["train", "val", "test"]:
    (OBSTRUCTION_MASTER / "images" / split).mkdir(parents=True, exist_ok=True)
    (OBSTRUCTION_MASTER / "labels" / split).mkdir(parents=True, exist_ok=True)

META_DIR = OBSTRUCTION_MASTER / "metadata"
META_DIR.mkdir(exist_ok=True)

# Tracking Lists for Reports
class_mapping_report = []
excluded_report = []
duplicate_report = []
conversion_report = []

img_stats = {"train": 0, "val": 0, "test": 0}
instance_stats = {0: 0, 1: 0, 2: 0, 3: 0}
source_stats = {"trafficcone": 0, "rock": 0, "debris": 0, "fallentree": 0}

seen_images = set()

# =====================================================================
# Extraction & Inspection
# =====================================================================
print("Extracting datasets...")
for source_prefix, zip_path in ZIPS.items():
    if zip_path.exists():
        extract_path = TEMP_DIR / source_prefix
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    else:
        print(f"ERROR: Could not find {zip_path}")

# =====================================================================
# Dynamic Class Mapping Logic
# =====================================================================
def get_final_class_mapping(source_classes):
    """Maps YOLO yaml class lists to our final 0-3 ontology."""
    mapping = {}
    for i, name in enumerate(source_classes):
        nl = name.lower()
        if "cone" in nl: mapping[i] = 0
        elif "rock" in nl: mapping[i] = 1
        elif "object" in nl or "debris" in nl: mapping[i] = 2
        elif "fallen" in nl or "tree" in nl: mapping[i] = 3
    return mapping

# =====================================================================
# Processing Engine
# =====================================================================
print("\nProcessing Datasets...")

for source_prefix in ZIPS.keys():
    source_root = TEMP_DIR / source_prefix
    if not source_root.exists(): continue
    
    yaml_path = list(source_root.glob("data.yaml"))
    if not yaml_path:
        print(f"Skipping {source_prefix}: No data.yaml found.")
        continue
        
    with open(yaml_path[0], 'r') as f:
        data_cfg = yaml.safe_load(f)
        
    source_names = data_cfg.get('names', [])
    if isinstance(source_names, dict):
        source_names = [source_names[i] for i in range(len(source_names))]
        
    class_map = get_final_class_mapping(source_names)
    
    # Log mappings and exclusions
    for i, name in enumerate(source_names):
        if i in class_map:
            final_id = class_map[i]
            class_mapping_report.append({
                "source_dataset": source_prefix,
                "source_class_name": name,
                "source_class_id": i,
                "final_class_name": FINAL_CLASSES[final_id],
                "final_class_id": final_id,
                "action": "keep",
                "notes": "auto-mapped"
            })
        else:
            excluded_report.append({
                "source_dataset": source_prefix,
                "source_class": name,
                "source_class_id": i,
                "reason": "Not in OBSTRUCTION_MASTER ontology",
                "action": "excluded"
            })

    # Roboflow usually exports train/valid/test folders
    splits_to_process = {"train": "train", "valid": "val", "test": "test"}
    
    for src_split, dst_split in splits_to_process.items():
        img_dir = source_root / src_split / "images"
        lbl_dir = source_root / src_split / "labels"
        
        if not img_dir.exists() or not lbl_dir.exists():
            continue
            
        for img_path in img_dir.glob("*.jpg"):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists(): continue
            
            # Simple duplicate check by filename (ignoring content hashing for speed)
            if img_path.name in seen_images:
                duplicate_report.append({
                    "image": img_path.name,
                    "source": source_prefix,
                    "duplicate_type": "filename collision",
                    "action": "renamed with prefix"
                })
            
            new_name = f"{source_prefix}_{img_path.name}"
            seen_images.add(new_name)
            
            # Read and filter labels
            valid_annotations = []
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        c_id = int(parts[0])
                        if c_id in class_map:
                            final_id = class_map[c_id]
                            # Coordinates
                            cx, cy, w, h = map(float, parts[1:5])
                            assert 0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1, "Coordinates out of bounds"
                            
                            valid_annotations.append(f"{final_id} {cx} {cy} {w} {h}")
                            instance_stats[final_id] += 1
                            
                            conversion_report.append({
                                "source_dataset": source_prefix,
                                "source_image": img_path.name,
                                "source_class": source_names[c_id],
                                "source_class_id": c_id,
                                "final_class": FINAL_CLASSES[final_id],
                                "final_class_id": final_id,
                                "split": dst_split,
                                "action": "converted",
                                "notes": ""
                            })
            
            # Only copy image if it has valid surviving annotations
            if valid_annotations:
                # Save new label
                new_lbl_path = OBSTRUCTION_MASTER / "labels" / dst_split / f"{source_prefix}_{img_path.stem}.txt"
                with open(new_lbl_path, 'w') as f:
                    f.write("\n".join(valid_annotations) + "\n")
                    
                # Copy image
                new_img_path = OBSTRUCTION_MASTER / "images" / dst_split / new_name
                shutil.copy(img_path, new_img_path)
                
                img_stats[dst_split] += 1
                source_stats[source_prefix] += 1

# =====================================================================
# Generate Data.yaml
# =====================================================================
yaml_content = f"""path: {OBSTRUCTION_MASTER.resolve()}
train: images/train
val: images/val
test: images/test

names:
  0: TrafficCone
  1: Rock
  2: RoadDebris
  3: FallenTree
"""
with open(OBSTRUCTION_MASTER / "data.yaml", 'w') as f:
    f.write(yaml_content)

# =====================================================================
# Generate Reports
# =====================================================================
pd.DataFrame(class_mapping_report).to_csv(META_DIR / "class_mapping.csv", index=False)
pd.DataFrame(excluded_report).to_csv(META_DIR / "excluded_classes_report.csv", index=False)
pd.DataFrame(duplicate_report).to_csv(META_DIR / "duplicate_report.csv", index=False)
pd.DataFrame(conversion_report).to_csv(META_DIR / "conversion_report.csv", index=False)

# Clean up temp
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)

# =====================================================================
# Final Print Summary
# =====================================================================
print("\n" + "="*60)
print("OBSTRUCTION_MASTER SUMMARY")
print("="*60)
print(f"Images:\nTrain: {img_stats['train']}\nValidation: {img_stats['val']}\nTest: {img_stats['test']}\n")
print(f"Instances:")
for k, v in FINAL_CLASSES.items():
    print(f"{v}: {instance_stats[k]}")
print("\nSource contribution:")
for k, v in source_stats.items():
    print(f"{k.capitalize()}: {v}")

print("\nFinal classes:")
for k, v in FINAL_CLASSES.items():
    print(f"{k} {v}")
    
total_images = sum(img_stats.values())
if total_images > 0:
    print("\nDataset ready for YOLO11n: YES")
else:
    print("\nDataset ready for YOLO11n: NO (0 images generated)")
print("="*60)
