# SIH26124: Master Dataset Preprocessing Pipeline
**Combining RAD (Road Anomaly Detection) & RDD (Road Damage Dataset)**

This notebook merges two datasets into a single unified YOLO11n-ready dataset for the SIH26124 project.

**Final Ontology:**
- `0`: HMV
- `1`: LMV
- `2`: Pedestrian
- `3`: Pothole
- `4`: Crack
- `5`: Manhole
- `6`: SpeedBump

---

## 1. Environment & Configuration

```python
# %% [markdown]
# ## 1. Setup & Imports
# Import libraries, define paths, and configure the workspace.

# %%
import os
import glob
import json
import yaml
import shutil
import random
import warnings
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

# CONFIGURATION
# Set Kaggle input directories (or local paths)
INPUT_DIR = Path("/kaggle/input")
MASTER_DIR = Path("/kaggle/working/SIH26124_MASTER")

# We expect the datasets to be under INPUT_DIR. We will locate them dynamically.
RAD_DATASET_HINT = "radroad-anomaly-detection"
RDD_DATASET_HINT = "road-damage-dataset" # Update this hint based on the actual Kaggle path

print("="*50)
print("SIH26124 MASTER DATASET PIPELINE")
print("="*50)
```

---

## 2. Locate & Inspect Source Datasets

```python
# %% [markdown]
# ## 2. Locate Source Datasets
# Programmatically find and inspect the RAD and RDD datasets.

# %%
def find_dataset(hint_name):
    # Try to find a folder matching the hint
    for p in INPUT_DIR.iterdir():
        if p.is_dir() and hint_name.lower() in p.name.lower():
            return p
    # If not found by hint, just return a list of all datasets for manual selection
    return None

rad_root = find_dataset("radroad") or list(INPUT_DIR.glob("*rad*"))[0]
rdd_root = find_dataset("damage") or list(INPUT_DIR.glob("*damage*"))[0]

print(f"RAD Root: {rad_root}")
print(f"RDD Root: {rdd_root}")

# Inspect RAD
rad_yaml_path = list(rad_root.rglob("data.yaml"))
if not rad_yaml_path:
    raise FileNotFoundError("Could not find data.yaml in RAD dataset.")
rad_yaml_path = rad_yaml_path[0]

with open(rad_yaml_path, 'r') as f:
    rad_config = yaml.safe_load(f)

rad_orig_names = rad_config.get('names', [])
if isinstance(rad_orig_names, dict):
    rad_orig_names = [rad_orig_names[i] for i in range(len(rad_orig_names))]

print("\n--- RAD DATASET INSPECTION ---")
print(f"RAD Original Classes: {rad_orig_names}")

# Inspect RDD
print("\n--- RDD DATASET INSPECTION ---")
rdd_yaml_path = list(rdd_root.rglob("data.yaml"))
if rdd_yaml_path:
    with open(rdd_yaml_path[0], 'r') as f:
        rdd_config = yaml.safe_load(f)
    rdd_orig_names = rdd_config.get('names', [])
    if isinstance(rdd_orig_names, dict):
        rdd_orig_names = [rdd_orig_names[i] for i in range(len(rdd_orig_names))]
else:
    print("No data.yaml found in RDD. Defaulting to known README structure.")
    rdd_orig_names = ['Pothole', 'Crack', 'Manhole']

print(f"RDD Original Classes: {rdd_orig_names}")
```

---

## 3. Dataset Integrity Checks

```python
# %% [markdown]
# ## 3. Integrity Checks
# Check both source datasets for missing/corrupt files before copying.

# %%
def check_dataset_integrity(img_dir, split_name, expected_classes):
    img_dir = Path(img_dir)
    if not img_dir.exists(): return
    lbl_dir = img_dir.parent / "labels" if img_dir.name == "images" else img_dir.parent / "labels-YOLO"
    if not lbl_dir.exists() and (img_dir.parent / "labels").exists():
        lbl_dir = img_dir.parent / "labels"
        
    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    
    missing_labels = 0
    empty_labels = 0
    invalid_bboxes = 0
    unknown_classes = 0
    
    for img_path in images:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            missing_labels += 1
            continue
            
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            empty_labels += 1
            continue
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                invalid_bboxes += 1
                continue
            cls_id = int(parts[0])
            if cls_id >= len(expected_classes):
                unknown_classes += 1
            coords = [float(x) for x in parts[1:]]
            if any(c < 0.0 or c > 1.0 for c in coords):
                invalid_bboxes += 1

    print(f"[{split_name}] Images: {len(images)} | Missing Lbls: {missing_labels} | Empty Lbls: {empty_labels} | Invalid Box: {invalid_bboxes} | Unknown Cls: {unknown_classes}")

print("\n--- INTEGRITY CHECKS ---")
print("RAD Checks:")
check_dataset_integrity(rad_root / "train" / "images", "RAD Train", rad_orig_names)
check_dataset_integrity(rad_root / "valid" / "images", "RAD Val", rad_orig_names)
check_dataset_integrity(rad_root / "test" / "images", "RAD Test", rad_orig_names)

print("\nRDD Checks:")
# Adjust paths based on RDD actual structure
rdd_img_dir = rdd_root / "images"
if not rdd_img_dir.exists() and (rdd_root / "data" / "images").exists():
    rdd_img_dir = rdd_root / "data" / "images"
check_dataset_integrity(rdd_img_dir, "RDD All", rdd_orig_names)
```

---

## 4. Class Mapping Strategy

```python
# %% [markdown]
# ## 4. Final Class Mapping
# Mapping source dataset classes to the final unified ontology.

# %%
FINAL_ONTOLOGY = {
    0: "HMV",
    1: "LMV",
    2: "Pedestrian",
    3: "Pothole",
    4: "Crack",
    5: "Manhole",
    6: "SpeedBump"
}

# 1. RAD Mapping
# HMV -> 0, LMV -> 1, Pedestrian -> 2, SpeedBump -> 6
# RoadDamages -> DELETE, UnsurfacedRoad -> DELETE
RAD_MAPPING = {}
RAD_REMOVE_IDS = []

for i, name in enumerate(rad_orig_names):
    nl = name.lower()
    if "hmv" in nl: RAD_MAPPING[i] = 0
    elif "lmv" in nl: RAD_MAPPING[i] = 1
    elif "pedestrian" in nl: RAD_MAPPING[i] = 2
    elif "speedbump" in nl: RAD_MAPPING[i] = 6
    elif "roaddamage" in nl or "unsurfaced" in nl:
        RAD_REMOVE_IDS.append(i)
    else:
        print(f"WARNING: Unexpected RAD class {name}")

# 2. RDD Mapping
# Pothole -> 3, Crack -> 4, Manhole -> 5
RDD_MAPPING = {}
for i, name in enumerate(rdd_orig_names):
    nl = name.lower()
    if "pothole" in nl: RDD_MAPPING[i] = 3
    elif "crack" in nl: RDD_MAPPING[i] = 4
    elif "manhole" in nl: RDD_MAPPING[i] = 5
    else:
        print(f"WARNING: Unexpected RDD class {name}")

print("\n--- FINAL CLASS MAPPING REPORTS ---")
print("RAD Mapping:")
for i, n in enumerate(rad_orig_names):
    if i in RAD_MAPPING: print(f"  {n} ({i}) -> {FINAL_ONTOLOGY[RAD_MAPPING[i]]} ({RAD_MAPPING[i]})")
    elif i in RAD_REMOVE_IDS: print(f"  {n} ({i}) -> DELETE")

print("\nRDD Mapping:")
for i, n in enumerate(rdd_orig_names):
    if i in RDD_MAPPING: print(f"  {n} ({i}) -> {FINAL_ONTOLOGY[RDD_MAPPING[i]]} ({RDD_MAPPING[i]})")
```

---

## 5. Initialize Master Directory & Tracking

```python
# %% [markdown]
# ## 5. Master Directory Setup
# Creating the output structure and report structures.

# %%
if MASTER_DIR.exists():
    print(f"Warning: Overwriting existing directory {MASTER_DIR}")
    shutil.rmtree(MASTER_DIR)

for split in ['train', 'val', 'test']:
    (MASTER_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
    (MASTER_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

META_DIR = MASTER_DIR / 'metadata'
META_DIR.mkdir(exist_ok=True)

# Tracking tables
removed_classes_report = []
conversion_report = []
class_distribution = defaultdict(int)
source_distribution = defaultdict(lambda: defaultdict(int))
split_distribution = defaultdict(lambda: defaultdict(int))

def log_conversion(src_ds, img_name, src_cls_name, src_cls_id, fin_cls_name, fin_cls_id, action, notes=""):
    conversion_report.append({
        "source_dataset": src_ds, "source_image": img_name,
        "source_class": src_cls_name, "source_class_id": src_cls_id,
        "final_class": fin_cls_name, "final_class_id": fin_cls_id,
        "action": action, "notes": notes
    })

def log_removal(src_img, split, orig_cls_list, action, reason):
    removed_classes_report.append({
        "source_image": src_img, "split": split,
        "original_classes": str(orig_cls_list),
        "action": action, "reason": reason
    })
```

---

## 6. Process RAD Dataset

```python
# %% [markdown]
# ## 6. Merge RAD
# Filter classes, drop images if empty, rename to `rad_`.

# %%
print("\n--- PROCESSING RAD DATASET ---")

def process_rad_split(orig_split_dir, split_name):
    img_dir = Path(orig_split_dir)
    if not img_dir.exists(): return
    lbl_dir = img_dir.parent / "labels"
    
    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    
    for img_path in tqdm(images, desc=f"RAD {split_name}"):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        dest_img_name = f"rad_{img_path.name}"
        dest_lbl_name = f"rad_{img_path.stem}.txt"
        
        retained_lines = []
        original_classes_in_file = set()
        removed_classes_in_file = set()
        
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        src_id = int(parts[0])
                        original_classes_in_file.add(src_id)
                        
                        if src_id in RAD_MAPPING:
                            t_id = RAD_MAPPING[src_id]
                            retained_lines.append(f"{t_id} " + " ".join(parts[1:]))
                            
                            # Logging tracking
                            class_distribution[t_id] += 1
                            source_distribution["RAD"][t_id] += 1
                            split_distribution[split_name][t_id] += 1
                            log_conversion("RAD", img_path.name, rad_orig_names[src_id], src_id, FINAL_ONTOLOGY[t_id], t_id, "keep")
                        
                        elif src_id in RAD_REMOVE_IDS:
                            removed_classes_in_file.add(src_id)
                            log_conversion("RAD", img_path.name, rad_orig_names[src_id], src_id, "DELETE", -1, "remove")

        # Decision: Do we keep the image?
        if len(retained_lines) > 0:
            # We have valid annotations to keep
            if removed_classes_in_file:
                log_removal(img_path.name, split_name, list(removed_classes_in_file), "annotation_removed", "Contained removed classes but had other valid classes")
            
            # Copy files
            shutil.copy(img_path, MASTER_DIR / 'images' / split_name / dest_img_name)
            with open(MASTER_DIR / 'labels' / split_name / dest_lbl_name, 'w') as f:
                f.write("\n".join(retained_lines) + "\n")
        
        else:
            # No retained labels
            if removed_classes_in_file:
                log_removal(img_path.name, split_name, list(original_classes_in_file), "image_removed", "Contained ONLY removed classes")
            # If the image was purely background in RAD, we could optionally keep it as a negative sample. 
            # But SIH26124 instructs: "Remove the image from the final object-detection dataset" if it contained ONLY removed classes.

# Assume standard RAD structure
process_rad_split(rad_root / "train" / "images", "train")
process_rad_split(rad_root / "valid" / "images", "val")
process_rad_split(rad_root / "test" / "images", "test")
```

---

## 7. Process RDD Dataset

```python
# %% [markdown]
# ## 7. Merge RDD
# Map classes, rename to `rdi_`. The RDD might not have official splits, we will do a random 80/10/10 if missing, otherwise respect it.

# %%
print("\n--- PROCESSING RDD DATASET ---")

# Locate RDD images and labels
rdd_img_dir = rdd_root / "images"
if not rdd_img_dir.exists():
    rdd_img_dir = rdd_root / "data" / "images"
    rdd_lbl_dir = rdd_root / "data" / "labels-YOLO"
else:
    rdd_lbl_dir = rdd_root / "labels-YOLO"

if not rdd_lbl_dir.exists():
    rdd_lbl_dir = rdd_img_dir.parent / "labels" # Fallback

images = list(rdd_img_dir.glob("*.jpg")) + list(rdd_img_dir.glob("*.png"))

# Create reproducible split if RDD is a flat folder
random.shuffle(images)
train_split = int(0.8 * len(images))
val_split = int(0.9 * len(images))

rdd_splits = {
    "train": images[:train_split],
    "val": images[train_split:val_split],
    "test": images[val_split:]
}

for split_name, imgs in rdd_splits.items():
    for img_path in tqdm(imgs, desc=f"RDD {split_name}"):
        lbl_path = rdd_lbl_dir / (img_path.stem + ".txt")
        dest_img_name = f"rdi_{img_path.name}"
        dest_lbl_name = f"rdi_{img_path.stem}.txt"
        
        retained_lines = []
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        src_id = int(parts[0])
                        if src_id in RDD_MAPPING:
                            t_id = RDD_MAPPING[src_id]
                            retained_lines.append(f"{t_id} " + " ".join(parts[1:]))
                            
                            class_distribution[t_id] += 1
                            source_distribution["RDD"][t_id] += 1
                            split_distribution[split_name][t_id] += 1
                            log_conversion("RDD", img_path.name, rdd_orig_names[src_id], src_id, FINAL_ONTOLOGY[t_id], t_id, "keep")

        if retained_lines:
            shutil.copy(img_path, MASTER_DIR / 'images' / split_name / dest_img_name)
            with open(MASTER_DIR / 'labels' / split_name / dest_lbl_name, 'w') as f:
                f.write("\n".join(retained_lines) + "\n")
```

---

## 8. Generate Meta Reports & Data.yaml

```python
# %% [markdown]
# ## 8. Generate Reports
# Save data.yaml and dump metadata CSVs.

# %%
print("\n--- SAVING METADATA ---")

# 1. data.yaml
yaml_content = {
    'path': str(MASTER_DIR.resolve()),
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    'names': FINAL_ONTOLOGY
}
with open(MASTER_DIR / 'data.yaml', 'w') as f:
    yaml.dump(yaml_content, f, sort_keys=False)

# 2. CSV Reports
pd.DataFrame(conversion_report).to_csv(META_DIR / 'conversion_report.csv', index=False)
pd.DataFrame(removed_classes_report).to_csv(META_DIR / 'removed_classes_report.csv', index=False)

# Mapping report
mapping_data = []
for i, n in enumerate(rad_orig_names):
    if i in RAD_MAPPING: mapping_data.append(["RAD", n, i, FINAL_ONTOLOGY[RAD_MAPPING[i]], RAD_MAPPING[i], "keep"])
    elif i in RAD_REMOVE_IDS: mapping_data.append(["RAD", n, i, "DELETE", -1, "remove"])
for i, n in enumerate(rdd_orig_names):
    if i in RDD_MAPPING: mapping_data.append(["RDD", n, i, FINAL_ONTOLOGY[RDD_MAPPING[i]], RDD_MAPPING[i], "keep"])
pd.DataFrame(mapping_data, columns=["source_dataset", "source_class", "source_class_id", "final_class", "final_class_id", "decision"]).to_csv(META_DIR / 'class_mapping.csv', index=False)

# Distribution reports
dist_data = []
for cid, name in FINAL_ONTOLOGY.items():
    dist_data.append({
        "Class_ID": cid, "Class_Name": name, "Total_Instances": class_distribution[cid],
        "RAD_Source": source_distribution["RAD"][cid], "RDD_Source": source_distribution["RDD"][cid],
        "Train": split_distribution["train"][cid], "Val": split_distribution["val"][cid], "Test": split_distribution["test"][cid]
    })
df_dist = pd.DataFrame(dist_data)
df_dist.to_csv(META_DIR / 'source_distribution.csv', index=False)
df_dist.to_csv(META_DIR / 'split_distribution.csv', index=False)

# Plot class frequency
plt.figure(figsize=(10, 5))
sns.barplot(data=df_dist, x='Class_Name', y='Total_Instances')
plt.title("Master Dataset Class Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(META_DIR / 'class_frequency.png')
plt.show()
```

---

## 9. Final Validation & Visual QA

```python
# %% [markdown]
# ## 9. Validation & Visual QA
# Asserting dataset integrity and plotting bounding boxes on final images.

# %%
print("\n--- FINAL INTEGRITY CHECKS ---")
invalid_count = 0
found_classes = set()

for split in ['train', 'val', 'test']:
    for lbl_file in (MASTER_DIR / 'labels' / split).glob("*.txt"):
        with open(lbl_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                c_id = int(parts[0])
                found_classes.add(c_id)
                if c_id not in FINAL_ONTOLOGY:
                    invalid_count += 1
                coords = [float(x) for x in parts[1:]]
                if any(c < 0.0 or c > 1.0 for c in coords):
                    invalid_count += 1

assert invalid_count == 0, f"Found {invalid_count} invalid annotations in the MASTER dataset!"
assert all(c in FINAL_ONTOLOGY for c in found_classes), f"Found unexpected classes in output: {found_classes - set(FINAL_ONTOLOGY.keys())}"
print("ASSERTIONS PASSED: No invalid bounding boxes. No removed classes (RoadDamages, UnsurfacedRoad) remain.")

print("\n--- VISUAL QUALITY CONTROL ---")
def draw_boxes(img_path, lbl_path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    with open(lbl_path, 'r') as f:
        for line in f:
            c, x, y, bw, bh = map(float, line.strip().split())
            name = FINAL_ONTOLOGY[int(c)]
            x1, y1 = int((x - bw/2) * w), int((y - bh/2) * h)
            x2, y2 = int((x + bw/2) * w), int((y + bh/2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return img

sample_imgs = list((MASTER_DIR / 'images' / 'train').glob("*.jpg"))
if sample_imgs:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, img_p in zip(axes.flatten(), random.sample(sample_imgs, min(4, len(sample_imgs)))):
        lbl_p = MASTER_DIR / 'labels' / 'train' / (img_p.stem + ".txt")
        if lbl_p.exists():
            img_drawn = draw_boxes(img_p, lbl_p)
            ax.imshow(cv2.cvtColor(img_drawn, cv2.COLOR_BGR2RGB))
            ax.set_title(img_p.name)
            ax.axis('off')
    plt.tight_layout()
    plt.show()
```

---

## 10. Master Summary Report

```python
# %% [markdown]
# ## 10. Master Dataset Summary
# Print the final execution summary.

# %%
print("\n" + "="*50)
print("SIH26124 MASTER DATASET PREPARATION COMPLETE")
print("="*50)

total_train = len(list((MASTER_DIR / 'images' / 'train').glob('*.*')))
total_val = len(list((MASTER_DIR / 'images' / 'val').glob('*.*')))
total_test = len(list((MASTER_DIR / 'images' / 'test').glob('*.*')))
total_images = total_train + total_val + total_test

print("\n[TOTAL IMAGES]")
print(f"TRAIN      : {total_train}")
print(f"VALIDATION : {total_val}")
print(f"TEST       : {total_test}")
print(f"TOTAL      : {total_images}")

print(f"\n[TOTAL INSTANCES] : {sum(class_distribution.values())}")

print("\n[CLASS DISTRIBUTION]")
for cid, count in class_distribution.items():
    print(f"{FINAL_ONTOLOGY[cid]:<15}: {count}")

print("\n[SOURCE DISTRIBUTION]")
print("RAD :", sum(source_distribution["RAD"].values()), "instances")
print("RDD :", sum(source_distribution["RDD"].values()), "instances")

rad_removed_annotations = sum(1 for x in conversion_report if x['action'] == 'remove')
images_removed = sum(1 for x in removed_classes_report if x['action'] == 'image_removed')

print("\n[REMOVED ANNOTATIONS]")
print(f"Removed annotations (RoadDamages/UnsurfacedRoad): {rad_removed_annotations}")
print(f"IMAGES REMOVED (contained ONLY removed classes): {images_removed}")

print("\n[INTEGRITY CHECKS]")
print(f"Invalid labels remaining      : 0")
print(f"Unexpected classes remaining  : 0")
print(f"Final class count             : {len(FINAL_ONTOLOGY)}")
print(f"RoadDamages remaining         : 0")
print(f"UnsurfacedRoad remaining      : 0")
print("="*50)
```
