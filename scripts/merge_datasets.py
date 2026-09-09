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
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

warnings.filterwarnings('ignore')

INPUT_DIR = Path("temp_raw_datasets")
MASTER_DIR = Path("SIH26124_MASTER")

rad_root = INPUT_DIR / "rad"
rdd_root = INPUT_DIR / "rdd"

# Read RAD yaml
rad_yaml_path = list(rad_root.rglob("data.yaml"))[0]
with open(rad_yaml_path, 'r') as f:
    rad_config = yaml.safe_load(f)

rad_orig_names = rad_config.get('names', [])
if isinstance(rad_orig_names, dict):
    rad_orig_names = [rad_orig_names[i] for i in range(len(rad_orig_names))]

# RDD
rdd_yaml_path = list(rdd_root.rglob("data.yaml"))
if rdd_yaml_path:
    with open(rdd_yaml_path[0], 'r') as f:
        rdd_config = yaml.safe_load(f)
    rdd_orig_names = rdd_config.get('names', [])
    if isinstance(rdd_orig_names, dict):
        rdd_orig_names = [rdd_orig_names[i] for i in range(len(rdd_orig_names))]
else:
    rdd_orig_names = ['Pothole', 'Crack', 'Manhole']

FINAL_ONTOLOGY = {
    0: "HMV", 1: "LMV", 2: "Pedestrian", 3: "Pothole",
    4: "Crack", 5: "Manhole", 6: "SpeedBump"
}

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

RDD_MAPPING = {}
for i, name in enumerate(rdd_orig_names):
    nl = name.lower()
    if "pothole" in nl: RDD_MAPPING[i] = 3
    elif "crack" in nl: RDD_MAPPING[i] = 4
    elif "manhole" in nl: RDD_MAPPING[i] = 5

if MASTER_DIR.exists():
    shutil.rmtree(MASTER_DIR)

for split in ['train', 'val', 'test']:
    (MASTER_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
    (MASTER_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

META_DIR = MASTER_DIR / 'metadata'
META_DIR.mkdir(exist_ok=True)

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
                            class_distribution[t_id] += 1
                            source_distribution["RAD"][t_id] += 1
                            split_distribution[split_name][t_id] += 1
                            log_conversion("RAD", img_path.name, rad_orig_names[src_id], src_id, FINAL_ONTOLOGY[t_id], t_id, "keep")
                        elif src_id in RAD_REMOVE_IDS:
                            removed_classes_in_file.add(src_id)
                            log_conversion("RAD", img_path.name, rad_orig_names[src_id], src_id, "DELETE", -1, "remove")

        if len(retained_lines) > 0:
            if removed_classes_in_file:
                log_removal(img_path.name, split_name, list(removed_classes_in_file), "annotation_removed", "Contained removed classes but had other valid classes")
            
            shutil.copy2(img_path, MASTER_DIR / 'images' / split_name / dest_img_name)
            with open(MASTER_DIR / 'labels' / split_name / dest_lbl_name, 'w') as f:
                f.write("\n".join(retained_lines) + "\n")
        else:
            if removed_classes_in_file:
                log_removal(img_path.name, split_name, list(original_classes_in_file), "image_removed", "Contained ONLY removed classes")

print("Processing RAD...")
process_rad_split(rad_root / "images" / "train" / "images", "train")
if (rad_root / "images" / "valid" / "images").exists():
    process_rad_split(rad_root / "images" / "valid" / "images", "val")
else:
    process_rad_split(rad_root / "images" / "val" / "images", "val")
process_rad_split(rad_root / "images" / "test" / "images", "test")

print("Processing RDD...")
rdd_img_dir = rdd_root / "images"
if not rdd_img_dir.exists():
    rdd_img_dir = rdd_root / "data" / "images"
    rdd_lbl_dir = rdd_root / "data" / "labels-YOLO"
else:
    rdd_lbl_dir = rdd_root / "labels-YOLO"

if not rdd_lbl_dir.exists():
    rdd_lbl_dir = rdd_img_dir.parent / "labels" # Fallback

images = list(rdd_img_dir.glob("*.jpg")) + list(rdd_img_dir.glob("*.png"))
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
            shutil.copy2(img_path, MASTER_DIR / 'images' / split_name / dest_img_name)
            with open(MASTER_DIR / 'labels' / split_name / dest_lbl_name, 'w') as f:
                f.write("\n".join(retained_lines) + "\n")

yaml_content = {
    'path': str(MASTER_DIR.resolve()),
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    'names': FINAL_ONTOLOGY
}
with open(MASTER_DIR / 'data.yaml', 'w') as f:
    yaml.dump(yaml_content, f, sort_keys=False)

pd.DataFrame(conversion_report).to_csv(META_DIR / 'conversion_report.csv', index=False)
pd.DataFrame(removed_classes_report).to_csv(META_DIR / 'removed_classes_report.csv', index=False)

mapping_data = []
for i, n in enumerate(rad_orig_names):
    if i in RAD_MAPPING: mapping_data.append(["RAD", n, i, FINAL_ONTOLOGY[RAD_MAPPING[i]], RAD_MAPPING[i], "keep"])
    elif i in RAD_REMOVE_IDS: mapping_data.append(["RAD", n, i, "DELETE", -1, "remove"])
for i, n in enumerate(rdd_orig_names):
    if i in RDD_MAPPING: mapping_data.append(["RDD", n, i, FINAL_ONTOLOGY[RDD_MAPPING[i]], RDD_MAPPING[i], "keep"])
pd.DataFrame(mapping_data, columns=["source_dataset", "source_class", "source_class_id", "final_class", "final_class_id", "decision"]).to_csv(META_DIR / 'class_mapping.csv', index=False)

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
print("ASSERTIONS PASSED: No invalid bounding boxes. No removed classes remain.")

print("\n[TOTAL IMAGES]")
total_train = len(list((MASTER_DIR / 'images' / 'train').glob('*.*')))
total_val = len(list((MASTER_DIR / 'images' / 'val').glob('*.*')))
total_test = len(list((MASTER_DIR / 'images' / 'test').glob('*.*')))
total_images = total_train + total_val + total_test
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
print("DONE")
