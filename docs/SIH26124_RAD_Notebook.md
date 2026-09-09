# SIH26124: AI-Powered Mobile Urban Intelligence Platform
**FIRST Unified Computer-Vision Model on RAD Dataset (YOLO11n)**

This notebook is generated for the Kaggle environment. It trains a unified YOLO11n detector for road hazards and urban traffic analysis.

---

## SECTION 1 — Environment setup

```python
# %% [markdown]
# ## 1. Environment Setup
# Import required packages, install dependencies, and print environment specs.

# %%
!pip install -q ultralytics supervision memory_profiler

import os
import sys
import glob
import json
import yaml
import time
import shutil
import random
import warnings
from pathlib import Path

import cv2
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO

warnings.filterwarnings('ignore')

# Set reproducible seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("="*50)
print("ENVIRONMENT SPECIFICATIONS")
print("="*50)
print(f"Python version: {sys.version.split(' ')[0]}")
print(f"PyTorch version: {torch.__version__}")
try:
    import ultralytics
    print(f"Ultralytics version: {ultralytics.__version__}")
except:
    pass

if torch.cuda.is_available():
    print("CUDA: AVAILABLE")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # GPU Memory
    t = torch.cuda.get_device_properties(0).total_memory
    print(f"Total GPU Memory: {t / 1024**3:.2f} GB")
else:
    print("CUDA: NOT AVAILABLE (Running on CPU)")
print("="*50)
```

---

## SECTION 2 — Locate and inspect RAD

```python
# %% [markdown]
# ## 2. Locate and Inspect RAD Dataset
# Programmatically find the dataset in `/kaggle/input`.

# %%
print("Scanning /kaggle/input for RAD dataset...")
INPUT_DIR = Path("/kaggle/input")
dataset_paths = list(INPUT_DIR.rglob("data.yaml"))

if not dataset_paths:
    raise FileNotFoundError("Could not find data.yaml in /kaggle/input. Ensure the dataset is attached.")

# Use the first found data.yaml to locate the dataset root
RAD_YAML = dataset_paths[0]
RAD_ROOT = RAD_YAML.parent
print(f"Found RAD Dataset root at: {RAD_ROOT}")

# Load the original YAML to understand structure
with open(RAD_YAML, 'r') as f:
    orig_yaml = yaml.safe_load(f)

# Resolve paths
orig_train_dir = (RAD_ROOT / orig_yaml.get('train', 'train/images')).resolve()
orig_val_dir = (RAD_ROOT / orig_yaml.get('val', 'valid/images')).resolve()
orig_test_dir = (RAD_ROOT / orig_yaml.get('test', 'test/images')).resolve()

orig_names = orig_yaml.get('names', [])
if isinstance(orig_names, dict):
    orig_names = [orig_names[i] for i in range(len(orig_names))]

print("\n--- ORIGINAL RAD DATASET METADATA ---")
print(f"Original Classes ({len(orig_names)}): {orig_names}")

def count_files(directory, ext):
    if not directory.exists(): return 0
    return len(list(directory.rglob(f"*.{ext}")))

print("\nFile Counts:")
for split_name, split_dir in zip(["Train", "Val", "Test"], [orig_train_dir, orig_val_dir, orig_test_dir]):
    img_cnt = count_files(split_dir, "jpg") + count_files(split_dir, "png")
    lbl_dir = split_dir.parent / 'labels' if split_dir.name == 'images' else split_dir
    lbl_cnt = count_files(lbl_dir, "txt")
    print(f"  {split_name}: {img_cnt} images, {lbl_cnt} labels")
```

---

## SECTION 3 — Dataset integrity checks

```python
# %% [markdown]
# ## 3. Dataset Integrity Checks
# Check for corrupt files, missing labels, and bounding box validity.

# %%
def check_dataset_integrity(img_dir):
    img_dir = Path(img_dir)
    if not img_dir.exists(): return
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
            if cls_id >= len(orig_names):
                unknown_classes += 1
                
            # Check normalized coords
            coords = [float(x) for x in parts[1:]]
            if any(c < 0.0 or c > 1.0 for c in coords):
                invalid_bboxes += 1

    print(f"Integrity Report for {img_dir.name}:")
    print(f"  Total Images: {len(images)}")
    print(f"  Missing Labels: {missing_labels}")
    print(f"  Empty Labels: {empty_labels}")
    print(f"  Invalid BBoxes (out of bounds/format): {invalid_bboxes}")
    print(f"  Unknown Classes: {unknown_classes}")
    print("-" * 30)

print("\n--- DATASET INTEGRITY CHECKS ---")
check_dataset_integrity(orig_train_dir)
check_dataset_integrity(orig_val_dir)
check_dataset_integrity(orig_test_dir)
```

---

## SECTION 4 — RAD class mapping

```python
# %% [markdown]
# ## 4. RAD Class Mapping
# Mapping original RAD classes to the SIH26124 Target Ontology.

# %%
TARGET_ONTOLOGY = {
    0: "LMV",
    1: "HMV",
    2: "pedestrian",
    3: "pothole",
    4: "crack",
    5: "protrusion",
    6: "manhole",
    7: "unsurfaced_road",
    8: "speed_bump"
}

# Reverse map for easier lookup
TARGET_NAME_TO_ID = {v: k for k, v in TARGET_ONTOLOGY.items()}

# We must dynamically map original names to target names.
# NOTE: The RAD dataset typically groups all road damages into "RoadDamages".
# We will map them as closely as possible, preserving missing classes as empty in the new ontology.
CLASS_MAPPING = {}

for orig_id, orig_name in enumerate(orig_names):
    name_lower = orig_name.lower()
    if name_lower == "lmv" or "car" in name_lower or "bike" in name_lower:
        CLASS_MAPPING[orig_id] = TARGET_NAME_TO_ID["LMV"]
    elif name_lower == "hmv" or "bus" in name_lower or "truck" in name_lower:
        CLASS_MAPPING[orig_id] = TARGET_NAME_TO_ID["HMV"]
    elif "pedestrian" in name_lower or "person" in name_lower:
        CLASS_MAPPING[orig_id] = TARGET_NAME_TO_ID["pedestrian"]
    elif "pothole" in name_lower:
        CLASS_MAPPING[orig_id] = TARGET_NAME_TO_ID["pothole"]
    elif "crack" in name_lower:
        CLASS_MAPPING[orig_id] = TARGET_NAME_TO_ID["crack"]
    elif "protrusion" in name_lower:
        CLASS_MAPPING[orig_id] = TARGET_NAME_TO_ID["protrusion"]
    elif "manhole" in name_lower:
        CLASS_MAPPING[orig_id] = TARGET_NAME_TO_ID["manhole"]
    elif "unsurfaced" in name_lower or "unpaved" in name_lower:
        CLASS_MAPPING[orig_id] = TARGET_NAME_TO_ID["unsurfaced_road"]
    elif "speedbump" in name_lower or "speed_bump" in name_lower or "breaker" in name_lower:
        CLASS_MAPPING[orig_id] = TARGET_NAME_TO_ID["speed_bump"]
    elif "roaddamages" in name_lower or "damage" in name_lower:
        # Fallback if the dataset groups all damages together. We map generic damage to pothole 
        # (or another representative class) but explicitly warn about it.
        print(f"WARNING: Original class '{orig_name}' is ambiguous. Mapping to 'pothole' as representative damage.")
        CLASS_MAPPING[orig_id] = TARGET_NAME_TO_ID["pothole"]
    else:
        print(f"WARNING: No explicit mapping for '{orig_name}'. Skipping labels.")
        CLASS_MAPPING[orig_id] = -1  # Skip

print("\n--- CLASS MAPPING REPORT ---")
print(f"{'Original ID':<15} | {'Original Name':<20} | {'Target ID':<10} | {'Target Name'}")
print("-" * 65)
for orig_id, orig_name in enumerate(orig_names):
    t_id = CLASS_MAPPING.get(orig_id, -1)
    t_name = TARGET_ONTOLOGY.get(t_id, "IGNORED")
    print(f"{orig_id:<15} | {orig_name:<20} | {t_id:<10} | {t_name}")
```

---

## SECTION 5 — Convert dataset to YOLO format

```python
# %% [markdown]
# ## 5. Convert Dataset
# Create a fresh directory and convert annotations to the unified 9-class ontology.

# %%
SIH_DIR = Path("/kaggle/working/SIH26124_RAD")

if SIH_DIR.exists():
    shutil.rmtree(SIH_DIR)

for split in ['train', 'val', 'test']:
    (SIH_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
    (SIH_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

def convert_split(orig_img_dir, split_name):
    if not orig_img_dir.exists(): return
    
    orig_lbl_dir = orig_img_dir.parent / "labels"
    dest_img_dir = SIH_DIR / 'images' / split_name
    dest_lbl_dir = SIH_DIR / 'labels' / split_name
    
    images = list(orig_img_dir.glob("*.jpg")) + list(orig_img_dir.glob("*.png"))
    
    for img_path in images:
        lbl_path = orig_lbl_dir / (img_path.stem + ".txt")
        
        # Symlink image to save space and time instead of copying
        dest_img_path = dest_img_dir / img_path.name
        if not dest_img_path.exists():
            os.symlink(img_path, dest_img_path)
            
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    orig_cls = int(parts[0])
                    target_cls = CLASS_MAPPING.get(orig_cls, -1)
                    if target_cls != -1:
                        new_lines.append(f"{target_cls} " + " ".join(parts[1:]))
                        
            if new_lines:
                with open(dest_lbl_dir / lbl_path.name, 'w') as f:
                    f.write("\n".join(new_lines) + "\n")

print("\nConverting Train split...")
convert_split(orig_train_dir, 'train')
print("Converting Validation split...")
convert_split(orig_val_dir, 'val')
print("Converting Test split...")
convert_split(orig_test_dir, 'test')
print(f"Dataset successfully converted to {SIH_DIR}")
```

---

## SECTION 6 & 7 — Train/Val/Test Split Stats & Data.yaml

```python
# %% [markdown]
# ## 6 & 7. Split Statistics & Generate data.yaml
# We retain the official RAD splits to prevent video-frame leakage.

# %%
print("\n--- SPLIT STATISTICS ---")
for split in ['train', 'val', 'test']:
    img_cnt = len(list((SIH_DIR / 'images' / split).glob("*.*")))
    lbl_cnt = len(list((SIH_DIR / 'labels' / split).glob("*.txt")))
    print(f"{split.upper()}: {img_cnt} images, {lbl_cnt} labels")

# Generate data.yaml
yaml_content = {
    'path': str(SIH_DIR),
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    'names': TARGET_ONTOLOGY
}

yaml_path = SIH_DIR / 'data.yaml'
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_content, f, sort_keys=False)

print("\n--- GENERATED DATA.YAML ---")
with open(yaml_path, 'r') as f:
    print(f.read())
```

---

## SECTION 8 — Dataset visualization

```python
# %% [markdown]
# ## 8. Dataset Visualization
# Displaying class distributions and analyzing imbalance.

# %%
# Count instances per class in the newly mapped dataset
class_counts = {k: 0 for k in TARGET_ONTOLOGY.keys()}

for lbl_file in (SIH_DIR / 'labels' / 'train').rglob("*.txt"):
    with open(lbl_file, 'r') as f:
        for line in f:
            cls_id = int(line.strip().split()[0])
            class_counts[cls_id] += 1

print("\n--- CLASS DISTRIBUTION (TRAIN SET) ---")
for cls_id, count in class_counts.items():
    print(f"{TARGET_ONTOLOGY[cls_id]:<15}: {count}")

plt.figure(figsize=(10, 5))
sns.barplot(x=list(TARGET_ONTOLOGY.values()), y=list(class_counts.values()))
plt.xticks(rotation=45)
plt.title("Class Distribution in SIH26124 Unified Dataset (Train)")
plt.ylabel("Number of Instances")
plt.tight_layout()
plt.savefig("/kaggle/working/class_distribution.png")
plt.show()

# Analysis Warning
zeros = [name for cid, name in TARGET_ONTOLOGY.items() if class_counts[cid] == 0]
if zeros:
    print(f"\nWARNING: The following target classes have ZERO instances in the mapped dataset: {zeros}")
    print("This is expected if the original RAD dataset grouped them into a single 'RoadDamages' class.")
```

---

## SECTION 9 & 10 — Model Initialization & Baseline Training (640)

```python
# %% [markdown]
# ## 9 & 10. Model Initialization & Baseline Training (imgsz=640)
# Transfer learning from COCO-pretrained YOLO11n.

# %%
print("\n--- MODEL INITIALIZATION ---")
# Load pretrained YOLO11n
model_640 = YOLO("yolo11n.pt") 
print(model_640.info())
print("\nInitiating Transfer Learning on RAD...")

# Training configuration
EPOCHS = 100
PATIENCE = 20
IMGSZ = 640
BATCH = 32 # Safe for Kaggle P100/T4
WORKERS = 4
PROJECT_DIR = "/kaggle/working/SIH_Runs"
RUN_NAME_640 = "yolo11n_rad_640"

# Execute training
# NOTE: inline wandb is disabled for kaggle notebook stability unless logged in
results_640 = model_640.train(
    data=str(yaml_path),
    epochs=EPOCHS,
    patience=PATIENCE,
    imgsz=IMGSZ,
    batch=BATCH,
    workers=WORKERS,
    project=PROJECT_DIR,
    name=RUN_NAME_640,
    pretrained=True,
    amp=True,
    seed=SEED,
    save=True,
    val=True,
    plots=True
)
```

---

## SECTION 11 & 12 — Training Analysis & Validation Metrics

```python
# %% [markdown]
# ## 11 & 12. Training Analysis & Validation Metrics
# Extract metrics from the best epoch.

# %%
run_dir_640 = Path(PROJECT_DIR) / RUN_NAME_640

# Show training plots generated by ultralytics
results_img = run_dir_640 / "results.png"
if results_img.exists():
    img = cv2.imread(str(results_img))
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Training Metrics (640)")
    plt.show()

print("\n--- VALIDATION METRICS (640) ---")
# Reload the best model for validation isolation
best_model_640 = YOLO(run_dir_640 / "weights" / "best.pt")
val_metrics_640 = best_model_640.val(data=str(yaml_path), split='val', plots=True)

print(f"Overall mAP50: {val_metrics_640.box.map50:.4f}")
print(f"Overall mAP50-95: {val_metrics_640.box.map:.4f}")

# Extract class-wise recall to emphasize hazards
class_indices = val_metrics_640.box.ap_class_index
recalls = val_metrics_640.box.r
precisions = val_metrics_640.box.p

print("\nClass-wise Performance:")
print(f"{'Class':<15} | {'Recall':<10} | {'Precision':<10}")
print("-" * 40)
for i, cls_idx in enumerate(class_indices):
    cls_name = TARGET_ONTOLOGY[cls_idx]
    print(f"{cls_name:<15} | {recalls[i]:.4f}     | {precisions[i]:.4f}")
```

---

## SECTION 13 & 14 — Test Set Evaluation & Error Analysis

```python
# %% [markdown]
# ## 13 & 14. Test Set Evaluation & Error Analysis
# Run the best 640 model on the held-out test set.

# %%
print("\n--- TEST SET EVALUATION (640) ---")
test_metrics_640 = best_model_640.val(data=str(yaml_path), split='test', task='test')

print("\nQualitative Error Analysis: (Generating Predictions on Test Samples)")
# Pick 3 random test images
test_imgs = list((SIH_DIR / 'images' / 'test').glob("*.jpg"))
if len(test_imgs) > 0:
    sample_imgs = random.sample(test_imgs, min(3, len(test_imgs)))
    for img_p in sample_imgs:
        res = best_model_640.predict(str(img_p), save=False, conf=0.25)
        # Display
        res_plotted = res[0].plot()
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
        plt.title(f"Test Prediction: {img_p.name}")
        plt.axis('off')
        plt.show()
```

---

## SECTION 15 — Inference Speed Benchmark

```python
# %% [markdown]
# ## 15. Inference Speed Benchmark
# Simulating deployment latency for the Mobile Urban Sensing units.

# %%
print("\n--- INFERENCE SPEED BENCHMARK ---")

def benchmark_model(model, imgsz, num_warmup=10, num_runs=50):
    dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8) # 16:9 source
    
    # Warmup
    for _ in range(num_warmup):
        model.predict(dummy_img, imgsz=imgsz, verbose=False)
        
    start_time = time.time()
    for _ in range(num_runs):
        model.predict(dummy_img, imgsz=imgsz, verbose=False)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency = (total_time / num_runs) * 1000 # ms
    fps = 1000 / avg_latency
    return avg_latency, fps

lat_512, fps_512 = benchmark_model(best_model_640, 512)
lat_640, fps_640 = benchmark_model(best_model_640, 640)

print(f"{'Resolution':<12} | {'Latency (ms)':<15} | {'FPS':<10}")
print("-" * 45)
print(f"{'512x512':<12} | {lat_512:.2f}           | {fps_512:.1f}")
print(f"{'640x640':<12} | {lat_640:.2f}           | {fps_640:.1f}")
```

---

## SECTION 16 & 17 — 640 vs 768 Experiment

```python
# %% [markdown]
# ## 16 & 17. 640 vs 768 Resolution Experiment
# Evaluating if higher resolution yields significantly better hazard recall to justify the speed trade-off.

# %%
print("\n--- TRAINING YOLO11N @ 768 ---")
RUN_NAME_768 = "yolo11n_rad_768"
model_768 = YOLO("yolo11n.pt") 

# Train at 768 (using fewer epochs for demonstration if time is constrained, but ideally full 100)
results_768 = model_768.train(
    data=str(yaml_path),
    epochs=EPOCHS,
    patience=PATIENCE,
    imgsz=768,
    batch=BATCH // 2, # Halve batch size to prevent OOM at higher res
    workers=WORKERS,
    project=PROJECT_DIR,
    name=RUN_NAME_768,
    pretrained=True,
    amp=True,
    seed=SEED,
    save=True,
    val=True
)

run_dir_768 = Path(PROJECT_DIR) / RUN_NAME_768
best_model_768 = YOLO(run_dir_768 / "weights" / "best.pt")
val_metrics_768 = best_model_768.val(data=str(yaml_path), split='val')

lat_768, fps_768 = benchmark_model(best_model_768, 768)

print("\n--- 640 vs 768 COMPARISON ---")
print(f"640 -> mAP50: {val_metrics_640.box.map50:.4f}, FPS: {fps_640:.1f}")
print(f"768 -> mAP50: {val_metrics_768.box.map50:.4f}, FPS: {fps_768:.1f}")

# Extract pothole/hazard recall comparison
def get_recall_for_class(metrics, class_name):
    try:
        idx = list(class_indices).index(TARGET_NAME_TO_ID[class_name])
        return metrics.box.r[idx]
    except (ValueError, KeyError):
        return 0.0

hazards = ['pothole', 'crack', 'protrusion', 'manhole', 'speed_bump']
print("\nHazard Recall Comparison:")
for h in hazards:
    r640 = get_recall_for_class(val_metrics_640, h)
    r768 = get_recall_for_class(val_metrics_768, h)
    if r640 > 0 or r768 > 0:
        print(f"  {h:<15}: 640={r640:.3f} | 768={r768:.3f}")
```

---

## SECTION 18 & 19 — Model Export & Save Artifacts

```python
# %% [markdown]
# ## 18 & 19. Export & Archiving
# Exporting models to ONNX and zipping up the run directories for download.

# %%
print("\n--- EXPORTING MODELS ---")
best_model_640.export(format="onnx", imgsz=640, half=True)
best_model_768.export(format="onnx", imgsz=768, half=True)
# Note for edge deployment: To export to TensorRT for NVIDIA devices (e.g., Jetson):
# best_model_640.export(format="engine", imgsz=640, half=True, workspace=4)

print("\n--- SAVING ARTIFACTS ---")
shutil.make_archive("/kaggle/working/SIH_Runs_Archive", 'zip', PROJECT_DIR)
print("Archived all training runs, weights, and plots to /kaggle/working/SIH_Runs_Archive.zip")
```

---

## SECTION 20 — FINAL AUTOMATIC REPORT

```python
# %% [markdown]
# ## 20. Final Automatic Report
# Summary of the experiment for SIH26124 decision making.

# %%
print("="*60)
print("FINAL SIH26124 UNIFIED DETECTOR REPORT")
print("="*60)
print(f"DATASET: {len(list((SIH_DIR/'images'/'train').glob('*.jpg')))} Train | {len(list((SIH_DIR/'images'/'val').glob('*.jpg')))} Val | {len(list((SIH_DIR/'images'/'test').glob('*.jpg')))} Test")

print("\n640 RESULTS:")
print(f"- Overall mAP50: {val_metrics_640.box.map50:.4f}")
for h in hazards:
    r = get_recall_for_class(val_metrics_640, h)
    if r > 0: print(f"- {h} recall: {r:.4f}")
print(f"- Latency: {lat_640:.2f}ms | FPS: {fps_640:.1f}")

print("\n768 RESULTS:")
print(f"- Overall mAP50: {val_metrics_768.box.map50:.4f}")
for h in hazards:
    r = get_recall_for_class(val_metrics_768, h)
    if r > 0: print(f"- {h} recall: {r:.4f}")
print(f"- Latency: {lat_768:.2f}ms | FPS: {fps_768:.1f}")

print("\nFINAL RECOMMENDATION:")
if fps_640 >= 30 and (get_recall_for_class(val_metrics_640, 'pothole') >= get_recall_for_class(val_metrics_768, 'pothole') - 0.05):
    print("Recommendation: YOLO11n @ 640")
    print("Reason: Achieves required real-time deployment FPS (>30) without significant degradation in critical hazard recall compared to 768.")
else:
    print("Recommendation: YOLO11n @ 768")
    print("Reason: 768 resolution yields significantly higher hazard recall, and if edge latency optimizations (TensorRT FP16) are applied, it can meet the speed requirement.")
print("="*60)
```
