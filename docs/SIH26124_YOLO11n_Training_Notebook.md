# SIH26124: YOLO11n Transfer Learning Notebook
**AI-Powered Mobile Urban Intelligence Platform Using Public Transport Fleet**

This notebook fine-tunes a pretrained YOLO11n model on the unified `SIH-data1` dataset for real-time edge deployment. This establishes a highly controlled and reproducible 640x640 baseline.

---

## 1. Environment & Setup

```python
# %% [markdown]
# ## 1. Environment Setup
# Verifying hardware, library versions, and configuring random seeds for reproducibility.

# %%
import os
import glob
import json
import yaml
import time
import shutil
import random
import cv2
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display, Image
import warnings

# Install ultralytics if not present (Kaggle usually requires this)
!pip install -q ultralytics supervision

import ultralytics
from ultralytics import YOLO

warnings.filterwarnings('ignore')

# Set deterministic random seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

print("="*50)
print("ENVIRONMENT SUMMARY")
print("="*50)
print(f"Python Version : {os.sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"Ultralytics    : {ultralytics.__version__}")
print(f"CUDA Available : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name       : {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory     : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("WARNING: No GPU detected. Training will be extremely slow.")
```

---

## 2. Locate and Verify Dataset

```python
# %% [markdown]
# ## 2. Locate and Verify SIH-data1 Master Dataset
# Programmatically search for the dataset and strictly assert the 7-class ontology.

# %%
INPUT_DIR = Path("/kaggle/input")
MASTER_DIR = None

# Search for the master dataset
for p in INPUT_DIR.rglob("data.yaml"):
    path_str = str(p).lower()
    if "sih-data1" in path_str and "sih26124_master" in path_str:
        MASTER_DIR = p.parent
        break

if not MASTER_DIR:
    # Fallback to current working directory if testing locally
    MASTER_DIR = Path("./SIH26124_MASTER")
    
if not MASTER_DIR.exists():
    raise FileNotFoundError("Could not locate SIH-data1 dataset directory in /kaggle/input!")

print(f"Dataset found at: {MASTER_DIR}")

# Read data.yaml
data_yaml_path = MASTER_DIR / "data.yaml"
with open(data_yaml_path, 'r') as f:
    dataset_config = yaml.safe_load(f)

# Assert Exact 7 Classes
EXPECTED_CLASSES = {
    0: 'HMV', 1: 'LMV', 2: 'Pedestrian', 3: 'Pothole',
    4: 'Crack', 5: 'Manhole', 6: 'SpeedBump'
}

names = dataset_config.get('names', {})
if isinstance(names, list):
    names = {i: n for i, n in enumerate(names)}

assert len(names) == 7, f"ERROR: Expected exactly 7 classes, found {len(names)}."
for k, v in EXPECTED_CLASSES.items():
    assert names.get(k) == v, f"ERROR: Class ID {k} must be {v}, found {names.get(k)}."

print("Ontology Verified: Exactly 7 SIH26124 classes found.")

# Update data.yaml path dynamically for Kaggle
dataset_config['path'] = str(MASTER_DIR.resolve())
# Ensure paths are relative to 'path'
dataset_config['train'] = 'images/train'
dataset_config['val'] = 'images/val'
dataset_config['test'] = 'images/test'

# Save a local working copy of data.yaml to avoid read-only errors on Kaggle input
WORKING_YAML = Path("/kaggle/working/data.yaml")
with open(WORKING_YAML, 'w') as f:
    yaml.dump(dataset_config, f, sort_keys=False)

def verify_split(split):
    img_dir = MASTER_DIR / "images" / split
    lbl_dir = MASTER_DIR / "labels" / split
    imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    lbls = list(lbl_dir.glob("*.txt"))
    print(f"[{split.upper()}] Images: {len(imgs)} | Labels: {len(lbls)}")
    if len(imgs) == 0:
        print(f"WARNING: No images found in {split} split!")
    return len(imgs), len(lbls)

print("\nDataset Splits:")
n_train, _ = verify_split("train")
n_val, _ = verify_split("val")
n_test, _ = verify_split("test")
```

---

## 3. Dataset Statistics

```python
# %% [markdown]
# ## 3. Dataset Statistics
# Analyzing the frequency of critical road hazards versus standard traffic objects.

# %%
class_counts = {v: 0 for k, v in EXPECTED_CLASSES.items()}
img_counts = {v: 0 for k, v in EXPECTED_CLASSES.items()}

for split in ["train", "val", "test"]:
    lbl_dir = MASTER_DIR / "labels" / split
    for lbl_file in lbl_dir.glob("*.txt"):
        with open(lbl_file, 'r') as f:
            lines = f.readlines()
        
        found_in_img = set()
        for line in lines:
            c_id = int(line.split()[0])
            c_name = EXPECTED_CLASSES[c_id]
            class_counts[c_name] += 1
            found_in_img.add(c_name)
        
        for c_name in found_in_img:
            img_counts[c_name] += 1

total_instances = sum(class_counts.values())
stats_df = pd.DataFrame({
    'Class': list(EXPECTED_CLASSES.values()),
    'Instances': [class_counts[c] for c in EXPECTED_CLASSES.values()],
    'Images_Containing': [img_counts[c] for c in EXPECTED_CLASSES.values()]
})
stats_df['Instance_Percentage'] = (stats_df['Instances'] / total_instances * 100).round(2)

print("="*50)
print("DATASET STATISTICS")
print("="*50)
display(stats_df.sort_values('Instances', ascending=False))

plt.figure(figsize=(12, 6))
sns.barplot(data=stats_df.sort_values('Instances', ascending=False), x='Class', y='Instances', palette='viridis')
plt.title('SIH26124 Class Instance Distribution')
plt.ylabel('Total Annotations')
plt.xlabel('Class Name')
plt.tight_layout()
plt.savefig('/kaggle/working/class_distribution.png')
plt.show()
```

---

## 4. Visual Dataset Inspection

```python
# %% [markdown]
# ## 4. Visual Dataset Inspection
# Confirming that bounding boxes are preserved and aligned properly.

# %%
def plot_samples(num_samples=4):
    train_imgs = list((MASTER_DIR / "images" / "train").glob("*.jpg"))
    if not train_imgs: return
    
    samples = random.sample(train_imgs, min(num_samples, len(train_imgs)))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, img_path in zip(axes.flatten(), samples):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        lbl_path = MASTER_DIR / "labels" / "train" / f"{img_path.stem}.txt"
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f:
                    c, x, y, bw, bh = map(float, line.strip().split())
                    name = EXPECTED_CLASSES[int(c)]
                    x1, y1 = int((x - bw/2) * w), int((y - bh/2) * h)
                    x2, y2 = int((x + bw/2) * w), int((y + bh/2) * h)
                    
                    # Distinguish hazards (red) from vehicles (blue)
                    color = (255, 0, 0) if int(c) >= 3 else (0, 0, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        ax.imshow(img)
        ax.set_title(img_path.name)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/sample_training_batch.png')
    plt.show()

plot_samples()
```

---

## 5. Initialize YOLO11n

```python
# %% [markdown]
# ## 5. Initialize YOLO11n (Transfer Learning)
# Loading COCO-pretrained weights. Do NOT train from scratch.

# %%
print("Loading COCO-pretrained YOLO11n...")
print("Concept: YOLO11n Pretrained -> Fine-Tuning -> SIH26124 Master Dataset")

model = YOLO("yolo11n.pt")
model.info(detailed=False)

print("\nModel successfully initialized with pretrained weights.")
```

---

## 6. Train YOLO11n @ 640

```python
# %% [markdown]
# ## 6. Train YOLO11n (Baseline @ 640x640)
# Establishing the primary robust baseline with 640 resolution, 100 epochs, and explicit optimizer configuration.

# %%
# Moderate augmentations for road scenes (aspect-ratio preserving)
augmentations = {
    'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, # Moderate lighting/color variation
    'degrees': 0.0, 'translate': 0.1, 'scale': 0.2, # Moderate affine
    'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5, # Safe horizontal flips
    'mosaic': 1.0, 'mixup': 0.1 # YOLO defaults
}

print("="*50)
print("STARTING TRAINING: YOLO11n @ 640")
print("="*50)

# Track time
start_time_640 = time.time()

# Train the model with Explicit Optimizer Settings
results_640 = model.train(
    data=str(WORKING_YAML),
    epochs=100,
    patience=20,
    imgsz=640,
    batch=-1, # AutoBatch to prevent OOM
    workers=4,
    cache=False,
    pretrained=True,
    amp=True,
    project="/kaggle/working/runs",
    name="yolo11n_640",
    seed=42,
    deterministic=True,
    # EXPLICIT OPTIMIZER CONFIGURATION
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    cos_lr=True,
    **augmentations
)

end_time_640 = time.time()
train_time_640 = (end_time_640 - start_time_640) / 60
print(f"Training completed in {train_time_640:.2f} minutes.")
```

---

## 7. Training Analysis

```python
# %% [markdown]
# ## 7. Training Analysis
# Visualizing loss curves and mAP progression from the 640 run.

# %%
run_dir_640 = Path("/kaggle/working/runs/yolo11n_640")

def display_results_plot(run_path):
    results_png = run_path / "results.png"
    if results_png.exists():
        shutil.copy(results_png, "/kaggle/working/training_curves.png")
        display(Image(filename=str(results_png)))
    else:
        print("results.png not found yet.")

display_results_plot(run_dir_640)

# Check early stopping
best_epoch_640 = 0
results_csv = run_dir_640 / "results.csv"
if results_csv.exists():
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    best_epoch_640 = df['metrics/mAP50-95(B)'].idxmax() + 1
    print(f"Best Epoch (mAP50-95): {best_epoch_640} / {len(df)}")
    if len(df) < 100:
        print(f"Early stopping triggered at epoch {len(df)}")
```

---

## 8. Validation Evaluation

```python
# %% [markdown]
# ## 8. Validation Evaluation (BEST Checkpoint)
# Evaluating the model on the validation split with emphasis on Road-Hazard Recall.

# %%
best_model_640_path = run_dir_640 / "weights" / "best.pt"
best_model_640 = YOLO(str(best_model_640_path))

print("="*50)
print("VALIDATION METRICS (640)")
print("="*50)

val_results = best_model_640.val(data=str(WORKING_YAML), split="val", imgsz=640)

def extract_metrics(metrics_obj, save_csv=False, csv_name="class_metrics.csv"):
    class_indices = metrics_obj.ap_class_index
    r = metrics_obj.results_dict
    
    print("\nOVERALL METRICS:")
    print(f"mAP50   : {r['metrics/mAP50(B)']:.4f}")
    print(f"mAP50-95: {r['metrics/mAP50-95(B)']:.4f}")
    print(f"Precision: {r['metrics/precision(B)']:.4f}")
    print(f"Recall   : {r['metrics/recall(B)']:.4f}")
    
    print("\nPER-CLASS METRICS:")
    metric_list = []
    
    for i, c_idx in enumerate(class_indices):
        c_name = EXPECTED_CLASSES[c_idx]
        recall = metrics_obj.box.r[i]
        precision = metrics_obj.box.p[i]
        ap50 = metrics_obj.box.ap50[i]
        ap = metrics_obj.box.ap[i]
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Highlight Hazards
        marker = "***" if c_idx >= 3 else ""
        print(f"{c_name:<12} {marker}: Recall = {recall:.4f} | Precision = {precision:.4f} | AP50 = {ap50:.4f} | F1 = {f1:.4f}")
        
        metric_list.append({
            "Class": c_name,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "AP50": round(ap50, 4),
            "AP50-95": round(ap, 4),
            "F1": round(f1, 4)
        })
        
    if save_csv:
        pd.DataFrame(metric_list).to_csv(f"/kaggle/working/{csv_name}", index=False)
        
    return metric_list

val_metrics = extract_metrics(val_results)

# Save confusion matrix
cm_path = run_dir_640 / "confusion_matrix.png"
if cm_path.exists():
    shutil.copy(cm_path, "/kaggle/working/confusion_matrix.png")
    display(Image(filename=str(cm_path), width=800))
```

---

## 9. Test Evaluation

```python
# %% [markdown]
# ## 9. Test Evaluation
# Final benchmark on the strictly held-out test set to ensure generalizability.

# %%
print("="*50)
print("TEST SET METRICS (640)")
print("="*50)

test_results = best_model_640.val(data=str(WORKING_YAML), split="test", imgsz=640)
test_metrics = extract_metrics(test_results, save_csv=True, csv_name="class_metrics_640.csv")
```

---

## 10. Error Analysis

```python
# %% [markdown]
# ## 10. Error Analysis
# Visualizing False Positives and False Negatives, specifically analyzing Hazard misclassifications.

# %%
val_preds = list(run_dir_640.glob("val_batch*_pred.jpg"))
error_dir = Path("/kaggle/working/error_analysis")
error_dir.mkdir(exist_ok=True)

if val_preds:
    print("Sample Validation Predictions saved to /kaggle/working/error_analysis/")
    for idx, p in enumerate(val_preds[:3]):
        shutil.copy(p, error_dir / f"sample_pred_{idx}.jpg")
    
    display(Image(filename=str(val_preds[0]), width=1200))
```

---

## 11. Inference Speed Benchmark

```python
# %% [markdown]
# ## 11. Inference Speed Benchmark
# Benchmark preprocessing, inference latency, and FPS at 512, 640, and 768.

# %%
print("="*50)
print("INFERENCE SPEED BENCHMARK (ms / FPS)")
print("="*50)

def run_speed_benchmark(model_path, metrics_data, resolutions=[512, 640, 768]):
    bench_model = YOLO(model_path)
    test_img = list((MASTER_DIR / "images" / "test").glob("*.jpg"))[0]
    
    results_list = []
    
    # Extract Hazard Recalls safely
    def get_recall(c_name):
        for m in metrics_data:
            if m["Class"] == c_name: return m["Recall"]
        return 0.0
    
    mAP50 = bench_model.metrics.box.map50 if hasattr(bench_model, 'metrics') else 0
    mAP50_95 = bench_model.metrics.box.map if hasattr(bench_model, 'metrics') else 0

    for res in resolutions:
        print(f"\nBenchmarking at {res}x{res}...")
        
        # Warmup
        for _ in range(5):
            bench_model(test_img, imgsz=res, verbose=False)
            
        # Benchmark
        latencies = []
        for _ in range(100):
            t0 = time.time()
            bench_model(test_img, imgsz=res, verbose=False)
            t1 = time.time()
            latencies.append((t1 - t0) * 1000) # ms
        
        avg_latency = np.mean(latencies)
        fps = 1000.0 / avg_latency if avg_latency > 0 else 0
        
        results_list.append({
            "Resolution": f"{res}x{res}",
            "mAP50": round(mAP50, 4),
            "mAP50-95": round(mAP50_95, 4),
            "Pothole Recall": get_recall("Pothole"),
            "Crack Recall": get_recall("Crack"),
            "Manhole Recall": get_recall("Manhole"),
            "SpeedBump Recall": get_recall("SpeedBump"),
            "Latency (ms)": round(avg_latency, 2),
            "FPS": round(fps, 1)
        })
        
    df_speed = pd.DataFrame(results_list)
    return df_speed

speed_df_640 = run_speed_benchmark(str(best_model_640_path), test_metrics)
display(speed_df_640)
speed_df_640.to_csv("/kaggle/working/speed_benchmark.csv", index=False)
```

---

## 12. Confidence Threshold Analysis

```python
# %% [markdown]
# ## 12. Confidence Analysis
# Testing threshold impact on precision/recall tradeoff.

# %%
def test_confidence(model, conf_list=[0.20, 0.25, 0.30, 0.40, 0.50]):
    print("Confidence Threshold Impact (640 Model):")
    for conf in conf_list:
        res = model.val(data=str(WORKING_YAML), split="test", imgsz=640, conf=conf, verbose=False)
        p = res.results_dict['metrics/precision(B)']
        r = res.results_dict['metrics/recall(B)']
        print(f"Conf {conf:.2f} -> Precision: {p:.4f} | Recall: {r:.4f}")

test_confidence(best_model_640)
```

---

## 13. Live Video Test Simulation

```python
# %% [markdown]
# ## 13. Live Video Test
# Simulating 1920x1080 frame-by-frame inference pipeline (No GPS fusion yet).

# %%
print("To simulate live video, upload a raw `.mp4` to `/kaggle/working/test_vid.mp4`.")
print("Command:")
print("""
import cv2
video_path = "/kaggle/working/test_vid.mp4"
if os.path.exists(video_path):
    # Ensure source aspect ratio is preserved
    best_model_640.predict(
        source=video_path,
        save=True,
        imgsz=640,
        conf=0.25,
        project="/kaggle/working",
        name="live_video_demo"
    )
""")

print("TEMPORAL TRACKING NOTE:")
print("In production, output from best.pt will be piped into ByteTrack (ultralytics tracker).")
print("Command: model.predict(source='...', tracker='bytetrack.yaml')")
```

---

## 14. Export to ONNX / TensorRT

```python
# %% [markdown]
# ## 14. Export
# Export the best baseline model to ONNX.

# %%
print("Exporting Best 640 Model to ONNX...")
exported_path = best_model_640.export(format="onnx", imgsz=640)
print(f"Exported to: {exported_path}")

# Copy final weights for easy access
shutil.copy(best_model_640_path, "/kaggle/working/best.pt")
shutil.copy(run_dir_640 / "weights" / "last.pt", "/kaggle/working/last.pt")
shutil.copy(exported_path, "/kaggle/working/best.onnx")

print("\nDEPLOYMENT NOTE: For Jetson/NVIDIA edge devices, compile to TensorRT locally using:")
print("yolo export model=best.pt format=engine half=True workspace=4")
```

---

## 15. Final Automatic Report

```python
# %% [markdown]
# ## 15. Final Automatic Report
# Printing the full project summary and final file structure.

# %%
def get_recall(c_name):
    for m in test_metrics:
        if m["Class"] == c_name: return m["Recall"]
    return 0.0

final_report = f"""
============================================================
SIH26124 FINAL REPORT
============================================================

DATASET:
Total Images: {n_train + n_val + n_test}
Train Images: {n_train}
Validation Images: {n_val}
Test Images: {n_test}

MODEL:
Architecture: YOLO11n
Pretrained: True
Optimizer: AdamW (lr0=0.001, lrf=0.01, wd=0.0005, warmup=3.0, cos_lr=True)
Epochs Requested: 100
Early Stopping: Epoch {best_epoch_640}

640 RESULTS (TEST SET):
mAP50    : {test_results.results_dict['metrics/mAP50(B)']:.4f}
mAP50-95 : {test_results.results_dict['metrics/mAP50-95(B)']:.4f}
Precision: {test_results.results_dict['metrics/precision(B)']:.4f}
Recall   : {test_results.results_dict['metrics/recall(B)']:.4f}
Pothole Recall  : {get_recall("Pothole"):.4f}
Crack Recall    : {get_recall("Crack"):.4f}
Manhole Recall  : {get_recall("Manhole"):.4f}
SpeedBump Recall: {get_recall("SpeedBump"):.4f}

FPS      : {speed_df_640.iloc[1]['FPS']}
Latency  : {speed_df_640.iloc[1]['Latency (ms)']} ms

FINAL RECOMMENDATION:
Review the training curves, error analysis, and per-class recall (especially Pothole/Crack) to determine if the 640 baseline meets the project requirements. 
Only proceed to a 768-resolution experiment if the baseline road-hazard recall is unsatisfactory and edge-hardware profiling confirms enough thermal/compute headroom to sacrifice latency for accuracy.
"""

print(final_report)

with open("/kaggle/working/final_report.txt", "w") as f:
    f.write(final_report)

print("="*50)
print("ALL REQUIRED ARTIFACTS SAVED TO /kaggle/working/")
"""
Expected output files:
- best.pt
- last.pt
- best.onnx
- results.csv (in runs folder)
- class_metrics_640.csv
- speed_benchmark.csv
- confusion_matrix.png
- training_curves.png
- class_distribution.png
- error_analysis/
- final_report.txt
"""
```
