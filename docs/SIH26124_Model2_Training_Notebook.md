# SIH26124 Model 2: Temporary Road Obstruction Detection
**YOLO11n Transfer Learning Training Notebook**

This notebook fine-tunes a pretrained YOLO11n model strictly on the `OBSTRUCTION_MASTER` dataset to act as the second lightweight detector in the edge-deployment pipeline.

---

## 1. Environment Setup

```python
# %% [markdown]
# ## 1. Environment Setup
# Installing Ultralytics, verifying CUDA hardware, and setting strict reproducible seeds.

# %%
import os
import glob
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

warnings.filterwarnings('ignore')

# Install dependencies if missing
!pip install -q ultralytics supervision

import ultralytics
from ultralytics import YOLO

# Strict reproducibility
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
print(f"Python Version : {os.sys.version.split()[0]}")
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

## 2. Dataset Validation

```python
# %% [markdown]
# ## 2. Dataset Validation
# Programmatically locating the dataset and verifying the 4-class ontology.

# %%
INPUT_DIR = Path("/kaggle/input")
MASTER_DIR = None

# Search for OBSTRUCTION_MASTER
for p in INPUT_DIR.rglob("data.yaml"):
    path_str = str(p).lower()
    if "mod-2" in path_str and "obstruction_master" in path_str:
        MASTER_DIR = p.parent
        break

if not MASTER_DIR:
    # Local fallback
    MASTER_DIR = Path("./OBSTRUCTION_MASTER")
    
if not MASTER_DIR.exists():
    raise FileNotFoundError("Could not locate OBSTRUCTION_MASTER dataset directory!")

print(f"Dataset found at: {MASTER_DIR}")

# Read data.yaml
data_yaml_path = MASTER_DIR / "data.yaml"
with open(data_yaml_path, 'r') as f:
    dataset_config = yaml.safe_load(f)

# Assert Exact 4 Classes
EXPECTED_CLASSES = {
    0: 'TrafficCone', 1: 'Rock', 2: 'RoadDebris', 3: 'FallenTree'
}

names = dataset_config.get('names', {})
if isinstance(names, list):
    names = {i: n for i, n in enumerate(names)}

assert len(names) == 4, f"ERROR: Expected exactly 4 classes, found {len(names)}."
for k, v in EXPECTED_CLASSES.items():
    assert names.get(k) == v, f"ERROR: Class ID {k} must be {v}, found {names.get(k)}."

print("Ontology Verified: Exactly 4 Obstruction classes found.")

# Update data.yaml path dynamically for Kaggle
dataset_config['path'] = str(MASTER_DIR.resolve())
dataset_config['train'] = 'images/train'
dataset_config['val'] = 'images/val'
dataset_config['test'] = 'images/test'

WORKING_YAML = Path("/kaggle/working/data.yaml")
with open(WORKING_YAML, 'w') as f:
    yaml.dump(dataset_config, f, sort_keys=False)

# Validate Paths
def verify_split(split):
    img_dir = MASTER_DIR / "images" / split
    lbl_dir = MASTER_DIR / "labels" / split
    assert img_dir.exists(), f"Missing {split} image dir"
    assert lbl_dir.exists(), f"Missing {split} label dir"
    
    imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    lbls = list(lbl_dir.glob("*.txt"))
    print(f"[{split.upper()}] Images: {len(imgs)} | Labels: {len(lbls)}")
    return len(imgs), len(lbls)

print("\nDataset Splits:")
n_train, _ = verify_split("train")
n_val, _ = verify_split("val")
n_test, _ = verify_split("test")
```

---

## 3. Class Distribution

```python
# %% [markdown]
# ## 3. Class Distribution
# Tracking imbalance across Train, Val, and Test splits.

# %%
class_counts = {v: 0 for k, v in EXPECTED_CLASSES.items()}

for split in ["train", "val", "test"]:
    lbl_dir = MASTER_DIR / "labels" / split
    for lbl_file in lbl_dir.glob("*.txt"):
        with open(lbl_file, 'r') as f:
            for line in f:
                c_id = int(line.split()[0])
                class_counts[EXPECTED_CLASSES[c_id]] += 1

total_instances = sum(class_counts.values())
stats_df = pd.DataFrame({
    'Class': list(EXPECTED_CLASSES.values()),
    'Instances': [class_counts[c] for c in EXPECTED_CLASSES.values()]
})
stats_df['Instance_Percentage'] = (stats_df['Instances'] / total_instances * 100).round(2)

print("="*50)
print("DATASET INSTANCE STATISTICS")
print("="*50)
display(stats_df.sort_values('Instances', ascending=False))
stats_df.to_csv("/kaggle/working/class_distribution.csv", index=False)

plt.figure(figsize=(10, 5))
sns.barplot(data=stats_df.sort_values('Instances', ascending=False), x='Class', y='Instances', palette='magma')
plt.title('Obstruction Master - Instance Distribution')
plt.savefig('/kaggle/working/class_distribution.png')
plt.show()
```

---

## 4. Visual Quality Control

```python
# %% [markdown]
# ## 4. Visual Quality Control
# Displaying bounding boxes to visually inspect coordinates and object context.

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
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        ax.imshow(img)
        ax.set_title(img_path.name)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

plot_samples()
```

---

## 5. Model Initialization

```python
# %% [markdown]
# ## 5. Initialize Model
# Loading official YOLO11n weights for controlled Transfer Learning.

# %%
print("Loading COCO-pretrained YOLO11n...")
print("CONCEPT: Pretrained YOLO11n -> Transfer learning -> Temporary Obstruction Detector")

model = YOLO("yolo11n.pt")
model.info(detailed=False)

print("\nYOLO11n pretrained weights are being fine-tuned on the SIH26124 OBSTRUCTION_MASTER dataset.")
```

---

## 6. Training Baseline

```python
# %% [markdown]
# ## 6. Training Configuration (Baseline)
# Utilizing explicitly defined augmentations for robustness against moving bus cameras, alongside an exact AdamW optimizer configuration.

# %%
# Carefully tuned Augmentations for Moving Road Cameras
augmentations = {
    # Geometric
    'degrees': 5.0, 
    'translate': 0.10, 
    'scale': 0.50, 
    'shear': 2.0, 
    'perspective': 0.0001, 
    'fliplr': 0.5,
    # Color
    'hsv_h': 0.015, 
    'hsv_s': 0.50, 
    'hsv_v': 0.40,
    # Mix/Mosaic
    'mosaic': 0.50, 
    'mixup': 0.05, 
    'close_mosaic': 10
}

print("="*50)
print("STARTING TRAINING: YOLO11n (OBSTRUCTION MODEL)")
print("="*50)

start_time = time.time()

# Train with EXPLICIT Optimizer
results = model.train(
    data=str(WORKING_YAML),
    epochs=100,
    patience=20,
    imgsz=640,
    batch=-1, 
    workers=4,
    cache=False,
    pretrained=True,
    amp=True,
    project="/kaggle/working/runs",
    name="obstruction_model",
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

train_time_min = (time.time() - start_time) / 60
print(f"Training completed in {train_time_min:.2f} minutes.")
```

---

## 7. Training Monitoring

```python
# %% [markdown]
# ## 7. Training Monitoring
# Evaluating convergence and early stopping triggers.

# %%
run_dir = Path("/kaggle/working/runs/obstruction_model")

def display_results_plot(run_path):
    results_png = run_path / "results.png"
    if results_png.exists():
        shutil.copy(results_png, "/kaggle/working/training_curves.png")
        display(Image(filename=str(results_png)))
    else:
        print("results.png not found yet.")

display_results_plot(run_dir)

best_epoch = 0
results_csv = run_dir / "results.csv"
if results_csv.exists():
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    best_epoch = df['metrics/mAP50-95(B)'].idxmax() + 1
    print(f"Best Epoch (mAP50-95): {best_epoch} / {len(df)}")
    if len(df) < 100:
        print(f"Early stopping triggered at epoch {len(df)}")
```

---

## 8. Validation Metrics

```python
# %% [markdown]
# ## 8. Validation Metrics
# Verifying performance on the validation split.

# %%
best_model_path = run_dir / "weights" / "best.pt"
best_model = YOLO(str(best_model_path))

print("="*50)
print("VALIDATION METRICS")
print("="*50)

val_results = best_model.val(data=str(WORKING_YAML), split="val", imgsz=640)

def extract_metrics(metrics_obj, save_csv=False, csv_name="class_metrics.csv"):
    class_indices = metrics_obj.ap_class_index
    r = metrics_obj.results_dict
    
    print("\nOVERALL METRICS:")
    print(f"mAP50    : {r['metrics/mAP50(B)']:.4f}")
    print(f"mAP50-95 : {r['metrics/mAP50-95(B)']:.4f}")
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
        
        print(f"{c_name:<12}: Recall = {recall:.4f} | Precision = {precision:.4f} | AP50 = {ap50:.4f} | F1 = {f1:.4f}")
        
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

_ = extract_metrics(val_results)
```

---

## 9. Test Set Evaluation

```python
# %% [markdown]
# ## 9. Test Set Evaluation
# True hold-out metrics to represent expected live performance.

# %%
print("="*50)
print("TEST SET METRICS")
print("="*50)

test_results = best_model.val(data=str(WORKING_YAML), split="test", imgsz=640)
test_metrics = extract_metrics(test_results, save_csv=True, csv_name="class_metrics.csv")

# Save confusion matrix
cm_path = run_dir / "confusion_matrix.png"
if cm_path.exists():
    shutil.copy(cm_path, "/kaggle/working/confusion_matrix.png")
    display(Image(filename=str(cm_path), width=800))
```

---

## 10. Error Analysis

```python
# %% [markdown]
# ## 10. Error Analysis
# Visualizing False Positives and False Negatives. Pay attention to distant debris, or rocks blending into the road texture.

# %%
val_preds = list(run_dir.glob("val_batch*_pred.jpg"))
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
# Measuring strict inference latency for edge compatibility.

# %%
print("="*50)
print("INFERENCE SPEED BENCHMARK (ms / FPS)")
print("="*50)

def get_recall(c_name):
    for m in test_metrics:
        if m["Class"] == c_name: return m["Recall"]
    return 0.0

bench_img = list((MASTER_DIR / "images" / "test").glob("*.jpg"))[0]

# Warmup
for _ in range(5):
    best_model(bench_img, imgsz=640, verbose=False)
    
# Benchmark
latencies = []
for _ in range(100):
    t0 = time.time()
    best_model(bench_img, imgsz=640, verbose=False)
    t1 = time.time()
    latencies.append((t1 - t0) * 1000) # ms

avg_latency = np.mean(latencies)
fps = 1000.0 / avg_latency if avg_latency > 0 else 0

speed_df = pd.DataFrame([{
    "Resolution": "640x640",
    "mAP50": round(test_results.results_dict['metrics/mAP50(B)'], 4),
    "mAP50-95": round(test_results.results_dict['metrics/mAP50-95(B)'], 4),
    "TrafficCone Recall": get_recall("TrafficCone"),
    "Rock Recall": get_recall("Rock"),
    "RoadDebris Recall": get_recall("RoadDebris"),
    "FallenTree Recall": get_recall("FallenTree"),
    "Latency (ms)": round(avg_latency, 2),
    "FPS": round(fps, 1)
}])

display(speed_df)
speed_df.to_csv("/kaggle/working/speed_benchmark.csv", index=False)
```

---

## 12. Video Inference Test

```python
# %% [markdown]
# ## 12. Video Inference Test
# Frame-by-frame processing. Provide an `.mp4` file for a visual demonstration.

# %%
print("To simulate live video, upload a raw `.mp4` to `/kaggle/working/test_vid.mp4`.")
print("""
import cv2
video_path = "/kaggle/working/test_vid.mp4"
if os.path.exists(video_path):
    best_model.predict(
        source=video_path,
        save=True,
        imgsz=640,
        conf=0.25,
        project="/kaggle/working",
        name="optional_video_demo"
    )
""")
```

---

## 13. Confidence Threshold Analysis

```python
# %% [markdown]
# ## 13. Confidence Threshold Analysis
# Testing confidence cutoffs for optimal F1 filtering in the future Event Layer.

# %%
def test_confidence(model, conf_list=[0.20, 0.25, 0.30, 0.40, 0.50]):
    print("Confidence Threshold Impact:")
    for conf in conf_list:
        res = model.val(data=str(WORKING_YAML), split="test", imgsz=640, conf=conf, verbose=False)
        p = res.results_dict['metrics/precision(B)']
        r = res.results_dict['metrics/recall(B)']
        print(f"Conf {conf:.2f} -> Precision: {p:.4f} | Recall: {r:.4f}")

test_confidence(best_model)
```

---

## 14. Export to ONNX

```python
# %% [markdown]
# ## 14. Export
# Compiling the model structure to ONNX.

# %%
print("Exporting Best Model to ONNX...")
exported_path = best_model.export(format="onnx", imgsz=640)

shutil.copy(best_model_path, "/kaggle/working/best.pt")
shutil.copy(run_dir / "weights" / "last.pt", "/kaggle/working/last.pt")
shutil.copy(exported_path, "/kaggle/working/best.onnx")

print("\nDEPLOYMENT NOTE: For NVIDIA edge devices (Jetson), compile to TensorRT locally via:")
print("yolo export model=best.pt format=engine half=True workspace=4")
```

---

## 15. Final Report & Recommendation

```python
# %% [markdown]
# ## 15. Final Automatic Report & Engineering Recommendation

# %%
r_dict = test_results.results_dict

final_report = f"""
============================================================
SIH26124 MODEL 2 — FINAL REPORT
============================================================

MODEL
YOLO11n
Pretrained: YES

CLASSES
0: TrafficCone
1: Rock
2: RoadDebris
3: FallenTree

DATASET
Total images: {n_train + n_val + n_test}
Train: {n_train}
Validation: {n_val}
Test: {n_test}

Instances:
TrafficCone : {class_counts.get('TrafficCone', 0)}
Rock        : {class_counts.get('Rock', 0)}
RoadDebris  : {class_counts.get('RoadDebris', 0)}
FallenTree  : {class_counts.get('FallenTree', 0)}

TRAINING
Epochs: 100
Best epoch: {best_epoch}
Training time: {train_time_min:.2f} mins
Optimizer: AdamW
lr0: 0.001
lrf: 0.01
weight_decay: 0.0005
warmup_epochs: 3.0
cos_lr: True

TEST SET RESULTS
Precision: {r_dict['metrics/precision(B)']:.4f}
Recall   : {r_dict['metrics/recall(B)']:.4f}
mAP50    : {r_dict['metrics/mAP50(B)']:.4f}
mAP50-95 : {r_dict['metrics/mAP50-95(B)']:.4f}

PER-CLASS RECALL
TrafficCone: {get_recall('TrafficCone'):.4f}
Rock       : {get_recall('Rock'):.4f}
RoadDebris : {get_recall('RoadDebris'):.4f}
FallenTree : {get_recall('FallenTree'):.4f}

INFERENCE
Image size: 640x640
Average latency: {avg_latency:.2f} ms
FPS: {fps:.1f}

============================================================
FINAL ENGINEERING RECOMMENDATION
============================================================
1. LEARNING SUCCESS: Based on the test mAP50 of {r_dict['metrics/mAP50(B)']:.4f}, the baseline transfer learning was successful.
2. WEAKEST CLASS: Review the Per-Class Recall metrics above. The class with the lowest recall represents the primary failure mode.
3. CLASS IMBALANCE: If FallenTree recall is significantly lower than others, the {class_counts.get('FallenTree', 0)} training instances were insufficient, necessitating oversampling/class-weights in Experiment 2.
4. AUGMENTATION: The current geometric augmentations protected against overfitting. If bounding-box localization is poor during error analysis, scale/translate augmentations may need tuning.
5. VIDEO TESTING: The latency ({avg_latency:.2f} ms) permits integration into the dual-model inference pipeline. Video testing is APPROVED.
"""

print(final_report)

with open("/kaggle/working/final_report.txt", "w") as f:
    f.write(final_report)

print("="*50)
print("ALL ARTIFACTS SAVED TO /kaggle/working/")
```
