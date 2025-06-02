# ASGM-Acne-severity-Grading-model
A model for grading severity on Acne images.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)  
[![PyPI](https://img.shields.io/pypi/v/fastapi.svg)](https://pypi.org/project/fastapi/)  
[![Build Status](https://img.shields.io/github/actions/workflow/status/<your-username>/ASGM-AcneSeverityGrading/ci.yml?branch=main)](https://github.com/<your-username>/ASGM-AcneSeverityGrading/actions)

**ASGM (Acne Severity Grading Model)** is a lightweight deep‐learning system that classifies facial acne into four severity grades (0–3). It uses a two‐stage distillation approach:

1. **Teacher**: a ResNet‐34 model trained on ACNE04 to predict acne grades.  
2. **Student**: a compact MobileNetV3‐Small network distilled from the teacher, offering near‐teacher accuracy with minimal inference cost.  
3. **API**: a FastAPI wrapper serves the student model as a REST endpoint—upload a face image and receive its predicted grade in JSON.

---

## 🌟 Features

- **Knowledge‐distillation pipeline** (ResNet‐34 → MobileNetV3‐Small)  
- **High accuracy** on ACNE04 (~75 % test accuracy)  
- **Small & efficient student** (≈2.4 M parameters)  
- **Easy installation** via `requirements.txt`  
- **RESTful API** (FastAPI + Uvicorn) for on‐demand grading  
- **CLI script** (`inference.py`) for offline evaluation  
- **Full metrics** (classification report + confusion matrix)  
- **MIT License** (free for research & commercial use)

---

## 📂 Repository Structure
ASGM-AcneSeverityGrading/
├─ app.py # FastAPI server for inference
├─ inference.py # CLI/SDK script to grade a single image
├─ metrics.py # Test‐split evaluation: classification report & confusion matrix
├─ model/
│ └─ student_best.pth # Trained student weights (~6 MB)
├─ requirements.txt # pip install -r requirements.txt
├─ LICENSE # MIT License
├─ README.md # This file
└─ utils/
├─ dataset.py # Dataset class & DataLoader (ACNE04 CSV → PyTorch)
├─ models.py # TeacherNet (ResNet‐34) & StudentNet (MobileNetV3‐Small) definitions
├─ train.py # Script to train teacher & student (distillation)
└─ transforms.py # Albumentations pipelines for train/val/test

---

## 🚀 Quick Start

### 0. Prerequisites

- Python 3.8+  
- Linux/macOS/Windows (Linux or macOS recommended)  
- GPU + CUDA (optional; CPU works for inference only)  
- ~5 GB disk space for model & data

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/ASGM-AcneSeverityGrading.git
cd ASGM-AcneSeverityGrading

# (Optional) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
.\venv\Scripts\activate    # Windows PowerShell

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```
2. Download & Prepare ACNE04
Download ACNE04 from Kaggle into data/ACNE04/raw/.

Unzip so you have:

```swift
data/ACNE04/raw/acne0_1024/
data/ACNE04/raw/acne1_1024/
data/ACNE04/raw/acne2_1024/
data/ACNE04/raw/acne3_1024/
```
Run the split script to generate CSVs:

```bash
python utils/dataset.py \
  --raw_dir data/ACNE04/raw \
  --out_dir data/ACNE04/splits \
  --split 0.70 0.15 0.15
```
This produces:

```kotlin
data/ACNE04/splits/train.csv
data/ACNE04/splits/val.csv
data/ACNE04/splits/test.csv
```
3. Train Teacher & Student
```bash
# 3A: Train ResNet‐34 teacher
python utils/train.py --mode teacher \
  --train_csv data/ACNE04/splits/train.csv \
  --val_csv   data/ACNE04/splits/val.csv \
  --img_root  data/ACNE04/raw \
  --epochs    100 \
  --batch     32 \
  --lr        1e-3 \
  --save_dir  model/teacher_checkpoints
````
# 3B: Distill to MobileNetV3‐Small student
python utils/train.py --mode student \
  --train_csv    data/ACNE04/splits/train.csv \
  --val_csv      data/ACNE04/splits/val.csv \
  --img_root     data/ACNE04/raw \
  --teacher_ckpt model/teacher_checkpoints/best_teacher.pth \
  --epochs       200 \
  --batch        64 \
  --lr           2e-4 \
  --alpha        0.5 \
  --temp         4.0 \
  --save_dir     model/student_checkpoints
Teacher mode: trains on ground‐truth grades using ResNet‐34.

Student mode: uses KL‐divergence + CE loss to distill from teacher, saving best weights to model/student_checkpoints/best_student.pth.

4. Evaluate on Test Split
```bash
python metrics.py \
  --test_csv     data/ACNE04/splits/test.csv \
  --img_root     data/ACNE04/raw \
  --student_ckpt model/student_checkpoints/best_student.pth
```
Outputs:

Overall Test Accuracy

Classification Report (precision, recall, F1, support per class + averages)

Confusion Matrix (plotted as a heatmap)

🖥️ Inference (Single Image)
Use inference.py to grade a single image from the command line:

```bash
python inference.py \
  --model_path model/student_checkpoints/best_student.pth \
  --img_path   /path/to/your_face.jpg
```
Expected output:

```pgsql
Loaded model from model/student_checkpoints/best_student.pth
Input image: /path/to/your_face.jpg
Predicted acne grade: 2
```
