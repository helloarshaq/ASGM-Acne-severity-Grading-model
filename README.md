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

