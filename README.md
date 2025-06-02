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

