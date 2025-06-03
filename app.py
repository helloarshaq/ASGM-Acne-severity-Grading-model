import os, io
import cv2
import torch
import timm
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ──────────────────────────────────────────────────────────────────────
# Device & model-file path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "model", "student_best.pth")

# ──────────────────────────────────────────────────────────────────────
# Define StudentNet architecture (must match training)
class StudentNet(nn.Module):
    def __init__(self, n_classes: int = 4):
        super().__init__()
        self.backbone = timm.create_model("mobilenetv3_small_050", pretrained=False)
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.backbone.classifier.in_features, n_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# Build model & load weights
model = StudentNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ──────────────────────────────────────────────────────────────────────
# Pre-processing (same as training)
infer_tf = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

# ──────────────────────────────────────────────────────────────────────
# FastAPI
app = FastAPI(
    title="ASGM – Acne Severity Grading Model",
    description="Upload a face image, get acne grade (0–3).",
    version="1.0"
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return JSONResponse(status_code=400, content={"error": "Bad image"})

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = infer_tf(image=rgb)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        grade = int(model(tensor).argmax(1).item())

    return {"filename": file.filename, "predicted_grade": grade}

# Local run:  uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
