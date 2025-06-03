import io
import os
import uvicorn
import torch
import cv2
import timm
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ──────────────────────────────────────────────────────────────────────────────
# 1) SETTINGS: adjust paths
ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "model", "student_best.pth")
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

# ──────────────────────────────────────────────────────────────────────────────
# 2) Define the student architecture (same as before)
class StudentNet(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.backbone = timm.create_model("mobilenetv3_small_050", pretrained=False)
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.backbone.classifier.in_features, n_classes)
        )
    def forward(self, x):
        return self.backbone(x)

# ──────────────────────────────────────────────────────────────────────────────
# 3) Load the trained student model
model = StudentNet(n_classes=4).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ──────────────────────────────────────────────────────────────────────────────
# 4) Define the same preprocessing you used in training/validation
infer_transform = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(),  # default ImageNet mean/std
    ToTensorV2()
])

# ──────────────────────────────────────────────────────────────────────────────
# 5) FastAPI setup
app = FastAPI(
    title="Acne Grading API",
    description="Upload a face image, get back an acne grade (0–3).",
    version="1.0"
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Expects: a multipart/form-data POST with an image file.
    Returns: JSON { "filename": str, "predicted_grade": int }.
    """
    # 5A) Read the raw bytes and decode to OpenCV BGR image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image file."})

    # 5B) Convert BGR → RGB, apply transforms
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = infer_transform(image=rgb)["image"].unsqueeze(0).to(DEVICE)

    # 5C) Forward pass & get prediction
    with torch.no_grad():
        logits = model(tensor)
        pred_class = int(logits.argmax(dim=1).item())

    return {"filename": file.filename, "predicted_grade": pred_class}

# ──────────────────────────────────────────────────────────────────────────────
# 6) Run with: uvicorn app:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
