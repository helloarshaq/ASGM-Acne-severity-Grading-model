# ────────────────────────────────────────────────────────────────────────────
# 1) Use a minimal Python base image (slim) to keep the image small
FROM python:3.9-slim

# Prevent Python from writing .pyc files and ensure stdout/stderr are unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ────────────────────────────────────────────────────────────────────────────
# 2) Install system dependencies needed for OpenCV, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
       git \
       libgl1 \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ────────────────────────────────────────────────────────────────────────────
# 3) Create and switch to the working directory
WORKDIR /app

# ────────────────────────────────────────────────────────────────────────────
# 4) Copy requirements file first (to leverage Docker layer caching)
COPY requirements.txt .

# ────────────────────────────────────────────────────────────────────────────
# 5) Install Python dependencies
#    - First install CPU‐only PyTorch and torchvision (to keep image size < 4 GB)
#    - Then install everything in requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.0.0+cpu torchvision==0.15.1+cpu \
       --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ────────────────────────────────────────────────────────────────────────────
# 6) Copy all application files into the container
COPY . .

# ────────────────────────────────────────────────────────────────────────────
# 7) Expose port 8000 (Railway will map $PORT to this)
EXPOSE 8000

# ────────────────────────────────────────────────────────────────────────────
# 8) Start the FastAPI app via Uvicorn, using shell form so $PORT is expanded
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
