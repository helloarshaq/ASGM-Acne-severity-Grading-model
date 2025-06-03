# ────────────────────────────────────────────────────────────────────────────
# Dockerfile for ASGM (CPU-only, < 4 GB)
# ────────────────────────────────────────────────────────────────────────────
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Our service will always listen on 8000
ENV PORT 8000   

# --- system deps for OpenCV -------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- install python deps ----------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.0.0+cpu torchvision==0.15.1+cpu \
      --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# --- copy code --------------------------------------------------------------
COPY . .

EXPOSE 8000

# Shell-form CMD so $PORT expands (we set it to 8000 above)
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
