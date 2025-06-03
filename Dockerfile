# ────────────────────────────────────────────────────────────────────────────
# Use slim Python as our base (smaller than the default Ubuntu image)
FROM python:3.9-slim

# ─── Set up environment variables ────────────────────────────────────────────
# Prevent Python from writing .pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
# Ensure output is sent straight to the terminal (no buffering)
ENV PYTHONUNBUFFERED=1

# ─── Install system dependencies ─────────────────────────────────────────────
# - git (in case you need it)
# - libgl1 and libglib2.0-0 (for OpenCV to work on FFMPEG)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
&& rm -rf /var/lib/apt/lists/*

# ─── Create and switch to /app folder ────────────────────────────────────────
WORKDIR /app

# ─── Copy requirements first (to leverage Docker layer caching) ───────────────
COPY requirements.txt .

# ─── Install Python dependencies ─────────────────────────────────────────────
# We explicitly install CPU‐only PyTorch wheels here to keep the image small.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.0.0+cpu torchvision==0.15.1+cpu \
      --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ─── Copy the entire repo into the container ─────────────────────────────────
COPY . .

# ─── Expose the port Uvicorn will run on ────────────────────────────────────
EXPOSE 8000

# ─── Start the FastAPI app via Uvicorn ───────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
