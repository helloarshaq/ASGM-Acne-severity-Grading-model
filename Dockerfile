# ---------- Dockerfile (root of repo) ----------
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.0.0+cpu torchvision==0.15.1+cpu \
      --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Hard-code 8000 so no env-var expansion is needed
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000"]
