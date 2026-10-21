# P2-A: reproducible single-container local demo (Python 3.11, CPU-capable)
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app
ARG DEBIAN_MIRROR=https://deb.debian.org

# FAISS / PyTorch runtime libraries; TLS roots for Hugging Face / DashScope.
RUN sed -i "s|http://deb.debian.org|${DEBIAN_MIRROR}|g" /etc/apt/sources.list.d/debian.sources \
    && apt-get update -o APT::Update::Error-Mode=any \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

# Configurable Python package index.
ARG PIP_INDEX_URL=https://pypi.org/simple

RUN python -m pip install --upgrade pip --index-url "${PIP_INDEX_URL}" \
    && python -m pip install -r requirements.txt --index-url "${PIP_INDEX_URL}"

COPY src/ ./src/
COPY frontend/ ./frontend/
COPY scripts/build_index.py ./scripts/build_index.py

RUN mkdir -p /app/data /opt/hf-cache

EXPOSE 8000

# Database initialization is idempotent (CREATE TABLE IF NOT EXISTS / INSERT OR IGNORE).
# RAGService loads before FastAPI accepts requests.
CMD ["sh", "-c", "python -m src.orders.init_db && exec python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 1"]
