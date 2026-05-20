# Streamlit app for the Deep Research Agent, packaged for Hugging Face Spaces.
# HF Spaces deprecated the Streamlit built-in SDK in April 2025, so we use Docker.

FROM python:3.11-slim

# Avoid python bytecode + interactive prompts; show logs immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# System packages needed by onnxruntime (libgomp1) and a few C-extension builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces convention: run as a non-root user with uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Cache locations that must be writable by the `user` user.
ENV XDG_CACHE_HOME=/home/user/.cache \
    HF_HOME=/home/user/.cache/huggingface \
    FASTEMBED_CACHE_DIR=/home/user/.cache/fastembed \
    SESSION_DB_PATH=/home/user/app/session.db

# Install Python deps first to maximise layer caching.
COPY --chown=user requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Pre-download the embedding model at build time so the first request is fast.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

# Copy the rest of the app.
COPY --chown=user . .

# HF Spaces routes external traffic to port 7860.
EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
