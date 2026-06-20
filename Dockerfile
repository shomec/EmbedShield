# Use a lightweight python base image
FROM python:3.11-slim

# Set environment variables for Python, pip, and Hugging Face caching
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install curl (used for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Pre-download the SentenceTransformer model weights during build to bake them into the image
COPY download_model.py .
RUN python download_model.py

# Copy application files
COPY safe_prompts.json .
COPY guard.py .
COPY app.py .
COPY dashboard.py .
COPY start.sh .

# Give execute permissions to the startup script and normalize line endings
RUN chmod +x start.sh && sed -i 's/\r$//' start.sh

# Expose ports
# 8000: FastAPI Gateway API
# 8501: Streamlit Dashboard UI
EXPOSE 8000
EXPOSE 8501

# Launch the unified gateway and dashboard services
CMD ["./start.sh"]
