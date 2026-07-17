# Multi-Agent Customer Support — Digital Music Store
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/app \
    GRADIO_ANALYTICS_ENABLED=False \
    HOST=0.0.0.0 \
    PORT=7860

# Install dependencies first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source.
COPY app.py .
COPY src/ src/
COPY pwa/ pwa/

# Run as an unprivileged user while keeping the runtime cache writable.
RUN addgroup --system app && \
    adduser --system --ingroup app --home /home/app app && \
    mkdir -p /home/app /app/data && \
    chown -R app:app /home/app /app/data
USER app

# Gradio listens on 7860 by default.
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; port = os.getenv('PORT', '7860'); urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=3).read(1)"]

CMD ["python", "app.py"]
