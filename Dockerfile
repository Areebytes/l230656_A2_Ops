# VisionGuard web app - production image (Task 6.2)
# Base image is a build argument so before/after sizes can be compared (Task 6.4)
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

# No .pyc files, logs appear immediately, no pip cache kept in the image, quieter TensorFlow
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# 1) Dependencies FIRST: this slow layer stays cached until requirements-serve.txt changes
COPY requirements_serve.txt .
RUN pip install -r requirements_serve.txt

# 2) Unprivileged user
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/instance && chown appuser:appuser /app/instance

# 3) Source code LAST: changes often, rebuilds in seconds
COPY src/inference.py src/inference.py
COPY app/ app/

USER appuser

# Model is MOUNTED read-only at /models. Secrets come at runtime, never baked in.
ENV SRC_DIR=/app/src \
    MODEL_PATH=/models/model.keras \
    MODEL_INFO_PATH=/models/model_info.json

EXPOSE 8000

# slim has no curl, so the health check uses Python's standard library
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", \
     "--timeout", "120", "--access-logfile", "-", "app.main:create_app()"]