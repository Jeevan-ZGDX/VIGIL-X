# ==============================================================================
# Production Dockerfile for VIGIL-X Pedestrian Safety Fusion System
# Supports RGB + Thermal Sensor Fusion, YOLOv5n Deep Learning, and Real-time Web HUD
# ==============================================================================

FROM python:3.11-slim-bookworm AS base

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install native OS dependencies required by OpenCV V4L2 capture and ONNX runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    v4l-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application modules, models, configuration, and synthetic benchmark datasets
COPY config/ ./config/
COPY models/ ./models/
COPY data/synthetic/ ./data/synthetic/
COPY scripts/ ./scripts/
COPY src/ ./src/

# Create runtime directory for event logs
RUN mkdir -p /app/data && touch /app/data/events.jsonl

# Expose FastAPI Dashboard port
EXPOSE 8000

# Container healthcheck querying internal ISO safety state
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default launch command: Real-time Pedestrian Safety Dashboard
CMD ["python", "scripts/run_system.py", "--mode", "dashboard", "--host", "0.0.0.0", "--port", "8000"]
