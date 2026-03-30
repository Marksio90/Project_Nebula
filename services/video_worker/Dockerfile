##############################################################################
# Video Worker — FFmpeg hardware-accelerated rendering & viral slicer
# Uses jrottenberg/ffmpeg as a proven FFmpeg base with codec support
##############################################################################
FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# FFmpeg with full codec support (h264, hevc, vp9, opus, aac)
# + libsndfile for audio analysis in the slicer
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY services/video_worker/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY shared /app/shared
COPY services/video_worker /app/services/video_worker

ENV PYTHONPATH=/app
