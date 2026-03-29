##############################################################################
# DSP Worker — librosa beat-matching + pedalboard mastering
# Inherits from a science/audio base to avoid compiling native libs from scratch
##############################################################################
FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Native audio/science deps required by librosa, soundfile, pedalboard
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    libfftw3-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY ../../shared /app/shared
COPY . /app/services/dsp_worker

ENV PYTHONPATH=/app
