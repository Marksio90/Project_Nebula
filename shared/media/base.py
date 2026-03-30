"""
shared/media/base.py
─────────────────────────────────────────────────────────────────────────────
Abstract base classes for every external media-generation provider.

Design principles:
  - All providers implement the same interface → swappable at runtime via
    a single env var (AUDIO_PROVIDER, IMAGE_PROVIDER, VIDEO_PROVIDER).
  - Each method is synchronous (called inside Celery tasks).
  - Each method writes the result to disk and returns the file path —
    callers never deal with raw bytes or provider-specific objects.
  - No business logic here; only I/O contracts.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GeneratedAudio:
    file_path: str       # Absolute path to WAV on disk
    duration_s: float
    sample_rate: int


@dataclass(frozen=True)
class GeneratedImage:
    file_path: str       # Absolute path to PNG on disk
    width: int
    height: int
    aspect_ratio: str    # "16:9" | "9:16" | "1:1"


@dataclass(frozen=True)
class GeneratedVideo:
    file_path: str       # Absolute path to MP4 on disk
    duration_s: float
    width: int
    height: int
    source: str          # "api" | "kenburns" — for QA/logging


class AudioGenerator(ABC):
    """Generate a single 30-second music stem from a text prompt."""

    @abstractmethod
    def generate_stem(
        self,
        prompt: str,
        bpm: float,
        duration_s: int,
        output_path: str,
    ) -> GeneratedAudio:
        """
        Generate audio and write it to output_path.

        Parameters
        ----------
        prompt      : English text prompt describing the musical content.
        bpm         : Target BPM — included in the prompt and used for
                      downstream beat alignment.
        duration_s  : Target duration in seconds (typically 30).
        output_path : Full path where the WAV file must be written.

        Returns
        -------
        GeneratedAudio with verified file_path and metadata.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str: ...


class ImageGenerator(ABC):
    """Generate a single static image from a text prompt."""

    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        aspect_ratio: str,
        output_path: str,
    ) -> GeneratedImage:
        """
        Generate an image and write it to output_path as PNG.

        Parameters
        ----------
        prompt       : English text prompt.
        aspect_ratio : "16:9" | "9:16" | "1:1"
        output_path  : Full path where the PNG must be written.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str: ...


class VideoGenerator(ABC):
    """Generate a short loopable video clip from a prompt or source image."""

    @abstractmethod
    def generate_video_loop(
        self,
        prompt: str,
        aspect_ratio: str,
        output_path: str,
        source_image_path: str | None = None,
    ) -> GeneratedVideo:
        """
        Generate a video loop and write it to output_path as MP4.

        Parameters
        ----------
        prompt            : English text prompt.
        aspect_ratio      : "16:9" | "9:16"
        output_path       : Full path where the MP4 must be written.
        source_image_path : Optional static image to animate (used by the
                            FFmpeg Ken Burns provider). If None and required,
                            the provider will generate its own image first.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str: ...
