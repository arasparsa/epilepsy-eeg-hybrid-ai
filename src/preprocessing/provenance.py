"""Utilities for preprocessing provenance and file integrity."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def calculate_sha256(
    file_path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate SHA-256 without loading the full file."""
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def get_git_commit_hash() -> str:
    """Return the Git commit associated with a run."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return "unavailable"