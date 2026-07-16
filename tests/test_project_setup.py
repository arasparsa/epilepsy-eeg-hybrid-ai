"""Basic tests for validating the initial project setup."""

from pathlib import Path


def test_required_directories_exist() -> None:
    """Check whether the main project directories exist."""
    required_directories = [
        Path("config"),
        Path("data/raw"),
        Path("data/interim"),
        Path("data/processed"),
        Path("metadata"),
        Path("notebooks"),
        Path("results"),
        Path("scripts"),
        Path("src"),
        Path("tests"),
    ]

    missing = [
        str(directory)
        for directory in required_directories
        if not directory.exists()
    ]

    assert not missing, f"Missing directories: {missing}"