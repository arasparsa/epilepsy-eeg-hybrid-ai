"""Verify the local structure of the CHB-MIT dataset."""

from pathlib import Path
import hashlib
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "chbmit"
REPORT_PATH = (
    PROJECT_ROOT / "reports" / "data_integrity" / "chbmit_inventory.json"
)


def calculate_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate SHA-256 checksum without loading the entire file into memory."""
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def build_inventory() -> dict:
    """Build an inventory of EDF and summary files."""
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {RAW_DIR}")

    edf_files = sorted(RAW_DIR.rglob("*.edf"))
    summary_files = sorted(RAW_DIR.rglob("*summary.txt"))

    inventory = {
        "raw_directory": str(RAW_DIR),
        "edf_file_count": len(edf_files),
        "summary_file_count": len(summary_files),
        "edf_files": [
            {
                "relative_path": str(path.relative_to(RAW_DIR)),
                "size_bytes": path.stat().st_size,
                "sha256": calculate_sha256(path),
            }
            for path in edf_files
        ],
        "summary_files": [
            str(path.relative_to(RAW_DIR))
            for path in summary_files
        ],
    }

    return inventory


def main() -> None:
    inventory = build_inventory()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(inventory, file, indent=2)

    print(f"EDF files: {inventory['edf_file_count']}")
    print(f"Summary files: {inventory['summary_file_count']}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()