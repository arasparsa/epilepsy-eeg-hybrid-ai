"""Parser for CHB-MIT case summary files.

The CHB-MIT annotation source consists of case-level text summary files.
Each summary may contain channel metadata followed by one or more recording
blocks. Each recording block describes one EDF file and may include seizure
onset and offset times in seconds from the start of that recording.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator


FILE_NAME_PATTERN = re.compile(
    r"^\s*File Name:\s*(?P<filename>.+?\.edf)\s*$",
    flags=re.IGNORECASE,
)

FILE_START_PATTERN = re.compile(
    r"^\s*File Start Time:\s*(?P<time>[0-9:]+)\s*$",
    flags=re.IGNORECASE,
)

FILE_END_PATTERN = re.compile(
    r"^\s*File End Time:\s*(?P<time>[0-9:]+)\s*$",
    flags=re.IGNORECASE,
)

SEIZURE_COUNT_PATTERN = re.compile(
    r"^\s*Number of Seizures in File:\s*(?P<count>\d+)\s*$",
    flags=re.IGNORECASE,
)

# Accepts both:
# Seizure 1 Start Time: 2996 seconds
# Seizure Start Time: 362 seconds
SEIZURE_START_PATTERN = re.compile(
    r"^\s*Seizure(?:\s+(?P<index>\d+))?\s+Start Time:\s*"
    r"(?P<seconds>\d+(?:\.\d+)?)\s*seconds?\s*$",
    flags=re.IGNORECASE,
)

SEIZURE_END_PATTERN = re.compile(
    r"^\s*Seizure(?:\s+(?P<index>\d+))?\s+End Time:\s*"
    r"(?P<seconds>\d+(?:\.\d+)?)\s*seconds?\s*$",
    flags=re.IGNORECASE,
)

CHANNEL_COUNT_PATTERN = re.compile(
    r"^\s*Channels in EDF Files:\s*$",
    flags=re.IGNORECASE,
)

CHANNEL_LINE_PATTERN = re.compile(
    r"^\s*Channel\s+(?P<index>\d+):\s*(?P<name>.+?)\s*$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedSeizure:
    """One seizure interval parsed from a CHB-MIT summary."""

    seizure_index: int
    onset_seconds: float
    offset_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.offset_seconds - self.onset_seconds

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True)
class ParsedRecording:
    """One EDF recording block parsed from a case summary."""

    edf_filename: str
    file_start_time: str | None
    file_end_time: str | None
    n_seizures_reported: int | None
    seizures: tuple[ParsedSeizure, ...]
    summary_filename: str
    summary_relative_path: str

    @property
    def n_seizures_parsed(self) -> int:
        return len(self.seizures)

    @property
    def total_ictal_seconds(self) -> float:
        return sum(
            seizure.duration_seconds
            for seizure in self.seizures
        )

    def to_dict(self) -> dict:
        return {
            "edf_filename": self.edf_filename,
            "file_start_time": self.file_start_time,
            "file_end_time": self.file_end_time,
            "n_seizures_reported": self.n_seizures_reported,
            "n_seizures_parsed": self.n_seizures_parsed,
            "total_ictal_seconds": self.total_ictal_seconds,
            "summary_filename": self.summary_filename,
            "summary_relative_path": self.summary_relative_path,
        }


@dataclass(frozen=True)
class ParsedSummary:
    """Parsed contents of one case-level summary file."""

    case_id: str
    summary_filename: str
    summary_relative_path: str
    declared_channels: tuple[str, ...]
    recordings: tuple[ParsedRecording, ...]


class SummaryParseError(ValueError):
    """Raised when a CHB-MIT summary cannot be parsed safely."""


def extract_case_id(path: Path) -> str:
    """Extract a CHB-MIT case identifier such as chb01."""
    candidates = [
        path.parent.name.lower(),
        path.stem.lower(),
    ]

    for candidate in candidates:
        match = re.search(r"chb\d{2}", candidate)
        if match:
            return match.group(0)

    raise SummaryParseError(
        f"Could not infer CHB-MIT case ID from: {path}"
    )


def parse_clock_time(value: str) -> int:
    """Convert CHB-MIT clock strings into seconds from a nominal day start.

    CHB-MIT may include hour values greater than 23 in some summary files.
    Therefore datetime.strptime is not used here.
    """
    parts = value.strip().split(":")

    if len(parts) != 3:
        raise SummaryParseError(
            f"Unexpected clock-time format: {value!r}"
        )

    try:
        hours, minutes, seconds = (
            int(part) for part in parts
        )
    except ValueError as exc:
        raise SummaryParseError(
            f"Non-integer clock-time component: {value!r}"
        ) from exc

    if hours < 0:
        raise SummaryParseError(
            f"Hour cannot be negative: {value!r}"
        )

    if not 0 <= minutes <= 59:
        raise SummaryParseError(
            f"Invalid minute in clock time: {value!r}"
        )

    if not 0 <= seconds <= 59:
        raise SummaryParseError(
            f"Invalid second in clock time: {value!r}"
        )

    return hours * 3600 + minutes * 60 + seconds


def calculate_clock_duration_seconds(
    start_time: str | None,
    end_time: str | None,
) -> tuple[float | None, bool | None]:
    """Calculate duration between summary clock times.

    Returns
    -------
    duration_seconds:
        Duration based on summary clocks.
    crosses_midnight:
        True when a standard 0–23 end hour precedes the start hour.
        False otherwise. None when either value is absent.

    Notes
    -----
    Some CHB-MIT summaries use hour values greater than 23. Those values are
    already extended beyond midnight and should not receive another 24 hours.
    """
    if start_time is None or end_time is None:
        return None, None

    start_seconds = parse_clock_time(start_time)
    end_seconds = parse_clock_time(end_time)

    start_hour = int(start_time.split(":")[0])
    end_hour = int(end_time.split(":")[0])

    crosses_midnight = False

    if end_seconds < start_seconds:
        if end_hour <= 23:
            end_seconds += 24 * 3600
            crosses_midnight = True
        else:
            raise SummaryParseError(
                "End time is earlier than start time even though "
                f"an extended-hour representation is used: "
                f"{start_time} -> {end_time}"
            )

    return float(end_seconds - start_seconds), crosses_midnight


def _finalize_recording(
    current: dict,
    summary_path: Path,
    raw_root: Path,
) -> ParsedRecording:
    """Validate and convert one temporary recording dictionary."""
    filename = current.get("edf_filename")

    if not filename:
        raise SummaryParseError(
            f"Recording block without EDF filename in {summary_path}"
        )

    starts: list[tuple[int | None, float]] = current.get(
        "seizure_starts",
        [],
    )
    ends: list[tuple[int | None, float]] = current.get(
        "seizure_ends",
        [],
    )

    if len(starts) != len(ends):
        raise SummaryParseError(
            f"Unequal seizure start/end counts for {filename}: "
            f"{len(starts)} starts and {len(ends)} ends"
        )

    seizures: list[ParsedSeizure] = []

    for position, (start_item, end_item) in enumerate(
        zip(starts, ends, strict=True),
        start=1,
    ):
        start_index, onset = start_item
        end_index, offset = end_item

        if (
            start_index is not None
            and end_index is not None
            and start_index != end_index
        ):
            raise SummaryParseError(
                f"Seizure index mismatch in {filename}: "
                f"start index={start_index}, end index={end_index}"
            )

        inferred_index = (
            start_index
            if start_index is not None
            else end_index
            if end_index is not None
            else position
        )

        if inferred_index is None:
            inferred_index = position

        seizures.append(
            ParsedSeizure(
                seizure_index=int(inferred_index),
                onset_seconds=float(onset),
                offset_seconds=float(offset),
            )
        )

    seizure_indices = [
        seizure.seizure_index for seizure in seizures
    ]

    if len(seizure_indices) != len(set(seizure_indices)):
        raise SummaryParseError(
            f"Duplicate seizure indices in {filename}: "
            f"{seizure_indices}"
        )

    seizures = sorted(
        seizures,
        key=lambda item: (
            item.seizure_index,
            item.onset_seconds,
        ),
    )

    reported_count = current.get("n_seizures_reported")

    relative_path = summary_path.relative_to(raw_root)

    return ParsedRecording(
        edf_filename=filename.strip(),
        file_start_time=current.get("file_start_time"),
        file_end_time=current.get("file_end_time"),
        n_seizures_reported=reported_count,
        seizures=tuple(seizures),
        summary_filename=summary_path.name,
        summary_relative_path=relative_path.as_posix(),
    )


def parse_summary_file(
    summary_path: Path,
    raw_root: Path,
) -> ParsedSummary:
    """Parse one CHB-MIT case summary file."""
    summary_path = summary_path.resolve()
    raw_root = raw_root.resolve()

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Summary file not found: {summary_path}"
        )

    text = summary_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    lines = text.splitlines()

    declared_channels: list[str] = []
    recordings: list[ParsedRecording] = []
    current: dict | None = None
    reading_channel_section = False

    for line in lines:
        file_match = FILE_NAME_PATTERN.match(line)

        if file_match:
            if current is not None:
                recordings.append(
                    _finalize_recording(
                        current=current,
                        summary_path=summary_path,
                        raw_root=raw_root,
                    )
                )

            current = {
                "edf_filename": file_match.group("filename"),
                "file_start_time": None,
                "file_end_time": None,
                "n_seizures_reported": None,
                "seizure_starts": [],
                "seizure_ends": [],
            }
            reading_channel_section = False
            continue

        if current is None:
            if CHANNEL_COUNT_PATTERN.match(line):
                reading_channel_section = True
                continue

            if reading_channel_section:
                channel_match = CHANNEL_LINE_PATTERN.match(line)

                if channel_match:
                    declared_channels.append(
                        channel_match.group("name").strip()
                    )
                    continue

                if line.strip() == "":
                    continue

            continue

        start_match = FILE_START_PATTERN.match(line)

        if start_match:
            current["file_start_time"] = start_match.group("time")
            continue

        end_match = FILE_END_PATTERN.match(line)

        if end_match:
            current["file_end_time"] = end_match.group("time")
            continue

        count_match = SEIZURE_COUNT_PATTERN.match(line)

        if count_match:
            current["n_seizures_reported"] = int(
                count_match.group("count")
            )
            continue

        seizure_start_match = SEIZURE_START_PATTERN.match(line)

        if seizure_start_match:
            index_text = seizure_start_match.group("index")
            current["seizure_starts"].append(
                (
                    int(index_text)
                    if index_text is not None
                    else None,
                    float(
                        seizure_start_match.group("seconds")
                    ),
                )
            )
            continue

        seizure_end_match = SEIZURE_END_PATTERN.match(line)

        if seizure_end_match:
            index_text = seizure_end_match.group("index")
            current["seizure_ends"].append(
                (
                    int(index_text)
                    if index_text is not None
                    else None,
                    float(
                        seizure_end_match.group("seconds")
                    ),
                )
            )
            continue

    if current is not None:
        recordings.append(
            _finalize_recording(
                current=current,
                summary_path=summary_path,
                raw_root=raw_root,
            )
        )

    if not recordings:
        raise SummaryParseError(
            f"No recording blocks found in {summary_path}"
        )

    case_id = extract_case_id(summary_path)

    return ParsedSummary(
        case_id=case_id,
        summary_filename=summary_path.name,
        summary_relative_path=summary_path.relative_to(
            raw_root
        ).as_posix(),
        declared_channels=tuple(declared_channels),
        recordings=tuple(recordings),
    )


def iter_summary_files(raw_root: Path) -> Iterator[Path]:
    """Yield likely CHB-MIT case summary files."""
    candidates = sorted(
        path
        for path in raw_root.rglob("*.txt")
        if "summary" in path.name.lower()
    )

    yield from candidates