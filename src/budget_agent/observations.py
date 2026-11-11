"""Parse the file tool's recorded output, never infer coverage from requests."""
from __future__ import annotations

from dataclasses import dataclass
import json
import re


_READ_LINE = re.compile(r"^(\d+): ")


@dataclass(frozen=True)
class ReadObservation:
    path: str
    lines: dict[int, str]
    truncated_line_content: bool | None


def parse_read_observation(output: str) -> ReadObservation | None:
    """Unsupported/partial legacy output has unknown coverage, not zero lines."""
    header, _, body = output.partition("\n")
    if not header.startswith("read_file "):
        return None
    try:
        metadata = json.loads(header.removeprefix("read_file "))
    except json.JSONDecodeError:
        return None
    if not isinstance(metadata, dict):
        return None
    path, total = metadata.get("path"), metadata.get("total_lines")
    start, end = metadata.get("start_line"), metadata.get("end_line")
    if not isinstance(path, str) or not path or type(total) is not int or total < 0:
        return None
    rows = body.splitlines()
    matches = [match for match in map(_READ_LINE.match, rows) if match is not None]
    if len(matches) != len(rows):
        return None
    numbers = [int(match[1]) for match in matches]
    if numbers:
        if (type(start) is not int or type(end) is not int or not 1 <= start <= end <= total
                or len(numbers) != end - start + 1
                or any(number != start + offset for offset, number in enumerate(numbers))):
            return None
    elif start is not None or end is not None:
        return None
    return ReadObservation(path, dict(zip(numbers, rows, strict=True)), metadata.get("truncated_line_content"))
