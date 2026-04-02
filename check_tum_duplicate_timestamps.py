#!/usr/bin/env python3
import argparse
import pathlib
import sys
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check a TUM trajectory file for duplicate timestamps."
    )
    parser.add_argument(
        "trajectory",
        type=pathlib.Path,
        help="Path to a TUM trajectory file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.trajectory.is_file():
        print(f"File not found: {args.trajectory}", file=sys.stderr)
        return 2

    timestamp_to_lines = defaultdict(list)
    data_line_count = 0

    with args.trajectory.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 8:
                print(
                    f"Invalid TUM trajectory line {line_no}: expected at least 8 columns, got {len(parts)}",
                    file=sys.stderr,
                )
                return 2

            timestamp = parts[0]
            timestamp_to_lines[timestamp].append(line_no)
            data_line_count += 1

    duplicates = {
        timestamp: line_nos
        for timestamp, line_nos in timestamp_to_lines.items()
        if len(line_nos) > 1
    }

    print(f"Scanned {data_line_count} trajectory entries from {args.trajectory}")

    if not duplicates:
        print("No duplicate timestamps found.")
        return 0

    print(f"Found {len(duplicates)} duplicate timestamp value(s):")
    for timestamp in sorted(duplicates):
        line_nos = duplicates[timestamp]
        line_str = ", ".join(str(line_no) for line_no in line_nos)
        print(f"{timestamp}: {len(line_nos)} entries on lines {line_str}")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
