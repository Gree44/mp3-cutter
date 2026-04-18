#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract cut timestamps from a rekordbox XML library and trim an MP3 "
            "with ffmpeg stream copy (-c copy, no re-encode)."
        )
    )
    parser.add_argument("--xml", required=True, type=Path, help="Path to rekordbox XML export")
    parser.add_argument("--track", required=True, type=Path, help="Path to source MP3")
    parser.add_argument("--output", type=Path, help="Output MP3 path (default: <track>_cut.mp3)")
    parser.add_argument(
        "--start-mark",
        default="",
        help="Optional POSITION_MARK Name to use as cut start (case-insensitive)",
    )
    parser.add_argument(
        "--end-mark",
        default="",
        help="Optional POSITION_MARK Name to use as cut end (case-insensitive)",
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg binary name/path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered timestamps and ffmpeg command without executing",
    )
    return parser.parse_args()


def decode_rekordbox_location(raw_location: str) -> str:
    parsed = urlparse(raw_location)
    if parsed.scheme.lower() == "file":
        path = unquote(parsed.path or "")
        if path.startswith("/") and len(path) > 2 and path[2] == ":":
            path = path[1:]
        return str(Path(path).expanduser().resolve())
    return str(Path(unquote(raw_location)).expanduser().resolve())


def normalize_for_match(path: Path | str) -> str:
    return str(Path(path).expanduser().resolve()).casefold()


def find_track(root: ET.Element, target_track: Path) -> ET.Element:
    target = normalize_for_match(target_track)
    collection = root.find("./COLLECTION")
    if collection is None:
        raise ValueError("Invalid rekordbox XML: missing COLLECTION element")

    for track in collection.findall("TRACK"):
        location = track.attrib.get("Location", "")
        if not location:
            continue
        try:
            track_path = decode_rekordbox_location(location)
        except OSError:
            continue
        if normalize_for_match(track_path) == target:
            return track

    raise ValueError(f"Track not found in rekordbox XML for path: {target_track}")


def get_marker_times(track: ET.Element, start_mark: str, end_mark: str) -> tuple[float, float]:
    markers: list[tuple[str, float]] = []
    for pm in track.findall("POSITION_MARK"):
        raw_start = pm.attrib.get("Start")
        if raw_start is None:
            continue
        try:
            start_value = float(raw_start)
        except ValueError:
            continue
        markers.append((pm.attrib.get("Name", ""), start_value))

    if len(markers) < 2:
        raise ValueError("Track does not have enough POSITION_MARK entries to derive a cut")

    if start_mark and end_mark:
        wanted_start = start_mark.casefold()
        wanted_end = end_mark.casefold()
        start_candidates = [t for name, t in markers if name.casefold() == wanted_start]
        end_candidates = [t for name, t in markers if name.casefold() == wanted_end]
        if not start_candidates or not end_candidates:
            raise ValueError(
                "Could not find both requested marker names "
                f"('{start_mark}', '{end_mark}')"
            )
        start, end = start_candidates[0], end_candidates[0]
    else:
        named_start = [t for name, t in markers if name.casefold() in {"start", "in"}]
        named_end = [t for name, t in markers if name.casefold() in {"end", "out"}]
        if named_start and named_end:
            start, end = named_start[0], named_end[0]
        else:
            ordered = sorted(t for _, t in markers)
            start, end = ordered[0], ordered[1]

    if end <= start:
        raise ValueError(f"Invalid cut markers: end ({end}) must be greater than start ({start})")
    return start, end


def cut_mp3(ffmpeg_bin: str, source: Path, output: Path, start: float, end: float, dry_run: bool) -> int:
    command = [
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(source),
        "-c",
        "copy",
        str(output),
    ]
    print(f"Cut start: {start:.3f}s")
    print(f"Cut end:   {end:.3f}s")
    print("Command:", " ".join(command))

    if dry_run:
        return 0

    completed = subprocess.run(command, check=False)
    return completed.returncode


def main() -> int:
    args = parse_args()
    if not args.xml.exists():
        print(f"XML file not found: {args.xml}", file=sys.stderr)
        return 2
    if not args.track.exists():
        print(f"Track file not found: {args.track}", file=sys.stderr)
        return 2

    output = args.output or args.track.with_name(f"{args.track.stem}_cut{args.track.suffix}")

    try:
        root = ET.parse(args.xml).getroot()
        track = find_track(root, args.track)
        start, end = get_marker_times(track, args.start_mark, args.end_mark)
    except (ET.ParseError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return cut_mp3(args.ffmpeg_bin, args.track, output, start, end, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
