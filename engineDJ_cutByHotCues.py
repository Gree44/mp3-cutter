#!/usr/bin/env python3
"""
Audio Cutter — Cut sections from MP3, FLAC, or WAV files using Engine DJ hotcues.
Removes audio between two hotcue positions without re-encoding.
"""

import struct
import sqlite3
import zlib
import os
import sys
import wave
from mutagen.flac import FLAC
from mutagen.id3 import ID3, TIT2
from mutagen.mp3 import MP3
from mutagen.wave import WAVE

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Edit these variables before running
# ═══════════════════════════════════════════════════════════════════════════════

TRACK_FILENAME = "01 - Titanium (feat. Sia) (Extended).flac"  # supports .mp3, .flac, .wav
HOTCUE_START = 5          # Hotcue number (1–8) — cut begins here
HOTCUE_END = 6            # Hotcue number (1–8) — cut ends here
OUTPUT_APPENDIX = "(Short Edit)"
OUTPUT_PATH = os.path.expanduser("~/Library/CloudStorage/OneDrive-Personal/DJing/Edits")
ENGINE_DB_PATH = os.path.expanduser("~/Music/Engine Library/Database2/m.db")

# Remap stale Engine DJ paths to their current locations when files have moved.
PATH_REMAPS = [
    ("~/Music/OneDrive", "~/Library/CloudStorage/OneDrive-Personal"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# MP3 constants
# ═══════════════════════════════════════════════════════════════════════════════

BITRATE_TABLE = {
    (1, 1): [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448],
    (1, 2): [32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384],
    (1, 3): [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
    (2, 1): [32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 320],
    (2, 2): [8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
    (2, 3): [8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
}

SAMPLE_RATE_TABLE = {
    1:   [44100, 48000, 32000],
    2:   [22050, 24000, 16000],
    2.5: [11025, 12000, 8000],
}

SAMPLES_PER_FRAME_TABLE = {
    (1, 1): 384, (1, 2): 1152, (1, 3): 1152,
    (2, 1): 384, (2, 2): 1152, (2, 3): 576,
    (2.5, 1): 384, (2.5, 2): 1152, (2.5, 3): 576,
}


# ═══════════════════════════════════════════════════════════════════════════════
# MP3 frame parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_frame_header(header_bytes):
    """Parse a 4-byte MP3 frame header. Returns a dict or None if invalid."""
    header = struct.unpack(">I", header_bytes)[0]

    if (header >> 21) != 0x7FF:
        return None

    version_bits = (header >> 19) & 0x03
    layer_bits = (header >> 17) & 0x03
    bitrate_idx = (header >> 12) & 0x0F
    srate_idx = (header >> 10) & 0x03
    padding = (header >> 9) & 0x01

    version = {0: 2.5, 2: 2, 3: 1}.get(version_bits)
    layer = {1: 3, 2: 2, 3: 1}.get(layer_bits)
    if version is None or layer is None:
        return None

    if bitrate_idx == 0 or bitrate_idx == 15 or srate_idx == 3:
        return None

    v_key = 1 if version == 1 else 2
    bitrate = BITRATE_TABLE.get((v_key, layer))
    if bitrate is None:
        return None
    bitrate = bitrate[bitrate_idx - 1] * 1000

    sample_rate = SAMPLE_RATE_TABLE[version][srate_idx]

    if layer == 1:
        frame_size = (12 * bitrate // sample_rate + padding) * 4
    elif version == 1:
        frame_size = 144 * bitrate // sample_rate + padding
    else:
        frame_size = 72 * bitrate // sample_rate + padding

    return {
        "frame_size": frame_size,
        "sample_rate": sample_rate,
        "samples_per_frame": SAMPLES_PER_FRAME_TABLE[(version, layer)],
    }


def read_id3v2_size(data):
    """Return total size of the ID3v2 tag (header + body), or 0 if absent."""
    if len(data) < 10 or data[:3] != b"ID3":
        return 0
    size = (
        ((data[6] & 0x7F) << 21)
        | ((data[7] & 0x7F) << 14)
        | ((data[8] & 0x7F) << 7)
        | (data[9] & 0x7F)
    )
    return size + 10


# ═══════════════════════════════════════════════════════════════════════════════
# FLAC constants and utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _make_crc8_table():
    t = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = ((c << 1) ^ 0x07) & 0xFF if c & 0x80 else (c << 1) & 0xFF
        t.append(c)
    return bytes(t)

_FLAC_CRC8_TABLE = _make_crc8_table()

FLAC_BLOCK_SIZE_TABLE = {
    0x00: None,   # reserved
    0x01: 192,
    0x02: 576, 0x03: 1152, 0x04: 2304, 0x05: 4608,
    0x06: None,   # 8-bit value follows header
    0x07: None,   # 16-bit value follows header
    0x08: 256,  0x09: 512,  0x0A: 1024, 0x0B: 2048,
    0x0C: 4096, 0x0D: 8192, 0x0E: 16384, 0x0F: 32768,
}


def _flac_crc8(data):
    crc = 0
    for b in data:
        crc = _FLAC_CRC8_TABLE[crc ^ b]
    return crc


def _flac_read_utf8_int(data, pos):
    """Read UTF-8 coded unsigned int from FLAC frame header. Returns (value, new_pos) or (None, pos)."""
    if pos >= len(data):
        return None, pos
    b0 = data[pos]
    if not (b0 & 0x80):
        return b0, pos + 1
    n = 0
    tmp = b0
    while tmp & 0x80:
        n += 1
        tmp = (tmp << 1) & 0xFF
    if n < 2 or n > 7 or pos + n > len(data):
        return None, pos
    val = b0 & (0x7F >> n)
    for i in range(1, n):
        cb = data[pos + i]
        if (cb & 0xC0) != 0x80:
            return None, pos
        val = (val << 6) | (cb & 0x3F)
    return val, pos + n


def _parse_flac_frame_header(data, pos):
    """
    Parse FLAC frame header at data[pos]. Validates CRC-8.
    Returns dict with block_size and header_end offset, or None if invalid.
    """
    if pos + 5 > len(data):
        return None
    # Sync word: 14 bits 0x3FFE, then reserved=0, then blocking_strategy
    if data[pos] != 0xFF or (data[pos + 1] & 0xFE) != 0xF8:
        return None

    byte2 = data[pos + 2]
    byte3 = data[pos + 3]

    if byte3 & 0x01:  # reserved bit must be 0
        return None

    block_size_bits = (byte2 >> 4) & 0x0F
    sample_rate_bits = byte2 & 0x0F

    if sample_rate_bits == 0x0F:  # invalid
        return None

    p = pos + 4

    # UTF-8 coded frame/sample number
    val, p = _flac_read_utf8_int(data, p)
    if val is None:
        return None

    # Optional block size
    if block_size_bits == 0x06:
        if p >= len(data):
            return None
        block_size = data[p] + 1
        p += 1
    elif block_size_bits == 0x07:
        if p + 1 >= len(data):
            return None
        block_size = struct.unpack(">H", data[p:p + 2])[0] + 1
        p += 2
    else:
        block_size = FLAC_BLOCK_SIZE_TABLE.get(block_size_bits)
        if block_size is None:
            return None

    # Optional sample rate bytes
    if sample_rate_bits == 0x0C:
        p += 1
    elif sample_rate_bits in (0x0D, 0x0E):
        p += 2

    if p >= len(data):
        return None

    # CRC-8 covers everything from sync to here (exclusive of crc byte itself)
    if _flac_crc8(data[pos:p]) != data[p]:
        return None

    return {"block_size": block_size, "header_end": p + 1}


# ═══════════════════════════════════════════════════════════════════════════════
# Engine DJ database access
# ═══════════════════════════════════════════════════════════════════════════════

def find_track(db_path, filename):
    """Look up a track by filename. Returns (id, path, filename)."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT id, path, filename FROM Track WHERE filename = ?", (filename,))
    results = cur.fetchall()

    if not results:
        cur.execute(
            "SELECT id, path, filename FROM Track WHERE filename LIKE ?",
            (f"%{filename}%",),
        )
        results = cur.fetchall()

    conn.close()

    if not results:
        print(f"Error: No track matching '{filename}' found in the Engine DJ library.")
        sys.exit(1)

    if len(results) > 1:
        print(f"\nWarning: {len(results)} tracks match '{filename}':")
        for i, (tid, path, fn) in enumerate(results):
            print(f"  [{i + 1}] {fn}  —  {path}")
        while True:
            try:
                choice = int(input("\nChoose a track number: "))
                if 1 <= choice <= len(results):
                    return results[choice - 1]
            except ValueError:
                pass
            print("Invalid choice, try again.")

    return results[0]


def get_hotcues(db_path, track_id):
    """
    Parse the quickCues blob for the given track.
    Returns a dict mapping cue number (1–8) to position in samples.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT quickCues FROM PerformanceData WHERE trackId = ?", (track_id,))
    row = cur.fetchone()
    conn.close()

    if not row or not row[0]:
        print("Error: No hotcue data found for this track.")
        sys.exit(1)

    blob = row[0]
    data = zlib.decompress(blob[4:])

    num_cues = struct.unpack(">q", data[:8])[0]
    pos = 8
    hotcues = {}

    for i in range(num_cues):
        if pos + 1 > len(data):
            break
        name_len = data[pos]
        pos += 1
        if pos + name_len + 12 > len(data):
            break
        pos += name_len
        position = struct.unpack(">d", data[pos : pos + 8])[0]
        pos += 8
        pos += 4  # skip ARGB color

        if position >= 0:
            hotcues[i + 1] = position

    return hotcues


def resolve_track_path(db_path, relative_path):
    """Resolve a track path from the DB to an existing absolute path if possible."""
    db_dir = os.path.dirname(os.path.abspath(db_path))
    resolved = os.path.normpath(os.path.expanduser(os.path.join(db_dir, relative_path)))

    if os.path.isfile(resolved):
        return resolved

    for legacy_root, current_root in PATH_REMAPS:
        legacy_abs = os.path.normpath(os.path.expanduser(legacy_root))
        current_abs = os.path.normpath(os.path.expanduser(current_root))
        try:
            if os.path.commonpath([resolved, legacy_abs]) == legacy_abs:
                rel = os.path.relpath(resolved, legacy_abs)
                remapped = os.path.normpath(os.path.join(current_abs, rel))
                if os.path.isfile(remapped):
                    return remapped
        except ValueError:
            # Paths on different mounts/drives cannot share a common root.
            continue

    return resolved


# ═══════════════════════════════════════════════════════════════════════════════
# Metadata updates
# ═══════════════════════════════════════════════════════════════════════════════

def update_track_title(output_path, new_title):
    """Update the embedded track title tag for supported output formats."""
    ext = os.path.splitext(output_path)[1].lower()

    try:
        if ext == ".mp3":
            audio = MP3(output_path, ID3=ID3)
            if audio.tags is None:
                audio.add_tags()
            audio.tags.setall("TIT2", [TIT2(encoding=3, text=new_title)])
            audio.save(v2_version=3)
        elif ext == ".flac":
            audio = FLAC(output_path)
            audio["title"] = [new_title]
            audio.save()
        elif ext == ".wav":
            audio = WAVE(output_path)
            if audio.tags is None:
                audio.add_tags()
            audio.tags.setall("TIT2", [TIT2(encoding=3, text=new_title)])
            audio.save(v2_version=3)
        else:
            return

        print(f"  Track title   : {new_title}")
    except Exception as exc:
        print(f"Warning: Could not update track title metadata: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# Frame-accurate MP3 cutting
# ═══════════════════════════════════════════════════════════════════════════════

def cut_mp3(input_path, output_path, cut_start_samples, cut_end_samples):
    """
    Remove a run of MP3 frames that best matches the region [cut_start, cut_end).
    The cut start snaps to the nearest frame boundary, and the number of frames
    removed is chosen so that the total cut length (in samples) stays as close
    as possible to the desired length — preserving beat-grid alignment.
    Preserves ID3v2/v1 tags and does not re-encode.
    """
    with open(input_path, "rb") as f:
        data = f.read()

    id3v2_size = read_id3v2_size(data)
    id3v2_data = data[:id3v2_size]

    id3v1_data = b""
    audio_end = len(data)
    if len(data) >= 128 and data[-128:-125] == b"TAG":
        id3v1_data = data[-128:]
        audio_end -= 128

    # --- Pass 1: collect every frame's position and byte range ----------------
    frames = []  # (file_offset, frame_size, start_sample)
    pos = id3v2_size
    current_sample = 0
    file_sample_rate = None
    spf = None

    while pos + 4 <= audio_end:
        info = parse_frame_header(data[pos : pos + 4])
        if info is None:
            pos += 1
            continue

        if file_sample_rate is None:
            file_sample_rate = info["sample_rate"]
            spf = info["samples_per_frame"]

        frame_end = pos + info["frame_size"]
        if frame_end > audio_end:
            break

        frames.append((pos, info["frame_size"], current_sample))
        current_sample += info["samples_per_frame"]
        pos = frame_end

    if not frames:
        print("Error: No valid MP3 frames found.")
        sys.exit(1)

    # --- Determine optimal cut region ----------------------------------------
    desired_length = cut_end_samples - cut_start_samples
    num_frames_to_cut = round(desired_length / spf)

    # Find the frame whose start is closest to cut_start_samples
    best_start_idx = min(
        range(len(frames)),
        key=lambda i: abs(frames[i][2] - cut_start_samples),
    )

    # Clamp so we don't run past the end of the file
    if best_start_idx + num_frames_to_cut > len(frames):
        num_frames_to_cut = len(frames) - best_start_idx

    end_idx = best_start_idx + num_frames_to_cut

    actual_start_sample = frames[best_start_idx][2]
    actual_end_sample = frames[end_idx][2] if end_idx < len(frames) else current_sample
    actual_length = actual_end_sample - actual_start_sample
    drift = actual_length - desired_length

    # --- Pass 2: write kept frames --------------------------------------------
    with open(output_path, "wb") as f:
        f.write(id3v2_data)
        for i, (foff, fsize, _) in enumerate(frames):
            if i < best_start_idx or i >= end_idx:
                f.write(data[foff : foff + fsize])
        f.write(id3v1_data)

    sr = file_sample_rate or 44100
    print(f"\n  Total frames  : {len(frames)}")
    print(f"  Cut frames    : {num_frames_to_cut}")
    print(f"  Desired cut   : {desired_length:.0f} samples ({desired_length / sr:.3f} s)")
    print(f"  Actual cut    : {actual_length:.0f} samples ({actual_length / sr:.3f} s)")
    print(f"  Drift         : {drift:+.0f} samples ({drift / sr * 1000:+.2f} ms)")
    print(f"  New duration  : ~{(len(frames) - num_frames_to_cut) * spf / sr:.2f} s")
    print(f"  Output file   : {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Frame-accurate FLAC cutting
# ═══════════════════════════════════════════════════════════════════════════════

def cut_flac(input_path, output_path, cut_start_samples, cut_end_samples):
    """
    Remove FLAC frames that best match the region [cut_start, cut_end).
    Validates each frame boundary with CRC-8. Updates STREAMINFO total_samples
    and replaces SEEKTABLE with PADDING (seek points become invalid after cutting).
    Does not re-encode audio.
    """
    with open(input_path, "rb") as f:
        data = f.read()

    if data[:4] != b"fLaC":
        print("Error: Not a valid FLAC file.")
        sys.exit(1)

    # --- Parse metadata blocks ---
    pos = 4
    output_meta = []  # list of [is_last_flag, block_type, bytearray_of_data]
    streaminfo_ref = None
    sample_rate = 44100  # fallback; will be overwritten from STREAMINFO

    while pos + 4 <= len(data):
        is_last = (data[pos] >> 7) & 1
        block_type = data[pos] & 0x7F
        length = (data[pos + 1] << 16) | (data[pos + 2] << 8) | data[pos + 3]
        block_data = bytearray(data[pos + 4: pos + 4 + length])

        if block_type == 0:  # STREAMINFO
            streaminfo_ref = block_data
            sample_rate = (block_data[10] << 12) | (block_data[11] << 4) | (block_data[12] >> 4)
            output_meta.append([is_last, 0, block_data])
        elif block_type == 3:  # SEEKTABLE — replace with PADDING; offsets change after cut
            output_meta.append([is_last, 1, bytearray(length)])
        else:
            output_meta.append([is_last, block_type, block_data])

        pos += 4 + length
        if is_last:
            break

    if streaminfo_ref is None:
        print("Error: No STREAMINFO block found in FLAC file.")
        sys.exit(1)

    audio_start = pos

    # --- Scan for frame boundaries using sync + CRC-8 validation ---
    frame_starts = []  # (file_offset, block_size)
    p = audio_start
    while p + 4 <= len(data):
        if data[p] == 0xFF and (data[p + 1] & 0xFE) == 0xF8:
            result = _parse_flac_frame_header(data, p)
            if result is not None:
                frame_starts.append((p, result["block_size"]))
                p += 2
                continue
        p += 1

    if not frame_starts:
        print("Error: No valid FLAC frames found.")
        sys.exit(1)

    # Build frame list: (offset, byte_size, start_sample, block_size)
    frames = []
    current_sample = 0
    for i, (fpos, block_size) in enumerate(frame_starts):
        next_pos = frame_starts[i + 1][0] if i + 1 < len(frame_starts) else len(data)
        frames.append((fpos, next_pos - fpos, current_sample, block_size))
        current_sample += block_size

    spf = frames[0][3]  # typical samples per frame (from first frame)

    # --- Determine optimal cut region (same logic as MP3) ---
    desired_length = cut_end_samples - cut_start_samples
    num_frames_to_cut = round(desired_length / spf)

    best_start_idx = min(
        range(len(frames)),
        key=lambda i: abs(frames[i][2] - cut_start_samples),
    )

    if best_start_idx + num_frames_to_cut > len(frames):
        num_frames_to_cut = len(frames) - best_start_idx

    end_idx = best_start_idx + num_frames_to_cut

    actual_start_sample = frames[best_start_idx][2]
    actual_end_sample = frames[end_idx][2] if end_idx < len(frames) else current_sample
    actual_length = actual_end_sample - actual_start_sample
    drift = actual_length - desired_length
    new_total_samples = current_sample - actual_length

    # --- Update STREAMINFO total_samples (36-bit field) and zero MD5 ---
    streaminfo_ref[13] = (streaminfo_ref[13] & 0xF0) | ((new_total_samples >> 32) & 0x0F)
    streaminfo_ref[14] = (new_total_samples >> 24) & 0xFF
    streaminfo_ref[15] = (new_total_samples >> 16) & 0xFF
    streaminfo_ref[16] = (new_total_samples >> 8) & 0xFF
    streaminfo_ref[17] = new_total_samples & 0xFF
    for i in range(18, 34):  # zero MD5: audio content changed
        streaminfo_ref[i] = 0

    # --- Write output ---
    with open(output_path, "wb") as f:
        f.write(b"fLaC")
        for k, (_, btype, bdata) in enumerate(output_meta):
            is_last_out = 1 if k == len(output_meta) - 1 else 0
            f.write(bytes([is_last_out << 7 | btype]))
            f.write(bytes([(len(bdata) >> 16) & 0xFF, (len(bdata) >> 8) & 0xFF, len(bdata) & 0xFF]))
            f.write(bdata)
        for i, (foff, fsize, _, _) in enumerate(frames):
            if i < best_start_idx or i >= end_idx:
                f.write(data[foff: foff + fsize])

    print(f"\n  Total frames  : {len(frames)}")
    print(f"  Cut frames    : {num_frames_to_cut}")
    print(f"  Desired cut   : {desired_length:.0f} samples ({desired_length / sample_rate:.3f} s)")
    print(f"  Actual cut    : {actual_length:.0f} samples ({actual_length / sample_rate:.3f} s)")
    print(f"  Drift         : {drift:+.0f} samples ({drift / sample_rate * 1000:+.2f} ms)")
    print(f"  New duration  : ~{new_total_samples / sample_rate:.2f} s")
    print(f"  Output file   : {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Sample-accurate WAV cutting
# ═══════════════════════════════════════════════════════════════════════════════

def cut_wav(input_path, output_path, cut_start_samples, cut_end_samples):
    """
    Remove PCM samples in [cut_start, cut_end) from a WAV file.
    Uses Python's stdlib wave module; no re-encoding needed since WAV is uncompressed.
    """
    with wave.open(input_path, "rb") as r:
        params = r.getparams()
        all_frames = r.readframes(params.nframes)

    sample_rate = params.framerate
    frame_width = params.nchannels * params.sampwidth  # bytes per sample (all channels)

    total_samples = params.nframes
    start = max(0, min(int(cut_start_samples), total_samples))
    end = max(start, min(int(cut_end_samples), total_samples))

    kept = all_frames[: start * frame_width] + all_frames[end * frame_width :]
    new_total = total_samples - (end - start)
    actual_length = end - start
    drift = actual_length - (cut_end_samples - cut_start_samples)

    with wave.open(output_path, "wb") as w:
        w.setparams(params._replace(nframes=new_total))
        w.writeframes(kept)

    print(f"\n  Total samples : {total_samples}")
    print(f"  Cut samples   : {actual_length}")
    print(f"  Desired cut   : {cut_end_samples - cut_start_samples:.0f} samples ({(cut_end_samples - cut_start_samples) / sample_rate:.3f} s)")
    print(f"  Actual cut    : {actual_length:.0f} samples ({actual_length / sample_rate:.3f} s)")
    print(f"  Drift         : {drift:+.0f} samples ({drift / sample_rate * 1000:+.2f} ms)")
    print(f"  New duration  : ~{new_total / sample_rate:.2f} s")
    print(f"  Output file   : {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if not os.path.isfile(ENGINE_DB_PATH):
        print(f"Error: Engine DJ database not found at:\n  {ENGINE_DB_PATH}")
        sys.exit(1)

    print(f"Looking up '{TRACK_FILENAME}' in Engine DJ library …")
    track_id, track_rel_path, track_fn = find_track(ENGINE_DB_PATH, TRACK_FILENAME)
    track_abs_path = resolve_track_path(ENGINE_DB_PATH, track_rel_path)

    print(f"  Found: {track_fn}")
    print(f"  Path : {track_abs_path}")

    if not os.path.isfile(track_abs_path):
        print(f"\nError: Audio file not found at resolved path:\n  {track_abs_path}")
        sys.exit(1)

    hotcues = get_hotcues(ENGINE_DB_PATH, track_id)
    print(f"\n  Available hotcues:")
    for num in sorted(hotcues):
        secs = hotcues[num] / 44100
        mins = int(secs) // 60
        remainder = secs - mins * 60
        print(f"    Cue {num}:  {mins}:{remainder:05.2f}  ({hotcues[num]:.0f} samples)")

    if HOTCUE_START not in hotcues:
        print(f"\nError: Hotcue {HOTCUE_START} is not set on this track.")
        sys.exit(1)
    if HOTCUE_END not in hotcues:
        print(f"\nError: Hotcue {HOTCUE_END} is not set on this track.")
        sys.exit(1)

    cut_start = hotcues[HOTCUE_START]
    cut_end = hotcues[HOTCUE_END]

    if cut_start >= cut_end:
        print(
            f"\nError: Hotcue {HOTCUE_START} ({cut_start:.0f} samples) must be "
            f"before Hotcue {HOTCUE_END} ({cut_end:.0f} samples)."
        )
        sys.exit(1)

    name, ext = os.path.splitext(track_fn)
    ext_lower = ext.lower()
    if ext_lower not in (".mp3", ".flac", ".wav"):
        print(f"Error: Unsupported file format '{ext}'. Supported: .mp3, .flac, .wav")
        sys.exit(1)

    output_title = f"{name} {OUTPUT_APPENDIX}"
    output_filename = f"{output_title}{ext}"
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_filepath = os.path.join(OUTPUT_PATH, output_filename)

    print(f"\nCutting between Cue {HOTCUE_START} and Cue {HOTCUE_END} …")
    if ext_lower == ".mp3":
        cut_mp3(track_abs_path, output_filepath, cut_start, cut_end)
    elif ext_lower == ".flac":
        cut_flac(track_abs_path, output_filepath, cut_start, cut_end)
    else:
        cut_wav(track_abs_path, output_filepath, cut_start, cut_end)

    update_track_title(output_filepath, output_title)


if __name__ == "__main__":
    main()
