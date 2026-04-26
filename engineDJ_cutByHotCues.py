#!/usr/bin/env python3
"""
Audio Editor — Edit MP3, FLAC, or WAV files using Engine DJ hotcues.

Three modes:
  CUT_BETWEEN_CUES — remove audio between two hotcue positions
  CUT_TO_END       — remove audio from a hotcue to the end of the track
  ADD_SILENCE      — insert a block of silence at a hotcue or a timestamp

FLAC and WAV are edited sample-accurately with zero-crossing snap.
MP3 is edited frame-accurately (lossless, no re-encoding).
"""

import struct
import sqlite3
import subprocess
import zlib
import os
import sys
import tempfile

try:
    import numpy as np
    import soundfile as sf
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\nInstall with: pip install soundfile numpy")

try:
    from pedalboard import Pedalboard, Reverb as PbReverb
    from pedalboard.io import AudioFile as PbAudioFile
    _HAS_PEDALBOARD = True
except ImportError:
    _HAS_PEDALBOARD = False

from mutagen.flac import FLAC
from mutagen.id3 import ID3, TIT2, TLEN
from mutagen.mp3 import MP3
from mutagen.wave import WAVE

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Edit these variables before running
# ═══════════════════════════════════════════════════════════════════════════════

# ── Operation mode ────────────────────────────────────────────────────────────
# Choose what the script should do, then fill in the matching section below:
#   "CUT_BETWEEN_CUES" — delete the audio between HOTCUE_START and HOTCUE_END
#   "CUT_TO_END"       — delete everything from HOTCUE_START to end of track
#   "ADD_SILENCE"      — insert a block of silence at SILENCE_CUE or SILENCE_TIMESTAMP
MODE = "ADD_SILENCE"

# ── Shared settings (used by every mode) ──────────────────────────────────────
TRACK_FILENAME  = "Scream & Shout.mp3"  # .mp3, .flac, or .wav
OUTPUT_APPENDIX = "(Added Silence before Start)"   # appended to the output filename and embedded title
OUTPUT_PATH     = os.path.expanduser("~/Library/CloudStorage/OneDrive-Personal/DJing/Edits")
ENGINE_DB_PATH  = os.path.expanduser("~/Music/Engine Library/Database2/m.db")

# ── CUT_BETWEEN_CUES settings ─────────────────────────────────────────────────
HOTCUE_START = 5   # hotcue number (1–8) where the cut begins
HOTCUE_END   = 6   # hotcue number (1–8) where the cut ends

# ── CUT_TO_END settings ───────────────────────────────────────────────────────
# (uses HOTCUE_START above — no additional variables needed)

# ── ADD_SILENCE settings ──────────────────────────────────────────────────────
# Set exactly one of SILENCE_CUE or SILENCE_TIMESTAMP to the insertion point;
# leave the other as None.  The silence is spliced in at that position; audio
# before and after plays normally.
SILENCE_CUE           = None   # hotcue number (1–8) that marks the insertion point, or None
SILENCE_TIMESTAMP     = 0   # insertion point in seconds (float, e.g. 95.5), or None
SILENCE_DURATION_SECS = 1    # length of the inserted silence in seconds

# ── Example — CUT_TO_END on a different track ─────────────────────────────────
# MODE            = "CUT_TO_END"
# TRACK_FILENAME  = "Britney Spears - Toxic.mp3"
# HOTCUE_START    = 8
# OUTPUT_APPENDIX = "(Cut End)"
# OUTPUT_PATH     = os.path.expanduser("~/Library/CloudStorage/OneDrive-Personal/DJing/Edits")
# ENGINE_DB_PATH  = os.path.expanduser("~/Music/Engine Library/Database2/m.db")


# Remap stale Engine DJ paths to their current locations when files have moved.
PATH_REMAPS = [
    ("~/Music/OneDrive", "~/Library/CloudStorage/OneDrive-Personal"),
]

# Reverb tail — only applied when cutting to end of track (HOTCUE_END = None).
REVERB_TAIL       = False   # set True to append reverb decay after the cut point
REVERB_ROOM_SIZE  = 0.75    # 0.0–1.0
REVERB_DAMPING    = 0.5     # 0.0–1.0
REVERB_WET_LEVEL  = 0.25    # 0.0–1.0  (tail amplitude)
REVERB_WIDTH      = 1.0     # stereo width 0.0–1.0
REVERB_TAIL_SECS  = 4.0     # seconds of decay to append
REVERB_BLEND_SECS = 2.0     # seconds before cut over which reverb fades in

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


def _patch_xing_header(frame_data, new_frame_count, new_byte_count):
    """Update the Xing/Info VBR header inside an MP3 frame with new totals.
    Returns patched bytes, or the original bytes unchanged if no header is found."""
    if len(frame_data) < 4:
        return frame_data
    header = struct.unpack(">I", frame_data[:4])[0]
    if (header >> 21) != 0x7FF:
        return frame_data
    version_bits = (header >> 19) & 0x03
    layer_bits   = (header >> 17) & 0x03
    version = {0: 2.5, 2: 2, 3: 1}.get(version_bits)
    layer   = {1: 3, 2: 2, 3: 1}.get(layer_bits)
    if layer != 3:
        return frame_data
    xing_off = 36 if version == 1 else 21  # 4-byte header + side-info bytes
    if len(frame_data) < xing_off + 8:
        return frame_data
    if frame_data[xing_off:xing_off + 4] not in (b"Xing", b"Info"):
        return frame_data
    buf   = bytearray(frame_data)
    flags = struct.unpack_from(">I", buf, xing_off + 4)[0]
    pos   = xing_off + 8
    if flags & 1:
        struct.pack_into(">I", buf, pos, new_frame_count)
        pos += 4
    if flags & 2:
        struct.pack_into(">I", buf, pos, new_byte_count)
    return bytes(buf)


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

def _read_title(path, ext):
    """Read the embedded track title, or return None if absent."""
    try:
        if ext == ".mp3":
            audio = MP3(path, ID3=ID3)
            if audio.tags and "TIT2" in audio.tags:
                return str(audio.tags["TIT2"])
        elif ext == ".flac":
            audio = FLAC(path)
            if audio.tags and "title" in audio:
                return audio["title"][0]
        elif ext == ".wav":
            audio = WAVE(path)
            if audio.tags and "TIT2" in audio.tags:
                return str(audio.tags["TIT2"])
    except Exception:
        pass
    return None


def _copy_metadata(src_path, dst_path, ext):
    """Copy all embedded tags and artwork from src to dst (FLAC and WAV only)."""
    try:
        if ext == ".flac":
            src = FLAC(src_path)
            dst = FLAC(dst_path)
            if src.tags:
                for key, values in src.tags.as_dict().items():
                    dst[key] = values
            for pic in src.pictures:
                dst.add_picture(pic)
            dst.save()
        elif ext == ".wav":
            src = WAVE(src_path)
            if src.tags:
                dst = WAVE(dst_path)
                if dst.tags is None:
                    dst.add_tags()
                for frame in src.tags.values():
                    dst.tags.add(frame)
                dst.save()
    except Exception as exc:
        print(f"Warning: Could not copy metadata: {exc}")


def update_track_title(output_path, new_title):
    """Update the embedded track title tag for supported output formats."""
    ext = os.path.splitext(output_path)[1].lower()

    try:
        if ext == ".mp3":
            audio = MP3(output_path, ID3=ID3)
            if audio.tags is None:
                audio.add_tags()
            audio.tags.setall("TIT2", [TIT2(encoding=3, text=new_title)])
            audio.tags.setall("TLEN", [TLEN(encoding=3, text=str(int(audio.info.length * 1000)))])
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

def cut_mp3(input_path, output_path, cut_start_samples, cut_end_samples, reverb_tail=False):
    """
    Remove a run of MP3 frames that best matches the region [cut_start, cut_end).
    The cut start snaps to the nearest frame boundary, and the number of frames
    removed is chosen so that the total cut length (in samples) stays as close
    as possible to the desired length — preserving beat-grid alignment.
    Preserves ID3v2/v1 tags and does not re-encode the kept frames.
    When reverb_tail=True the removed section is replaced with a reverb decay tail
    (decoded from the kept frames, re-encoded as new MP3 frames).
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

    # --- Generate reverb tail (MP3 only, decoded → reverb → re-encoded) -------
    tail_mp3_bytes = b""
    if reverb_tail:
        if not _HAS_PEDALBOARD:
            sys.exit("Missing dependency: pedalboard\nInstall with: pip install pedalboard")
        print("  Generating reverb tail …", flush=True)
        pre_cut_n = frames[best_start_idx][2]
        with PbAudioFile(input_path) as af:
            pre_pcm = af.read(pre_cut_n).T  # (n_samples, n_ch) float32
        _, tail_pcm = _reverb_outro(pre_pcm.astype(np.float64), file_sample_rate, blend_in=False)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(tmp_fd)
        try:
            with PbAudioFile(tmp_path, "w", samplerate=file_sample_rate,
                             num_channels=tail_pcm.shape[1]) as af:
                af.write(tail_pcm.T.astype(np.float32))
            with open(tmp_path, "rb") as f:
                raw = f.read()
        finally:
            os.unlink(tmp_path)
        tail_start = read_id3v2_size(raw)
        tail_mp3_bytes = raw[tail_start:]
        if len(tail_mp3_bytes) >= 128 and tail_mp3_bytes[-128:-125] == b"TAG":
            tail_mp3_bytes = tail_mp3_bytes[:-128]

    # --- Pass 2: write kept frames --------------------------------------------
    kept_frame_bytes = sum(frames[i][1] for i in range(len(frames))
                           if i < best_start_idx or i >= end_idx)
    new_frame_count = len(frames) - num_frames_to_cut
    new_byte_count  = kept_frame_bytes + len(tail_mp3_bytes)  # audio bytes only, no ID3 tags

    with open(output_path, "wb") as f:
        f.write(id3v2_data)
        first_kept = True
        for i, (foff, fsize, _) in enumerate(frames):
            if i < best_start_idx or i >= end_idx:
                frame_bytes = data[foff : foff + fsize]
                if first_kept:
                    frame_bytes = _patch_xing_header(frame_bytes, new_frame_count, new_byte_count)
                    first_kept = False
                f.write(frame_bytes)
        if tail_mp3_bytes:
            f.write(tail_mp3_bytes)
        f.write(id3v1_data)

    sr = file_sample_rate or 44100
    kept_dur = best_start_idx * spf / sr
    print(f"\n  Total frames  : {len(frames)}")
    print(f"  Cut frames    : {num_frames_to_cut}")
    print(f"  Desired cut   : {desired_length:.0f} samples ({desired_length / sr:.3f} s)")
    print(f"  Actual cut    : {actual_length:.0f} samples ({actual_length / sr:.3f} s)")
    print(f"  Drift         : {drift:+.0f} samples ({drift / sr * 1000:+.2f} ms)")
    if reverb_tail:
        print(f"  New duration  : ~{kept_dur + REVERB_TAIL_SECS:.2f} s")
        print(f"  Reverb tail   : {REVERB_TAIL_SECS:.1f} s ({len(tail_mp3_bytes) / 1024:.1f} KB)")
    else:
        print(f"  New duration  : ~{(len(frames) - num_frames_to_cut) * spf / sr:.2f} s")
    print(f"  Output file   : {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Zero-crossing helper (shared by FLAC and WAV)
# ═══════════════════════════════════════════════════════════════════════════════

_ZC_WINDOW = 500  # samples to search either side of the target position

def _nearest_zero_crossing(mono, pos):
    """Return the nearest zero-crossing index to pos within ±_ZC_WINDOW samples.

    A zero crossing is defined as the first sample of a new half-cycle (sign
    change between adjacent samples). Returns pos unchanged if none is found.
    mono must be a 1-D float array.
    """
    lo = max(0, pos - _ZC_WINDOW)
    hi = min(len(mono), pos + _ZC_WINDOW + 1)
    signs = np.sign(mono[lo:hi])
    local = np.where(np.diff(signs))[0]  # indices just before each crossing
    if not len(local):
        return pos
    global_idx = local + lo + 1  # first sample on the new half-cycle
    return int(global_idx[np.argmin(np.abs(global_idx - pos))])


def _reverb_outro(pcm, sample_rate, blend_in=True):
    """Process pre-cut PCM through reverb and return (modified_pre_cut, tail).

    When blend_in=True (FLAC/WAV): the reverb is gradually mixed into the last
    REVERB_BLEND_SECS of modified_pre_cut so the transition sounds natural.
    The tail then continues at the same level — no abrupt onset.

    When blend_in=False (MP3, pre-cut frames kept lossless): modified_pre_cut is
    the original pcm unchanged; the tail is faded in over REVERB_BLEND_SECS instead.

    pcm: (n_samples, n_ch) float64
    Returns: (modified_pre_cut, tail) both (n_samples, n_ch) float64.
    """
    if not _HAS_PEDALBOARD:
        sys.exit("Missing dependency: pedalboard\nInstall with: pip install pedalboard")

    board = Pedalboard([PbReverb(
        room_size=REVERB_ROOM_SIZE,
        damping=REVERB_DAMPING,
        wet_level=REVERB_WET_LEVEL,
        dry_level=0.0,
        width=REVERB_WIDTH,
    )])

    n_ch = pcm.shape[1] if pcm.ndim > 1 else 1
    pcm_2d = pcm if pcm.ndim > 1 else pcm[:, np.newaxis]
    audio_f32 = pcm_2d.T.astype(np.float32)  # (n_ch, n_samples)

    reverb_out = board(audio_f32, sample_rate, reset=True)  # (n_ch, n_samples)

    modified = pcm_2d.copy()  # (n_samples, n_ch)
    if blend_in:
        blend_n = min(len(modified), int(sample_rate * REVERB_BLEND_SECS))
        ramp = np.linspace(0.0, 1.0, blend_n, dtype=np.float32)
        modified[-blend_n:] += (reverb_out[:, -blend_n:].T * ramp[:, np.newaxis]).astype(np.float64)

    # Flush reverb state with silence to get the decay tail
    tail_samples = int(sample_rate * REVERB_TAIL_SECS)
    silence = np.zeros((n_ch, tail_samples), dtype=np.float32)
    tail = board(silence, sample_rate, reset=False)  # (n_ch, tail_samples)

    if not blend_in:
        fade_in_n = min(tail.shape[1], int(sample_rate * REVERB_BLEND_SECS))
        tail[:, :fade_in_n] *= np.linspace(0.0, 1.0, fade_in_n, dtype=np.float32)

    fade_out_n = min(tail.shape[1], int(sample_rate * 0.5))
    tail[:, -fade_out_n:] *= np.linspace(1.0, 0.0, fade_out_n, dtype=np.float32)

    return modified.astype(np.float64), tail.T.astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# Sample-accurate FLAC cutting
# ═══════════════════════════════════════════════════════════════════════════════

def cut_flac(input_path, output_path, cut_start_samples, cut_end_samples, reverb_tail=False):
    """
    Remove exact samples [cut_start, cut_end) from a FLAC file.
    Both cut points are snapped to the nearest zero crossing (within ±11 ms)
    to eliminate clicks at the splice. Re-encodes losslessly via soundfile;
    decoded audio for kept samples is bit-for-bit identical to the original.
    When reverb_tail=True the post-cut silence is replaced with a reverb decay.
    """
    info = sf.info(input_path)
    sample_rate = info.samplerate
    subtype = info.subtype

    pcm, _ = sf.read(input_path, dtype="float64", always_2d=True)
    total_samples = len(pcm)

    start = max(0, min(int(round(cut_start_samples)), total_samples))
    end   = max(0, min(int(round(cut_end_samples)),   total_samples))

    mono = pcm[:, 0]
    snapped_start = _nearest_zero_crossing(mono, start)
    snapped_end   = _nearest_zero_crossing(mono, end)
    snapped_end   = max(snapped_end, snapped_start + 1)

    desired_length = cut_end_samples - cut_start_samples
    actual_length  = snapped_end - snapped_start
    drift          = actual_length - desired_length

    pre_cut = pcm[:snapped_start]
    if reverb_tail:
        print("  Generating reverb tail …", flush=True)
        pre_cut_blended, tail = _reverb_outro(pre_cut, sample_rate, blend_in=True)
        kept = np.concatenate([pre_cut_blended, tail])
    else:
        tail = None
        kept = np.concatenate([pre_cut, pcm[snapped_end:]])

    sf.write(output_path, kept, sample_rate, subtype=subtype, format="FLAC")
    _copy_metadata(input_path, output_path, ".flac")

    zc_start_shift = snapped_start - start
    zc_end_shift   = snapped_end   - end

    print(f"\n  Total samples : {total_samples}")
    print(f"  ZC snap (start): {zc_start_shift:+d} samples ({zc_start_shift / sample_rate * 1000:+.2f} ms)")
    if not reverb_tail:
        print(f"  ZC snap (end)  : {zc_end_shift:+d} samples ({zc_end_shift / sample_rate * 1000:+.2f} ms)")
    print(f"  Desired cut   : {desired_length:.0f} samples ({desired_length / sample_rate:.3f} s)")
    print(f"  Actual cut    : {actual_length} samples ({actual_length / sample_rate:.3f} s)")
    print(f"  Drift         : {drift:+.0f} samples ({drift / sample_rate * 1000:+.2f} ms)")
    print(f"  New duration  : ~{len(kept) / sample_rate:.2f} s")
    if reverb_tail:
        print(f"  Reverb tail   : {len(tail) / sample_rate:.2f} s")
    print(f"  Output file   : {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Sample-accurate WAV cutting
# ═══════════════════════════════════════════════════════════════════════════════

def cut_wav(input_path, output_path, cut_start_samples, cut_end_samples, reverb_tail=False):
    """
    Remove exact samples [cut_start, cut_end) from a WAV file.
    Both cut points are snapped to the nearest zero crossing (within ±11 ms)
    to eliminate clicks at the splice. No re-encoding needed (WAV is uncompressed).
    When reverb_tail=True the post-cut silence is replaced with a reverb decay.
    """
    info = sf.info(input_path)
    sample_rate = info.samplerate
    subtype = info.subtype

    pcm, _ = sf.read(input_path, dtype="float64", always_2d=True)
    total_samples = len(pcm)

    start = max(0, min(int(round(cut_start_samples)), total_samples))
    end   = max(0, min(int(round(cut_end_samples)),   total_samples))

    mono = pcm[:, 0]
    snapped_start = _nearest_zero_crossing(mono, start)
    snapped_end   = _nearest_zero_crossing(mono, end)
    snapped_end   = max(snapped_end, snapped_start + 1)

    desired_length = cut_end_samples - cut_start_samples
    actual_length  = snapped_end - snapped_start
    drift          = actual_length - desired_length

    pre_cut = pcm[:snapped_start]
    if reverb_tail:
        print("  Generating reverb tail …", flush=True)
        pre_cut_blended, tail = _reverb_outro(pre_cut, sample_rate, blend_in=True)
        kept = np.concatenate([pre_cut_blended, tail])
    else:
        tail = None
        kept = np.concatenate([pre_cut, pcm[snapped_end:]])

    sf.write(output_path, kept, sample_rate, subtype=subtype, format="WAV")
    _copy_metadata(input_path, output_path, ".wav")

    zc_start_shift = snapped_start - start
    zc_end_shift   = snapped_end   - end

    print(f"\n  Total samples : {total_samples}")
    print(f"  ZC snap (start): {zc_start_shift:+d} samples ({zc_start_shift / sample_rate * 1000:+.2f} ms)")
    if not reverb_tail:
        print(f"  ZC snap (end)  : {zc_end_shift:+d} samples ({zc_end_shift / sample_rate * 1000:+.2f} ms)")
    print(f"  Desired cut   : {desired_length:.0f} samples ({desired_length / sample_rate:.3f} s)")
    print(f"  Actual cut    : {actual_length} samples ({actual_length / sample_rate:.3f} s)")
    print(f"  Drift         : {drift:+.0f} samples ({drift / sample_rate * 1000:+.2f} ms)")
    print(f"  New duration  : ~{len(kept) / sample_rate:.2f} s")
    if reverb_tail:
        print(f"  Reverb tail   : {len(tail) / sample_rate:.2f} s")
    print(f"  Output file   : {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Silence insertion
# ═══════════════════════════════════════════════════════════════════════════════

def _insert_silence_pcm(input_path, output_path, insert_at_samples, silence_duration_secs, fmt):
    """Insert silence_duration_secs of digital silence into a PCM audio file.

    insert_at_samples is snapped to the nearest zero crossing so there are no
    clicks at the splice points.  fmt must be "FLAC" or "WAV".  The original
    bit depth (subtype) and all embedded metadata/artwork are preserved.
    """
    info = sf.info(input_path)
    sample_rate = info.samplerate
    subtype = info.subtype

    pcm, _ = sf.read(input_path, dtype="float64", always_2d=True)
    total_samples = len(pcm)

    # Clamp to valid range then snap to the nearest zero crossing
    pos = max(0, min(int(round(insert_at_samples)), total_samples))
    snapped = _nearest_zero_crossing(pcm[:, 0], pos)
    shift = snapped - pos  # how far the snap moved us

    # Build the block of silence (all channels, float64 zeros)
    n_ch = pcm.shape[1]
    silence_n = int(round(silence_duration_secs * sample_rate))
    silence = np.zeros((silence_n, n_ch), dtype=np.float64)

    # Stitch: audio before insertion point | silence | audio after insertion point
    result = np.concatenate([pcm[:snapped], silence, pcm[snapped:]])

    ext = ".flac" if fmt == "FLAC" else ".wav"
    sf.write(output_path, result, sample_rate, subtype=subtype, format=fmt)
    _copy_metadata(input_path, output_path, ext)

    print(f"\n  Total samples  : {total_samples}")
    print(f"  Insertion pt   : sample {snapped}  ({snapped / sample_rate:.3f} s)")
    print(f"  ZC snap        : {shift:+d} samples ({shift / sample_rate * 1000:+.2f} ms)")
    print(f"  Silence        : {silence_n} samples ({silence_duration_secs:.3f} s)")
    print(f"  New duration   : ~{len(result) / sample_rate:.2f} s")
    print(f"  Output file    : {output_path}")


def insert_silence_flac(input_path, output_path, insert_at_samples, silence_duration_secs):
    """Insert silence into a FLAC file at insert_at_samples (sample-accurate)."""
    _insert_silence_pcm(input_path, output_path, insert_at_samples, silence_duration_secs, "FLAC")


def insert_silence_wav(input_path, output_path, insert_at_samples, silence_duration_secs):
    """Insert silence into a WAV file at insert_at_samples (sample-accurate)."""
    _insert_silence_pcm(input_path, output_path, insert_at_samples, silence_duration_secs, "WAV")


def insert_silence_mp3(input_path, output_path, insert_at_samples, silence_duration_secs):
    """Insert silence into an MP3 file at insert_at_samples (frame-accurate, no re-encoding).

    Constructs silent MP3 frames by copying the header of the frame at the
    insertion point and zero-filling the body.  An all-zero body means the side
    information allocates zero Huffman bits to every subband, so decoders output
    silence.  The ID3 tags and Xing/Info VBR header are preserved/updated.
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

    # --- Pass 1: collect every frame's byte position and sample offset ---------
    frames = []   # list of (file_offset, frame_size, start_sample)
    pos = id3v2_size
    current_sample = 0
    file_sample_rate = None
    spf = None   # samples per frame (constant for a given Layer/version)

    while pos + 4 <= audio_end:
        info = parse_frame_header(data[pos:pos + 4])
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

    # --- Find the frame boundary closest to the target sample offset -----------
    insert_idx = min(
        range(len(frames)),
        key=lambda i: abs(frames[i][2] - insert_at_samples),
    )
    # If the target lands past the midpoint of the last frame, append at end
    if insert_at_samples >= frames[-1][2] + spf / 2:
        insert_idx = len(frames)

    # --- Build the block of silent frames -------------------------------------
    # Take the header (4 bytes) of the nearest real frame and zero-fill the rest.
    # An all-zero frame body means no bits are allocated → decoder outputs silence.
    tmpl_idx = min(insert_idx, len(frames) - 1)
    tmpl_pos, tmpl_size, _ = frames[tmpl_idx]
    tmpl_header = data[tmpl_pos:tmpl_pos + 4]
    silent_frame = tmpl_header + bytes(tmpl_size - 4)   # header + zeroed body

    n_silent_frames = max(1, round(silence_duration_secs * file_sample_rate / spf))
    silent_block = silent_frame * n_silent_frames

    # --- Update VBR header totals so players report the correct duration -------
    new_frame_count = len(frames) + n_silent_frames
    new_byte_count = sum(f[1] for f in frames) + len(silent_block)

    # --- Write output: original frames with the silent block spliced in --------
    insert_sample = (
        frames[insert_idx][2] if insert_idx < len(frames) else current_sample
    )
    with open(output_path, "wb") as f:
        f.write(id3v2_data)
        first_kept = True
        for i, (foff, fsize, _) in enumerate(frames):
            # Splice the silent block in immediately before the target frame
            if i == insert_idx:
                f.write(silent_block)
            frame_bytes = data[foff:foff + fsize]
            if first_kept:
                # Patch the Xing/Info VBR header in the first frame
                frame_bytes = _patch_xing_header(frame_bytes, new_frame_count, new_byte_count)
                first_kept = False
            f.write(frame_bytes)
        # Handle the insert-at-end case (insert_idx == len(frames))
        if insert_idx >= len(frames):
            f.write(silent_block)
        f.write(id3v1_data)

    sr = file_sample_rate or 44100
    actual_silence_secs = n_silent_frames * spf / sr
    print(f"\n  Total frames   : {len(frames)}")
    print(f"  Silent frames  : {n_silent_frames}")
    print(f"  Silence        : {actual_silence_secs:.3f} s")
    print(f"  Insertion pt   : {insert_sample / sr:.3f} s (frame {insert_idx})")
    print(f"  New duration   : ~{(len(frames) + n_silent_frames) * spf / sr:.2f} s")
    print(f"  Output file    : {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Validate Engine DJ database ───────────────────────────────────────────
    if not os.path.isfile(ENGINE_DB_PATH):
        print(f"Error: Engine DJ database not found at:\n  {ENGINE_DB_PATH}")
        sys.exit(1)

    # ── Look up the track in the Engine DJ library ────────────────────────────
    print(f"Looking up '{TRACK_FILENAME}' in Engine DJ library …")
    track_id, track_rel_path, track_fn = find_track(ENGINE_DB_PATH, TRACK_FILENAME)
    track_abs_path = resolve_track_path(ENGINE_DB_PATH, track_rel_path)

    print(f"  Found: {track_fn}")
    print(f"  Path : {track_abs_path}")

    if not os.path.isfile(track_abs_path):
        print(f"\nError: Audio file not found at resolved path:\n  {track_abs_path}")
        sys.exit(1)

    # ── List all hotcues stored in the database ───────────────────────────────
    hotcues = get_hotcues(ENGINE_DB_PATH, track_id)
    print(f"\n  Available hotcues:")
    for num in sorted(hotcues):
        secs = hotcues[num] / 44100
        mins = int(secs) // 60
        remainder = secs - mins * 60
        print(f"    Cue {num}:  {mins}:{remainder:05.2f}  ({hotcues[num]:.0f} samples)")

    # ── Validate audio format ─────────────────────────────────────────────────
    name, ext = os.path.splitext(track_fn)
    ext_lower = ext.lower()
    if ext_lower not in (".mp3", ".flac", ".wav"):
        print(f"Error: Unsupported file format '{ext}'. Supported: .mp3, .flac, .wav")
        sys.exit(1)

    # ── Prepare output path ───────────────────────────────────────────────────
    output_title    = f"{_read_title(track_abs_path, ext_lower) or name} {OUTPUT_APPENDIX}"
    output_filename = f"{name} {OUTPUT_APPENDIX}{ext}"
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_filepath = os.path.join(OUTPUT_PATH, output_filename)

    # ── Dispatch on MODE ──────────────────────────────────────────────────────

    if MODE in ("CUT_BETWEEN_CUES", "CUT_TO_END"):
        # Validate the start cue (required by both cut modes)
        if HOTCUE_START not in hotcues:
            print(f"\nError: Hotcue {HOTCUE_START} is not set on this track.")
            sys.exit(1)

        cut_start = hotcues[HOTCUE_START]

        if MODE == "CUT_TO_END":
            # Determine the sample offset of the very last sample in the file
            if ext_lower in (".flac", ".wav"):
                cut_end = sf.info(track_abs_path).frames
            else:
                mp3 = MP3(track_abs_path)
                cut_end = int(mp3.info.length * mp3.info.sample_rate)
            print(f"\nCutting from Cue {HOTCUE_START} to end of track …")

        else:  # CUT_BETWEEN_CUES
            if HOTCUE_END not in hotcues:
                print(f"\nError: Hotcue {HOTCUE_END} is not set on this track.")
                sys.exit(1)
            cut_end = hotcues[HOTCUE_END]
            if cut_start >= cut_end:
                print(
                    f"\nError: Hotcue {HOTCUE_START} ({cut_start:.0f} samples) must be "
                    f"before Hotcue {HOTCUE_END} ({cut_end:.0f} samples)."
                )
                sys.exit(1)
            print(f"\nCutting between Cue {HOTCUE_START} and Cue {HOTCUE_END} …")

        # Reverb tail only makes sense when cutting to the end of the track
        add_reverb = REVERB_TAIL and (MODE == "CUT_TO_END")

        if ext_lower == ".mp3":
            cut_mp3(track_abs_path, output_filepath, cut_start, cut_end, reverb_tail=add_reverb)
        elif ext_lower == ".flac":
            cut_flac(track_abs_path, output_filepath, cut_start, cut_end, reverb_tail=add_reverb)
        else:
            cut_wav(track_abs_path, output_filepath, cut_start, cut_end, reverb_tail=add_reverb)

    elif MODE == "ADD_SILENCE":
        # Exactly one insertion-point source must be specified
        if SILENCE_CUE is None and SILENCE_TIMESTAMP is None:
            print("Error: In ADD_SILENCE mode set either SILENCE_CUE or SILENCE_TIMESTAMP.")
            sys.exit(1)
        if SILENCE_CUE is not None and SILENCE_TIMESTAMP is not None:
            print("Error: Set only one of SILENCE_CUE or SILENCE_TIMESTAMP, not both.")
            sys.exit(1)

        if SILENCE_CUE is not None:
            # Resolve the insertion point from an Engine DJ hotcue
            if SILENCE_CUE not in hotcues:
                print(f"\nError: Hotcue {SILENCE_CUE} is not set on this track.")
                sys.exit(1)
            insert_at = hotcues[SILENCE_CUE]
            mins = int(insert_at / 44100) // 60
            secs = (insert_at / 44100) - mins * 60
            label = f"Cue {SILENCE_CUE} ({mins}:{secs:05.2f})"
        else:
            # Resolve the insertion point from a wall-clock timestamp in seconds
            if ext_lower in (".flac", ".wav"):
                sr = sf.info(track_abs_path).samplerate
            else:
                mp3 = MP3(track_abs_path)
                sr = mp3.info.sample_rate
            insert_at = SILENCE_TIMESTAMP * sr
            mins = int(SILENCE_TIMESTAMP) // 60
            secs = SILENCE_TIMESTAMP - mins * 60
            label = f"{mins}:{secs:05.2f} ({SILENCE_TIMESTAMP:.3f} s)"

        print(f"\nInserting {SILENCE_DURATION_SECS:.1f} s of silence at {label} …")

        if ext_lower == ".mp3":
            insert_silence_mp3(track_abs_path, output_filepath, insert_at, SILENCE_DURATION_SECS)
        elif ext_lower == ".flac":
            insert_silence_flac(track_abs_path, output_filepath, insert_at, SILENCE_DURATION_SECS)
        else:
            insert_silence_wav(track_abs_path, output_filepath, insert_at, SILENCE_DURATION_SECS)

    else:
        print(f"Error: Unknown MODE '{MODE}'. Valid options: CUT_BETWEEN_CUES, CUT_TO_END, ADD_SILENCE")
        sys.exit(1)

    # ── Update the embedded track title ───────────────────────────────────────
    update_track_title(output_filepath, output_title)

    # ── Trigger Spotlight re-index on macOS ───────────────────────────────────
    if sys.platform == "darwin":
        subprocess.run(["mdimport", output_filepath], capture_output=True)


if __name__ == "__main__":
    main()
