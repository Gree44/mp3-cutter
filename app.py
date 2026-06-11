#!/usr/bin/env python3
"""
Flask web UI for engineDJ_cutByHotCues.py
"""

import csv
import json
import os
import glob
import sys
import threading
import sqlite3
import zlib
import struct

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

REPO_ROOT   = os.path.dirname(os.path.abspath(__file__))
STATE_FILE  = os.path.join(REPO_ROOT, "ui_state.json")


def _engine_module():
    """Return the engineDJ_cutByHotCues module, loading it once and caching it."""
    if not hasattr(_engine_module, "_mod"):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "engineDJ", os.path.join(REPO_ROOT, "engineDJ_cutByHotCues.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _engine_module._mod = mod
    return _engine_module._mod

DEFAULT_STATE = {
    "track_filename": "",
    "output_appendix": "",
    "search_full_db": True,
    "output_path": os.path.expanduser("~/Library/CloudStorage/OneDrive-Personal/DJing/Edits"),
    "engine_db_path": os.path.expanduser("~/Music/Engine Library/Database2/m.db"),
    "mode": "CUT_BETWEEN_CUES",
    "cut_between_start": 1,
    "cut_between_end": 2,
    "cut_to_end_hotcue": 1,
    "cut_to_end_reverb_tail": True,
    "cut_to_end_reverb_room_size": 1.0,
    "cut_to_end_reverb_damping": 0.5,
    "cut_to_end_reverb_wet_level": 0.2,
    "cut_to_end_reverb_width": 1.0,
    "cut_to_end_reverb_tail_secs": 4.0,
    "cut_to_end_reverb_blend_secs": 0.1,
    "add_silence_use_cue": True,
    "add_silence_cue": 1,
    "add_silence_timestamp": 0.0,
    "add_silence_duration_secs": 1.0,
    "compress_bitrate": 320,
    "compress_remove_artwork": False,
    "copy_src_start_cue": 1,
    "copy_src_end_cue": 2,
    "copy_dst_start_cue": 3,
    "copy_dst_end_cue": None,
    "copy_repeat_count": 1,
    "copy_paste_mode": "ADD",
}


def load_state():
    if os.path.isfile(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                data = json.load(f)
            merged = {**DEFAULT_STATE, **data}
            return merged
        except Exception:
            pass
    return dict(DEFAULT_STATE)


def save_state(updates: dict):
    state = load_state()
    state.update(updates)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_csv_tracks():
    """Return list of {label, filename} from all CSVs in repo root."""
    tracks = []
    for csv_path in sorted(glob.glob(os.path.join(REPO_ROOT, "*.csv"))):
        playlist_name = os.path.splitext(os.path.basename(csv_path))[0]
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # The "File name" column contains a full path — we only need the basename
                full_path = (row.get("File name") or "").strip()
                if not full_path:
                    continue
                filename = os.path.basename(full_path)
                if not filename:
                    continue
                title  = (row.get("Title") or filename).strip()
                artist = (row.get("Artist") or "").strip()
                label  = f"{title}{' — ' + artist if artist else ''}"
                tracks.append({
                    "playlist": playlist_name,
                    "label":    label,
                    "filename": filename,
                })
    # Deduplicate by filename, preserving order
    seen = set()
    result = []
    for t in tracks:
        if t["filename"] not in seen:
            seen.add(t["filename"])
            result.append(t)
    return result


def get_db_tracks(db_path):
    """Return list of {label, filename} for every track in the Engine DJ database."""
    if not os.path.isfile(db_path):
        return []
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("SELECT filename, title, artist FROM Track ORDER BY title COLLATE NOCASE")
    rows = cur.fetchall()
    conn.close()
    tracks = []
    seen   = set()
    for full_path, title, artist in rows:
        if not full_path:
            continue
        filename = os.path.basename(full_path.replace("\\", "/"))
        if not filename or filename in seen:
            continue
        seen.add(filename)
        title  = (title or filename).strip()
        artist = (artist or "").strip()
        label  = f"{title}{' — ' + artist if artist else ''}"
        tracks.append({"label": label, "filename": filename})
    return tracks


def _parse_hotcue_blob(blob):
    """Parse a quickCues blob.  Returns dict {cue_number: seconds} or raises."""
    data     = zlib.decompress(blob[4:])
    num_cues = struct.unpack(">q", data[:8])[0]
    pos      = 8
    hotcues  = {}
    for i in range(num_cues):
        if pos + 1 > len(data):
            break
        name_len = data[pos]
        pos += 1
        if pos + name_len + 12 > len(data):
            break
        pos += name_len
        position = struct.unpack(">d", data[pos: pos + 8])[0]
        pos += 8
        pos += 4
        if position >= 0:
            hotcues[i + 1] = round(position / 44100, 3)
    return hotcues


def get_hotcues_for_track(db_path, filename, track_id=None):
    """Return hotcue data for a track.

    When *track_id* is supplied the lookup is exact (no disambiguation needed).

    Otherwise the function searches by filename.  If multiple DB rows match it
    returns ``{"multiple": [{"track_id": …, "path": …, "exists": bool}, …]}``
    so the caller can let the user choose.

    The normal success response adds ``_track_id`` and ``_file_exists`` keys so
    the UI can warn when the audio file is missing on disk.
    """
    if not os.path.isfile(db_path):
        return {"error": f"Engine DJ database not found: {db_path}"}

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    if track_id is not None:
        cur.execute("SELECT id, path, filename FROM Track WHERE id = ?", (int(track_id),))
        rows = cur.fetchall()
    else:
        cur.execute("SELECT id, path, filename FROM Track WHERE filename = ?", (filename,))
        rows = cur.fetchall()
        if not rows:
            cur.execute("SELECT id, path, filename FROM Track WHERE filename LIKE ?",
                        (f"%{filename}%",))
            rows = cur.fetchall()

    if not rows:
        conn.close()
        return {"error": f"Track '{filename}' not found in Engine DJ library"}

    # Multiple matches → ask the user to disambiguate
    if len(rows) > 1:
        eng = _engine_module()
        choices = []
        for tid, rel_path, fn in rows:
            abs_path = eng.resolve_track_path(db_path, rel_path)
            choices.append({
                "track_id": tid,
                "filename": fn,
                "path":     abs_path,
                "exists":   os.path.isfile(abs_path),
            })
        conn.close()
        return {"multiple": choices}

    chosen_id, rel_path, _ = rows[0]
    eng        = _engine_module()
    abs_path   = eng.resolve_track_path(db_path, rel_path)
    file_exists = os.path.isfile(abs_path)

    cur.execute("SELECT quickCues FROM PerformanceData WHERE trackId = ?", (chosen_id,))
    row = cur.fetchone()
    conn.close()

    if not row or not row[0]:
        result = {"error": "No hotcue data found for this track"}
        result["_track_id"]    = chosen_id
        result["_file_exists"] = file_exists
        result["_abs_path"]    = abs_path
        return result

    try:
        hotcues = _parse_hotcue_blob(row[0])
    except Exception as exc:
        return {"error": f"Failed to parse hotcue data: {exc}",
                "_track_id": chosen_id, "_file_exists": file_exists, "_abs_path": abs_path}

    # Stringify int cue-number keys so jsonify (sort_keys=True) doesn't try to
    # compare them against the "_track_id" / "_file_exists" / "_abs_path" keys.
    result = {str(k): v for k, v in hotcues.items()}
    result["_track_id"]    = chosen_id
    result["_file_exists"] = file_exists
    result["_abs_path"]    = abs_path
    return result


# ── job execution (runs in background thread) ────────────────────────────────

_job_lock   = threading.Lock()
_job_status = {"running": False, "log": [], "success": None}


def _stream_log(line: str):
    with _job_lock:
        _job_status["log"].append(line)


def run_job(params: dict):
    """Execute the cut/silence/compress operation in a background thread."""
    import importlib.util, io, contextlib

    spec = importlib.util.spec_from_file_location(
        "engineDJ", os.path.join(REPO_ROOT, "engineDJ_cutByHotCues.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # When the UI has already resolved a unique track_id (e.g. after disambiguation)
    # we patch find_track to return that row directly without any interactive prompt.
    chosen_track_id = params.get("track_id")
    if chosen_track_id is not None:
        try:
            chosen_track_id = int(chosen_track_id)
        except (TypeError, ValueError):
            chosen_track_id = None

    if chosen_track_id is not None:
        db_path_for_patch = params.get("engine_db_path",
                                       os.path.expanduser("~/Music/Engine Library/Database2/m.db"))

        def _patched_find_track(db_path, filename):
            conn = sqlite3.connect(db_path)
            cur  = conn.cursor()
            cur.execute("SELECT id, path, filename FROM Track WHERE id = ?", (chosen_track_id,))
            row = cur.fetchone()
            conn.close()
            if row:
                return row
            import sys as _sys
            print(f"Error: track_id {chosen_track_id} not found in database.")
            _sys.exit(1)

        mod.find_track = _patched_find_track

    # Patch all config globals on the imported module
    mod.TRACK_FILENAME  = params["track_filename"]
    mod.OUTPUT_APPENDIX = params.get("output_appendix", "")
    mod.OUTPUT_PATH     = params["output_path"]
    mod.ENGINE_DB_PATH  = params["engine_db_path"]
    mod.MODE            = params["mode"]

    if params["mode"] == "CUT_BETWEEN_CUES":
        mod.CUT_BETWEEN_CUES_START = int(params["cut_between_start"])
        mod.CUT_BETWEEN_CUES_END   = int(params["cut_between_end"])

    elif params["mode"] == "CUT_TO_END":
        mod.CUT_TO_END_HOTCUE            = int(params["cut_to_end_hotcue"])
        mod.CUT_TO_END_REVERB_TAIL       = bool(params.get("cut_to_end_reverb_tail", True))
        mod.CUT_TO_END_REVERB_ROOM_SIZE  = float(params.get("cut_to_end_reverb_room_size", 1.0))
        mod.CUT_TO_END_REVERB_DAMPING    = float(params.get("cut_to_end_reverb_damping", 0.5))
        mod.CUT_TO_END_REVERB_WET_LEVEL  = float(params.get("cut_to_end_reverb_wet_level", 0.2))
        mod.CUT_TO_END_REVERB_WIDTH      = float(params.get("cut_to_end_reverb_width", 1.0))
        mod.CUT_TO_END_REVERB_TAIL_SECS  = float(params.get("cut_to_end_reverb_tail_secs", 4.0))
        mod.CUT_TO_END_REVERB_BLEND_SECS = float(params.get("cut_to_end_reverb_blend_secs", 0.1))

    elif params["mode"] == "ADD_SILENCE":
        use_cue = params.get("add_silence_use_cue", True)
        if use_cue:
            mod.ADD_SILENCE_CUE       = int(params["add_silence_cue"])
            mod.ADD_SILENCE_TIMESTAMP = None
        else:
            mod.ADD_SILENCE_CUE       = None
            mod.ADD_SILENCE_TIMESTAMP = float(params["add_silence_timestamp"])
        mod.ADD_SILENCE_DURATION_SECS = float(params.get("add_silence_duration_secs", 1.0))

    elif params["mode"] == "COMPRESS":
        mod.COMPRESS_BITRATE        = int(params.get("compress_bitrate", 320))
        mod.COMPRESS_REMOVE_ARTWORK = bool(params.get("compress_remove_artwork", False))

    elif params["mode"] == "COPY_BEATS_BETWEEN_CUES":
        mod.COPY_SRC_START_CUE = int(params["copy_src_start_cue"])
        mod.COPY_SRC_END_CUE   = int(params["copy_src_end_cue"])
        mod.COPY_DST_START_CUE = int(params["copy_dst_start_cue"])
        raw_dst_end = params.get("copy_dst_end_cue")
        mod.COPY_DST_END_CUE   = int(raw_dst_end) if raw_dst_end not in (None, "", 0, "0") else None
        mod.COPY_REPEAT_COUNT  = max(1, int(params.get("copy_repeat_count") or 1))
        raw_mode = (params.get("copy_paste_mode") or "ADD").upper()
        mod.COPY_PASTE_MODE    = "REPLACE" if raw_mode == "REPLACE" else "ADD"

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            mod.main()
        output = buf.getvalue()
        with _job_lock:
            for line in output.splitlines():
                _job_status["log"].append(line)
            _job_status["running"] = False
            _job_status["success"] = True
    except SystemExit as exc:
        output = buf.getvalue()
        with _job_lock:
            for line in output.splitlines():
                _job_status["log"].append(line)
            _job_status["log"].append(f"[exited with code {exc.code}]")
            _job_status["running"] = False
            _job_status["success"] = (str(exc.code) == "0" or exc.code == 0)
    except Exception as exc:
        output = buf.getvalue()
        with _job_lock:
            for line in output.splitlines():
                _job_status["log"].append(line)
            _job_status["log"].append(f"[error] {exc}")
            _job_status["running"] = False
            _job_status["success"] = False


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    return jsonify(load_state())


@app.route("/api/csv_tracks")
def api_csv_tracks():
    return jsonify(get_csv_tracks())


@app.route("/api/db_tracks")
def api_db_tracks():
    db_path = request.args.get("db_path", "").strip()
    if not db_path:
        db_path = os.path.expanduser("~/Music/Engine Library/Database2/m.db")
    return jsonify(get_db_tracks(db_path))


@app.route("/api/hotcues")
def api_hotcues():
    filename = request.args.get("filename", "").strip()
    db_path  = request.args.get("db_path", "").strip()
    track_id = request.args.get("track_id", "").strip() or None
    if not filename and not track_id:
        return jsonify({"error": "filename or track_id required"}), 400
    if not db_path:
        db_path = os.path.expanduser("~/Music/Engine Library/Database2/m.db")
    result = get_hotcues_for_track(db_path, filename, track_id=track_id)
    return jsonify(result)


@app.route("/api/run", methods=["POST"])
def api_run():
    with _job_lock:
        if _job_status["running"]:
            return jsonify({"error": "A job is already running"}), 409

    params = request.get_json(force=True)
    save_state(params)

    with _job_lock:
        _job_status["running"] = True
        _job_status["log"]     = ["Starting …"]
        _job_status["success"] = None

    t = threading.Thread(target=run_job, args=(params,), daemon=True)
    t.start()
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    with _job_lock:
        return jsonify(dict(_job_status))


if __name__ == "__main__":
    app.run(debug=True, port=5055)
