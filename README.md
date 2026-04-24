# mp3-cutter

Losslessly cut a section from an MP3 file using hotcue positions from your [Engine DJ](https://enginedj.com/) library. Pure Python — no ffmpeg or other external tools required.

---

## How it works

The script reads hotcue timestamps directly from the Engine DJ SQLite database, then walks the raw MP3 frame headers to find the two frame boundaries that most closely match the requested cut range. It writes a new file with those frames removed, preserving ID3v2 and ID3v1 tags. No audio is re-encoded.

Because MP3 is a frame-based format the cut snaps to the nearest frame boundary. The script reports the resulting drift (typically < 26 ms at 44.1 kHz).

---

## Requirements

- Python 3.8+
- Engine DJ installed with an active library database at:  
  `~/Music/Engine Library/Database2/m.db`

---

## Setup

Edit the configuration block at the top of `engineDJ_cutByHotCues.py`:

```python
TRACK_FILENAME  = "example_track.mp3"  # filename to look up in the Engine DJ library
HOTCUE_START    = 2                    # hotcue number where the cut begins (1–8)
HOTCUE_END      = 5                    # hotcue number where the cut ends (1–8)
OUTPUT_APPENDIX = "(Short Edit)"       # appended to the output filename
OUTPUT_PATH     = os.path.dirname(os.path.abspath(__file__))  # output directory
ENGINE_DB_PATH  = os.path.expanduser("~/Music/Engine Library/Database2/m.db")
```

---

## Usage

```bash
python engineDJ_cutByHotCues.py
```

The script will:
1. Look up the track in the Engine DJ library (prompts for a choice if multiple matches exist).
2. Print all available hotcues and their positions.
3. Cut the MP3 between the two configured hotcues.
4. Report cut length, actual cut length, and drift.

---

## License

MIT — see [LICENSE](LICENSE).
