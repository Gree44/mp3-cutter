# mp3-cutter
lightweight tool to cut out sections of mp3-files lossless, based on prepared cuts from the rekordbox library

## Usage

```bash
python3 rekordbox_cut.py \
  --xml /path/to/rekordbox.xml \
  --track /path/to/song.mp3 \
  --output /path/to/song_cut.mp3
```

The script will:
- locate the track in the rekordbox XML by matching `TRACK Location` to `--track`
- extract cut timestamps from `POSITION_MARK` entries
- prefer markers named `start`/`end` (or `in`/`out`) when present
- otherwise use the earliest and latest marker timestamps
- call `ffmpeg -c copy` so the mp3 is cut without re-encoding

### Dry-run

```bash
python3 rekordbox_cut.py \
  --xml /path/to/rekordbox.xml \
  --track /path/to/song.mp3 \
  --dry-run
```

### Marker names

If your cut markers have custom names, pass them explicitly:

```bash
python3 rekordbox_cut.py \
  --xml /path/to/rekordbox.xml \
  --track /path/to/song.mp3 \
  --start-mark intro \
  --end-mark outro
```
