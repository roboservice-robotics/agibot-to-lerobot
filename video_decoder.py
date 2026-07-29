"""Decodes Genie Studio's raw H.265 camera streams into per-frame arrays.

Each camera's `.h265` file is a completely standard Annex-B HEVC elementary
stream (confirmed: it starts with a normal `00 00 00 01` start code and
`ffprobe` reads codec/resolution/frame-rate directly from it, no custom
framing at all) - so this just shells out to `ffmpeg`/`ffprobe`, it does not
need to reverse-engineer anything.

Per-frame timestamps come from the camera's sidecar `.txt` file (one line per
frame: `<timestamp_ns> <I|P>`), paired with decoded frames in order - the
stream has no B-frames (only I/P, confirmed from the sidecar contents), so
decode order matches presentation order and this pairing is safe.

Requires `ffmpeg`/`ffprobe` on PATH (e.g. `apt install ffmpeg`).
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np


def probe_video_info(h265_path: Path) -> tuple[int, int, float]:
    """Returns (width, height, fps) for a raw H.265 elementary stream via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "json",
            str(h265_path),
        ],
        capture_output=True, check=True, text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return int(stream["width"]), int(stream["height"]), float(num) / float(den)


def read_frame_timestamps(sidecar_txt_path: Path) -> list[int]:
    """Returns the per-frame timestamp_ns list from a camera's `.txt` sidecar."""
    timestamps = []
    for line in sidecar_txt_path.read_text().splitlines():
        if not line.strip():
            continue
        timestamp_str, _frame_type = line.split()
        timestamps.append(int(timestamp_str))
    return timestamps


def decode_h265_frames(h265_path: Path, sidecar_txt_path: Path) -> Iterator[tuple[int, np.ndarray]]:
    """Yields (timestamp_ns, frame) for every frame in a raw H.265 stream.

    `frame` is an (height, width, 3) uint8 RGB array.
    """
    width, height, _fps = probe_video_info(h265_path)
    timestamps = read_frame_timestamps(sidecar_txt_path)
    frame_bytes = width * height * 3

    process = subprocess.Popen(
        [
            "ffmpeg", "-v", "error",
            "-i", str(h265_path),
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    assert process.stdout is not None

    frame_index = 0
    try:
        while True:
            chunk = process.stdout.read(frame_bytes)
            if len(chunk) < frame_bytes:
                break
            frame = np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3)
            if frame_index >= len(timestamps):
                raise ValueError(
                    f"{h265_path}: ffmpeg decoded more frames than the sidecar "
                    f"{sidecar_txt_path} has timestamps for"
                )
            yield timestamps[frame_index], frame
            frame_index += 1
    finally:
        process.stdout.close()
        process.wait()
