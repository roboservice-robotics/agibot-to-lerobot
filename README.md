# agibot-to-lerobot

Converts a raw Genie Studio recording from the Agibot G2 into a [LeRobot](https://github.com/huggingface/lerobot)
v2.1 dataset, ready for training (pi0.5, GR00T, GO-1, etc.) — no cloud conversion service
required.

## Why this exists

Genie Studio (the recording tool used during teleoperation) can convert its raw recordings
to LeRobot format itself, but only via its cloud platform, which costs money per conversion.
This tool does the same conversion locally, for free, by reading the raw recording files
directly.

## What it reads from a raw recording

A raw Genie Studio recording directory looks like this:

```
<recording-id>/
  meta_info.json          # task instruction text + data_validate/integrity flags
  record/
    -hal-joint_state.pbdat     # measured joint positions (observation.state)
    -hal-joint_cmd.pbdat       # commanded joint positions (action)
    -hal-left_ee_data.pbdat    # measured left gripper position
    -hal-right_ee_data.pbdat   # measured right gripper position
    -remote-vr_data.pbdat      # real-time commanded gripper openness (VR controller triggers)
    ... (other topics — chassis, lidar, IMU, TF, etc. — not used by this tool)
  camera/
    head_color/head_color.h265       (+ .txt sidecar: per-frame timestamp + I/P flag)
    hand_left_color/hand_left_color.h265
    hand_right_color/hand_right_color.h265
    ... (other camera streams — fisheye, stereo, depth — not used by this tool)
```

`.pbdat` files are raw per-topic dumps of the robot's internal `cosine_bus` messages
(protobuf-encoded, one topic per file). `.h265` files are standard Annex-B HEVC video
streams, decoded via `ffmpeg`. See the docstrings in `pbdat_reader.py`, `video_decoder.py`,
and `vr_trigger_parser.py` for exactly how each format was reverse-engineered and verified.

## What it writes

One LeRobot v2.1 dataset per recording, with:

- `observation.state` / `action` — arm, body, and head joint positions, plus left/right
  gripper openness (measured for `observation.state`, commanded for `action` — sourced
  from the VR controller trigger signal in `-remote-vr_data.pbdat`, scaled into the same
  per-recording open/closed range as the measured gripper position).
- `observation.images.head_color`, `observation.images.hand_left_color`,
  `observation.images.hand_right_color` — RGB video only (no depth, no fisheye/stereo,
  no lidar/IMU/chassis/nav topics — out of scope for manipulation training).

Frame timing is driven by the head camera's own frame timestamps; other streams are
aligned to each head-camera frame via nearest-timestamp lookup. See the comment in
`episode_builder.py`'s `build_episode()` for why frames are added without explicit
timestamps (real recordings have camera-drop gaps too large for LeRobot's strict
load-time tolerance check).

A recording that fails Genie Studio's own `data_validate` or `integrity` check (recorded
in its `meta_info.json`) is skipped with an error, not silently converted.

## Setup

```bash
pip install -r requirements.txt
pip install vendor/genie_msgs_pb-1.4.3-py3-none-any.whl
```

You'll also need `ffmpeg`/`ffprobe` on `PATH` (e.g. `apt install ffmpeg`) for camera decoding.

`requirements.txt` pins `lerobot==0.3.3` deliberately — it's the last release that writes
LeRobot v2.1 by default (v0.4.0 switched to always writing v3.0, with no way to opt back
into v2.1).

`genie_msgs_pb` (the protobuf message definitions for `cosine_bus` topics) isn't published
on PyPI — it's vendored here as a wheel copied out of the GDK SDK cache.

## Usage

```bash
python convert.py --input /path/to/<recording-id>
```

This converts the recording and writes it to `lerobot_format/<recording-id>/` next to
this script — that's the folder [`vla-app`](../vla-app)'s Dataset Review tool mounts by
default, so converted episodes show up there automatically.

To write somewhere else instead:

```bash
python convert.py --input /path/to/<recording-id> --output /path/to/output_dir
```

Full options:

```
--input     Raw Genie Studio recording directory (required)
--output    Output LeRobot dataset directory (default: lerobot_format/<recording-name>)
--repo-id   Local repo id passed to LeRobotDataset.create() (default: local/agibot_g2)
```

Exit code is `1` (with a logged reason) if the recording fails Genie Studio's own
validation checks; `0` on success.

## Files

| File | Purpose |
|------|---------|
| `convert.py` | CLI entry point |
| `episode_builder.py` | Orchestrates the conversion — reads all input streams, aligns them by timestamp, writes the LeRobot dataset |
| `pbdat_reader.py` | Parses raw `.pbdat` protobuf frame dumps |
| `video_decoder.py` | Decodes `.h265` camera streams into per-frame arrays via `ffmpeg` |
| `vr_trigger_parser.py` | Extracts the commanded gripper-openness signal from `/remote/vr_data` (reverse-engineered — see its docstring) |
| `vendor/genie_msgs_pb-1.4.3-py3-none-any.whl` | Vendored protobuf message definitions (not on PyPI) |

## Verified against

Tested end-to-end against 3 real recordings from the G2, including cross-checking the
gripper open/close signal against recordings with visible single-arm and dual-arm gripper
movement, and confirming the converted output loads correctly and plays back joint motion
+ video in sync in [`vla-app`](../vla-app)'s Dataset Review tool.
