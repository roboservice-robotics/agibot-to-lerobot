"""Builds one LeRobot v2.1 episode from a raw Genie Studio recording directory.

Pulls together `pbdat_reader.py` (joint/gripper state+command), `vr_trigger_parser.py`
(the real-time gripper command signal, reverse-engineered - see that module's
docstring), and `video_decoder.py` (camera frames), and feeds them one frame at
a time through LeRobot's own `LeRobotDataset` writer, using the head camera's
frame timestamps as the episode's reference timeline.

Scope (see the project's planning notes for why): joint_state -> observation.state,
joint_cmd -> action, for arms/body/head; left/right gripper measured position folded
into observation.state, commanded gripper openness (from vr_data) folded into action.
RGB cameras only (head_color, hand_left_color, hand_right_color) - no depth, no
lidar/imu/chassis/nav topics.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from pbdat_reader import read_pbdat_frames, read_pbdat_raw_frames
from video_decoder import decode_h265_frames, probe_video_info
from vr_trigger_parser import gripper_openness_command

logger = logging.getLogger(__name__)

HEAD_CAMERA = "head_color"
SECONDARY_CAMERAS = ["hand_left_color", "hand_right_color"]
ALL_CAMERAS = [HEAD_CAMERA, *SECONDARY_CAMERAS]

T = TypeVar("T")


class NearestCursor:
    """Looks up the value nearest a forward-moving target timestamp in a
    timestamp-ordered stream, without rescanning from the start each call.

    Assumes `get()` is called with non-decreasing `target_ts` values, which
    holds here since every caller drives off the head camera's own increasing
    frame timestamps.
    """

    def __init__(self, items: Iterable[tuple[int, T]]) -> None:
        self._it = iter(items)
        self._prev = next(self._it, None)
        self._next = next(self._it, None)

    def get(self, target_ts: int) -> T | None:
        while self._next is not None and abs(self._next[0] - target_ts) <= abs(self._prev[0] - target_ts):
            self._prev = self._next
            self._next = next(self._it, None)
        return self._prev[1] if self._prev is not None else None


def _load_meta_info(recording_dir: Path) -> dict[str, Any]:
    meta = json.loads((recording_dir / "meta_info.json").read_text())
    if not meta.get("data_validate", {}).get("validate", False):
        raise ValueError(f"{recording_dir}: failed Genie Studio's data_validate check")
    integrity = meta.get("integrity", {})
    if not integrity.get("integrity", False):
        raise ValueError(
            f"{recording_dir}: failed Genie Studio's integrity check (reason={integrity.get('reason')})"
        )
    return meta


def _task_instruction(meta: dict[str, Any]) -> str:
    """Task/language instruction for the episode, read from meta_info.json's `text` field.

    `text` holds a JSON document stored as a *string*, so it needs parsing a second time.
    Recordings in the wild have been seen with it empty (`""`), which `json.loads` rejects -
    and an unusable instruction is a metadata gap, not a reason to throw away an otherwise
    good trajectory. So anything missing, empty, or unparseable yields "" with a warning
    rather than an exception. Pass `--instruction` to `convert.py` to supply one by hand.
    """
    raw = meta.get("text") or ""
    if not raw.strip():
        logger.warning("meta_info.json has no task instruction text - converting with an empty instruction")
        return ""

    try:
        text = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "meta_info.json 'text' is not valid JSON (%r) - converting with an empty instruction", raw[:80]
        )
        return ""

    if not isinstance(text, dict):
        logger.warning(
            "meta_info.json 'text' parsed to %s, expected an object - converting with an empty instruction",
            type(text).__name__,
        )
        return ""

    parts = [text.get("description", ""), *text.get("extra", [])]
    instruction = ". ".join(p.strip() for p in parts if isinstance(p, str) and p.strip())
    if not instruction:
        logger.warning("meta_info.json 'text' has no description/extra content - empty instruction")
    return instruction


def _read_joint_series(path: Path, message_cls: type, position_attr: str) -> list[tuple[int, dict[str, float]]]:
    """For JointState/JointCommand: parallel `name`/<position_attr> repeated fields."""
    frames = []
    for ts, msg in read_pbdat_frames(path, message_cls):
        positions = getattr(msg, position_attr)
        frames.append((ts, dict(zip(msg.name, positions))))
    return frames


def _read_end_state_series(path: Path) -> list[tuple[int, dict[str, float]]]:
    """For EndState (left/right_ee_data): repeated `name` + repeated `end_state` submessage."""
    from genie_msgs_pb.msg import EndState_pb2

    frames = []
    for ts, msg in read_pbdat_frames(path, EndState_pb2.EndState):
        frames.append((ts, {name: es.position for name, es in zip(msg.name, msg.end_state)}))
    return frames


def _read_vr_gripper_series(path: Path) -> list[tuple[int, tuple[float | None, float | None]]]:
    return [(ts, gripper_openness_command(payload)) for ts, payload in read_pbdat_raw_frames(path)]


# Recorded timestamps are epoch nanoseconds (verified against real recordings: values ~1.78e18,
# consecutive head-camera frames ~33.3e6 ns apart at 30 fps).
NS_PER_S = 1_000_000_000

# A frame whose gap from its predecessor exceeds this multiple of the nominal frame period is
# treated as a camera drop rather than normal jitter. Real recordings show drops up to ~1.4 s
# against a 33 ms nominal period, so anything past 2x is unambiguous.
GAP_FACTOR = 2.0


def _detect_gaps(timestamps_ns: list[int], fps: float) -> list[dict[str, float]]:
    """Finds camera-drop gaps in a frame timestamp series.

    LeRobot is handed uniform synthetic timestamps (see the comment in `build_episode`'s
    frame loop), so the real timing is lost in the dataset itself. Anything computing a rate
    of change - joint velocity, jerk - would silently treat a 1.4 s drop as one frame period
    and report a huge false spike. Recording the gaps here lets consumers exclude those
    frames instead.

    Returns one entry per gap: the index of the frame *after* the gap, and the real elapsed
    time in seconds.
    """
    if fps <= 0:
        return []
    nominal_dt_s = 1.0 / fps
    gaps = []
    for i in range(1, len(timestamps_ns)):
        dt_s = (timestamps_ns[i] - timestamps_ns[i - 1]) / NS_PER_S
        if dt_s > nominal_dt_s * GAP_FACTOR:
            gaps.append({"frame_index": i, "dt_s": round(dt_s, 6)})
    return gaps


def _gripper_mapping_verifiable(openness_series: list[tuple[float | None, float | None]]) -> bool:
    """Whether this recording could confirm the VR controller left/right assignment.

    `vr_trigger_parser` maps controller index 0 to the left gripper and 1 to the right, but
    that mapping is an unverified guess (see its docstring) - no recording tested so far had
    the two hands do different things, and while both grippers move together a swap is
    invisible. If the two signals differ meaningfully somewhere in this episode, then this
    recording *would* expose a swap, so the mapping is verifiable from it.
    """
    diffs = [abs(l - r) for l, r in openness_series if l is not None and r is not None]
    return bool(diffs) and max(diffs) > 0.05


def _write_conversion_sidecar(output_root: Path, payload: dict[str, Any]) -> None:
    """Writes provenance that LeRobot's own schema has nowhere to put.

    Three things a consumer needs but the dataset cannot express: which action dimensions are
    real vs zero-filled, the true frame timing (LeRobot only gets uniform synthetic
    timestamps), and whether the gripper left/right mapping was verifiable. Kept in our own
    file under `meta/` rather than bent into `info.json`, whose schema belongs to lerobot.
    """
    sidecar = output_root / "meta" / "agibot_conversion.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote conversion sidecar %s", sidecar)


def build_episode(
    recording_dir: Path, output_root: Path, repo_id: str, instruction: str | None = None
) -> None:
    """Converts one raw Genie Studio recording into a LeRobot v2.1 dataset at `output_root`.

    `instruction` overrides the task text from meta_info.json. Needed because recordings are
    routinely made with no instruction text at all, and a VLA trained on an empty instruction
    learns nothing from the language channel.
    """
    from genie_msgs_pb.msg import JointCommand_pb2, JointState_pb2
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    meta = _load_meta_info(recording_dir)
    instruction = instruction if instruction is not None else _task_instruction(meta)
    record_dir = recording_dir / "record"
    camera_dir = recording_dir / "camera"

    joint_state_frames = _read_joint_series(record_dir / "-hal-joint_state.pbdat", JointState_pb2.JointState, "motor_position")
    joint_cmd_frames = _read_joint_series(record_dir / "-hal-joint_cmd.pbdat", JointCommand_pb2.JointCommand, "position")
    left_ee_frames = _read_end_state_series(record_dir / "-hal-left_ee_data.pbdat")
    right_ee_frames = _read_end_state_series(record_dir / "-hal-right_ee_data.pbdat")
    vr_frames = _read_vr_gripper_series(record_dir / "-remote-vr_data.pbdat")

    joint_names = list(joint_state_frames[0][1].keys())
    left_gripper_name = next(iter(left_ee_frames[0][1].keys()))
    right_gripper_name = next(iter(right_ee_frames[0][1].keys()))
    state_names = [*joint_names, left_gripper_name, right_gripper_name]

    # This episode's own observed open/closed range, used to scale vr_data's 0-1
    # commanded-openness value into the same physical (radian) unit as state -
    # deliberately not a hardcoded constant, since gripper hardware (and so its
    # position range) differs between recordings (omnipicker vs. ctek90d seen so far).
    left_positions = [v[left_gripper_name] for _, v in left_ee_frames]
    right_positions = [v[right_gripper_name] for _, v in right_ee_frames]
    left_open, left_closed = max(left_positions), min(left_positions)
    right_open, right_closed = max(right_positions), min(right_positions)

    joint_state_cursor = NearestCursor(joint_state_frames)
    joint_cmd_cursor = NearestCursor(joint_cmd_frames)
    left_ee_cursor = NearestCursor(left_ee_frames)
    right_ee_cursor = NearestCursor(right_ee_frames)
    vr_cursor = NearestCursor(vr_frames)

    # NearestCursor consumes these lazily (it only needs an iterable), so the
    # hand cameras' decoded frames are never all held in memory at once.
    secondary_cursors = {
        cam: NearestCursor(decode_h265_frames(
            camera_dir / cam / f"{cam}.h265", camera_dir / cam / f"{cam}.txt",
        ))
        for cam in SECONDARY_CAMERAS
    }

    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
        "action": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
    }
    fps = None
    for cam in ALL_CAMERAS:
        width, height, cam_fps = probe_video_info(camera_dir / cam / f"{cam}.h265")
        if cam == HEAD_CAMERA:
            fps = cam_fps
        features[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channels"],
        }
    assert fps is not None

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=round(fps),
        features=features,
        root=output_root,
        robot_type="agibot_g2",
        use_videos=True,
    )

    head_h265 = camera_dir / HEAD_CAMERA / f"{HEAD_CAMERA}.h265"
    head_txt = camera_dir / HEAD_CAMERA / f"{HEAD_CAMERA}.txt"

    # Accumulated for the conversion sidecar written after the episode is saved.
    head_timestamps_ns: list[int] = []
    openness_series: list[tuple[float | None, float | None]] = []
    # Joints seen in joint_cmd at least once. Anything in joint_names but missing here is
    # zero-filled into `action` below, which would be a fabricated ground-truth value.
    cmd_covered: set[str] = set()

    frame_count = 0
    for ts, head_frame in decode_h265_frames(head_h265, head_txt):
        state_joints = joint_state_cursor.get(ts) or {}
        cmd_joints = joint_cmd_cursor.get(ts) or {}
        head_timestamps_ns.append(ts)
        cmd_covered.update(cmd_joints)
        left_state = (left_ee_cursor.get(ts) or {}).get(left_gripper_name, left_closed)
        right_state = (right_ee_cursor.get(ts) or {}).get(right_gripper_name, right_closed)
        left_openness, right_openness = vr_cursor.get(ts) or (None, None)
        openness_series.append((left_openness, right_openness))

        left_action = left_closed + left_openness * (left_open - left_closed) if left_openness is not None else left_state
        right_action = right_closed + right_openness * (right_open - right_closed) if right_openness is not None else right_state

        state_vec = [state_joints.get(n, 0.0) for n in joint_names] + [left_state, right_state]
        action_vec = [cmd_joints.get(n, 0.0) for n in joint_names] + [left_action, right_action]

        frame = {
            "observation.state": np.array(state_vec, dtype=np.float32),
            "action": np.array(action_vec, dtype=np.float32),
            f"observation.images.{HEAD_CAMERA}": head_frame,
        }
        for cam in SECONDARY_CAMERAS:
            frame[f"observation.images.{cam}"] = secondary_cursors[cam].get(ts)

        # Deliberately not passing an explicit `timestamp=` here: the real recorded
        # timestamps (`ts`, used above for cross-stream alignment) include genuine
        # camera frame drops with gaps up to ~1.4s in the recordings tested - too
        # large for LeRobotDataset's own strict load-time tolerance check
        # (`tolerance_s`, default 1e-4s) to accept at any reasonable setting. Letting
        # it assign uniform frame_index/fps timestamps instead matches how virtually
        # every LeRobot dataset (including this app's own in-app recorder) is
        # structured, and avoids fighting that validator.
        dataset.add_frame(frame, task=instruction)
        frame_count += 1

    dataset.save_episode()

    uncovered = [n for n in joint_names if n not in cmd_covered]
    if uncovered:
        logger.warning(
            "joint_cmd never reported %d of %d joints (%s) - their `action` values are "
            "zero-filled, not real commands. Recorded in the sidecar so downstream consumers "
            "can exclude them.",
            len(uncovered), len(joint_names), ", ".join(uncovered),
        )

    gaps = _detect_gaps(head_timestamps_ns, fps)
    if gaps:
        logger.warning(
            "%d camera-drop gap(s) detected, largest %.2fs - LeRobot's timestamps are uniform "
            "and hide these; real timing is in the sidecar.",
            len(gaps), max(g["dt_s"] for g in gaps),
        )

    mapping_verifiable = _gripper_mapping_verifiable(openness_series)
    if not mapping_verifiable:
        logger.warning(
            "Both grippers moved together throughout this recording, so it cannot confirm the "
            "VR controller left/right mapping (see vr_trigger_parser). Gripper actions may be "
            "swapped; treat gripper metrics as unverified."
        )

    _write_conversion_sidecar(output_root, {
        "schema_version": 1,
        "source_recording": recording_dir.name,
        "instruction": instruction,
        "state_names": state_names,
        "nominal_fps": round(fps),
        "frame_count": frame_count,
        # Action dims backed by real commands vs zero-filled. Metrics computed against a
        # zero-filled dimension are meaningless, so consumers must exclude `action_uncovered`.
        "action_covered": [n for n in joint_names if n in cmd_covered],
        "action_uncovered": uncovered,
        # Real recorded frame times, seconds from episode start. LeRobot only stores uniform
        # frame_index/fps timestamps, so this is the only record of true timing.
        "timestamps_s": [
            round((ts - head_timestamps_ns[0]) / NS_PER_S, 6) for ts in head_timestamps_ns
        ] if head_timestamps_ns else [],
        "start_timestamp_ns": head_timestamps_ns[0] if head_timestamps_ns else None,
        "gap_frames": gaps,
        "gripper": {
            "left_name": left_gripper_name,
            "right_name": right_gripper_name,
            "left_open": left_open,
            "left_closed": left_closed,
            "right_open": right_open,
            "right_closed": right_closed,
            "lr_mapping_verifiable_from_this_recording": mapping_verifiable,
        },
    })

    logger.info("Converted %s -> %s (%d frames)", recording_dir, output_root, frame_count)
