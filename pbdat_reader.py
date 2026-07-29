"""Parses Genie Studio's raw `.pbdat` recording files.

Each `.pbdat` file is a raw dump of one cosine_bus topic: an 8-byte frame-count
header, followed by repeating frames of an 8-byte ptp timestamp, 24 reserved
bytes (a second timestamp-like value plus an incrementing sequence counter -
not needed here), an 8-byte payload size, and the raw protobuf payload itself.

This exact byte layout was verified against a real recording (parsed every
frame of a 2100-frame `-hal-joint_state.pbdat` and landed precisely at EOF,
matching the header's declared count) - it is not officially documented
anywhere, so if a future Genie Studio version changes it, this is the module
to revisit.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar

from google.protobuf.message import Message

_HEADER_SIZE = 8
_TIMESTAMP_SIZE = 8
_RESERVED_SIZE = 24
_FRAME_SIZE_FIELD = 8

MessageT = TypeVar("MessageT", bound=Message)


def read_pbdat_frames(path: Path, message_cls: type[MessageT]) -> Iterator[tuple[int, MessageT]]:
    """Yields (timestamp_ns, decoded_message) for every frame in a `.pbdat` file.

    Args:
        path: path to the `.pbdat` file for a single cosine_bus topic.
        message_cls: the protobuf message class to parse each frame's payload as
            (e.g. `genie_msgs_pb.msg.JointState_pb2.JointState`).

    Returns:
        An iterator of (timestamp_ns, message) tuples, in recorded order.
    """
    data = path.read_bytes()
    (declared_count,) = struct.unpack_from("<q", data, 0)

    offset = _HEADER_SIZE
    n = 0
    while offset < len(data):
        (timestamp_ns,) = struct.unpack_from("<q", data, offset)
        offset += _TIMESTAMP_SIZE + _RESERVED_SIZE
        (frame_size,) = struct.unpack_from("<q", data, offset)
        offset += _FRAME_SIZE_FIELD

        payload = data[offset : offset + frame_size]
        offset += frame_size

        message = message_cls()
        message.ParseFromString(payload)
        yield timestamp_ns, message
        n += 1

    if n != declared_count:
        raise ValueError(
            f"{path}: header declared {declared_count} frames but parsed {n} "
            "- the file may be truncated or the frame layout has changed"
        )


def read_pbdat_raw_frames(path: Path) -> Iterator[tuple[int, bytes]]:
    """Yields (timestamp_ns, raw_payload_bytes) for every frame, unparsed.

    Same framing as `read_pbdat_frames`, for topics with no known message
    class (e.g. `/remote/vr_data` - see `vr_trigger_parser.py`).
    """
    data = path.read_bytes()
    (declared_count,) = struct.unpack_from("<q", data, 0)

    offset = _HEADER_SIZE
    n = 0
    while offset < len(data):
        (timestamp_ns,) = struct.unpack_from("<q", data, offset)
        offset += _TIMESTAMP_SIZE + _RESERVED_SIZE
        (frame_size,) = struct.unpack_from("<q", data, offset)
        offset += _FRAME_SIZE_FIELD

        payload = data[offset : offset + frame_size]
        offset += frame_size

        yield timestamp_ns, payload
        n += 1

    if n != declared_count:
        raise ValueError(
            f"{path}: header declared {declared_count} frames but parsed {n} "
            "- the file may be truncated or the frame layout has changed"
        )
