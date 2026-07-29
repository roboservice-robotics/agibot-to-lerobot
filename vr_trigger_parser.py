"""Extracts the real-time commanded gripper signal from `/remote/vr_data`.

This topic does NOT match any message class in `genie_msgs_pb` (tried ~10
candidates including `VRData`/`VRControllerState` - all either fail to parse
or silently populate none of their fields, which is protobuf's normal
behavior when the wrong message class is used against bytes that happen to
still be valid-but-meaningless wire data). Since no matching schema could be
found, this module manually walks the raw protobuf wire format instead - a
legitimate, well-defined thing to do since protobuf's wire format is
self-describing (this is exactly what `protoc --decode_raw` does).

What we're reading (raw field numbers, not names - the real field names are
unknown): a repeated submessage at field 103, each occurrence holding a
controller index (field 1) and several per-controller values, of which field
6 is the one that matters here.

This mapping was verified empirically, not sourced from documentation (none
exists that we could find - see the project's research notes): field 6 was
observed to hold at 1.0 while the gripper sat fully open (measured position
0.0), ramp smoothly down to 0.0 over ~0.7s, and the gripper's *measured*
position followed the same ramp starting ~30-40ms later - consistent with
this being the real commanded input, not a coincidence. Confirmed
reproducible across two independent recordings (a third recording showed no
correlation, but that recording is the one Genie Studio's own integrity check
already flags as bad - separately excluded, see episode_builder.py).

Two caveats to know about if this ever needs revisiting:
- The value's direction is inverted relative to the "squeeze to close"
  convention used by most VR teleoperation integrations elsewhere in the
  LeRobot ecosystem (1.0 = open, 0.0 = closed here). Verified against real
  measured motion for this system, so left as-is rather than "corrected."
- Which controller index (0 or 1) is the left vs. right hand could not be
  verified - every recording checked had both hands moving in lockstep, with
  no moment where they diverged. Assumed 0=left, 1=right (matching the
  left-then-right ordering used everywhere else in this data), but this is
  an unverified guess, not a confirmed mapping.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator

_CONTROLLER_SUBMESSAGE_FIELD = 103
_CONTROLLER_INDEX_FIELD = 1
_TRIGGER_VALUE_FIELD = 6

_LEFT_CONTROLLER_INDEX = 0
_RIGHT_CONTROLLER_INDEX = 1


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result, offset
        shift += 7


def _skip_field(data: bytes, offset: int, wire_type: int) -> int:
    if wire_type == 0:  # varint
        _, offset = _read_varint(data, offset)
    elif wire_type == 1:  # 64-bit fixed
        offset += 8
    elif wire_type == 2:  # length-delimited
        length, offset = _read_varint(data, offset)
        offset += length
    elif wire_type == 5:  # 32-bit fixed
        offset += 4
    else:
        raise ValueError(f"unsupported protobuf wire type {wire_type}")
    return offset


def _iter_top_level_fields(data: bytes) -> Iterator[tuple[int, int, int]]:
    """Yields (field_number, wire_type, value_start_offset) for each top-level field."""
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        field_number = tag >> 3
        wire_type = tag & 0x7
        yield field_number, wire_type, offset
        offset = _skip_field(data, offset, wire_type)


def parse_gripper_trigger_values(payload: bytes) -> dict[int, float]:
    """Returns {controller_index: trigger_value} for one `/remote/vr_data` frame.

    `trigger_value` is on the raw 0.0-1.0 scale this field uses (1.0 = open,
    0.0 = closed - see module docstring for how that direction was verified).
    """
    results: dict[int, float] = {}
    offset = 0
    while offset < len(payload):
        tag, offset = _read_varint(payload, offset)
        field_number = tag >> 3
        wire_type = tag & 0x7
        if field_number == _CONTROLLER_SUBMESSAGE_FIELD and wire_type == 2:
            length, offset = _read_varint(payload, offset)
            sub_end = offset + length
            sub_offset = offset
            index: int | None = None
            value: float | None = None
            while sub_offset < sub_end:
                sub_tag, sub_offset = _read_varint(payload, sub_offset)
                sub_field = sub_tag >> 3
                sub_wire = sub_tag & 0x7
                if sub_field == _CONTROLLER_INDEX_FIELD and sub_wire == 0:
                    index, sub_offset = _read_varint(payload, sub_offset)
                elif sub_field == _TRIGGER_VALUE_FIELD and sub_wire == 5:
                    (value,) = struct.unpack_from("<f", payload, sub_offset)
                    sub_offset += 4
                else:
                    sub_offset = _skip_field(payload, sub_offset, sub_wire)
            if index is not None and value is not None:
                results[index] = value
            offset = sub_end
        else:
            offset = _skip_field(payload, offset, wire_type)
    return results


def gripper_openness_command(payload: bytes) -> tuple[float | None, float | None]:
    """Returns (left_openness, right_openness) commanded values for one vr_data frame.

    Either value is None if that controller wasn't present in this frame.
    """
    values = parse_gripper_trigger_values(payload)
    return values.get(_LEFT_CONTROLLER_INDEX), values.get(_RIGHT_CONTROLLER_INDEX)
