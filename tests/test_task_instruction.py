"""Tests for the task-instruction extraction in `episode_builder._task_instruction`.

The instruction lives in meta_info.json's `text` field as a JSON document stored inside a
string, so it gets parsed twice. Real recordings from the G2 have been seen with that field
empty, which used to crash the whole conversion - these tests pin the behaviour that a
missing or malformed instruction is a metadata gap, not a reason to discard a good
trajectory.
"""

from __future__ import annotations

import json

import pytest

from episode_builder import _task_instruction


def meta(text: object) -> dict:
    """A minimal meta_info.json-shaped dict carrying just the `text` field."""
    return {"text": text}


def test_reads_description_and_extra():
    raw = json.dumps({"description": "Pick up the bottle", "extra": ["Place it on the tray"]})
    assert _task_instruction(meta(raw)) == "Pick up the bottle. Place it on the tray"


def test_description_only():
    assert _task_instruction(meta(json.dumps({"description": "Pick up the bottle"}))) == "Pick up the bottle"


def test_strips_whitespace_and_drops_blank_parts():
    raw = json.dumps({"description": "  Pick up the bottle  ", "extra": ["", "   ", "Then stop"]})
    assert _task_instruction(meta(raw)) == "Pick up the bottle. Then stop"


# --- the cases that used to raise ---------------------------------------------------------


def test_empty_string_does_not_raise():
    """The case seen in real recordings: `"text": ""`. json.loads rejects it."""
    assert _task_instruction(meta("")) == ""


def test_whitespace_only_does_not_raise():
    assert _task_instruction(meta("   \n ")) == ""


def test_missing_field_does_not_raise():
    assert _task_instruction({}) == ""


def test_null_field_does_not_raise():
    assert _task_instruction(meta(None)) == ""


def test_malformed_json_does_not_raise():
    assert _task_instruction(meta("{not valid json")) == ""


def test_non_object_json_does_not_raise():
    """`text` holding a bare JSON scalar or list rather than an object."""
    assert _task_instruction(meta(json.dumps(["a", "b"]))) == ""
    assert _task_instruction(meta(json.dumps("just a string"))) == ""


def test_object_with_no_usable_content():
    assert _task_instruction(meta(json.dumps({"description": "", "extra": []}))) == ""


def test_non_string_parts_are_ignored():
    """Guards against a numeric/null entry in `extra` blowing up the join."""
    raw = json.dumps({"description": "Pick up", "extra": [None, 42, "and place"]})
    assert _task_instruction(meta(raw)) == "Pick up. and place"


@pytest.mark.parametrize("bad", ["", "   ", "{oops", json.dumps(42)])
def test_warns_rather_than_raising(bad, caplog):
    """Every degraded path must log, so a silently empty instruction is never a surprise."""
    with caplog.at_level("WARNING"):
        assert _task_instruction(meta(bad)) == ""
    assert caplog.records, f"no warning logged for {bad!r}"
