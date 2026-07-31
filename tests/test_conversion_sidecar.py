"""Tests for the provenance recorded alongside a converted episode.

LeRobot's own schema has nowhere to record three things a validation consumer needs: which
action dimensions are real commands vs zero-filled, the true frame timing (LeRobot stores
uniform synthetic timestamps), and whether the gripper left/right mapping could be confirmed.
These cover the helpers that produce that provenance.
"""

from __future__ import annotations

import json

from episode_builder import (
    NS_PER_S,
    _detect_gaps,
    _gripper_mapping_verifiable,
    _write_conversion_sidecar,
)

FPS = 30.0
NOMINAL_DT_NS = int(NS_PER_S / FPS)  # ~33.3 ms


def steady(n: int, start: int = 1_785_250_228_257_431_038) -> list[int]:
    """A gap-free timestamp series at the nominal frame rate."""
    return [start + i * NOMINAL_DT_NS for i in range(n)]


# --- gap detection ------------------------------------------------------------------------


def test_no_gaps_in_steady_series():
    assert _detect_gaps(steady(100), FPS) == []


def test_detects_a_dropped_frame_run():
    ts = steady(10)
    # Simulate a 1.4s drop before frame 5 - the magnitude seen in real recordings.
    ts = ts[:5] + [t + int(1.4 * NS_PER_S) for t in ts[5:]]
    gaps = _detect_gaps(ts, FPS)
    assert len(gaps) == 1
    assert gaps[0]["frame_index"] == 5
    assert gaps[0]["dt_s"] > 1.4


def test_normal_jitter_is_not_a_gap():
    """Sub-threshold timing wobble must not be reported, or every frame becomes a 'gap'."""
    ts = steady(50)
    ts = [t + (i % 3) * 1_000_000 for i, t in enumerate(ts)]  # +-1ms jitter
    assert _detect_gaps(ts, FPS) == []


def test_multiple_gaps_all_reported():
    ts = steady(20)
    ts = ts[:5] + [t + int(0.5 * NS_PER_S) for t in ts[5:]]
    ts = ts[:12] + [t + int(0.8 * NS_PER_S) for t in ts[12:]]
    assert [g["frame_index"] for g in _detect_gaps(ts, FPS)] == [5, 12]


def test_degenerate_inputs():
    assert _detect_gaps([], FPS) == []
    assert _detect_gaps(steady(1), FPS) == []
    assert _detect_gaps(steady(10), 0) == []  # unknown fps -> no nominal period to compare


# --- gripper mapping verifiability --------------------------------------------------------


def test_identical_grippers_cannot_verify_mapping():
    """Both hands moving together: a left/right swap would be invisible."""
    series = [(0.2, 0.2), (0.5, 0.5), (0.9, 0.9)]
    assert _gripper_mapping_verifiable(series) is False


def test_diverging_grippers_can_verify_mapping():
    series = [(0.2, 0.2), (0.9, 0.1), (0.5, 0.5)]
    assert _gripper_mapping_verifiable(series) is True


def test_tiny_differences_do_not_count_as_divergence():
    """Noise-level differences prove nothing about the mapping."""
    assert _gripper_mapping_verifiable([(0.50, 0.51), (0.30, 0.29)]) is False


def test_missing_values_are_skipped():
    assert _gripper_mapping_verifiable([(None, None), (None, 0.5)]) is False
    assert _gripper_mapping_verifiable([(None, None), (0.9, 0.1)]) is True


def test_empty_series_is_not_verifiable():
    assert _gripper_mapping_verifiable([]) is False


# --- sidecar writing ----------------------------------------------------------------------


def test_sidecar_lands_under_meta_and_round_trips(tmp_path):
    payload = {"schema_version": 1, "action_uncovered": ["idx01_body_joint1"]}
    _write_conversion_sidecar(tmp_path, "agibot_conversion.json", payload)

    written = tmp_path / "meta" / "agibot_conversion.json"
    assert written.is_file()
    assert json.loads(written.read_text()) == payload


def test_sidecar_creates_meta_dir_if_absent(tmp_path):
    _write_conversion_sidecar(tmp_path / "fresh", "agibot_conversion.json", {"schema_version": 1})
    assert (tmp_path / "fresh" / "meta" / "agibot_conversion.json").is_file()


def test_sidecar_honors_custom_name(tmp_path):
    # Batch conversion writes one sidecar per source recording under distinct names.
    _write_conversion_sidecar(tmp_path, "agibot_conversion.rec-a.json", {"schema_version": 1})
    assert (tmp_path / "meta" / "agibot_conversion.rec-a.json").is_file()
