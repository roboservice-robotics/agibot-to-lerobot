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
