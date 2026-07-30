"""Tests for output directory naming.

Raw recordings are named from their task text, so they contain spaces and punctuation.
vla-app uses the dataset directory name as an episode id in URLs and rejects anything
outside [a-z0-9_-], so an unsanitised name yields a dataset it silently refuses to serve.
"""

from __future__ import annotations

import re

import pytest

from convert import slugify

# The character class vla-app's EpisodeService._validate_dir_name accepts.
VLA_APP_SAFE = re.compile(r"^[a-z0-9_-]+$")


@pytest.mark.parametrize("name", [
    "Arranging the internals_20260728225028",  # the real recording that prompted this
    "pick up the bottle / place on tray",
    "TASK.WITH.DOTS_123",
    "  leading and trailing  ",
    "Ünïcödé task",
    "multiple   spaces",
])
def test_output_is_always_safe_for_vla_app(name):
    assert VLA_APP_SAFE.match(slugify(name)), f"{name!r} -> {slugify(name)!r}"


def test_real_recording_name():
    assert slugify("Arranging the internals_20260728225028") == "arranging_the_internals_20260728225028"


def test_already_safe_names_are_untouched():
    assert slugify("pick_bottle_001") == "pick_bottle_001"
    assert slugify("task-2026-07-28") == "task-2026-07-28"


def test_runs_of_unsafe_characters_collapse():
    assert slugify("a   ///  b") == "a_b"


def test_leading_and_trailing_separators_are_trimmed():
    assert slugify("  spaced  ") == "spaced"
    assert slugify("///weird///") == "weird"


def test_name_that_sanitises_to_nothing_still_yields_a_usable_directory():
    """An all-punctuation name must not produce an empty path segment."""
    assert slugify("///") == "recording"
    assert slugify("") == "recording"


def test_distinct_recordings_do_not_collide_on_their_timestamp():
    """Recording names end in a timestamp, so even if the task text slugifies identically
    the directories stay distinct."""
    a = slugify("Pick bottle_20260728225028")
    b = slugify("Pick bottle_20260728225100")
    assert a != b
