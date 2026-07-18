"""Narrative layer: computed, evidence-bearing scene notes."""

import io
import re

import numpy as np
import pytest

import statefile
from config import MarbleConfig
from narrate import scene_notes
from ui import run_pipeline, run_scene

PROMPTS = ["the capital of france is", "the capital of germany is"]


@pytest.fixture(scope="module")
def scene_result():
    cfg = MarbleConfig(model="synthetic", use_cache=False)
    return run_scene(cfg, PROMPTS)


def test_notes_are_declarative_and_grounded(scene_result):
    notes = scene_notes(scene_result)
    assert 2 <= len(notes) <= 4
    text = " ".join(notes)
    assert "density estimate" in text            # what the terrain is
    assert "not a property of the model" in text  # epistemic honesty
    assert re.search(r"\d+%", text)              # numbers, not vibes
    assert "layers" in text                       # what the basin is made of
    assert "!" not in text and "🚀" not in text   # copy rules hold


def test_basin_share_matches_the_measurement(scene_result):
    notes = scene_notes(scene_result)
    share = int(re.search(r"holds (\d+)% of the (\d+) captured states",
                          " ".join(notes)).group(1))
    landscape = scene_result["landscape"]
    dens = np.asarray(landscape.point_density)
    expected = (dens >= 0.5 * dens.max()).mean() * 100
    assert abs(share - expected) <= 1  # the sentence is the measurement


def test_entropy_note_present_for_collapsing_runs(scene_result):
    text = " ".join(scene_notes(scene_result))
    assert "entropy falls" in text
    assert "interpretation" in text  # narrative flagged as narrative


def test_notes_survive_single_run_and_missing_artifacts():
    cfg = MarbleConfig(model="synthetic", use_cache=False)
    single = run_pipeline(cfg, PROMPTS[0])
    assert len(scene_notes(single)) >= 2
    assert scene_notes({"traj": single["traj"]}) == []   # no landscape: no claims
    assert scene_notes({"landscape": single["landscape"]}) == []


def test_notes_travel_in_the_scene_file(scene_result):
    buf = io.BytesIO()
    statefile.save_scene(scene_result, buf)
    buf.seek(0)
    scene = statefile.load_scene(buf)
    assert scene["notes"] == scene_notes(scene_result)
