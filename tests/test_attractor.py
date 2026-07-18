"""Attractor analysis: why the basin forms and what it is made of."""

import numpy as np

import attractor as A
import density as D
import projection as P
from models import synthetic

PROMPT = "the capital of france is"


def _pipeline(prompt=PROMPT, **kw):
    traj = synthetic.capture(prompt, **kw)
    coords, _ = P.project(traj.hidden)
    landscape = D.compute_density(coords)
    return traj, coords, landscape


def test_analyze_report_is_sane():
    traj, coords, landscape = _pipeline(capture_components=True)
    r = A.analyze(traj, coords, landscape)
    L, T = traj.n_layers, traj.n_tokens
    assert r.token == T - 1
    assert r.n_states == L * T
    assert r.n_members >= 1 and 0 < r.member_share <= 1
    assert 0 <= r.layer_range[0] <= r.layer_range[1] <= L - 1
    assert 0.0 <= r.late_share <= 1.0
    assert r.step.shape == (L - 1,) and (r.step >= 0).all()
    assert 1 <= r.peak_step_layer <= L - 1
    if r.settle_layer is not None:
        assert 0 <= r.settle_layer <= L - 2
    assert r.entropy is not None and r.entropy.shape == (L,)
    assert r.top_token is not None
    assert 0 <= r.readout_stable_from <= L - 1
    assert r.late_attn_share is not None and 0.0 <= r.late_attn_share <= 1.0
    assert landscape.grid_x[0] <= r.center[0] <= landscape.grid_x[-1]
    assert landscape.grid_y[0] <= r.center[1] <= landscape.grid_y[-1]


def test_members_meet_the_density_cut():
    traj, coords, landscape = _pipeline()
    r = A.analyze(traj, coords, landscape, member_threshold=0.6)
    dens = A.density_at(landscape, coords.reshape(-1, 2)).reshape(
        traj.n_layers, traj.n_tokens)
    cut = 0.6 * dens.max()
    member_set = {tuple(m) for m in r.members}
    for l in range(traj.n_layers):
        for t in range(traj.n_tokens):
            assert ((l, t) in member_set) == (dens[l, t] >= cut)


def test_settle_layer_definition():
    traj, coords, landscape = _pipeline()
    r = A.analyze(traj, coords, landscape, settle_frac=0.5)
    small = 0.5 * r.step.max()
    if r.settle_layer is None:
        assert r.step[-1] > small  # the tail never stays below the cut
    else:
        assert r.step[r.settle_layer:].max() <= small
        if r.settle_layer > 0:
            assert r.step[r.settle_layer - 1:].max() > small


def test_readout_stability_is_the_minimal_suffix():
    traj, coords, landscape = _pipeline()
    r = A.analyze(traj, coords, landscape)
    top1 = [traj.state(l, r.token).topk[0][0] for l in range(traj.n_layers)]
    assert r.top_token == top1[-1]
    assert all(t == r.top_token for t in top1[r.readout_stable_from:])
    if r.readout_stable_from > 0:
        assert top1[r.readout_stable_from - 1] != r.top_token


def test_explain_contains_the_measurements():
    traj, coords, landscape = _pipeline(capture_components=True)
    r = A.analyze(traj, coords, landscape)
    text = A.explain(r, traj)
    assert f"{r.n_members} of {r.n_states}" in text
    assert f"layers {r.layer_range[0]}–{r.layer_range[1]}" in text
    assert "density" in text
    assert r.top_token in text
    assert "MLP" in text  # components were captured
    # deterministic prose from deterministic measurements
    assert text == A.explain(A.analyze(traj, coords, landscape), traj)


def test_density_at_matches_grid_nodes():
    _, coords, landscape = _pipeline()
    gx, gy = landscape.grid_x, landscape.grid_y
    nodes = np.array([[gx[3], gy[5]], [gx[10], gy[0]]])
    got = A.density_at(landscape, nodes)
    assert np.allclose(got, [landscape.density[5, 3], landscape.density[0, 10]],
                       atol=1e-6)
    # outside the grid the field is empty, not extrapolated
    far = np.array([[gx[-1] + 10 * (gx[-1] - gx[0]), gy[0]]])
    assert A.density_at(landscape, far)[0] == 0.0


def test_render_pins_the_basin_annotation():
    from config import MarbleConfig
    from ui import render, run_pipeline

    cfg = MarbleConfig(model="synthetic", use_cache=False)
    result = run_pipeline(cfg, PROMPT)
    r = A.analyze(result["traj"], result["coords"], result["landscape"])
    fig = render(result["traj"], result["mesh"], result["trajectories"],
                 result["fine_paths"], basin=r)
    assert len(fig.layout.scene.annotations) == 1
    note = fig.layout.scene.annotations[0].text
    assert "attractor basin" in note and f"{r.n_members}/{r.n_states}" in note
    assert any(tr.name == "attractor" for tr in fig.data)
    # without a report the scene stays unannotated
    plain = render(result["traj"], result["mesh"], result["trajectories"],
                   result["fine_paths"])
    assert len(plain.layout.scene.annotations) == 0
