"""Measurements -> declarative sentences: the narrative layer, kept separate.

Mottled's philosophy splits data (a capture), measurements (densities,
entropies, distances) and narratives (readings of those measurements).  This
module is the narrative layer made explicit: it turns a scene's measurements
into short, declarative, evidence-bearing sentences that the viewers show
next to the picture — so the design explains *why the terrain looks like
that* instead of leaving the reader to guess.

Every sentence is computed from the scene it describes, follows the design
language's copy rules (declarative, data-first, no hype), and is honest
about its epistemic status: the terrain is a density estimate over THIS
capture's states, not a property of the model.
"""

from __future__ import annotations

import numpy as np


def scene_notes(result: dict) -> list[str]:
    """Declarative reading of a pipeline result (run_pipeline / run_scene /
    run_compare / run_intervention).  Returns [] when the artifacts needed
    to support a sentence are missing — never a claim without evidence.
    """
    landscape = result.get("landscape")
    trajs = result.get("trajs") or ([result["traj"]] if "traj" in result else [])
    if landscape is None or not trajs:
        return []

    notes = [
        "Terrain height is the concentration of captured hidden states in "
        "this scene's shared projection — a density estimate over every "
        "(layer, token) state of these runs, not a property of the model "
        "itself. Peaks exist because many states land close together."
    ]

    # Per-state layer index in the density fit's order (run-major,
    # layer-major within a run) — must mirror how the union was assembled.
    density = np.asarray(landscape.point_density, dtype=np.float64)
    layers = np.concatenate(
        [np.repeat(np.arange(t.n_layers), t.n_tokens) for t in trajs])
    if len(layers) != len(density):
        return notes  # unfamiliar assembly: keep only the always-true note

    peak = float(density.max())
    if peak > 0:
        basin = density >= 0.5 * peak
        if basin.any() and not basin.all():
            share = float(basin.mean())
            l25, l75 = (int(v) for v in np.percentile(layers[basin], [25, 75]))
            depth = trajs[0].n_layers - 1
            notes.append(
                f"The densest region (at least half of peak density) holds "
                f"{share:.0%} of the {len(density)} captured states; half of "
                f"its mass sits between layers {l25} and {l75} of {depth}. "
                f"The basin is where late-layer states crowd together."
            )

    drops = [float(t.entropy[0].mean() - t.entropy[-1].mean())
             for t in trajs if t.entropy is not None]
    if drops and np.mean(drops) > 0:
        notes.append(
            f"Predictive entropy falls by {np.mean(drops):.1f} nats between "
            f"the first and last layer, averaged over these runs — "
            f"trajectories converge as predictions sharpen. Reading the "
            f"basin as that commitment is an interpretation; the density "
            f"and the entropy drop are the measurements."
        )
    return notes
