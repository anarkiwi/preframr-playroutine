"""Real-tune end-to-end tests, parametrized over tests/fixtures/tunes.json.

The whole module is skipped when the catalog is missing, the ``sidtrace`` binary
is absent, or no tune can be fetched. Each tune is rendered for its full song
length and analysed; recovery must classify at least one SID register, and a
re-run must be byte-identical (determinism).
"""

import os
import shutil
import subprocess

import numpy as np
import pytest

from preframr_playroutine import Trace, analyze

from _hvsc import ensure_tune, fetchable, load_catalog

SIDTRACE = shutil.which("sidtrace") or "/usr/local/bin/sidtrace"
HAVE_SIDTRACE = os.path.exists(SIDTRACE)

CATALOG = load_catalog()

if not CATALOG:
    pytest.skip("tests/fixtures/tunes.json missing", allow_module_level=True)
if not HAVE_SIDTRACE:
    pytest.skip("sidtrace binary not available", allow_module_level=True)


def _ids(entry):
    return f"{entry.get('family', '?')}:{os.path.basename(entry['path'])}:{entry.get('subtune', 1)}"


def _run_sidtrace(sid_path, prefix, seconds, subtune):
    subprocess.run(
        [
            SIDTRACE,
            "--seconds",
            str(seconds),
            "--song",
            str(subtune),
            "--out",
            prefix,
            sid_path,
        ],
        check=True,
        capture_output=True,
    )
    return Trace.load(prefix)


@pytest.mark.parametrize("entry", CATALOG, ids=[_ids(e) for e in CATALOG])
def test_real_tune(entry, tmp_path_factory):
    if not fetchable(entry):
        pytest.skip(f"tune not fetchable: {entry['path']}")
    sid = ensure_tune(entry)
    seconds = entry["seconds"]
    subtune = entry.get("subtune", 1)

    work = tmp_path_factory.mktemp("hvsc")
    prefix_a = str(work / "a")
    trace = _run_sidtrace(sid, prefix_a, seconds, subtune)

    assert len(trace.events) > 0
    assert len(trace.ram_writes()) > 0
    assert len(trace.coverage_pcs()) > 0
    img = trace.ram_image()
    assert img is not None and len(img) == 65536

    result = analyze(trace)
    classified = sum(v for k, v in result["summary"].items())
    assert classified >= 1

    # Determinism: a second render is byte-identical across event + RAM streams.
    prefix_b = str(work / "b")
    trace2 = _run_sidtrace(sid, prefix_b, seconds, subtune)
    assert np.array_equal(trace.events.view(np.uint8), trace2.events.view(np.uint8))
    assert np.array_equal(trace.ram_writes().view(np.uint8), trace2.ram_writes().view(np.uint8))
