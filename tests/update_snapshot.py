"""Regenerate the per-register fidelity snapshot ratchet.

Renders every catalog fixture whole-song (no reads, CI conditions), runs
``round_trip``, and writes ``tests/fixtures/fidelity_snapshot.json`` keyed by
``<family>:<basename>:<subtune>`` with the overall and per-register fidelities
(floats rounded to 6 places, keys sorted). ``tests/test_hvsc.py`` then asserts
no recorded fidelity regresses. NOT named ``test_*`` so pytest does not collect
it. Mirrors ``tests/straggler_report.py``; run inside the Docker image with the
HVSC mirror available, e.g.:

    docker run --rm -e HVSC_BASE_URL=https://hvsc.c64.org/download/C64Music \\
      --entrypoint python3 preframr-playroutine:snap tests/update_snapshot.py
"""

import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from preframr_playroutine import Trace, analyze, round_trip  # noqa: E402
from _hvsc import ensure_tune, fetchable, load_catalog  # noqa: E402
import subprocess  # noqa: E402

SIDTRACE = "/usr/local/bin/sidtrace"
_FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
OUT = os.path.join(_FIX, "fidelity_snapshot.json")
CLASS_OUT = os.path.join(_FIX, "classification_snapshot.json")


def _key(entry):
    return f"{entry.get('family', '?')}:{os.path.basename(entry['path'])}:{entry.get('subtune', 1)}"


def _render(entry, work):
    sid = ensure_tune(entry)
    prefix = os.path.join(work, "a")
    subprocess.run(
        [
            SIDTRACE,
            "--seconds",
            str(entry["seconds"]),
            "--song",
            str(entry.get("subtune", 1)),
            "--out",
            prefix,
            sid,
        ],
        check=True,
        capture_output=True,
    )
    return Trace.load(prefix)


def _snapshot_one(entry):
    """Render one fixture -> (key, fidelity_entry, classification_entry) or None.

    ``fidelity_entry`` feeds the fidelity ratchet; ``classification_entry`` (the
    per-register descriptor ``type``) feeds the Phase-2 arbiter calibration test.
    """
    if not fetchable(entry):
        return None
    with tempfile.TemporaryDirectory() as work:
        trace = _render(entry, work)
        rt = round_trip(trace)
        res = analyze(trace)
    regs = {hex(a): round(f, 6) for a, f in rt.items() if isinstance(a, int)}
    types = {hex(a): d.get("type") for a, d in res.items() if isinstance(a, int)}
    fid_entry = {"overall": round(rt["overall"], 6), "regs": dict(sorted(regs.items()))}
    class_entry = {"regs": dict(sorted(types.items()))}
    return _key(entry), fid_entry, class_entry


def main():
    catalog = load_catalog()
    # Tunes are independent; render them across a process pool (each sidtrace +
    # round_trip is CPU-bound) rather than serially.
    workers = min(len(catalog), os.cpu_count() or 1) or 1
    snapshot = {}
    classification = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_snapshot_one, catalog):
            if result is not None:
                snapshot[result[0]] = result[1]
                classification[result[0]] = result[2]
    for path, data in ((OUT, snapshot), (CLASS_OUT, classification)):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(dict(sorted(data.items())), fh, indent=2, sort_keys=True)
            fh.write("\n")
    print(f"wrote {OUT} + {CLASS_OUT} ({len(snapshot)} tunes)")


if __name__ == "__main__":
    main()
