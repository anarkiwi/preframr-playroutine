"""Diagnostic: per-fixture round-trip straggler report.

Renders every catalog fixture whole-song WITH ``--reads`` (CI conditions) and
prints, for each: overall round-trip and the sorted ``unmodeled`` register list
(addr/type/fidelity/example_frames) from ``round_trip``. Turns the imperfect set
into a ranked table of exactly which register blocks each fixture.

Run inside the Docker image with the repo + HVSC mirror mounted, e.g.:

    docker run --rm \
      -v "$PWD:/work" -v /scratch/hvsc/C64Music:/sids:ro \
      -e HVSC_ROOT=/sids --entrypoint python3 preframr-playroutine:v2g \
      tests/straggler_report.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from preframr_playroutine import Trace, analyze, round_trip  # noqa: E402
from _hvsc import ensure_tune, fetchable, load_catalog  # noqa: E402
import subprocess  # noqa: E402

SIDTRACE = "/usr/local/bin/sidtrace"


def render(entry, work):
    sid = ensure_tune(entry)
    prefix = os.path.join(work, "a")
    subprocess.run(
        [
            SIDTRACE,
            "--seconds",
            str(entry["seconds"]),
            "--song",
            str(entry.get("subtune", 1)),
            "--reads",
            "--out",
            prefix,
            sid,
        ],
        check=True,
        capture_output=True,
    )
    return Trace.load(prefix)


def main():
    catalog = load_catalog()
    rows = []
    for entry in catalog:
        fam = entry.get("family")
        name = os.path.basename(entry["path"])
        if not fetchable(entry):
            rows.append((1.0, fam, f"{name} SKIP (not fetchable)", []))
            continue
        with tempfile.TemporaryDirectory() as work:
            trace = render(entry, work)
            analyze(trace)
            rt = round_trip(trace)
        rows.append((rt["overall"], fam, name, rt["unmodeled"]))

    rows.sort(key=lambda r: r[0])
    print("\n==== straggler report (sorted worst overall first) ====\n")
    for overall, fam, name, unmodeled in rows:
        print(f"{overall:.4f}  {fam}:{name}")
        for u in unmodeled:
            print(
                f"        {hex(u['addr'])}  {u['type']:<11} "
                f"fid={u['fidelity']:.4f}  frames={u['example_frames']}"
            )
        print()


if __name__ == "__main__":
    main()
