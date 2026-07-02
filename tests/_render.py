"""Shared HVSC render plumbing for the real-tune test modules.

Both ``test_hvsc`` (the anchor/ratchet suite) and ``test_retrackers`` (the
license-walled cross-check) render a catalog fixture whole-song, WITHOUT a RAM
read log (exactly as CI renders), then analyse it. The rendering itself is
identical, so it lives here once and both modules import it -- there is no
second copy of the sidtrace invocation or the trace-sanity asserts.
"""

import os
import shutil
import subprocess

from preframr_playroutine import Trace

from _hvsc import ensure_tune

SIDTRACE = shutil.which("sidtrace") or "/usr/local/bin/sidtrace"
HAVE_SIDTRACE = os.path.exists(SIDTRACE)


def run_sidtrace(sid_path, prefix, seconds, subtune):
    """Run sidtrace for ``seconds`` of ``subtune`` and load the resulting trace."""
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


def render_tune(entry, work):
    """Render a fixture whole-song, no reads (CI conditions); return (sid, prefix, trace)."""
    sid = ensure_tune(entry)
    prefix = str(work / "a")
    trace = run_sidtrace(sid, prefix, entry["seconds"], entry.get("subtune", 1))
    assert len(trace.events) > 0
    assert len(trace.ram_writes()) > 0
    assert len(trace.coverage_pcs()) > 0
    img = trace.ram_image()
    assert img is not None and len(img) == 65536
    return sid, prefix, trace
