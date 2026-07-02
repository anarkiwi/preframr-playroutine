"""Shared HVSC render plumbing for the real-tune test modules.

Both ``test_hvsc`` (the anchor/ratchet suite) and ``test_retrackers`` (the
license-walled cross-check) render a catalog fixture whole-song WITH the RAM read
log (``--reads``), then analyse it. Rendering with ``--reads`` feeds the read-log
dataflow narrowing (Phase 4) and the Tier-3 witness's per-frame read-PC input cone
(GENERIC_RECOVERY.md 3.5): the totality backstop needs the code-derived input cone
to retire XSTATE. The invocation lives here once and both modules import it -- there
is no second copy of the sidtrace command or the trace-sanity asserts.
"""

import os
import shutil
import subprocess

from preframr_playroutine import Trace

from _hvsc import ensure_tune

SIDTRACE = shutil.which("sidtrace") or "/usr/local/bin/sidtrace"
HAVE_SIDTRACE = os.path.exists(SIDTRACE)


def run_sidtrace(sid_path, prefix, seconds, subtune, reads=True):
    """Run sidtrace for ``seconds`` of ``subtune`` and load the resulting trace.

    ``reads`` (default True) emits the RAM read log (``--reads``); the I/O-read log
    is always emitted regardless.
    """
    cmd = [SIDTRACE, "--seconds", str(seconds), "--song", str(subtune)]
    if reads:
        cmd.append("--reads")
    cmd += ["--out", prefix, sid_path]
    subprocess.run(cmd, check=True, capture_output=True)
    return Trace.load(prefix)


def render_tune(entry, work):
    """Render a fixture whole-song WITH ``--reads``; return (sid, prefix, trace)."""
    sid = ensure_tune(entry)
    prefix = str(work / "a")
    trace = run_sidtrace(sid, prefix, entry["seconds"], entry.get("subtune", 1))
    assert len(trace.events) > 0
    assert len(trace.ram_writes()) > 0
    assert len(trace.ram_reads()) > 0
    assert len(trace.coverage_pcs()) > 0
    img = trace.ram_image()
    assert img is not None and len(img) == 65536
    return sid, prefix, trace
