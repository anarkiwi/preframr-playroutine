"""End-to-end oracle tests: run sidtrace on hand-built PSID tunes.

Skipped when the ``sidtrace`` binary is unavailable (e.g. running the python
suite outside the Docker image).
"""

import os
import shutil
import subprocess

import numpy as np
import pytest

from preframr_playroutine import Trace

from _minisid import build_psid

SIDTRACE = shutil.which("sidtrace") or "/usr/local/bin/sidtrace"
HAVE_SIDTRACE = os.path.exists(SIDTRACE)

pytestmark = pytest.mark.skipif(not HAVE_SIDTRACE, reason="sidtrace binary not available")


def _run(tmp_path, speed, seconds=2.0):
    sid = tmp_path / "mini.sid"
    sid.write_bytes(build_psid(speed=speed))
    prefix = str(tmp_path / "mini")
    subprocess.run(
        [SIDTRACE, "--seconds", str(seconds), "--out", prefix, str(sid)],
        check=True,
        capture_output=True,
    )
    return Trace.load(prefix)


def test_vbi_oracle(tmp_path):
    trace = _run(tmp_path, speed=0)
    assert len(trace.events) > 0
    writes = trace.sid_writes()
    assert len(writes) > 0
    # The player writes voice-1 freq lo ($D400) and volume ($D418) each frame.
    assert np.any(writes["addr"] == 0xD400)
    assert np.any(writes["addr"] == 0xD418)
    info = trace.classify()
    assert info["driver"] == "RASTER"
    assert info["interrupt_sources"]["vic_raster"] > 0


def test_vbi_frame_counter_recovered(tmp_path):
    trace = _run(tmp_path, speed=0, seconds=2.0)
    _, frames = trace.register_frames(chip=0)
    assert len(frames) > 50
    # Volume held at 0x0f every frame.
    assert np.all(frames[1:, 24] == 0x0F)
    # Voice-1 freq lo is a per-frame increment (mod 256): consecutive deltas
    # are overwhelmingly +1.
    col = frames[:, 0].astype(np.int16)
    deltas = np.diff(col) % 256
    assert np.mean(deltas == 1) > 0.9


def test_cia_oracle(tmp_path):
    trace = _run(tmp_path, speed=1)
    info = trace.classify()
    assert info["driver"] == "CIA"
    assert info["interrupt_sources"]["cia1_irq"] > 0
    assert "cia_timer_latch" in info


def test_metadata_sidecar(tmp_path):
    trace = _run(tmp_path, speed=0)
    assert "PSID" in trace.meta["format"]
    assert trace.meta["effective_model"] == "PAL"
    assert trace.meta["deterministic"] is True
    assert trace.meta["num_records"] == len(trace.events)
    assert trace.meta["record_size"] == 16


def test_determinism(tmp_path):
    # Same input must produce a byte-identical oracle (deterministic emulation).
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    t1 = _run(a, speed=0)
    t2 = _run(b, speed=0)
    assert np.array_equal(
        t1.events.view(np.uint8).reshape(-1, 16),
        t2.events.view(np.uint8).reshape(-1, 16),
    )
