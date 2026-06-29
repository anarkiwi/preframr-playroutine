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

from preframr_playroutine import Trace, analyze, round_trip

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


# DMC per-voice register strides: $D400 + 7*voice.
_AD = (0xD405, 0xD40C, 0xD413)
_SR = (0xD406, 0xD40D, 0xD414)
_PW = (0xD402, 0xD403, 0xD409, 0xD40A, 0xD410, 0xD411)
_CTRL = (0xD404, 0xD40B, 0xD412)


def _types(result, addrs):
    return {a: result.get(a, {}).get("type") for a in addrs}


def _assert_register_classes(entry, trace, result):
    """Anchor recovery quality + round-trip fidelity on the DMC and GT2 tunes.

    Only the two named tunes are checked against their reverse-engineering docs
    (``re-trackers/DMC`` and ``re-trackers/GoatTracker2``); every other tune keeps
    the generic ``classified >= 1`` assertion. Round-trip fidelity (regenerated
    descriptor vs oracle) is the real correctness metric: the recovered registers
    must reconstruct essentially exactly and the whole tune near-perfectly. The
    thresholds (per-register >= 0.99, overall >= 0.95) leave headroom only for
    registers not yet decomposed; on these tunes the recovered ones hit 1.0.
    """
    family = entry.get("family")
    base = os.path.basename(entry["path"])
    if not (
        (family == "DMC" and base == "Doctagop.sid")
        or (family == "GoatTracker2" and base == "Raindrops.sid")
    ):
        return
    fid = round_trip(trace)

    if family == "DMC":
        # dmc-generators.md: AD/SR per-note SEQ; PW 16-bit up/down BACC; CTRL a
        # waveform table walk (AND-ed with the gate mask). A sustained voice can
        # be a 1-entry waveform loop, presenting as a CONST CTRL -- allow it, but
        # require at least one active voice to recover as a real TABLE_WALK. The
        # whole-song fixture renders without --reads, so this exercises the
        # image-based scan (cursor state cell + ram_image, masked) path.
        for addr, t in _types(result, _AD + _SR).items():
            assert t in ("SEQ", "CONST"), (hex(addr), t)
        for addr, t in _types(result, _PW).items():
            assert t == "BACC", (hex(addr), t)
        ctrl_types = _types(result, _CTRL)
        for addr, t in ctrl_types.items():
            assert t in ("TABLE_WALK", "CONST"), (hex(addr), t)
        assert "TABLE_WALK" in ctrl_types.values(), ctrl_types
        for addr in _AD + _SR + _PW + _CTRL:
            assert fid[addr] >= 0.99, (hex(addr), result[addr]["type"], fid[addr])

    if family == "GoatTracker2":
        # goattracker2-generators.md: AD/SR per-note SEQ (+ hard restart); the
        # verified voice-1 PW is a 16-bit up BACC.
        for addr, t in _types(result, _AD + _SR).items():
            assert t in ("SEQ", "CONST"), (hex(addr), t)
        for addr, t in _types(result, (0xD409, 0xD40A)).items():
            assert t == "BACC", (hex(addr), t)
        for addr in _AD + _SR + (0xD409, 0xD40A):
            assert fid[addr] >= 0.99, (hex(addr), result[addr]["type"], fid[addr])

    assert fid["overall"] >= 0.95, fid["overall"]


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
    _assert_register_classes(entry, trace, result)

    # Determinism: a second render is byte-identical across event + RAM streams.
    prefix_b = str(work / "b")
    trace2 = _run_sidtrace(sid, prefix_b, seconds, subtune)
    assert np.array_equal(trace.events.view(np.uint8), trace2.events.view(np.uint8))
    assert np.array_equal(trace.ram_writes().view(np.uint8), trace2.ram_writes().view(np.uint8))
