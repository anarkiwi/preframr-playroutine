"""Real-tune end-to-end tests, parametrized over tests/fixtures/tunes.json.

The whole module is skipped when the catalog is missing, the ``sidtrace`` binary
is absent, or no tune can be fetched. Each tune is rendered for its full song
length WITHOUT a RAM read log (exactly as CI renders), then analysed.

``test_real_tune_perfect`` is the ratchet: it asserts *perfect* recovery for
every fixture -- ``round_trip(trace)['overall'] == 1.0`` and zero ``XSTATE``
registers -- and marks the tunes that do not yet meet that bar with
``xfail(strict=True)``. When a recovery improvement makes an xfail tune perfect
it flips to XPASS and CI fails until the marker is removed (the intended
ratchet). ``test_real_tune_anchors`` independently pins the DMC and GoatTracker2
register classifications against their reverse-engineering docs, guarding those
specifics regardless of the perfect gate.
"""

import json
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

_SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "fidelity_snapshot.json")
try:
    with open(_SNAPSHOT_PATH, encoding="utf-8") as _fh:
        _SNAPSHOT = json.load(_fh)
except (OSError, ValueError):
    _SNAPSHOT = {}

if not CATALOG:
    pytest.skip("tests/fixtures/tunes.json missing", allow_module_level=True)
if not HAVE_SIDTRACE:
    pytest.skip("sidtrace binary not available", allow_module_level=True)


def _ids(entry):
    return f"{entry.get('family', '?')}:{os.path.basename(entry['path'])}:{entry.get('subtune', 1)}"


def _key(entry):
    return (entry.get("family"), os.path.basename(entry["path"]))


# Fixtures that already round-trip PERFECTLY (overall == 1.0, no XSTATE register)
# on a whole-song, no-reads render -- determined empirically, not guessed. Every
# other fixture is xfail(strict=True): improve its recovery to perfect and remove
# it here (CI XPASS-fails until you do).
_PERFECT = {
    ("DMC", "Doctagop.sid"),
    ("DMC", "In_My_Head.sid"),
    ("Soundmonitor", "Only_3.sid"),
    ("Soundmonitor", "Denarius.sid"),
    ("Soundmonitor", "Tom_Tom.sid"),
    ("JCH_NewPlayer", "24th_Amaranth_Grand_Prix_3.sid"),
    ("JCH_NewPlayer", "Dreams.sid"),
    ("FutureComposer", "Hawkeye.sid"),
    ("GoatTracker2", "Grid_Runner.sid"),
    ("GoatTracker2", "Day_6_in_Kleve_Hades.sid"),
    ("GoatTracker2", "Raindrops.sid"),
    ("GoatTracker2", "Tunnelbound.sid"),
    ("GoatTracker2", "Cauldron_II_Remix.sid"),
    ("MusicAssembler", "Let_It_Bee.sid"),
    ("MusicAssembler", "Torpedo.sid"),
    ("MusicAssembler", "Pozitronic.sid"),
    ("defMON", "Wasps.sid"),
    ("defMON", "Vacuole.sid"),
    ("defMON", "Stargazer.sid"),
    ("defMON", "Automatas.sid"),
}

# DMC / GoatTracker2 per-voice register strides: $D400 + 7*voice.
_AD = (0xD405, 0xD40C, 0xD413)
_SR = (0xD406, 0xD40D, 0xD414)
_PW = (0xD402, 0xD403, 0xD409, 0xD40A, 0xD410, 0xD411)
_CTRL = (0xD404, 0xD40B, 0xD412)

# The two tunes pinned against their reverse-engineering docs.
_ANCHORS = [
    e for e in CATALOG if _key(e) in {("DMC", "Doctagop.sid"), ("GoatTracker2", "Raindrops.sid")}
]


def _types(result, addrs):
    return {a: result.get(a, {}).get("type") for a in addrs}


def _assert_register_classes(entry, trace, result):
    """Anchor recovery quality + round-trip fidelity on the DMC and GT2 tunes.

    Both named tunes are checked against their reverse-engineering docs
    (``re-trackers/DMC`` and ``re-trackers/GoatTracker2``). Round-trip fidelity
    (regenerated descriptor vs oracle) is the real correctness metric: the
    recovered registers must reconstruct essentially exactly (per-register
    >= 0.99) and the whole tune near-perfectly (overall >= 0.95).
    """
    family = entry.get("family")
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


def _render(entry, work):
    """Render a fixture whole-song, no reads (CI conditions); return the trace."""
    sid = ensure_tune(entry)
    prefix = str(work / "a")
    trace = _run_sidtrace(sid, prefix, entry["seconds"], entry.get("subtune", 1))
    assert len(trace.events) > 0
    assert len(trace.ram_writes()) > 0
    assert len(trace.coverage_pcs()) > 0
    img = trace.ram_image()
    assert img is not None and len(img) == 65536
    return sid, prefix, trace


def _perfect_param(entry):
    marks = []
    if _key(entry) not in _PERFECT:
        marks = [
            pytest.mark.xfail(
                strict=True,
                reason="round_trip not yet perfect (overall<1.0 or XSTATE present)",
            )
        ]
    return pytest.param(entry, marks=marks, id=_ids(entry))


@pytest.mark.parametrize("entry", [_perfect_param(e) for e in CATALOG])
def test_real_tune_perfect(entry, tmp_path_factory):
    """Every fixture must recover PERFECTLY: overall == 1.0 and no XSTATE."""
    if not fetchable(entry):
        pytest.skip(f"tune not fetchable: {entry['path']}")
    work = tmp_path_factory.mktemp("hvsc")
    _sid, _prefix, trace = _render(entry, work)

    result = analyze(trace)
    classified = sum(v for k, v in result["summary"].items())
    assert classified >= 1
    xstate = sorted(
        a for a, d in result.items() if isinstance(a, int) and d.get("type") == "XSTATE"
    )
    rt = round_trip(trace)

    # Ratchet: no recorded per-register or overall fidelity may regress. Absent
    # key (e.g. the committed empty snapshot) -> no assertion, so CI stays green
    # until the snapshot is populated in the Docker/HVSC environment.
    snap = _SNAPSHOT.get(_ids(entry))
    if snap is not None:
        for reg, recorded in snap.get("regs", {}).items():
            addr = int(reg, 16)
            assert rt.get(addr, 0.0) >= recorded - 1e-9, (reg, rt.get(addr), recorded)
        assert rt["overall"] >= snap["overall"] - 1e-9, (rt["overall"], snap["overall"])

    assert not xstate, [hex(a) for a in xstate]
    assert rt["overall"] == 1.0, (rt["overall"], rt["unmodeled"][:4])


@pytest.mark.parametrize("entry", _ANCHORS, ids=[_ids(e) for e in _ANCHORS])
def test_real_tune_anchors(entry, tmp_path_factory):
    """Pin DMC/GT2 register classifications + per-register fidelity (non-xfail)."""
    if not fetchable(entry):
        pytest.skip(f"tune not fetchable: {entry['path']}")
    work = tmp_path_factory.mktemp("hvsc")
    sid, prefix_a, trace = _render(entry, work)

    result = analyze(trace)
    _assert_register_classes(entry, trace, result)

    # Determinism: a second render is byte-identical across event + RAM streams.
    prefix_b = str(work / "b")
    trace2 = _run_sidtrace(sid, prefix_b, entry["seconds"], entry.get("subtune", 1))
    assert np.array_equal(trace.events.view(np.uint8), trace2.events.view(np.uint8))
    assert np.array_equal(trace.ram_writes().view(np.uint8), trace2.ram_writes().view(np.uint8))
    assert np.array_equal(trace.io_reads().view(np.uint8), trace2.io_reads().view(np.uint8))
    assert np.array_equal(trace.io_writes().view(np.uint8), trace2.io_writes().view(np.uint8))
