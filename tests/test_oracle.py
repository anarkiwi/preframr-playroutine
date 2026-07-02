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

from _minisid import build_ioprobe_psid, build_psid

SIDTRACE = shutil.which("sidtrace") or "/usr/local/bin/sidtrace"
HAVE_SIDTRACE = os.path.exists(SIDTRACE)

pytestmark = pytest.mark.skipif(not HAVE_SIDTRACE, reason="sidtrace binary not available")


def _trace(tmp_path, sid_bytes, name="mini", seconds=2.0, extra=()):
    sid = tmp_path / f"{name}.sid"
    sid.write_bytes(sid_bytes)
    prefix = str(tmp_path / name)
    subprocess.run(
        [SIDTRACE, "--seconds", str(seconds), "--out", prefix, *extra, str(sid)],
        check=True,
        capture_output=True,
    )
    return prefix


def _run(tmp_path, speed, seconds=2.0):
    return Trace.load(_trace(tmp_path, build_psid(speed=speed), seconds=seconds))


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
    # Determinism must cover the v2 RAM-write stream too.
    assert np.array_equal(
        t1.ram_writes().view(np.uint8),
        t2.ram_writes().view(np.uint8),
    )


def test_v2_artifacts_present(tmp_path):
    trace = _run(tmp_path, speed=0)
    # RAM write log: the player INCs/loads $FB each frame, so it must be non-empty.
    assert len(trace.ram_writes()) > 0
    # Executed-PC coverage of the play window.
    pcs = trace.coverage_pcs()
    assert len(pcs) > 0
    assert pcs.dtype == np.uint16
    assert np.all(np.diff(pcs.astype(np.int64)) > 0)  # sorted, unique
    # RAM image dump.
    img = trace.ram_image()
    assert img is not None
    assert len(img) == 65536
    # Store-site PC of each SID write.
    pc_col = trace.sid_write_pc()
    assert len(pc_col) == len(trace.sid_writes())
    assert np.any(pc_col != 0)


def test_io_probe(tmp_path):
    trace = Trace.load(_trace(tmp_path, build_ioprobe_psid(), name="probe"))
    iord = trace.io_reads()
    assert len(iord) > 0
    read_addrs = set(iord["addr"].tolist())
    assert {0xD41B, 0xDC04, 0xD020} <= read_addrs
    # The noise oscillator moves: osc3 readbacks are not constant.
    osc3 = iord[iord["addr"] == 0xD41B]
    assert len(np.unique(osc3["value"])) >= 2
    write_addrs = set(trace.io_writes()["addr"].tolist())
    assert {0xD020, 0xD402} <= write_addrs
    # osc3 -> PW-lo copy: each $D402 SID write equals the value of the nearest
    # preceding $D41B read (the chip-node ground truth).
    pw = trace.sid_writes()
    pw = pw[pw["addr"] == 0xD402]
    assert len(pw) > 0
    idx = np.searchsorted(osc3["cycle"], pw["cycle"], side="left") - 1
    assert np.all(idx >= 0)
    assert np.array_equal(pw["value"], osc3["value"][idx])


def test_io_logs_deterministic(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    pa = _trace(a, build_ioprobe_psid(), name="probe")
    pb = _trace(b, build_ioprobe_psid(), name="probe")
    for suffix in (".bin", ".ramwr.bin", ".cov.bin", ".ram", ".iord.bin", ".iowr.bin"):
        with open(pa + suffix, "rb") as fa, open(pb + suffix, "rb") as fb:
            assert fa.read() == fb.read(), suffix


def test_ram_log_purity(tmp_path):
    # Byte-identity contract: default flags keep I/O and stack out of the RAM log.
    trace = _run(tmp_path, speed=0)
    addrs = trace.ram_writes()["addr"]
    assert len(addrs) > 0
    assert not np.any((addrs & 0xF000) == 0xD000)
    assert not np.any((addrs & 0xFF00) == 0x0100)


def test_stack_flag(tmp_path):
    default_prefix = _trace(tmp_path, build_psid(speed=0), name="default")
    plain_prefix = _trace(tmp_path, build_psid(speed=0), name="plain")
    stack_prefix = _trace(tmp_path, build_psid(speed=0), name="stack", extra=("--stack",))
    stack_addrs = Trace.load(stack_prefix).ram_writes()["addr"]
    assert np.any((stack_addrs & 0xFF00) == 0x0100)
    plain_addrs = Trace.load(plain_prefix).ram_writes()["addr"]
    assert not np.any((plain_addrs & 0xFF00) == 0x0100)
    with (
        open(plain_prefix + ".ramwr.bin", "rb") as fa,
        open(default_prefix + ".ramwr.bin", "rb") as fb,
    ):
        assert fa.read() == fb.read()


def test_v2_analyze_classifies(tmp_path):
    from preframr_playroutine import analyze

    trace = _run(tmp_path, speed=0, seconds=2.0)
    result = analyze(trace)
    # At least one SID register classified, with a generator-type summary.
    assert result["summary"]
    assert sum(result["summary"].values()) >= 1


def test_reads_flag_narrowing(tmp_path):
    # The --reads fixture: the player LDA $FB / STA $D400, so the read log names
    # $FB as the one cell the $D400 store consumed. Narrowing must activate and
    # must not drop fidelity versus the non-reads (fallback) recovery.
    from preframr_playroutine import analyze, round_trip
    from preframr_playroutine.recover import _build_context

    reads = Trace.load(_trace(tmp_path, build_psid(speed=0), name="reads", extra=("--reads",)))
    plain = Trace.load(_trace(tmp_path, build_psid(speed=0), name="plain"))

    assert len(reads.ram_reads()) > 0  # --reads populated the read log
    assert len(plain.ram_reads()) == 0  # default render omits it (fallback path)

    ctx = _build_context(reads)
    assert ctx.reads_near is not None  # read log -> narrowing active
    assert 0x00FB in ctx.candidates(0xD400)  # the cell the store actually read
    assert ctx.candidate_cols(0xD400)  # non-empty narrowed set

    ctx_plain = _build_context(plain)
    assert ctx_plain.reads_near is None  # no read log -> fallback
    assert ctx_plain.candidates(0xD400) is None

    # Same recovery quality with narrowing as without: $D400 stays perfect.
    assert analyze(reads)["summary"]
    assert round_trip(reads)[0xD400] == round_trip(plain)[0xD400] == 1.0
