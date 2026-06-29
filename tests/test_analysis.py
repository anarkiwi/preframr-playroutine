"""Unit tests for the numpy analysis layer using synthetic event arrays."""

import numpy as np
import pytest

from preframr_playroutine import (
    CIA_IRQ,
    CPU_VECTOR,
    EVENT_DTYPE,
    SID_WRITE,
    VEC_IRQ,
    VIC_IRQ,
    Trace,
    decode_voices,
)
from preframr_playroutine.trace import SRC_CIA1

PAL_FRAME = 985248.444 / 50.0  # ~19705 cycles
PAL_META = {"cpu_hz": 985248.444, "effective_model": "PAL", "speed_string": "50 Hz VBI"}


def _rec(cycle, etype, chip=0, reg=0, value=0, addr=0, aux=0):
    return (cycle, etype, chip, reg, value, addr, aux)


def _build(records):
    arr = np.array(records, dtype=EVENT_DTYPE)
    arr.sort(order="cycle", kind="stable")
    return arr


def _vbi_trace(n_frames=64, ramp_reg=0):
    """VBI tune: one play per frame, raster IRQ, ramp written to ramp_reg."""
    recs = []
    for i in range(n_frames):
        tick = int(round(i * PAL_FRAME))
        recs.append(_rec(tick, VIC_IRQ, chip=3, addr=10, aux=10))
        recs.append(_rec(tick + 2, CPU_VECTOR, value=VEC_IRQ, addr=0x1003))
        # play body writes a ramp and volume a few cycles later
        recs.append(
            _rec(tick + 20, SID_WRITE, reg=ramp_reg, value=i & 0xFF, addr=0xD400 + ramp_reg)
        )
        recs.append(_rec(tick + 24, SID_WRITE, reg=24, value=0x0F, addr=0xD418))
    return Trace.from_events(_build(recs), PAL_META)


def _cia_trace(n_frames=64, calls_per_frame=2):
    """Multispeed CIA tune: N plays per frame, CIA1 timer IRQ."""
    period = PAL_FRAME / calls_per_frame
    latch = int(round(period))
    recs = []
    for i in range(n_frames * calls_per_frame):
        tick = int(round(i * period))
        recs.append(_rec(tick, CIA_IRQ, chip=SRC_CIA1, addr=latch, aux=0))
        recs.append(_rec(tick + 2, CPU_VECTOR, value=VEC_IRQ, addr=0x1003))
        recs.append(_rec(tick + 20, SID_WRITE, reg=0, value=i & 0xFF, addr=0xD400))
    return Trace.from_events(_build(recs), PAL_META)


def test_vbi_classification():
    trace = _vbi_trace()
    info = trace.classify()
    assert info["driver"] == "RASTER"
    assert info["speed"] == 1
    assert not info["multispeed"]
    assert info["interrupt_sources"]["vic_raster"] == 64
    assert info["raster_lines"] == [10]


def test_cia_classification():
    trace = _cia_trace(calls_per_frame=2)
    info = trace.classify()
    assert info["driver"] == "CIA"
    assert info["multispeed"]
    assert info["speed"] == 2
    assert info["interrupt_sources"]["cia1_irq"] == 128
    assert "cia_timer_latch" in info
    assert info["cia_timer_latch"] == pytest.approx(round(PAL_FRAME / 2), abs=2)


def test_interval_stats():
    trace = _vbi_trace(n_frames=50)
    stats = trace.interval_stats()
    assert stats["count"] == 50
    assert stats["period"] == pytest.approx(PAL_FRAME, abs=1)
    assert stats["calls_per_frame"] == pytest.approx(1.0, abs=0.01)


def test_register_frames_recovers_ramp():
    n = 40
    trace = _vbi_trace(n_frames=n, ramp_reg=0)
    ticks, frames = trace.register_frames(chip=0)
    assert len(ticks) == n
    assert frames.shape == (n, 32)
    # voice-1 freq lo follows the frame index; volume reg constant.
    assert np.array_equal(frames[:, 0], np.arange(n, dtype=np.uint8))
    assert np.all(frames[:, 24] == 0x0F)


def test_register_frames_carry_forward():
    # A register written once should carry forward to later frames.
    recs = []
    for i in range(10):
        tick = int(round(i * PAL_FRAME))
        recs.append(_rec(tick + 2, CPU_VECTOR, value=VEC_IRQ))
        if i == 3:
            recs.append(_rec(tick + 20, SID_WRITE, reg=5, value=0xAB, addr=0xD405))
    trace = Trace.from_events(_build(recs), PAL_META)
    _, frames = trace.register_frames()
    assert np.all(frames[:3, 5] == 0)
    assert np.all(frames[3:, 5] == 0xAB)


def test_decode_voices():
    n = 16
    frames = np.zeros((n, 32), dtype=np.uint8)
    frames[:, 0] = 0x34  # v1 freq lo
    frames[:, 1] = 0x12  # v1 freq hi -> 0x1234
    frames[:, 4] = 0x41  # v1 ctrl: gate + sawtooth(0x40)
    frames[:, 5] = 0x2A  # v1 AD: attack 2 decay 10
    frames[:, 6] = 0xF8  # v1 SR: sustain 15 release 8
    frames[:, 24] = 0x1F  # volume 15, lowpass mode bit
    voices = decode_voices(frames)
    v1 = voices["voices"][0]
    assert v1["freq"][0] == 0x1234
    assert v1["gate"][0] == 1
    assert v1["waveform"][0] == 0x4
    assert v1["attack"][0] == 2
    assert v1["decay"][0] == 10
    assert v1["sustain"][0] == 15
    assert v1["release"][0] == 8
    assert voices["volume"][0] == 15
    assert voices["filter_mode"][0] == 1


def test_tick_cycles_auto_prefers_busier_source():
    # Few NMIs, many IRQs -> auto picks IRQ cadence.
    recs = []
    for i in range(30):
        recs.append(_rec(i * 1000, CPU_VECTOR, value=VEC_IRQ))
    recs.append(_rec(5, CPU_VECTOR, value=0xFA))  # one NMI
    trace = Trace.from_events(_build(recs), PAL_META)
    assert len(trace.tick_cycles("auto")) == 30
    assert len(trace.tick_cycles("nmi")) == 1


def test_empty_trace():
    trace = Trace.from_events(np.empty(0, dtype=EVENT_DTYPE), PAL_META)
    assert trace.interval_stats()["period"] is None
    ticks, frames = trace.register_frames()
    assert len(ticks) == 0
    assert frames.shape == (0, 32)
    info = trace.classify()
    assert info["driver"] == "UNKNOWN"


def test_dtype_is_16_bytes():
    assert EVENT_DTYPE.itemsize == 16


def test_load_roundtrip(tmp_path):
    import json

    trace = _vbi_trace(n_frames=8)
    base = tmp_path / "t"
    trace.events.tofile(str(base) + ".bin")
    with open(str(base) + ".json", "w", encoding="utf-8") as handle:
        json.dump({**PAL_META, "num_records": len(trace.events)}, handle)

    for arg in (str(base), str(base) + ".bin", str(base) + ".json"):
        loaded = Trace.load(arg)
        assert len(loaded.events) == len(trace.events)
        assert loaded.meta["num_records"] == len(trace.events)


def test_load_missing_json(tmp_path):
    trace = _vbi_trace(n_frames=4)
    base = tmp_path / "nojson"
    trace.events.tofile(str(base) + ".bin")
    loaded = Trace.load(str(base))
    assert loaded.meta == {}
    assert len(loaded.events) == len(trace.events)


def test_tick_kinds_irq_and_both():
    recs = []
    for i in range(10):
        recs.append(_rec(i * 1000, CPU_VECTOR, value=VEC_IRQ))
        recs.append(_rec(i * 1000 + 1, CPU_VECTOR, value=0xFA))  # NMI same cadence
    trace = Trace.from_events(_build(recs), PAL_META)
    assert len(trace.tick_cycles("irq")) == 10
    assert len(trace.tick_cycles("both")) == 20


def test_single_tick_no_period():
    recs = [_rec(100, CPU_VECTOR, value=VEC_IRQ)]
    trace = Trace.from_events(_build(recs), PAL_META)
    assert trace.interval_stats()["period"] is None


def test_ntsc_refresh():
    trace = Trace.from_events(
        np.empty(0, dtype=EVENT_DTYPE),
        {"cpu_hz": 1022727.14, "effective_model": "NTSC"},
    )
    assert trace.refresh_hz == 60.0
    assert trace.frame_cycles == pytest.approx(1022727.14 / 60.0)
