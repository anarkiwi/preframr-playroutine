"""Synthetic unit tests for the generator-recovery layer (pure numpy, fast)."""

import numpy as np
import pytest

from preframr_playroutine import (
    EVENT_DTYPE,
    RAMACCESS_DTYPE,
    SID_WRITE,
    VEC_IRQ,
    WIN_IRQ,
    Trace,
    analyze,
    classify_register,
    combine_lohi,
    correlate_event_reset,
    detect_table_walk,
    fit_bacc,
    reconstruct_register,
    round_trip,
    segmented_bacc,
    state_sequence,
    voice_events,
)
from preframr_playroutine.trace import CPU_VECTOR

PAL_FRAME = 985248.444 / 50.0
PAL_META = {"cpu_hz": 985248.444, "effective_model": "PAL"}


def _ev(cycle, etype, chip=0, reg=0, value=0, addr=0, aux=0):
    return (cycle, etype, chip, reg, value, addr, aux)


def _ra(cycle, addr, value, pc=0x1234, kind=WIN_IRQ):
    return (cycle, pc, addr, value, kind, 0)


def _frame_cycle(i):
    return int(round(i * PAL_FRAME))


def _build_trace(records, ram_writes=None, ram=None):
    evs = np.array(records, dtype=EVENT_DTYPE)
    evs.sort(order="cycle", kind="stable")
    kwargs = {}
    if ram_writes is not None:
        kwargs["ramwr"] = np.array(ram_writes, dtype=RAMACCESS_DTYPE)
    if ram is not None:
        kwargs["ram"] = np.asarray(ram, dtype=np.uint8)
    return Trace.from_events(evs, PAL_META, **kwargs)


def _trace_with_register(values, sid_addr=0xD400, pcs=None, ram_writes=None, ram=None):
    """One IRQ per frame, SID write of values[i] to sid_addr each frame."""
    recs = []
    for i, v in enumerate(values):
        tick = _frame_cycle(i)
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ, addr=0x1003))
        pc = pcs[i] if pcs is not None else 0x1100
        recs.append(
            _ev(
                tick + 20,
                SID_WRITE,
                reg=sid_addr & 0x1F,
                value=int(v) & 0xFF,
                addr=sid_addr,
                aux=pc,
            )
        )
    return _build_trace(recs, ram_writes=ram_writes, ram=ram)


# -- fit_bacc -------------------------------------------------------------


def test_fit_bacc_saw():
    series = (np.arange(200) * 3) % 64  # saw ramp, step 3, wraps at 64
    fit = fit_bacc(series)
    assert fit is not None
    assert fit["type"] == "BACC"
    assert fit["step"] == 3
    assert fit["residual"] > 0.95


def test_fit_bacc_reflect_triangle():
    span = 40
    up = np.arange(0, span + 1, 2)
    down = np.arange(span - 2, 0, -2)
    series = np.tile(np.concatenate([up, down]), 6)
    fit = fit_bacc(series)
    assert fit is not None
    assert fit["mode"] == "reflect"
    assert fit["step"] == 2
    assert fit["period"] == pytest.approx(span // 2, abs=1)
    assert fit["residual"] > 0.95


def test_fit_bacc_wrap():
    series = (np.arange(300) * 5) % 256  # 8-bit wrapping accumulator
    fit = fit_bacc(series)
    assert fit is not None
    assert fit["mode"] in ("wrap", "saw")
    assert fit["step"] == 5
    assert fit["residual"] > 0.95


def test_fit_bacc_16bit_vibrato():
    # 16-bit reflecting accumulator from lo/hi cells (vibrato-style).
    span = 600
    up = np.arange(0, span, 24)
    down = np.arange(span, 0, -24)
    raw = np.tile(np.concatenate([up, down]), 5)
    lo = raw & 0xFF
    hi = (raw >> 8) & 0xFF
    series = combine_lohi(lo, hi)
    assert np.array_equal(series, raw)
    fit = fit_bacc(series)
    assert fit is not None
    assert fit["mode"] == "reflect"
    assert fit["step"] == 24
    assert fit["residual"] > 0.9


def test_fit_bacc_constant_returns_none():
    assert fit_bacc(np.full(50, 7)) is None


def test_fit_bacc_noise_returns_none():
    rng = np.random.default_rng(0)
    assert fit_bacc(rng.integers(0, 256, size=100)) is None


# -- detect_table_walk ----------------------------------------------------


def test_detect_table_walk_basic():
    ram = np.zeros(65536, dtype=np.uint8)
    base = 0x2000
    table = np.array([0x41, 0x11, 0x21, 0x81, 0x09], dtype=np.uint8)
    ram[base : base + len(table)] = table
    # cursor walks 0..4 looping back to 0
    cursor = np.array([0, 1, 2, 3, 4] * 8, dtype=np.int64)
    values = table[cursor % len(table)]
    res = detect_table_walk(cursor, ram, value_series=values)
    assert res is not None
    assert res["base"] == base
    assert res["length"] == 5
    assert res["loop"] == 0
    assert res["residual"] == 1.0
    assert np.array_equal(res["table"], table)


def test_detect_table_walk_loopback_marker():
    ram = np.zeros(65536, dtype=np.uint8)
    base = 0x3000
    table = np.arange(0x50, 0x50 + 8, dtype=np.uint8)
    ram[base : base + len(table)] = table
    # cursor runs to 6 then loops back to 2 (loop marker target).
    pattern = [0, 1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6]
    cursor = np.array(pattern * 4, dtype=np.int64)
    values = table[cursor]
    res = detect_table_walk(cursor, ram, value_series=values, stride=1)
    assert res is not None
    assert res["base"] == base
    assert res["loop"] == 2
    assert res["residual"] == 1.0


def test_detect_table_walk_constant_cursor_none():
    ram = np.zeros(65536, dtype=np.uint8)
    cursor = np.full(20, 3, dtype=np.int64)
    assert detect_table_walk(cursor, ram, value_series=np.zeros(20)) is None


def test_detect_table_walk_structural_only():
    cursor = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    res = detect_table_walk(cursor, None)
    assert res is not None
    assert res["base"] is None
    assert res["length"] == 3


# -- state_sequence -------------------------------------------------------


def test_state_sequence_carry_forward():
    recs = []
    for i in range(10):
        recs.append(_ev(_frame_cycle(i) + 2, CPU_VECTOR, value=VEC_IRQ))
    ramwr = [
        _ra(_frame_cycle(0) + 5, 0xFB, 0),
        _ra(_frame_cycle(3) + 5, 0xFB, 9),
        _ra(_frame_cycle(7) + 5, 0xFB, 20),
        _ra(_frame_cycle(0) + 6, 0xFC, 100),  # constant cell -> excluded
    ]
    trace = _build_trace(recs, ram_writes=ramwr)
    ss = state_sequence(trace)
    assert list(ss.addrs) == [0xFB]  # 0xFC constant, excluded
    col = ss.grid[:, 0]
    assert np.all(col[:3] == 0)
    assert np.all(col[3:7] == 9)
    assert np.all(col[7:] == 20)


def test_state_sequence_explicit_addrs():
    recs = [_ev(_frame_cycle(i) + 2, CPU_VECTOR, value=VEC_IRQ) for i in range(5)]
    ramwr = [_ra(_frame_cycle(2) + 5, 0xC000, 42, kind=WIN_IRQ)]
    trace = _build_trace(recs, ram_writes=ramwr)
    ss = state_sequence(trace, addrs=[0xC000])
    assert list(ss.addrs) == [0xC000]
    assert ss.grid[2, 0] == 42


# -- segmented BACC / feeder cells / read-log table walk ------------------


def _note_reseeded_triangle(notes, note_len):
    """Reflecting triangle reseeded to 0x10 at the start of every note."""
    up = list(range(0x10, 0x41, 6))  # 0x10..0x40
    down = list(range(0x3A, 0x10, -6))  # 0x3a..0x16
    one = (up + down)[:note_len]
    return (one * notes)[: notes * note_len]


def test_segmented_bacc_recovers_reseeded_step():
    vals = np.array(_note_reseeded_triangle(6, 13), dtype=np.int64)
    resets = list(range(0, len(vals), 13))
    fit = segmented_bacc(vals, resets)
    assert fit is not None
    assert fit["type"] == "BACC"
    assert fit["step"] == 6
    assert fit["segmented"] is True
    assert fit["n_fit"] >= 3
    # The same series with no resets has discontinuities at each note edge that
    # defeat a single global accumulator fit.
    assert fit_bacc(vals) is None


def test_classify_bacc_note_reseeded_feeder_cell():
    cell = 0x1750  # DMC PW-lo accumulator cell, mirrored to $D402.
    note_len = 13
    vals = _note_reseeded_triangle(6, note_len)
    recs = []
    ramwr = []
    for i, v in enumerate(vals):
        tick = _frame_cycle(i)
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ))
        ctrl = 0x40 if (i % note_len) == note_len - 1 else 0x41
        recs.append(_ev(tick + 8, SID_WRITE, reg=4, value=ctrl, addr=0xD404, aux=0x1500))
        recs.append(_ev(tick + 12, SID_WRITE, reg=2, value=int(v), addr=0xD402, aux=0x1388))
        ramwr.append(_ra(tick + 10, cell, int(v)))
    trace = _build_trace(recs, ram_writes=ramwr)
    res = classify_register(trace, 0xD402)
    assert res["type"] == "BACC"
    assert res["step"] == 6
    assert res["cell_addr"] == cell  # feeder state cell recovered


def test_classify_table_walk_readlog_mask():
    ram = np.zeros(65536, dtype=np.uint8)
    base = 0x18AD
    table = np.array([0x41, 0x11, 0x21, 0x81], dtype=np.uint8)
    ram[base : base + len(table)] = table
    cursor_cell = 0x177A
    n = 64
    recs = []
    ramwr = []
    ramrd = []
    for i in range(n):
        tick = _frame_cycle(i)
        cur = i % len(table)
        gate_off = (i % 10) < 3  # force the gate bit off ~30% of frames
        ctrl = int(table[cur]) & (0xFE if gate_off else 0xFF)
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ))
        recs.append(_ev(tick + 14, SID_WRITE, reg=4, value=ctrl, addr=0xD404, aux=0x1628))
        ramwr.append(_ra(tick + 6, cursor_cell, cur))
        ramrd.append(_ra(tick + 8, base + cur, int(table[cur])))  # LDA table,X
    evs = np.array(recs, dtype=EVENT_DTYPE)
    evs.sort(order="cycle", kind="stable")
    trace = Trace.from_events(
        evs,
        PAL_META,
        ramwr=np.array(ramwr, dtype=RAMACCESS_DTYPE),
        ramrd=np.array(ramrd, dtype=RAMACCESS_DTYPE),
        ram=ram,
    )
    res = classify_register(trace, 0xD404)
    assert res["type"] == "TABLE_WALK"
    assert res["base"] == base
    assert res["mask"] == 0xFE  # the gate bit is masked out
    assert res["cursor_addr"] == cursor_cell
    assert np.array_equal(res["table"], table)


def test_classify_table_walk_noread_image_mask():
    # Same masked waveform table walk, but WITHOUT a read log: recovery must use
    # the cursor state cell + ram_image scan and still recover base/mask/cursor.
    ram = np.zeros(65536, dtype=np.uint8)
    base = 0x18AD
    table = np.array([0x41, 0x11, 0x21, 0x81], dtype=np.uint8)
    ram[base : base + len(table)] = table
    cursor_cell = 0x177A
    n = 64
    recs = []
    ramwr = []
    for i in range(n):
        tick = _frame_cycle(i)
        cur = i % len(table)
        gate_off = (i % 10) < 3
        ctrl = int(table[cur]) & (0xFE if gate_off else 0xFF)
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ))
        recs.append(_ev(tick + 14, SID_WRITE, reg=4, value=ctrl, addr=0xD404, aux=0x1628))
        ramwr.append(_ra(tick + 6, cursor_cell, cur))
    trace = _build_trace(recs, ram_writes=ramwr, ram=ram)  # no ramrd -> scan path
    assert len(trace.ram_reads()) == 0
    res = classify_register(trace, 0xD404)
    assert res["type"] == "TABLE_WALK"
    assert res["base"] == base
    assert res["mask"] == 0xFE
    assert res["cursor_addr"] == cursor_cell
    assert np.array_equal(res["table"], table)


def test_classify_per_note_ad_cell_seq():
    n = 80
    note_len = 16
    recs = []
    for i in range(n):
        tick = _frame_cycle(i)
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ))
        if i % note_len == 0:
            recs.append(_ev(tick + 6, SID_WRITE, reg=4, value=0x41, addr=0xD404, aux=0x1500))
            ad = (0x10 + (i // note_len) * 0x11) & 0xFF
            recs.append(_ev(tick + 8, SID_WRITE, reg=5, value=ad, addr=0xD405, aux=0x1230))
        elif i % note_len == note_len - 1:
            recs.append(_ev(tick + 6, SID_WRITE, reg=4, value=0x40, addr=0xD404, aux=0x1500))
    trace = _build_trace(recs)
    res = classify_register(trace, 0xD405)
    assert res["type"] == "SEQ"  # AD written once per note -> event-latched


# -- classify_register ----------------------------------------------------


def test_classify_const():
    trace = _trace_with_register([0x0F] * 30, sid_addr=0xD418)
    res = classify_register(trace, 0xD418)
    assert res["type"] == "CONST"
    assert res["value"] == 0x0F


def test_classify_bacc():
    values = (np.arange(120) * 2) % 64
    trace = _trace_with_register(values, sid_addr=0xD400)
    res = classify_register(trace, 0xD400)
    assert res["type"] == "BACC"
    assert res["step"] == 2
    assert 0x1100 in res["store_pcs"]


def test_classify_table_walk():
    ram = np.zeros(65536, dtype=np.uint8)
    base = 0x2200
    table = np.array([0x41, 0x11, 0x21, 0x81], dtype=np.uint8)
    ram[base : base + len(table)] = table
    n = 40
    cursor_vals = np.arange(n) % len(table)
    reg_vals = table[cursor_vals]
    recs = []
    ramwr = []
    for i in range(n):
        tick = _frame_cycle(i)
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ))
        recs.append(
            _ev(tick + 20, SID_WRITE, reg=4, value=int(reg_vals[i]), addr=0xD404, aux=0x1591)
        )
        ramwr.append(_ra(tick + 10, 0x80, int(cursor_vals[i])))
    trace = _build_trace(recs, ram_writes=ramwr, ram=ram)
    res = classify_register(trace, 0xD404)
    assert res["type"] == "TABLE_WALK"
    assert res["base"] == base
    assert res["cursor_addr"] == 0x80


def test_classify_seq_sparse():
    # Register written only on a few note boundaries (sparse) -> SEQ.
    recs = []
    n = 60
    for i in range(n):
        recs.append(_ev(_frame_cycle(i) + 2, CPU_VECTOR, value=VEC_IRQ))
    for i, v in ((0, 0x21), (20, 0x41), (40, 0x11)):
        recs.append(_ev(_frame_cycle(i) + 20, SID_WRITE, reg=5, value=v, addr=0xD405, aux=0x1230))
    trace = _build_trace(recs)
    res = classify_register(trace, 0xD405)
    assert res["type"] == "SEQ"


def test_classify_xstate_dense_irregular():
    rng = np.random.default_rng(1)
    values = rng.integers(0, 256, size=80)
    trace = _trace_with_register(values, sid_addr=0xD402)
    res = classify_register(trace, 0xD402)
    assert res["type"] == "XSTATE"


# -- correlate_event_reset ------------------------------------------------


def _accumulator_with_resets(reset_frames, n=80, step=3, modulus=200):
    """Build (records, ramwr) where cell 0x90 accumulates and resets to 0."""
    recs = []
    ramwr = []
    val = 0
    reset_set = set(reset_frames)
    for i in range(n):
        tick = _frame_cycle(i)
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ))
        if i in reset_set:
            val = 0
            recs.append(_ev(tick + 4, SID_WRITE, reg=4, value=0x41, addr=0xD404, aux=0x11DB))
        else:
            val = (val + step) % modulus
        ramwr.append(_ra(tick + 10, 0x90, val))
    return recs, ramwr


def _gate_on(ev):
    return ev["etype"] == SID_WRITE and ev["addr"] == 0xD404 and (ev["value"] & 1)


def test_correlate_event_reset_positive():
    recs, ramwr = _accumulator_with_resets([20, 40, 60])
    trace = _build_trace(recs, ram_writes=ramwr)
    res = correlate_event_reset(trace, _gate_on, 0x90)
    assert res["n_triggers"] == 3
    assert res["correlation"] >= 0.9


def test_correlate_event_reset_negative():
    # Gate writes happen, but the accumulator never resets at them.
    recs = []
    ramwr = []
    n = 80
    for i in range(n):
        tick = _frame_cycle(i)
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ))
        ramwr.append(_ra(tick + 10, 0x90, (i * 3) % 200))
        if i in (20, 40, 60):
            recs.append(_ev(tick + 4, SID_WRITE, reg=4, value=0x41, addr=0xD404))
    trace = _build_trace(recs, ram_writes=ramwr)
    res = correlate_event_reset(trace, _gate_on, 0x90)
    assert res["n_triggers"] == 3
    assert res["correlation"] < 0.5


def test_correlate_event_reset_no_triggers():
    recs, ramwr = _accumulator_with_resets([])
    trace = _build_trace(recs, ram_writes=ramwr)
    res = correlate_event_reset(trace, _gate_on, 0x90)
    assert res["n_triggers"] == 0
    assert res["correlation"] == 0.0


# -- voice_events ---------------------------------------------------------


def test_voice_events_gate_transitions():
    # Voice 0 CTRL gate goes on at frame 1, off at frame 10.
    recs = []
    n = 20
    for i in range(n):
        recs.append(_ev(_frame_cycle(i) + 2, CPU_VECTOR, value=VEC_IRQ))
    recs.append(_ev(_frame_cycle(1) + 20, SID_WRITE, reg=4, value=0x41, addr=0xD404))
    recs.append(_ev(_frame_cycle(10) + 20, SID_WRITE, reg=4, value=0x40, addr=0xD404))
    trace = _build_trace(recs)
    ev = voice_events(trace)
    types = [(e["frame"], e["type"]) for e in ev[0]]
    assert (1, "note_on") in types
    assert (10, "note_off") in types
    assert ev[1] == [] and ev[2] == []


# -- analyze --------------------------------------------------------------


def test_analyze_summary():
    ram = np.zeros(65536, dtype=np.uint8)
    base = 0x2400
    table = np.array([0x41, 0x11, 0x21, 0x81], dtype=np.uint8)
    ram[base : base + len(table)] = table
    n = 48
    recs = []
    ramwr = []
    saw = (np.arange(n) * 2) % 64
    cursor_vals = np.arange(n) % len(table)
    ctrl_vals = table[cursor_vals]
    for i in range(n):
        tick = _frame_cycle(i)
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ))
        recs.append(_ev(tick + 12, SID_WRITE, reg=0, value=int(saw[i]), addr=0xD400, aux=0x1400))
        recs.append(
            _ev(tick + 14, SID_WRITE, reg=4, value=int(ctrl_vals[i]), addr=0xD404, aux=0x1591)
        )
        recs.append(_ev(tick + 16, SID_WRITE, reg=24, value=0x0F, addr=0xD418, aux=0x105C))
        ramwr.append(_ra(tick + 10, 0x80, int(cursor_vals[i])))
    trace = _build_trace(recs, ram_writes=ramwr, ram=ram)
    result = analyze(trace)
    assert result[0xD400]["type"] == "BACC"
    assert result[0xD404]["type"] == "TABLE_WALK"
    assert result[0xD418]["type"] == "CONST"
    summary = result["summary"]
    assert summary["BACC"] >= 1
    assert summary["TABLE_WALK"] >= 1
    assert summary["CONST"] >= 1


def test_analyze_empty():
    trace = Trace.from_events(np.empty(0, dtype=EVENT_DTYPE), PAL_META)
    result = analyze(trace)
    assert result["summary"] == {}


# -- v2 trace contract ----------------------------------------------------


def test_ramaccess_dtype_is_16_bytes():
    assert RAMACCESS_DTYPE.itemsize == 16


def test_v2_selectors_default_empty():
    trace = Trace.from_events(np.empty(0, dtype=EVENT_DTYPE), PAL_META)
    assert len(trace.ram_writes()) == 0
    assert len(trace.ram_reads()) == 0
    assert len(trace.coverage_pcs()) == 0
    assert trace.ram_image() is None


def test_coverage_pcs_bit_indexing():
    cov = np.zeros(8192, dtype=np.uint8)
    cov[0] = 1 << 5  # PC 5
    cov[1025] = 1 << 0  # PC 8200 = 1025*8 + 0
    trace = Trace.from_events(np.empty(0, dtype=EVENT_DTYPE), PAL_META, coverage=cov)
    assert list(trace.coverage_pcs()) == [5, 8200]


def test_ram_writes_kind_filter():
    ramwr = np.array(
        [_ra(0, 0x10, 1, kind=WIN_IRQ), _ra(1, 0x11, 2, kind=1)],
        dtype=RAMACCESS_DTYPE,
    )
    trace = Trace.from_events(np.empty(0, dtype=EVENT_DTYPE), PAL_META, ramwr=ramwr)
    assert len(trace.ram_writes()) == 2
    assert len(trace.ram_writes(kind=WIN_IRQ)) == 1
    assert trace.ram_writes(kind=WIN_IRQ)["addr"][0] == 0x10


# -- round-trip reconstruction --------------------------------------------


def _ticks(n):
    return np.array([_frame_cycle(i) for i in range(n)], dtype=np.uint64)


def test_reconstruct_const_roundtrip():
    n = 40
    desc = {"type": "CONST", "value": 0x1F}
    recon = reconstruct_register(desc, _ticks(n))
    assert np.array_equal(recon, np.full(n, 0x1F))


def test_reconstruct_seq_roundtrip():
    # Event-latched series held between note changes.
    series = np.array([0x00] * 5 + [0x21] * 7 + [0x41] * 8 + [0x11] * 10, dtype=np.int64)
    frames = [0, 5, 12, 20]
    values = [0x00, 0x21, 0x41, 0x11]
    desc = {"type": "SEQ", "latch_frames": frames, "latch_values": values}
    recon = reconstruct_register(desc, _ticks(len(series)))
    assert np.array_equal(recon, series)


def test_reconstruct_bacc_reflect_roundtrip():
    # Reseeded reflecting triangle (no holds) regenerates exactly from seeds.
    up = list(range(0, 41, 4))
    down = list(range(36, 0, -4))
    period = up + down  # length 20
    n = len(period) * 6
    series = np.array(period * 6, dtype=np.int64)
    resets = list(range(0, n, len(period)))
    seeds = [0] * len(resets)
    desc = {
        "type": "BACC",
        "mode": "reflect",
        "step": 4,
        "lo": 0,
        "hi": 40,
        "resets": resets,
        "seeds": seeds,
        "byte_role": "full",
    }
    recon = reconstruct_register(desc, _ticks(n))
    assert np.array_equal(recon, series)


def test_reconstruct_bacc_saw_roundtrip():
    n = 90
    series = (np.arange(n) * 5) % 60
    desc = {
        "type": "BACC",
        "mode": "saw",
        "step": 5,
        "lo": 0,
        "hi": 55,
        "modulus": 60,
        "resets": [0],
        "seeds": [0],
        "byte_role": "full",
    }
    recon = reconstruct_register(desc, _ticks(n))
    assert np.array_equal(recon, series)


def test_reconstruct_table_walk_roundtrip():
    table = np.array([0x41, 0x11, 0x21, 0x81], dtype=np.uint8)
    cursor = np.tile([0, 1, 2, 3], 10).astype(np.int64)
    desc = {
        "type": "TABLE_WALK",
        "base": 0x2000,
        "stride": 1,
        "table": table,
        "mask": 0xFF,
        "cursor": cursor,
        "cursor_offset": 0,
    }
    recon = reconstruct_register(desc, _ticks(len(cursor)))
    assert np.array_equal(recon, table[cursor % len(table)])


def test_reconstruct_table_walk_masked_roundtrip():
    table = np.array([0x41, 0x11, 0x21, 0x81], dtype=np.uint8)
    cursor = np.tile([0, 1, 2, 3], 10).astype(np.int64)
    desc = {
        "type": "TABLE_WALK",
        "base": 0x2000,
        "stride": 1,
        "table": table,
        "mask": 0xFE,
        "cursor": cursor,
        "cursor_offset": 0,
    }
    recon = reconstruct_register(desc, _ticks(len(cursor)))
    assert np.array_equal(recon, table[cursor % len(table)] & 0xFE)


def test_reconstruct_composite_series_roundtrip():
    n = 60
    base = (np.arange(n) % 50).astype(np.int64)
    mod = np.zeros(n, dtype=np.int64)
    desc = {
        "type": "COMPOSITE",
        "byte_role": "full",
        "width_mask": 0xFF,
        "base": {"series": base},
        "mod": {"series": mod},
        "overrides": [],
    }
    recon = reconstruct_register(desc, _ticks(n))
    assert np.array_equal(recon, base & 0xFF)


def test_round_trip_composite_freq_from_trace():
    # Real-trace-style COMPOSITE: FREQ = base_cell + accum_cell, with a $FFFF
    # hard-restart override gated by a flag cell. round_trip must reconstruct it
    # exactly from the recovered descriptor.
    n = 80
    note_len = 20
    base_lo, base_hi = 0x10, 0x11
    acc_lo, acc_hi = 0x12, 0x13
    flag = 0x20
    recs = []
    ramwr = []
    for i in range(n):
        tick = _frame_cycle(i)
        note = i // note_len
        base16 = 0x0480 + note * 0x140
        acc16 = (i % 5) * 6
        forced = i % note_len == 1
        freq = 0xFFFF if forced else (base16 + acc16) & 0xFFFF
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ))
        ramwr.append(_ra(tick + 3, flag, 1 if forced else 0))
        ramwr.append(_ra(tick + 4, base_lo, base16 & 0xFF))
        ramwr.append(_ra(tick + 4, base_hi, (base16 >> 8) & 0xFF))
        ramwr.append(_ra(tick + 5, acc_lo, acc16 & 0xFF))
        ramwr.append(_ra(tick + 5, acc_hi, (acc16 >> 8) & 0xFF))
        # gate so a voice/note structure exists (note-on each note)
        gate = 0x41 if i % note_len else 0x40
        recs.append(_ev(tick + 8, SID_WRITE, reg=4, value=gate, addr=0xD404, aux=0x1500))
        recs.append(_ev(tick + 10, SID_WRITE, reg=0, value=freq & 0xFF, addr=0xD400, aux=0x1606))
        recs.append(
            _ev(tick + 10, SID_WRITE, reg=1, value=(freq >> 8) & 0xFF, addr=0xD401, aux=0x1609)
        )
    trace = _build_trace(recs, ram_writes=ramwr)
    res = analyze(trace)
    assert res[0xD400]["type"] == "COMPOSITE"
    assert res[0xD401]["type"] == "COMPOSITE"
    rt = round_trip(trace)
    assert rt[0xD400] == 1.0
    assert rt[0xD401] == 1.0


def test_round_trip_reports_overall_and_unmodeled():
    trace = _trace_with_register([0x0F] * 30, sid_addr=0xD418)
    rt = round_trip(trace)
    assert rt["overall"] == 1.0
    assert rt["unmodeled"] == []
