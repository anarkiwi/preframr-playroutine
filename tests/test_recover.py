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
    recover_tuning,
    round_trip,
    state_sequence,
    voice_detune,
    voice_events,
)
from preframr_playroutine.recover import (
    _recur_const,
    _recur_product,
    _segment_bounds,
    _segmented_recur,
)
from preframr_playroutine.trace import CPU_VECTOR
from _minisid import recur_series

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


# -- state_sequence group-by optimization parity --------------------------

from preframr_playroutine.recover import (  # noqa: E402
    _carry_series,
    _changing_cells,
    _window_kind,
)


def _ref_state_sequence_grid(trace, kind, addrs):
    """Frozen pre-optimization per-address boolean scan of recover.state_sequence."""
    ticks = trace.tick_cycles(kind).astype(np.uint64)
    wr = trace.ram_writes(_window_kind(kind))
    if addrs is None:
        addrs = _changing_cells(wr)
    else:
        addrs = np.array(sorted({int(a) for a in addrs}), dtype=np.uint16)
    grid = np.zeros((len(ticks), len(addrs)), dtype=np.uint8)
    if len(ticks) and len(addrs) and len(wr):
        for j, a in enumerate(addrs):
            sel = wr[wr["addr"] == a]
            if len(sel):
                grid[:, j] = _carry_series(sel["cycle"], sel["value"], ticks).astype(np.uint8)
    return addrs, grid


def _make_state_seq_trace(rng):
    """Random IRQ ticks plus RAM writes across many cells (some constant)."""
    n_frames = int(rng.integers(20, 120))
    recs = [_ev(_frame_cycle(i) + 2, CPU_VECTOR, value=VEC_IRQ) for i in range(n_frames)]
    n_cells = int(rng.integers(1, 12))
    cells = rng.choice(np.arange(0x0002, 0x0400), size=n_cells, replace=False)
    ramwr = []
    for a in cells:
        n_wr = int(rng.integers(0, 6))
        const = rng.integers(0, 2) == 0  # a cell that never changes -> excluded
        val0 = int(rng.integers(0, 256))
        for _ in range(n_wr):
            f = int(rng.integers(0, n_frames))
            off = int(rng.integers(3, 40))
            v = val0 if const else int(rng.integers(0, 256))
            ramwr.append(_ra(_frame_cycle(f) + off, int(a), v))
    if not ramwr:
        ramwr.append(_ra(_frame_cycle(0) + 5, int(cells[0]), 7))
    return _build_trace(recs, ram_writes=ramwr), [int(a) for a in cells]


@pytest.mark.parametrize("seed", range(10))
def test_state_sequence_groupby_parity(seed):
    rng = np.random.default_rng(2000 + seed)
    for _ in range(6):
        trace, cells = _make_state_seq_trace(rng)
        for addrs in (None, cells, cells[:1], [int(cells[0]), 0xFFFF]):
            ref_addrs, ref_grid = _ref_state_sequence_grid(trace, "auto", addrs)
            ss = state_sequence(trace, addrs=addrs)
            assert np.array_equal(np.asarray(ss.addrs), np.asarray(ref_addrs)), (seed, addrs)
            assert np.array_equal(ss.grid, ref_grid), (seed, addrs)


# -- segmented BACC / feeder cells / read-log table walk ------------------


def _note_reseeded_triangle(notes, note_len):
    """Reflecting triangle reseeded to 0x10 at the start of every note."""
    up = list(range(0x10, 0x41, 6))  # 0x10..0x40
    down = list(range(0x3A, 0x10, -6))  # 0x3a..0x16
    one = (up + down)[:note_len]
    return (one * notes)[: notes * note_len]


def test_segmented_recur_recovers_reseeded_step():
    # The unified fitter's const-step vote recovers a note-reseeded accumulator.
    vals = np.array(_note_reseeded_triangle(6, 13), dtype=np.int64)
    resets = list(range(0, len(vals), 13))
    fit = _recur_const(vals, _segment_bounds(vals, resets))
    assert fit is not None
    assert fit["type"] == "BACC"
    assert fit["step"] == 6
    assert fit["segmented"] is True
    assert fit["n_fit"] >= 3
    # It is offered as a candidate by the unified proposer entry point too.
    cands = _segmented_recur(vals, resets)
    assert any(c["type"] == "BACC" and c["step"] == 6 for c in cands)
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


def _pitch_table(length):
    """A monotonic 16-bit pitch table with high-entropy lo bytes (FC-style)."""
    pitch = (0x0100 + np.arange(length) * 0x57).astype(np.int64) & 0xFFFF
    return pitch


def _build_pitchwalk_trace(idx_series, idx_cells, lo_base=0x1564, hi_base=0x15C4, note_len=16):
    """Trace whose FREQ is ``pitchtable[sum(idx_cells)]`` (FC-style pitch walk).

    ``idx_series`` is a list of per-frame index-cell-value tuples (one per cell in
    ``idx_cells``); the emitted FREQ is the pitch-table entry at their sum.
    """
    n = len(idx_series)
    length = max(int(sum(v)) for v in idx_series) + 1
    pitch = _pitch_table(length)
    lo_tab = (pitch & 0xFF).astype(np.uint8)
    hi_tab = ((pitch >> 8) & 0xFF).astype(np.uint8)
    ram = np.zeros(65536, dtype=np.uint8)
    ram[lo_base : lo_base + length] = lo_tab
    ram[hi_base : hi_base + length] = hi_tab
    recs = []
    ramwr = []
    for i, parts in enumerate(idx_series):
        tick = _frame_cycle(i)
        k = int(sum(parts))
        freq = int(pitch[k])
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ))
        gate = 0x41 if i % note_len else 0x40
        recs.append(_ev(tick + 6, SID_WRITE, reg=4, value=gate, addr=0xD404, aux=0x1552))
        recs.append(_ev(tick + 10, SID_WRITE, reg=0, value=freq & 0xFF, addr=0xD400, aux=0x190E))
        recs.append(
            _ev(tick + 10, SID_WRITE, reg=1, value=(freq >> 8) & 0xFF, addr=0xD401, aux=0x190E)
        )
        for cell, val in zip(idx_cells, parts):
            ramwr.append(_ra(tick + 4, cell, int(val)))
    return _build_trace(recs, ram_writes=ramwr, ram=ram)


def test_classify_pitchwalk_single_index_cell():
    rng = np.random.default_rng(3)
    idx = rng.integers(0, 16, size=96)
    trace = _build_pitchwalk_trace([(int(k),) for k in idx], [0x1930])
    res = classify_register(trace, 0xD400)
    assert res["type"] == "PITCHWALK", res["type"]
    assert res["lo_base"] == 0x1564
    assert res["hi_base"] == 0x15C4
    assert res["index_cells"] == [0x1930]
    rt = round_trip(trace)
    assert rt[0xD400] == 1.0
    assert rt[0xD401] == 1.0


def test_classify_pitchwalk_additive_offset():
    # idx = note ($1930) + arp offset ($0041); the additive index must be recovered.
    # The arp offset is 0 on the held note (dominant) with periodic +4/+7 jumps,
    # exactly as a Future Composer arpeggio drives the pitch index.
    rng = np.random.default_rng(7)
    note = rng.integers(2, 12, size=160)
    arp = np.array([0, 0, 0, 0, 4, 0, 0, 7] * 20)[:160]
    trace = _build_pitchwalk_trace(list(zip(note.tolist(), arp.tolist())), [0x1930, 0x0041])
    res = classify_register(trace, 0xD400)
    assert res["type"] == "PITCHWALK"
    assert set(res["index_cells"]) == {0x1930, 0x0041}
    rt = round_trip(trace)
    assert rt[0xD400] == 1.0
    assert rt[0xD401] == 1.0


def test_classify_pitchwalk_with_override():
    # Pitch walk plus a hard-restart override: on flag frames the player forces
    # FREQ to $FFFF (off table). The override predicate must be recovered so the
    # walk round-trips exactly.
    rng = np.random.default_rng(11)
    idx = rng.integers(0, 14, size=140)
    flag_cell = 0x0050
    length = 16
    pitch = _pitch_table(length)
    lo_tab = (pitch & 0xFF).astype(np.uint8)
    hi_tab = ((pitch >> 8) & 0xFF).astype(np.uint8)
    ram = np.zeros(65536, dtype=np.uint8)
    ram[0x1564 : 0x1564 + length] = lo_tab
    ram[0x15C4 : 0x15C4 + length] = hi_tab
    recs = []
    ramwr = []
    for i, k in enumerate(idx):
        tick = _frame_cycle(i)
        forced = i % 20 == 1
        freq = 0xFFFF if forced else int(pitch[int(k)])
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ))
        gate = 0x41 if i % 16 else 0x40
        recs.append(_ev(tick + 6, SID_WRITE, reg=4, value=gate, addr=0xD404, aux=0x1552))
        recs.append(_ev(tick + 10, SID_WRITE, reg=0, value=freq & 0xFF, addr=0xD400, aux=0x190E))
        recs.append(
            _ev(tick + 10, SID_WRITE, reg=1, value=(freq >> 8) & 0xFF, addr=0xD401, aux=0x190E)
        )
        ramwr.append(_ra(tick + 4, 0x1930, int(k)))
        ramwr.append(_ra(tick + 3, flag_cell, 1 if forced else 0))
    trace = _build_trace(recs, ram_writes=ramwr, ram=ram)
    res = classify_register(trace, 0xD400)
    assert res["type"] == "PITCHWALK"
    assert res["overrides"]
    rt = round_trip(trace)
    assert rt[0xD400] == 1.0
    assert rt[0xD401] == 1.0


def test_reconstruct_pitchwalk_no_sampler_zeros():
    desc = {
        "type": "PITCHWALK",
        "byte_role": "lo",
        "lo_table": np.arange(8, dtype=np.uint8),
        "hi_table": np.zeros(8, dtype=np.uint8),
        "index_cells": [0x90],
        "overrides": [],
    }
    recon = reconstruct_register(desc, _ticks(10))  # no trace/sampler -> zeros
    assert np.array_equal(recon, np.zeros(10, dtype=np.int64))


def test_reconstruct_pitchwalk_roundtrip():
    length = 16
    pitch = _pitch_table(length)
    desc = {
        "type": "PITCHWALK",
        "byte_role": "hi",
        "lo_base": 0x1564,
        "hi_base": 0x15C4,
        "lo_table": (pitch & 0xFF).astype(np.uint8),
        "hi_table": ((pitch >> 8) & 0xFF).astype(np.uint8),
        "index_cells": [0x90],
        "overrides": [],
    }
    cursor = np.tile(np.arange(length), 5).astype(np.int64)
    n = len(cursor)
    recs = [_ev(_frame_cycle(i) + 2, CPU_VECTOR, value=VEC_IRQ) for i in range(n)]
    ramwr = [_ra(_frame_cycle(i) + 4, 0x90, int(cursor[i])) for i in range(n)]
    trace = _build_trace(recs, ram_writes=ramwr)
    recon = reconstruct_register(desc, _ticks(n), trace=trace)
    assert np.array_equal(recon, (pitch[cursor] >> 8) & 0xFF)


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


def test_arbiter_report_keys():
    # Every arbitrated register carries the MDL report keys.
    values = (np.arange(120) * 2) % 64
    trace = _trace_with_register(values, sid_addr=0xD400)
    res = classify_register(trace, 0xD400)
    assert res["type"] == "BACC"
    for key in ("score", "complexity", "captured_frames"):
        assert key in res, key
    # A clean recurrence wins with no replayed captured state.
    assert res["captured_frames"] == 0
    assert res["score"] == pytest.approx(1.0 - 1e-3 * res["complexity"])


def test_mdl_cost_latch_cheap_replay_expensive():
    # A SEQ latch list (few changes) is far cheaper than a per-frame feeder cell
    # replaying a stream that changes every frame -- the captured-state cost is
    # charged per change, not per frame held.
    from preframr_playroutine import ir  # noqa: PLC0415

    n = 200

    class _Sampler:
        def eof(self, _addr):
            return np.arange(n, dtype=np.int64)  # changes every frame

        def at_write(self, _addr, _sid):
            return np.arange(n, dtype=np.int64)

        def operand(self, _addr, _sid):
            return np.arange(n, dtype=np.int64)

        def written_mask(self, _sid):
            return np.ones(n, dtype=bool)

    smp = _Sampler()
    seq = {
        "op": "post",
        "addr": 0xD404,
        "expr": {"op": "seq", "frames": [0, 100], "values": [1, 2]},
    }
    feeder = {
        "op": "post",
        "addr": 0xD404,
        "expr": {"op": "cell", "addr": 0x40, "sample": "write", "sid": 0xD404},
    }
    cx_seq, cap_seq = ir.cost_captured(seq, smp, n)
    cx_feeder, cap_feeder = ir.cost_captured(feeder, smp, n)
    assert cap_seq == 0 and cap_feeder == n
    assert cx_seq < cx_feeder


def test_classify_bacc():
    values = (np.arange(120) * 2) % 64
    trace = _trace_with_register(values, sid_addr=0xD400)
    res = classify_register(trace, 0xD400)
    assert res["type"] == "BACC"
    assert res["step"] == 2
    assert 0x1100 in res["store_pcs"]


def test_classify_bacc_held_seed_prelude_before_cell_write():
    # A note-reseeded PW accumulator captured by a feeder cell, preceded by a
    # pre-dwell hold at the note-on seed where the cell is not yet written (the
    # MusicAssembler PW case). The cell drives the modulated frames; the held-seed
    # prelude must reconstruct from latches so the register round-trips exactly.
    cell = 0x40
    seed = 0x08
    note_len = 8
    predwell = 9
    # Per-note saw whose stride varies between notes: the modal-step recurrence
    # mis-fits the off-modal notes while the captured cell reproduces every
    # modulated frame, so the cell attaches.
    steps = [6, 6, 5, 6, 5, 6]
    ramp = [(0x10 + st * t) & 0xFF for st in steps for t in range(note_len)]
    values = [seed] * predwell + ramp
    recs = []
    ramwr = []
    for i, v in enumerate(values):
        tick = _frame_cycle(i)
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ))
        if i < predwell:
            ctrl = 0x40  # gate off during the held-seed pre-dwell
        else:
            j = i - predwell
            ctrl = 0x40 if (j % note_len) == note_len - 1 else 0x41
            ramwr.append(_ra(tick + 10, cell, int(v)))  # cell written only while modulating
        recs.append(_ev(tick + 8, SID_WRITE, reg=4, value=ctrl, addr=0xD404, aux=0x1500))
        recs.append(_ev(tick + 12, SID_WRITE, reg=2, value=int(v), addr=0xD402, aux=0x1388))
    trace = _build_trace(recs, ram_writes=ramwr)
    res = classify_register(trace, 0xD402)
    assert res["type"] == "BACC"
    assert res["cell"] == cell
    assert res["prelude_end"] == predwell
    rt = round_trip(trace)
    assert rt[0xD402] == 1.0


def test_classify_bacc_output_then_compute_latency_feeder():
    # defMON output-then-compute: each call writes SID from a self-modified operand
    # (current value) then computes the NEXT value into the feeder cell, so the
    # cell's end-of-frame value LEADS the register by one call while the value read
    # at the write instant is exact. A leading held-seed prelude precedes the first
    # cell write. The feeder must still be recovered (its end-of-frame match to the
    # register is ~0; it matches the one-call-shifted register) and round-trip
    # exactly once the prelude fills the pre-modulation hold.
    cell = 0x1023
    seed = 0xD3
    note_len = 8
    predwell = 11
    steps = [19, 19, 18, 19, 18, 19]  # per-note stride varies: recurrence mis-fits
    ramp = [(0x90 - st * t) & 0xFF for st in steps for t in range(note_len)]
    # The held seed runs through frame ``predwell`` inclusive; the cell is written
    # only once per modulating frame, late (after the SID store) with the NEXT
    # value, so its end-of-frame value leads the register by one call and the value
    # read at the write instant lags to the current one.
    values = [seed] * (predwell + 1) + ramp
    recs = []
    ramwr = []
    for i, v in enumerate(values):
        tick = _frame_cycle(i)
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ))
        if i < predwell:
            ctrl = 0x40  # gate off during the held-seed pre-dwell
        else:
            j = i - predwell
            ctrl = 0x40 if (j % note_len) == note_len - 1 else 0x41
            nxt = values[i + 1] if i + 1 < len(values) else v
            ramwr.append(_ra(tick + 40, cell, int(nxt)))  # next value, late -> leads eof
        recs.append(_ev(tick + 8, SID_WRITE, reg=4, value=ctrl, addr=0xD404, aux=0x1500))
        recs.append(_ev(tick + 12, SID_WRITE, reg=2, value=int(v), addr=0xD402, aux=0x1388))
    trace = _build_trace(recs, ram_writes=ramwr)
    res = classify_register(trace, 0xD402)
    assert res["type"] == "BACC"
    assert res["cell"] == cell  # the one-call-latency feeder is recovered
    # The first cell write is late in its frame, so the held-seed prelude must
    # extend one frame past the cell's first-write frame (the first-live handoff).
    assert res["prelude_end"] == predwell + 1
    rt = round_trip(trace)
    assert rt[0xD402] == 1.0


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


def test_classify_table_walk_three_overrides():
    # A CTRL waveform table walk whose player forces three distinct off-table
    # values (one per gating cell): a $0A-style wavetable command resolved to a
    # real waveform plus two key/gate forces. Recovering all three requires more
    # than two overrides; each is kept only when it raises reproduction, so the
    # walk round-trips exactly.
    ram = np.zeros(65536, dtype=np.uint8)
    base = 0x18AD
    table = np.array([0x41, 0x11, 0x21, 0x81], dtype=np.uint8)
    ram[base : base + len(table)] = table
    cell_a, cell_b, cell_c = 0x0050, 0x0051, 0x0052
    # Decreasing force counts (20/8/4) so each greedy pass has a dominant residual
    # value; the firing frames are disjoint by construction.
    n = 240
    recs = []
    ramwr = []
    for i in range(n):
        tick = _frame_cycle(i)
        cur = i % len(table)
        flags = {cell_a: 0, cell_b: 0, cell_c: 0}
        ctrl = int(table[cur])
        if i % 12 == 1:
            flags[cell_a], ctrl = 1, 0x09
        elif i % 30 == 2:
            flags[cell_b], ctrl = 1, 0x80
        elif i % 60 == 3:
            flags[cell_c], ctrl = 1, 0x44
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ))
        recs.append(_ev(tick + 20, SID_WRITE, reg=4, value=ctrl, addr=0xD404, aux=0x1591))
        ramwr.append(_ra(tick + 8, 0x80, cur))
        for cell, flag in flags.items():
            ramwr.append(_ra(tick + 6, cell, flag))
    trace = _build_trace(recs, ram_writes=ramwr, ram=ram)
    res = classify_register(trace, 0xD404)
    assert res["type"] == "TABLE_WALK"
    assert res["cursor_addr"] == 0x80
    assert len(res["overrides"]) == 3
    assert round_trip(trace)[0xD404] == 1.0


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


def test_classify_filter_feeder_latch():
    # Global filter cutoff-high ($D416) is an irregular per-frame copy of a RAM
    # feeder cell -> FEEDER primitive, round-tripping exactly.
    rng = np.random.default_rng(7)
    values = rng.integers(0, 256, size=80)
    cell = 0x40
    recs = []
    ramwr = []
    for i, v in enumerate(values):
        tick = _frame_cycle(i)
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ))
        ramwr.append(_ra(tick + 10, cell, int(v)))
        recs.append(_ev(tick + 20, SID_WRITE, reg=0x16, value=int(v), addr=0xD416, aux=0x1700))
    trace = _build_trace(recs, ram_writes=ramwr)
    res = classify_register(trace, 0xD416)
    assert res["type"] == "FEEDER", res["type"]
    assert res["cell"] == cell
    assert res["sid"] == 0xD416
    assert res["cell_frac"] == 1.0
    rt = round_trip(trace)
    assert rt[0xD416] == 1.0
    ticks = trace.tick_cycles("auto")
    recon = reconstruct_register(res, ticks, trace=trace)
    assert np.array_equal(recon, np.asarray(values, dtype=np.int64))


def test_classify_ctrl_feeder_overrides_table_walk():
    # A voice CTRL ($D404) waveform table walk plus an irregular per-frame gate
    # bit the table can't reproduce -> imperfect TABLE_WALK. A captured RAM cell
    # holds the exact written value, so the case-2 FEEDER upgrade replaces the
    # over-fit table on this NON-filter register and round-trips exactly.
    ram = np.zeros(65536, dtype=np.uint8)
    base = 0x18AD
    table = np.array([0x40, 0x10, 0x20, 0x80], dtype=np.uint8)  # waveform, gate=0
    ram[base : base + len(table)] = table
    cursor_cell = 0x177A
    feeder_cell = 0x1762
    n = 80
    rng = np.random.default_rng(11)
    gate = rng.integers(0, 2, size=n)
    recs = []
    ramwr = []
    for i in range(n):
        tick = _frame_cycle(i)
        cur = i % len(table)
        ctrl = int(table[cur]) | int(gate[i])
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ))
        ramwr.append(_ra(tick + 6, cursor_cell, cur))
        ramwr.append(_ra(tick + 8, feeder_cell, ctrl))
        recs.append(_ev(tick + 14, SID_WRITE, reg=4, value=ctrl, addr=0xD404, aux=0x1628))
    trace = _build_trace(recs, ram_writes=ramwr, ram=ram)
    res = classify_register(trace, 0xD404)
    assert res["type"] == "FEEDER", res["type"]
    assert res["cell"] == feeder_cell
    assert res["sid"] == 0xD404
    assert res["cell_frac"] == 1.0
    assert round_trip(trace)[0xD404] == 1.0


def test_classify_ctrl_table_walk_without_feeder_is_imperfect():
    # Same waveform-plus-gate trace WITHOUT the captured cell still classifies as
    # an imperfect TABLE_WALK -- proving the FEEDER upgrade above fires on the
    # case-2 (table/composite replacement) branch, not the XSTATE relabel.
    ram = np.zeros(65536, dtype=np.uint8)
    base = 0x18AD
    table = np.array([0x40, 0x10, 0x20, 0x80], dtype=np.uint8)
    ram[base : base + len(table)] = table
    cursor_cell = 0x177A
    n = 80
    rng = np.random.default_rng(11)
    gate = rng.integers(0, 2, size=n)
    recs = []
    ramwr = []
    for i in range(n):
        tick = _frame_cycle(i)
        cur = i % len(table)
        ctrl = int(table[cur]) | int(gate[i])
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ))
        ramwr.append(_ra(tick + 6, cursor_cell, cur))
        recs.append(_ev(tick + 14, SID_WRITE, reg=4, value=ctrl, addr=0xD404, aux=0x1628))
    trace = _build_trace(recs, ram_writes=ramwr, ram=ram)
    res = classify_register(trace, 0xD404)
    assert res["type"] == "TABLE_WALK", res["type"]
    assert round_trip(trace)[0xD404] < 1.0


def test_reconstruct_feeder_no_sampler_zeros():
    desc = {"type": "FEEDER", "cell": 0x40, "sid": 0xD416}
    recon = reconstruct_register(desc, _ticks(10))
    assert np.array_equal(recon, np.zeros(10, dtype=np.int64))


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


def test_composite_freq_base_only_no_spurious_mod():
    # Output-then-compute style: the operand cell already carries the whole FREQ.
    # An unrelated varying cell must NOT be grafted on as an additive modulation
    # (that would lower fidelity) -- the composite stays base-only and exact.
    n = 80
    note_len = 20
    op_lo, op_hi, noise = 0x10, 0x11, 0x22
    recs = []
    ramwr = []
    for i in range(n):
        tick = _frame_cycle(i)
        note = i // note_len
        freq = (0x0500 + note * 0x130 + (i % 7) * 11) & 0xFFFF
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ, addr=0x1003))
        ramwr.append(_ra(tick + 4, op_lo, freq & 0xFF))
        ramwr.append(_ra(tick + 4, op_hi, (freq >> 8) & 0xFF))
        ramwr.append(_ra(tick + 5, noise, (i * 37) & 0xFF))
        gate = 0x41 if i % note_len else 0x40
        recs.append(_ev(tick + 8, SID_WRITE, reg=4, value=gate, addr=0xD404, aux=0x1500))
        recs.append(_ev(tick + 10, SID_WRITE, reg=0, value=freq & 0xFF, addr=0xD400, aux=0x1606))
        recs.append(
            _ev(tick + 10, SID_WRITE, reg=1, value=(freq >> 8) & 0xFF, addr=0xD401, aux=0x1609)
        )
    trace = _build_trace(recs, ram_writes=ramwr)
    res = analyze(trace)
    # The operand cell already carries the whole FREQ (no override needed), so the
    # arbiter's minimal exact model is the single captured cell -- a spurious
    # additive modulation would only lower fidelity, so it is never grafted on.
    # (Pre-Phase-2 the cascade fixed this as a base-only COMPOSITE; the MDL arbiter
    # now prefers the simpler single-cell description, still exact.)
    for addr in (0xD400, 0xD401):
        assert res[addr].get("mod") is None
    rt = round_trip(trace)
    assert rt[0xD400] == 1.0
    assert rt[0xD401] == 1.0


def test_xor_ctrl_recovers_base_eor():
    # CTRL written as ``base XOR eor`` (the defMON gate/waveform idiom): neither
    # captured cell alone reproduces it, but the exact XOR of the pair does. Gate
    # (bit 0) stays on while the waveform/sync bits move every frame, so the
    # register is a per-frame generator (not a note-gated SEQ latch).
    n = 60
    base_cell, eor_cell = 0x30, 0x31
    waveforms = [0x11, 0x21, 0x41, 0x81]  # bit0 (gate) always set
    recs = []
    ramwr = []
    for i in range(n):
        tick = _frame_cycle(i)
        base = waveforms[i % len(waveforms)]
        eor = 0x02 if i % 3 else 0x00  # sync bit toggled by flipping the eor mask
        ctrl = base ^ eor
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ, addr=0x1003))
        ramwr.append(_ra(tick + 4, base_cell, base))
        ramwr.append(_ra(tick + 5, eor_cell, eor))
        recs.append(_ev(tick + 10, SID_WRITE, reg=4, value=ctrl, addr=0xD404, aux=0x1500))
    trace = _build_trace(recs, ram_writes=ramwr)
    res = analyze(trace)
    assert res[0xD404]["type"] == "XOR", res[0xD404]["type"]
    assert {res[0xD404]["cell_a"], res[0xD404]["cell_b"]} == {base_cell, eor_cell}
    rt = round_trip(trace)
    assert rt[0xD404] == 1.0


def test_and_ctrl_recovers_wave_gate():
    # CTRL written as ``chnwave AND chngate`` (the GoatTracker2 idiom): a waveform
    # shadow cell masked by a gate cell holding $FF (pass) or $FE (force gate
    # off). Neither cell alone reproduces CTRL, but the exact AND of the pair
    # does, sampled at the SID-write instant.
    rng = np.random.default_rng(7)
    n = 60
    wave_cell, gate_cell = 0x30, 0x31
    waveforms = [0x11, 0x21, 0x41, 0x81]  # waveform+gate-bit shadow values
    recs = []
    ramwr = []
    for i in range(n):
        tick = _frame_cycle(i)
        wave = waveforms[i % len(waveforms)]
        gate = 0xFF if rng.integers(0, 2) else 0xFE  # $FE forces gate (bit0) off
        ctrl = wave & gate
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ, addr=0x1003))
        ramwr.append(_ra(tick + 4, wave_cell, wave))
        ramwr.append(_ra(tick + 5, gate_cell, gate))
        recs.append(_ev(tick + 10, SID_WRITE, reg=4, value=ctrl, addr=0xD404, aux=0x1500))
    trace = _build_trace(recs, ram_writes=ramwr)
    res = analyze(trace)
    assert res[0xD404]["type"] == "AND", res[0xD404]["type"]
    assert {res[0xD404]["cell_a"], res[0xD404]["cell_b"]} == {wave_cell, gate_cell}
    rt = round_trip(trace)
    assert rt[0xD404] == 1.0


def test_and_ctrl_recovers_wave_gate_with_onset_overrides():
    # CTRL = wave AND gate, but every note-onset frame forces a control byte the
    # shadow never carries ($08 = test/gate-off), gated by an onset cell. The AND
    # pair reproduces the steady frames; the onset frames are recovered as a
    # value-forcing override, so CTRL reconstructs byte-exact (the DMC case).
    rng = np.random.default_rng(11)
    n = 72
    wave_cell, gate_cell, onset_cell = 0x30, 0x31, 0x32
    waveforms = [0x11, 0x21, 0x41, 0x81]
    recs = []
    ramwr = []
    for i in range(n):
        tick = _frame_cycle(i)
        wave = waveforms[i % len(waveforms)]
        gate = 0xFF if rng.integers(0, 2) else 0xFE
        onset = i % 12 == 0
        ctrl = 0x08 if onset else (wave & gate)
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ, addr=0x1003))
        ramwr.append(_ra(tick + 4, wave_cell, wave))
        ramwr.append(_ra(tick + 5, gate_cell, gate))
        ramwr.append(_ra(tick + 6, onset_cell, 9 if onset else 0))
        recs.append(_ev(tick + 10, SID_WRITE, reg=4, value=ctrl, addr=0xD404, aux=0x1500))
    trace = _build_trace(recs, ram_writes=ramwr)
    res = analyze(trace)
    assert res[0xD404]["type"] == "AND", res[0xD404]["type"]
    assert res[0xD404]["overrides"], res[0xD404]
    rt = round_trip(trace)
    assert rt[0xD404] == 1.0


def _ctx_with_cells(cell_series, sid_addr=0xD404, reg_values=None):
    """A RecoverContext over a trace that writes each named cell every frame."""
    from preframr_playroutine.recover import _build_context

    n = len(next(iter(cell_series.values())))
    if reg_values is None:
        reg_values = [i % 7 for i in range(n)]
    recs = []
    ramwr = []
    for i in range(n):
        tick = _frame_cycle(i)
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ, addr=0x1003))
        for addr, series in cell_series.items():
            ramwr.append(_ra(tick + 4, addr, int(series[i])))
        recs.append(
            _ev(
                tick + 10,
                SID_WRITE,
                reg=sid_addr & 0x1F,
                value=int(reg_values[i]),
                addr=sid_addr,
                aux=0x1500,
            )
        )
    trace = _build_trace(recs, ram_writes=ramwr)
    return _build_context(trace)


def test_find_override_value_membership():
    # A force gated by a cell holding one of a few values that no single
    # equality/bit test can isolate ({2,5,7} vs the rest): _find_override must
    # recover a membership term, and _apply_overrides must evaluate it.
    from preframr_playroutine.recover import _find_override, _apply_overrides

    n = 40
    wave = np.empty(n, dtype=np.int64)
    forced = np.zeros(n, dtype=bool)
    member = [2, 5, 7]
    other = [0, 1, 3, 4, 6]
    for i in range(n):
        if i % 4 == 0:
            wave[i] = member[(i // 4) % 3]
            forced[i] = True
        else:
            wave[i] = other[i % len(other)]
    ctx = _ctx_with_cells({0x30: wave})
    terms = _find_override(forced, ctx)
    assert terms is not None
    assert any(t["kind"] == "in" and set(t["values"]) == set(member) for t in terms), terms
    out = np.zeros(n, dtype=np.int64)
    ov = {"predicate": terms, "force": 0x99}
    applied = _apply_overrides(out, [ov], ctx.sampler)
    assert np.array_equal(applied == 0x99, forced)


def test_override_descriptor_uses_membership_predicate():
    # A composite-style residual force gated by a cell holding one of a few values
    # (the per-voice waveform shadow that flags hard-restart): _override_descriptor
    # recovers it via a membership predicate, and _apply_overrides replays it.
    from preframr_playroutine.recover import _apply_overrides, _override_descriptor

    n = 48
    wave = np.empty(n, dtype=np.int64)
    forced = np.full(n, -1, dtype=np.int64)
    member = [0x41, 0x49, 0x89]
    other = [0x00, 0x09]
    for i in range(n):
        if i % 4 == 0:
            wave[i] = member[(i // 4) % 3]
            forced[i] = 0  # forced to value 0
        else:
            wave[i] = other[i % len(other)]
    ctx = _ctx_with_cells({0x30: wave})
    ov = _override_descriptor(forced, ctx)
    assert ov is not None and ov["force"] == 0, ov
    out = np.full(n, 0x77, dtype=np.int64)
    applied = _apply_overrides(out, [ov], ctx.sampler)
    assert np.array_equal(applied == 0, forced == 0)


def test_round_trip_reports_overall_and_unmodeled():
    trace = _trace_with_register([0x0F] * 30, sid_addr=0xD418)
    rt = round_trip(trace)
    assert rt["overall"] == 1.0
    assert rt["unmodeled"] == []


# -- global tuning + detune ------------------------------------------------

_VOICE_REGS = {
    0: (0xD400, 0xD401, 0xD404),
    1: (0xD407, 0xD408, 0xD40B),
    2: (0xD40E, 0xD40F, 0xD412),
}


def _sidfreq(midi, a4=440.0, cpu_hz=985248.444):
    hz = a4 * 2.0 ** ((midi - 69) / 12.0)
    return int(round(hz * (1 << 24) / cpu_hz)) & 0xFFFF


def _held_notes_trace(voice_notes, voice_a4=None, note_len=8):
    """Trace of sustained, gate-on notes per voice (16-bit freq held across frames)."""
    voice_a4 = voice_a4 or {}
    n_notes = max(len(v) for v in voice_notes.values())
    recs = []
    frame = 0
    for ni in range(n_notes):
        for fi in range(note_len):
            tick = _frame_cycle(frame)
            recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ))
            if fi == 0:
                for voice, notes in voice_notes.items():
                    if ni < len(notes):
                        flo, fhi, fctl = _VOICE_REGS[voice]
                        fv = _sidfreq(notes[ni], voice_a4.get(voice, 440.0))
                        recs.append(
                            _ev(tick + 10, SID_WRITE, reg=flo & 0x1F, value=fv & 0xFF, addr=flo)
                        )
                        recs.append(
                            _ev(
                                tick + 11,
                                SID_WRITE,
                                reg=fhi & 0x1F,
                                value=(fv >> 8) & 0xFF,
                                addr=fhi,
                            )
                        )
                        recs.append(
                            _ev(tick + 12, SID_WRITE, reg=fctl & 0x1F, value=0x11, addr=fctl)
                        )
            frame += 1
    return _build_trace(recs)


_SCALE = [48, 52, 55, 60, 64, 67, 72, 36, 79]


def _chromatic_pitch_table(base_midi, n, a4=440.0):
    """Clean chromatic note->freq ladder as ``(lo, hi)`` uint8 arrays."""
    pitch = np.array([_sidfreq(base_midi + k, a4) for k in range(n)], dtype=np.int64)
    lo = (pitch & 0xFF).astype(np.uint8)
    hi = ((pitch >> 8) & 0xFF).astype(np.uint8)
    return lo, hi


def test_recover_tuning_a440():
    tu = recover_tuning(_held_notes_trace({0: _SCALE}, voice_a4={0: 440.0}))
    assert tu is not None
    assert abs(tu["cents_from_a440"]) < 3.0
    assert tu["residual_cents"] < 3.0
    assert tu["temperament"] == "12-TET"
    assert tu["note_numbers"] == sorted(set(_SCALE))
    assert tu["note_range"] == [36, 79]
    assert tu["source"] == "live_freq"


def test_recover_tuning_note_table_source():
    trace = _held_notes_trace({0: _SCALE}, voice_a4={0: 440.0})
    lo, hi = _chromatic_pitch_table(36, 44)
    tu = recover_tuning(trace, note_tables=[(lo, hi)])
    assert tu["source"] == "note_table"
    assert abs(tu["cents_from_a440"]) < 3.0
    assert tu["temperament"] == "12-TET"
    assert tu["note_numbers"] == sorted(set(_SCALE))


def test_recover_tuning_note_table_nonchromatic_fallback():
    trace = _held_notes_trace({0: _SCALE}, voice_a4={0: 440.0})
    pitch = _pitch_table(64)
    lo = (pitch & 0xFF).astype(np.uint8)
    hi = ((pitch >> 8) & 0xFF).astype(np.uint8)
    tu = recover_tuning(trace, note_tables=[(lo, hi)])
    assert tu["source"] == "live_freq"
    assert abs(tu["cents_from_a440"]) < 3.0


def test_recover_tuning_detuned():
    a4 = 440.0 * 2.0 ** (25.0 / 1200.0)
    tu = recover_tuning(_held_notes_trace({0: _SCALE}, voice_a4={0: a4}))
    assert abs(tu["cents_from_a440"] - 25.0) < 3.0


def test_recover_tuning_insufficient():
    assert recover_tuning(_held_notes_trace({0: [60]}, note_len=2)) is None


def test_voice_detune_same_note_offset():
    sharp = 440.0 * 2.0 ** (18.0 / 1200.0)
    trace = _held_notes_trace({0: _SCALE, 1: _SCALE}, voice_a4={0: 440.0, 1: sharp})
    det = voice_detune(trace)
    assert det["detuned"] is True
    assert 14.0 < det["median_cents"] < 22.0
    assert "0-1" in det["pairs"]


def test_voice_detune_unison_not_detuned():
    trace = _held_notes_trace({0: _SCALE, 1: _SCALE}, voice_a4={0: 440.0, 1: 440.0})
    assert voice_detune(trace)["detuned"] is False


def test_analyze_includes_tuning_and_detune():
    result = analyze(_held_notes_trace({0: _SCALE}))
    assert result["tuning"] is not None and "a4_hz" in result["tuning"]
    assert "note_numbers" in result["tuning"]
    assert "source" in result["tuning"]
    assert "detune" in result and "detuned" in result["detune"]


# -- _table_walk_scan optimization parity ---------------------------------

from preframr_playroutine.recover import (  # noqa: E402
    RecoverContext,
    StateSequence,
    _anchor_positions,
    _bits_set,
    _table_walk_scan,
    _varying_bits,
)


def _ref_score_cursor(series, cur, ram, lo, hi, mask, n):
    """Frozen pre-optimization copy of recover._score_cursor."""
    span = hi - lo + 1
    if int(cur.max()) >= span + 2:
        return -1.0, 0
    best_res, best_off = -1.0, 0
    for off in (-2, -1, 0, 1, 2):
        idx = cur + off
        ok = (idx >= 0) & (lo + idx <= hi)
        if int(ok.sum()) < n * 0.8:
            continue
        tv = ram[lo + np.clip(idx, 0, span - 1)]
        if len(np.unique(tv[ok] & mask)) < 2:
            continue
        res = float(np.mean((series[ok] & mask) == (tv[ok] & mask)))
        if res > best_res:
            best_res, best_off = res, off
    return best_res, best_off


def _ref_table_walk_scan(series, ctx, min_res=0.8, max_bases=96):
    """Frozen pre-optimization copy of recover._table_walk_scan."""
    ram = ctx.ram
    if ram is None or not ctx.cursor_cols:
        return None
    series = np.asarray(series, dtype=np.int64).ravel()
    vbits = _varying_bits(series)
    if _bits_set(vbits) < 2:
        return None
    grid = ctx.stateseq.grid.astype(np.int64)
    n = len(series)
    best = None
    best_res = min_res
    for mask in (0xFE, 0xFF):
        if _bits_set(vbits & mask) < 2:
            continue
        anchor = _anchor_positions(series, ram & mask, mask)
        if anchor is None:
            continue
        anchor_frame, positions = anchor
        for j in ctx.cursor_cols:
            cur = grid[:, j]
            cmax = int(cur.max())
            bases = positions - int(cur[anchor_frame])
            bases = bases[(bases >= 0) & (bases + cmax < len(ram))]
            for base in bases[:max_bases]:
                res, off = _ref_score_cursor(series, cur, ram, int(base), int(base) + cmax, mask, n)
                if res >= best_res:
                    best_res = res
                    best = (res, int(base), int(base) + cmax, int(ctx.stateseq.addrs[j]), off, mask)
        if best is not None and best[0] >= 0.999:
            break
    if best is None:
        return None
    res, lo, hi, cursor, off, mask = best
    return {
        "type": "TABLE_WALK",
        "base": int(lo),
        "stride": 1,
        "length": int(hi - lo + 1),
        "loop": 0,
        "table": ram[lo : hi + 1].copy(),
        "mask": int(mask),
        "cursor_addr": int(cursor),
        "cursor_offset": int(off),
        "residual": float(res),
    }


def _make_scan_ctx(rng):
    """Synthetic RecoverContext exercising the no-read-log table-walk scan.

    ``ram`` is mostly zero with a primary table plus duplicate decoy tables (to
    force residual ties across bases) and sparse noise (to vary anchor counts).
    """
    ram = np.zeros(65536, dtype=np.uint8)
    noise_pos = rng.integers(0, 65536, size=int(rng.integers(200, 1200)))
    ram[noise_pos] = rng.integers(1, 256, size=len(noise_pos)).astype(np.uint8)
    length = int(rng.integers(4, 24))
    table = rng.integers(0, 256, size=length).astype(np.uint8)
    base = int(rng.integers(0x1000, 0xF000))
    ram[base : base + length] = table
    for _ in range(int(rng.integers(0, 3))):
        dbase = int(rng.integers(0x1000, 0xF000))
        ram[dbase : dbase + length] = table
    n = int(rng.integers(200, 700))
    n_cols = int(rng.integers(3, 7))
    cols, addrs = [], []
    for _ in range(n_cols):
        cmax = min(length - 1, int(rng.integers(2, length)))
        cols.append(rng.integers(0, cmax + 1, size=n))
        addrs.append(int(rng.integers(0xC000, 0xCFFF)))
    grid = np.stack(cols, axis=1)
    stateseq = StateSequence(ticks=np.arange(n), addrs=np.asarray(addrs, dtype=np.int64), grid=grid)
    ctx = RecoverContext(
        kind="auto",
        stateseq=stateseq,
        ram=ram,
        tables=[],
        cursor_cols=list(range(n_cols)),
        note_on={},
        all_on=[],
        n_frames=n,
        sampler=None,
    )
    return ctx, base, table, length


def _series_from_walk(ctx, base, length, rng, noisy):
    """Series = ram[base + cursor] & mask for a random cell/mask, maybe noised."""
    grid = ctx.stateseq.grid.astype(np.int64)
    j = int(rng.integers(0, grid.shape[1]))
    cur = grid[:, j]
    cur = np.clip(cur, 0, length - 1)
    mask = int(rng.choice([0xFE, 0xFF]))
    series = (ctx.ram[base + cur].astype(np.int64)) & mask
    if noisy:
        flip = rng.random(len(series)) < rng.uniform(0.1, 0.5)
        series[flip] = rng.integers(0, 256, size=int(flip.sum()))
    return series


def _walk_eq(a, b):
    if a is None or b is None:
        return a is None and b is None
    if set(a) != set(b):
        return False
    for key, val in a.items():
        if key == "table":
            if not np.array_equal(val, b[key]):
                return False
        elif val != b[key]:
            return False
    return True


@pytest.mark.parametrize("seed", range(8))
def test_table_walk_scan_parity(seed):
    rng = np.random.default_rng(1000 + seed)
    for _ in range(6):
        ctx, base, table, length = _make_scan_ctx(rng)
        noisy = bool(rng.integers(0, 2))
        if rng.integers(0, 4) == 0:
            series = rng.integers(0, 256, size=ctx.n_frames)  # fully random
        else:
            series = _series_from_walk(ctx, base, length, rng, noisy)
        ref = _ref_table_walk_scan(series, ctx)
        opt = _table_walk_scan(series, ctx)
        assert _walk_eq(ref, opt), (seed, ref, opt)


# -- ping-pong (clamp-and-flip) BACC --------------------------------------

from preframr_playroutine.recover import (  # noqa: E402
    _recon_bacc_full,
    _recur_updown,
    _simulate_pingpong,
    _simulate_reflect,
)


def test_fit_bacc_pingpong_clamp_flip():
    # Asymmetric clamp-and-flip sweep (defMON PW): saturates at the bounds and
    # reverses, with different up/down step magnitudes.
    lo, hi, up, down = 10, 250, 5, 7
    series, _ = _simulate_pingpong(lo, hi, lo, hi, up, down, lo, 1, 600)
    fit = fit_bacc(series)
    assert fit is not None
    assert fit["type"] == "BACC"
    assert fit["mode"] == "pingpong"
    assert fit["step"] == up
    assert fit["down_step"] == down
    assert fit["lo"] == lo and fit["hi"] == hi
    assert fit["clamp_lo"] == lo and fit["clamp_hi"] == hi
    assert fit["residual"] >= 0.99


def test_reconstruct_bacc_pingpong_roundtrip():
    lo, hi, up, down = 10, 250, 5, 7
    n = 600
    series, _ = _simulate_pingpong(lo, hi, lo, hi, up, down, lo, 1, n)
    fit = fit_bacc(series)
    assert fit is not None and fit["mode"] == "pingpong"
    desc = dict(fit, byte_role="full")
    recon = reconstruct_register(desc, _ticks(n))
    assert np.array_equal(recon, series)


def test_recur_updown_per_note_roundtrip():
    # Two notes with different seeds and rates, reseeded; the unified up/down
    # clamp-flip fit keeps a per-segment step/direction and regenerates exactly.
    s1, _ = _simulate_pingpong(0, 200, 0, 200, 4, 6, 0, 1, 150)
    s2, _ = _simulate_pingpong(0, 200, 0, 200, 8, 5, 50, 1, 150)
    series = np.concatenate([s1, s2])
    fit = _recur_updown(series, _segment_bounds(series, [0, 150]))
    assert fit is not None
    assert fit["mode"] == "pingpong"
    assert fit["clamp_lo"] == 0 and fit["clamp_hi"] == 200
    recon = _recon_bacc_full(fit, len(series))
    assert np.array_equal(recon, series)
    desc = dict(fit, byte_role="full")
    recon2 = reconstruct_register(desc, _ticks(len(series)))
    assert np.array_equal(recon2, series)


def test_pingpong_does_not_steal_mirror_reflect():
    # A true mirror reflect (overshoot mirrored, not clamped) must still fit as
    # reflect -- the new clamp mode must not over-fire on clean reflect/saw/wrap.
    series, _ = _simulate_reflect(0, 20, 3, 0, 1, 400)
    fit = fit_bacc(np.asarray(series))
    assert fit is not None
    assert fit["mode"] == "reflect"


# -- tick-banded (table-indexed-stride) reflect BACC ----------------------

from preframr_playroutine.recover import (  # noqa: E402
    _recon_tickband,
    _recur_table,
    _simulate_tickreflect,
)


def _tickband_series(rate, lo, hi, seed, n_notes, note_len, direction=1):
    """Concatenated tick-banded reflect notes: stride = rate[tick], reseed each note.

    Each note reseeds to ``seed`` at tick 0 and steps by ``rate[tick]`` per frame
    (Future Composer PW), mirror-reflecting at ``lo``/``hi``. Returns the 16-bit
    series and the per-note reset frames.
    """
    rate = np.asarray(rate, dtype=np.int64)
    series, resets = [], []
    for _ in range(n_notes):
        resets.append(len(series))
        seg = _simulate_tickreflect(lo, hi, rate, seed, direction, note_len)
        series.extend(int(x) for x in seg)
    return np.array(series, dtype=np.int64), resets


def test_recur_table_recovers_rate_table():
    # Stride varies with the tick (96,96,64,64,96,128...) and reflects at the PW
    # bounds; the fitter recovers a single shared rate table and reconstructs the
    # whole reseeded series exactly.
    rate = [96, 96, 64, 64, 96, 128, 128, 128, 128, 128]
    series, resets = _tickband_series(rate, 1536, 2592, 1536, 12, 14)
    fit = _recur_table(series, _segment_bounds(series, resets))
    assert fit is not None
    assert fit["type"] == "BACC"
    assert fit["mode"] == "tickband"
    assert fit["segmented"] is True
    assert len(fit["rate_tables"]) == 1  # one shared program across all notes
    assert fit["lo"] == 1536 and fit["hi"] == 2592
    assert fit["residual"] == 1.0
    recon = _recon_tickband(fit, len(series))
    assert np.array_equal(recon, series)
    # The reflect was actually exercised (the long notes turn around at the top).
    assert series.max() == 2592 and np.any(np.diff(series) < 0)


def test_classify_tickband_pw_roundtrip():
    # A full PW LO/HI register pair (D402/D403) driven by a tick-banded sweep
    # classifies as a tickband BACC and round-trips exactly on both bytes.
    rate = [96, 96, 64, 64, 96, 128, 128, 128, 128, 128]
    note_len = 14
    pw, resets = _tickband_series(rate, 1536, 2592, 1536, 12, note_len)
    lo = pw & 0xFF
    hi = (pw >> 8) & 0xFF
    n = len(pw)
    reset_set = set(resets)
    recs = []
    for i in range(n):
        tick = _frame_cycle(i)
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ))
        is_last = (i + 1) in reset_set or i == n - 1
        ctrl = 0x40 if is_last else 0x41  # gate off at note end -> note_on next frame
        recs.append(_ev(tick + 8, SID_WRITE, reg=4, value=ctrl, addr=0xD404, aux=0x1500))
        recs.append(_ev(tick + 12, SID_WRITE, reg=2, value=int(lo[i]), addr=0xD402, aux=0x1388))
        recs.append(_ev(tick + 12, SID_WRITE, reg=3, value=int(hi[i]), addr=0xD403, aux=0x1390))
    trace = _build_trace(recs)
    res = classify_register(trace, 0xD402)
    assert res["type"] == "BACC"
    assert res["mode"] == "tickband"
    rt = round_trip(trace)
    assert rt[0xD402] == 1.0
    assert rt[0xD403] == 1.0


def test_tickband_does_not_steal_constant_step():
    # A constant per-note stride is a plain reflect, NOT tick-banded: the table
    # mode must reject it (no within-note stride variation) and leave it to the
    # scalar reflect/saw modes.
    series, resets = _tickband_series([6] * 14, 0, 240, 0, 12, 16)
    assert _recur_table(series, _segment_bounds(series, resets)) is None
    plain, _ = _simulate_reflect(0, 240, 6, 0, 1, 400)
    fit = fit_bacc(np.asarray(plain))
    assert fit is not None and fit["mode"] == "reflect"


def test_tickband_rejects_noise():
    # Per-note unique diff vectors (noise) are not a reused program: rejected by
    # the shared-table guard.
    rng = np.random.default_rng(5)
    series = rng.integers(0, 4096, size=600).astype(np.int64)
    resets = list(range(0, 600, 6))
    assert _recur_table(series, _segment_bounds(series, resets)) is None


def test_or_modevol_cell_or_const():
    # MODE/VOL ($D418) blitted as ``volume | filter_mode`` where the mode nibble is
    # a constant ($10) and the volume is a captured cell (JCH idiom). Neither a
    # single feeder nor a table walk reproduces it; ``cell | const`` does. The
    # volume cell is not written for the first few frames, so the register holds its
    # note-on seed ($1F) there -- exercising the held-seed prelude.
    rng = np.random.default_rng(3)
    n = 60
    vol_cell, mode = 0x30, 0x10
    seed = mode | 0x0F  # $1F held until the volume cell starts updating
    prelude_frames = 5
    recs = []
    ramwr = []
    for i in range(n):
        tick = _frame_cycle(i)
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ, addr=0x1003))
        if i < prelude_frames:
            value = seed
        else:
            vol = int(rng.integers(1, 16))  # moving volume nibble, not an accumulator
            ramwr.append(_ra(tick + 4, vol_cell, vol))
            value = mode | vol
        recs.append(_ev(tick + 20, SID_WRITE, reg=0x18, value=value, addr=0xD418, aux=0x1500))
    trace = _build_trace(recs, ram_writes=ramwr)
    res = analyze(trace)
    assert res[0xD418]["type"] == "OR", res[0xD418]["type"]
    assert res[0xD418]["cell_a"] == vol_cell
    assert res[0xD418]["const"] == mode
    assert res[0xD418]["prelude_end"] == prelude_frames
    rt = round_trip(trace)
    assert rt[0xD418] == 1.0


def test_or_modevol_cell_pair():
    # MODE/VOL ($D418) blitted as ``mode_cell | volume_cell``: a moving filter-mode
    # hi-nibble OR-ed with a moving volume lo-nibble. Neither cell alone reproduces
    # it, but the exact OR of the pair does, sampled at the SID-write instant.
    rng = np.random.default_rng(5)
    n = 60
    mode_cell, vol_cell = 0x30, 0x31
    modes = [0x10, 0x20, 0x30]
    recs = []
    ramwr = []
    for i in range(n):
        tick = _frame_cycle(i)
        mode = modes[int(rng.integers(0, len(modes)))]
        vol = int(rng.integers(1, 16))
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ, addr=0x1003))
        ramwr.append(_ra(tick + 4, mode_cell, mode))
        ramwr.append(_ra(tick + 5, vol_cell, vol))
        recs.append(_ev(tick + 20, SID_WRITE, reg=0x18, value=mode | vol, addr=0xD418, aux=0x1500))
    trace = _build_trace(recs, ram_writes=ramwr)
    res = analyze(trace)
    assert res[0xD418]["type"] == "OR", res[0xD418]["type"]
    assert {res[0xD418]["cell_a"], res[0xD418]["cell_b"]} == {mode_cell, vol_cell}
    rt = round_trip(trace)
    assert rt[0xD418] == 1.0


def _cutoff_image(store_pc, scale_byte=0xEA, base=0x02, reg_byte=0x16):
    """A 64K RAM image carrying the defMON cutoff micro-routine signature."""
    img = np.zeros(0x10000, dtype=np.uint8)
    p = store_pc
    img[p], img[p + 1], img[p + 2] = 0x8D, reg_byte, 0xD4  # sta $d4xx
    img[p - 32] = 0xA9  # lda #lo
    img[p - 30] = 0x18  # clc
    img[p - 29] = 0x69  # adc/sbc #steplo (op cell default)
    img[p - 27] = 0x8D  # sta lo
    img[p - 24] = 0xA9  # lda #hi
    img[p - 22] = 0x69  # adc/sbc #stephi (op cell default)
    img[p - 20] = 0x10  # bpl
    img[p - 15] = 0x8D  # sta hi
    img[p - 12] = 0x69  # adc #imm
    img[p - 10] = 0x30  # bmi
    img[p - 8] = 0xC9  # cmp #base
    img[p - 7] = base  # bound-load / clamp base cell
    img[p - 1] = scale_byte  # nop (x1) / asl (x2)
    return img


def _cutoff_sim(op, slo, shi, imm, seed_lo, seed_hi, base, scale):
    """6502-accurate frame loop for the cutoff routine -> stored (lo, hi, d416).

    ``op``/``slo``/``shi``/``imm`` are per-frame operand values. Returns the arrays
    the player would store to the accumulator cells and the SID each frame.
    """
    lo, hi = seed_lo, seed_hi
    los, his, outs = [], [], []
    for i in range(len(op)):
        ilo, ihi = lo, hi
        if op[i] == 0x69:
            s = ilo + slo[i]
            carry_lo = 1 if s > 0xFF else 0
            lo = s & 0xFF
            s2 = ihi + shi[i] + carry_lo
            carry_hi = 1 if s2 > 0xFF else 0
            hi = s2 & 0xFF
        else:
            s = ilo - slo[i] - 1
            carry_lo = 1 if s >= 0 else 0
            lo = s & 0xFF
            s2 = ihi - shi[i] - (1 - carry_lo)
            carry_hi = 1 if s2 >= 0 else 0
            hi = s2 & 0xFF
        a = (hi + imm[i] + carry_hi) & 0xFF
        emit = base if (a < base or a >= 0x80) else a
        los.append(lo)
        his.append(hi)
        outs.append((emit * scale) & 0xFF)
    return np.array(los), np.array(his), np.array(outs)


def _cutoff_cells_for(store_pc):
    return {
        "lo": store_pc - 31,
        "op_lo": store_pc - 29,
        "slo": store_pc - 28,
        "hi": store_pc - 23,
        "op_hi": store_pc - 22,
        "shi": store_pc - 21,
        "imm": store_pc - 11,
        "base": store_pc - 7,
        "scale": store_pc - 1,
    }


def test_cutoff_opcode_directed_sweep():
    # Opcode-directed 16-bit accumulator (ADC up / SBC down triangle) emitted as
    # clamp(hi + imm + carry) with a per-instrument imm (SEQ filter base). A whole
    # number of periods (with warm-up) makes the frame-0 predecessor consistent, so
    # the recovered CUTOFF descriptor regenerates every frame.
    store_pc = 0x1200
    base, scale = 0x02, 1
    period = 60  # 30 up + 30 down
    warm, n = period, period * 3
    total = warm + n
    op = np.empty(total, dtype=np.int64)
    slo = np.empty(total, dtype=np.int64)
    shi = np.empty(total, dtype=np.int64)
    imm = np.empty(total, dtype=np.int64)
    for i in range(total):
        ph = i % period
        if ph < period // 2:  # ADC: hi += 1, lo stays 0
            op[i], slo[i], shi[i] = 0x69, 0x00, 0x01
        else:  # SBC: hi -= 1 (slo=0xFF keeps lo at 0), carry gives the +1 emit bias
            op[i], slo[i], shi[i] = 0xE9, 0xFF, 0x00
        imm[i] = 0 if (i // period) % 2 == 0 else 5  # SEQ base steps per "instrument"
    los, his, outs = _cutoff_sim(op, slo, shi, imm, 0x00, 0x14, base, scale)
    cells = _cutoff_cells_for(store_pc)
    recs, ramwr = [], []
    for j in range(n):
        i = warm + j
        tick = _frame_cycle(j)
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ, addr=0x1003))
        ramwr.append(_ra(tick + 4, cells["lo"], int(los[i])))
        ramwr.append(_ra(tick + 5, cells["hi"], int(his[i])))
        ramwr.append(_ra(tick + 6, cells["op_lo"], int(op[i])))
        ramwr.append(_ra(tick + 7, cells["op_hi"], int(op[i])))
        ramwr.append(_ra(tick + 8, cells["slo"], int(slo[i])))
        ramwr.append(_ra(tick + 9, cells["shi"], int(shi[i])))
        ramwr.append(_ra(tick + 10, cells["imm"], int(imm[i])))
        recs.append(
            _ev(tick + 20, SID_WRITE, reg=0x16, value=int(outs[i]), addr=0xD416, aux=store_pc)
        )
    trace = _build_trace(recs, ram_writes=ramwr, ram=_cutoff_image(store_pc))
    res = analyze(trace)
    assert res[0xD416]["type"] == "CUTOFF", res[0xD416]["type"]
    assert res[0xD416]["cells"]["hi"] == cells["hi"]
    # both emit levels present (clamp not hit): the sweep really moved.
    assert len(np.unique(outs[warm:])) > 3
    rt = round_trip(trace)
    assert rt[0xD416] == 1.0


def test_cutoff_scale_x2_model():
    # 6581 SID-model post-scale ($10d4 = asl) doubles the emitted cutoff.
    store_pc = 0x1200
    base, scale = 0x02, 2
    warm, period, n = 40, 40, 80
    total = warm + n
    op = np.full(total, 0x69, dtype=np.int64)
    slo = np.zeros(total, dtype=np.int64)
    shi = np.where((np.arange(total) % period) < period // 2, 1, 0).astype(np.int64)
    shi[(np.arange(total) % period) >= period // 2] = 0
    # ramp up then hold: keep hi in a small band that x2 leaves < 0x80.
    imm = np.zeros(total, dtype=np.int64)
    los, his, outs = _cutoff_sim(op, slo, shi, imm, 0x00, 0x08, base, scale)
    cells = _cutoff_cells_for(store_pc)
    recs, ramwr = [], []
    for j in range(n):
        i = warm + j
        tick = _frame_cycle(j)
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ, addr=0x1003))
        for off, arr in (
            ("lo", los),
            ("hi", his),
            ("slo", slo),
            ("shi", shi),
            ("imm", imm),
        ):
            ramwr.append(_ra(tick + 4, cells[off], int(arr[i])))
        ramwr.append(_ra(tick + 5, cells["op_lo"], int(op[i])))
        ramwr.append(_ra(tick + 6, cells["op_hi"], int(op[i])))
        recs.append(
            _ev(tick + 20, SID_WRITE, reg=0x16, value=int(outs[i]), addr=0xD416, aux=store_pc)
        )
    trace = _build_trace(recs, ram_writes=ramwr, ram=_cutoff_image(store_pc, scale_byte=0x0A))
    res = analyze(trace)
    assert res[0xD416]["type"] == "CUTOFF"
    assert res[0xD416]["scale"] == 2
    assert round_trip(trace)[0xD416] == 1.0


def test_cutoff_signature_gate_rejects_non_routine():
    # A corrupted store-site opcode (no cutoff routine) must not classify as CUTOFF;
    # the register falls through to the generic paths without error.
    store_pc = 0x1200
    img = _cutoff_image(store_pc)
    img[store_pc - 10] = 0x00  # break the BMI-clamp signature byte
    op = np.full(40, 0x69, dtype=np.int64)
    slo = np.zeros(40, dtype=np.int64)
    shi = np.ones(40, dtype=np.int64)
    imm = np.zeros(40, dtype=np.int64)
    los, his, outs = _cutoff_sim(op, slo, shi, imm, 0x00, 0x10, 0x02, 1)
    cells = _cutoff_cells_for(store_pc)
    recs, ramwr = [], []
    for i in range(40):
        tick = _frame_cycle(i)
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ, addr=0x1003))
        ramwr.append(_ra(tick + 4, cells["hi"], int(his[i])))
        recs.append(
            _ev(tick + 20, SID_WRITE, reg=0x16, value=int(outs[i]), addr=0xD416, aux=store_pc)
        )
    trace = _build_trace(recs, ram_writes=ramwr, ram=img)
    assert analyze(trace)[0xD416]["type"] != "CUTOFF"


def test_sampler_operand_pre_store():
    # operand() returns the cell write immediately before the register's store
    # instant (the pre-update value the routine read), while at_write() returns the
    # post-update store -- the distinction that makes the cutoff carry exact across
    # a note reseed (which writes the accumulator cell earlier in the frame).
    from preframr_playroutine.recover import _CellSampler

    cell, sid = 0x40, 0xD416
    recs, ramwr = [], []
    for i in range(5):
        tick = _frame_cycle(i)
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ, addr=0x1003))
        ramwr.append(_ra(tick + 4, cell, 10 + i))  # reseed / carried operand
        ramwr.append(_ra(tick + 8, cell, 100 + i))  # routine store (post-update)
        recs.append(_ev(tick + 20, SID_WRITE, reg=0x16, value=0, addr=sid, aux=0x1200))
    trace = _build_trace(recs, ram_writes=ramwr)
    samp = _CellSampler(trace, trace.tick_cycles("auto"))
    assert list(samp.at_write(cell, sid)) == [100, 101, 102, 103, 104]
    assert list(samp.operand(cell, sid)) == [10, 11, 12, 13, 14]


# -- step x boundary product fitter (all 12 cells, randomized params) ------

from preframr_playroutine import ir as _ir  # noqa: E402

# Diverse, deterministic parameterizations proven to round-trip per product cell
# (a fitter that only recovers the HVSC magic constants fails these). Scalar cells
# carry (lo, hi, up, |down|); table cells carry (lo, hi, rate) with a per-note
# tick ramp settling to a constant tail (the FC/tickband regime). Each cell has a
# small fixed set of variants -- the anti-overfitting mechanism (section 5b).
_SCALAR_VARIANTS = [(0, 48, 6, 8), (0, 60, 5, 10), (0, 120, 8, 12)]
_TABLE_VARIANTS = {
    "wrap": [
        (0, 48, [4, 8, 12, 24, 24, 24, 24, 24, 24, 24]),
        (8, 104, [16, 32, 48, 48, 48, 48, 48]),
        (0, 96, [8, 16, 24, 48, 48, 48, 48, 48]),
    ],
    "saw": [
        (0, 48, [4, 8, 12, 24, 24, 24, 24, 24, 24, 24]),
        (0, 60, [10, 20, 20, 20, 20, 20, 20, 20]),
        (0, 96, [16, 16, 32, 48, 48, 48, 48, 48]),
    ],
    "reflect": [
        (0, 48, [4, 8, 12, 24, 24, 24, 24, 24, 24, 24]),
        (8, 104, [16, 32, 48, 48, 48, 48, 48]),
        (0, 84, [12, 12, 24, 24, 24, 24, 24]),
    ],
    "clampflip": [
        (0, 48, [4, 8, 12, 24, 24, 24, 24, 24, 24, 24]),
        (0, 60, [10, 20, 20, 20, 20, 20, 20, 20]),
        (0, 96, [16, 16, 32, 48, 48, 48, 48, 48]),
    ],
}
_STEP_KINDS = ("const", "updown", "table")
_BOUNDARIES = ("wrap", "saw", "reflect", "clampflip")


def _product_cases():
    for step_kind in _STEP_KINDS:
        for boundary in _BOUNDARIES:
            variants = _TABLE_VARIANTS[boundary] if step_kind == "table" else _SCALAR_VARIANTS
            for vi, spec in enumerate(variants):
                yield pytest.param(step_kind, boundary, spec, id=f"{step_kind}-{boundary}-{vi}")


@pytest.mark.parametrize("step_kind,boundary,spec", list(_product_cases()))
def test_product_fitter_recovers_cell(step_kind, boundary, spec):
    # The unified recurrence fitter recovers EVERY cell of the
    # {const,updown,table} x {wrap,saw,reflect,clampflip} product on a reseeded
    # synthetic series and reconstructs it frame-exact. Reconstruction (not a
    # magic-value match) is the anti-overfitting gate across the diverse variants.
    if step_kind == "table":
        lo, hi, rate = spec
        up = down = 0
        seeds = [lo] * 10
    else:
        lo, hi, up, down = spec
        rate = None
        seeds = [lo + (k * up) % (hi - lo) for k in range(10)]
    series, resets = recur_series(lo, hi, seeds, 30, boundary, step_kind, up, down, rate)
    # Feed the known note bounds so each segment is intact (the product fitter's
    # per-segment job); segmentation is exercised separately by the real-trace
    # tests. A wrap/saw reset mid-note otherwise reads as a discontinuity cut.
    bounds = list(resets) + [len(series)]
    desc = _recur_product(series, bounds)
    assert desc is not None
    assert desc["type"] == "BACC" and desc["mode"] == "product"
    recon = _ir._recon_product(desc, len(series))  # noqa: SLF001
    assert np.array_equal(recon, series), (step_kind, boundary, spec)
    # The boundary was genuinely exercised (the sweep turns / resets, not a short
    # monotone ramp) -- so the cell is really tested, not trivially reproduced.
    assert np.any(np.diff(series) < 0)
    # Round-trips through the public reconstruction path (to_ir -> evaluate).
    rt = reconstruct_register(desc, _ticks(len(series)))
    assert np.array_equal(rt, series)


def test_product_table_step_genuinely_varies():
    # A table-step cell must recover a per-tick VARYING stride, not a constant
    # (which the scalar modes already cover) -- proves the table axis is real.
    lo, hi, rate = _TABLE_VARIANTS["reflect"][1]
    series, resets = recur_series(lo, hi, [lo] * 8, 30, "reflect", "table", 0, 0, rate)
    desc = _recur_product(series, list(resets) + [len(series)])
    assert desc["step_kind"] == "table"
    table = np.asarray(desc["rate_tables"][0])
    assert len(np.unique(table[table != 0])) > 1


def test_product_fitter_loses_to_constant_step_on_cost():
    # On a clean constant-step reflect the product cell must not out-score the
    # scalar recurrence: the rate-table capture is MDL-charged, so a per-frame
    # stride replay is more expensive than the closed-form generator.
    series, resets = recur_series(0, 60, [0, 20, 40, 10, 30], 40, "reflect", "const", 5, 5, None)
    bounds = _segment_bounds(series, resets)
    scalar = _recur_const(series, bounds)
    product = _recur_product(series, bounds)
    assert scalar is not None and product is not None
    # Both reconstruct the series, but the product (captured rate tables) costs more.
    n = len(series)
    assert _ir.complexity(_ir.to_ir(product), None, n) > _ir.complexity(_ir.to_ir(scalar), None, n)


# -- unified binop proposer (add / sub / or / and / xor) -------------------


def _binop_reg_trace(target, cells, n=60, sid_addr=0xD418):
    """Trace where ``target[i]`` is written to ``sid_addr`` each frame and every
    ``(addr, series)`` in ``cells`` is a RAM feeder written just before the store."""
    recs, ramwr = [], []
    for i in range(n):
        tick = _frame_cycle(i)
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ, addr=0x1003))
        for off, (addr, ser) in enumerate(cells):
            ramwr.append(_ra(tick + 4 + off, addr, int(ser[i])))
        recs.append(
            _ev(
                tick + 20,
                SID_WRITE,
                reg=sid_addr & 0x1F,
                value=int(target[i]) & 0xFF,
                addr=sid_addr,
                aux=0x1500,
            )
        )
    return _build_trace(recs, ram_writes=ramwr)


# A control-like operand carries only a handful of distinct values (the binop
# prefilter -- like the former pair searches -- scans low-entropy cells).
_LOWENT = [0x02, 0x10, 0x12, 0x20, 0x22, 0x30, 0x32, 0x40, 0x42, 0x50, 0x52, 0x60]


def test_binop_add_cell_const():
    # $D418 written as ``cell + const`` (an additive bias on a low-entropy byte):
    # neither the cell nor a single feeder reproduces it, but the exact 8-bit add
    # with a carry (so it is not merely an OR) does.
    rng = np.random.default_rng(1)
    n = 60
    base = np.array([_LOWENT[int(rng.integers(0, len(_LOWENT)))] for _ in range(n)])
    target = (base + 0x1F) & 0xFF  # low-bit carry -> a genuine add, not an OR
    trace = _binop_reg_trace(target, [(0x40, base)], n=n)
    res = analyze(trace)
    assert res[0xD418]["type"] == "ADD", res[0xD418]["type"]
    assert res[0xD418]["cell_a"] == 0x40 and res[0xD418]["const"] == 0x1F
    assert round_trip(trace)[0xD418] == 1.0


def test_binop_add_cell_cell():
    # $D418 written as ``cellA + cellB`` (two low-entropy operands summed with
    # carry -- distinct from an OR of the same pair).
    rng = np.random.default_rng(2)
    n = 60
    a = np.array([_LOWENT[int(rng.integers(0, len(_LOWENT)))] for _ in range(n)])
    b = np.array([(0x0C, 0x14, 0x1C)[int(rng.integers(0, 3))] for _ in range(n)])
    target = (a + b) & 0xFF
    trace = _binop_reg_trace(target, [(0x40, a), (0x41, b)], n=n)
    res = analyze(trace)
    assert res[0xD418]["type"] == "ADD", res[0xD418]["type"]
    assert {res[0xD418]["cell_a"], res[0xD418]["cell_b"]} == {0x40, 0x41}
    assert round_trip(trace)[0xD418] == 1.0


def test_binop_sub_cell_cell():
    # $D418 written as ``cellA - cellB``: subtraction of two cells is the one fold
    # an add cannot express (a const sub folds into an add mod 256), so SUB wins.
    rng = np.random.default_rng(4)
    n = 60
    a = np.array([_LOWENT[int(rng.integers(6, len(_LOWENT)))] for _ in range(n)])  # larger
    b = np.array([(0x02, 0x10, 0x12, 0x20)[int(rng.integers(0, 4))] for _ in range(n)])
    target = (a - b) & 0xFF
    trace = _binop_reg_trace(target, [(0x40, a), (0x41, b)], n=n)
    res = analyze(trace)
    assert res[0xD418]["type"] == "SUB", res[0xD418]["type"]
    assert res[0xD418]["cell_a"] == 0x40 and res[0xD418]["cell_b"] == 0x41
    assert round_trip(trace)[0xD418] == 1.0


def test_binop_add_sub_randomized_constants():
    # Parameter randomization: the add/sub const search recovers the offset from
    # the residual histogram across many constants (not a hardcoded magic value).
    for seed, const in ((10, 0x07), (11, 0x1D), (12, 0x25), (13, 0x2B)):
        rng = np.random.default_rng(seed)
        n = 64
        base = np.array([_LOWENT[int(rng.integers(0, len(_LOWENT)))] for _ in range(n)])
        target = (base + const) & 0xFF
        trace = _binop_reg_trace(target, [(0x40, base)], n=n)
        res = analyze(trace)
        assert res[0xD418]["type"] in ("ADD", "SUB"), (const, res[0xD418]["type"])
        assert round_trip(trace)[0xD418] == 1.0


# -- read-log dataflow narrowing (Phase 4) --------------------------------

from preframr_playroutine.recover import (  # noqa: E402
    _build_context,
    _propose_binop,
    _reads_near_store,
    _window_cells,
)


def _narrowing_trace(target, feeders, noise, n=64, sid_addr=0xD418, with_reads=True):
    """Trace whose store reads ``feeders`` in its local slice; ``noise`` cells

    change every frame but are written only mid-frame, after the store (outside
    the backward-slice window). With ``with_reads`` a read log records the feeder
    reads; otherwise the trace has no read log (the CI fallback)."""
    recs, ramwr, ramrd = [], [], []
    for i in range(n):
        tick = _frame_cycle(i)
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ, addr=0x1003))
        for off, (addr, ser) in enumerate(feeders):
            ramwr.append(_ra(tick + 4 + off, addr, int(ser[i])))  # feeder computed
            ramrd.append(_ra(tick + 10 + off, addr, int(ser[i])))  # LDA feeder
        recs.append(
            _ev(
                tick + 20,
                SID_WRITE,
                reg=sid_addr & 0x1F,
                value=int(target[i]) & 0xFF,
                addr=sid_addr,
                aux=0x1500,
            )
        )
        for off, (addr, ser) in enumerate(noise):
            ramwr.append(_ra(tick + 300 + off, addr, int(ser[i])))  # unrelated bookkeeping
    evs = np.array(recs, dtype=EVENT_DTYPE)
    evs.sort(order="cycle", kind="stable")
    kwargs = {"ramwr": np.array(ramwr, dtype=RAMACCESS_DTYPE)}
    if with_reads:
        kwargs["ramrd"] = np.array(ramrd, dtype=RAMACCESS_DTYPE)
    return Trace.from_events(evs, PAL_META, **kwargs)


def test_window_cells_vectorized():
    # A cell whose access sits within the window before a store is kept; one
    # outside it is dropped -- no per-store loop, pure searchsorted.
    stores = np.array([1000, 2000, 3000], dtype=np.int64)
    acc_c = np.array([950, 400, 1990, 2600], dtype=np.int64)  # deltas 50,600,10,400
    acc_a = np.array([0x10, 0x11, 0x12, 0x13], dtype=np.uint16)
    got = _window_cells(stores, acc_c, acc_a, 512)
    assert got == {0x10, 0x12, 0x13}  # 0x11 is 600 cycles before -> excluded


def test_reads_near_store_narrows_candidates():
    rng = np.random.default_rng(7)
    n = 80
    a = np.array([_LOWENT[int(rng.integers(0, len(_LOWENT)))] for _ in range(n)])
    b = np.array([(0x0C, 0x14, 0x1C)[int(rng.integers(0, 3))] for _ in range(n)])
    target = (a + b) & 0xFF
    noise = [(0x20 + k, (rng.integers(0, 200, n)).astype(np.int64)) for k in range(6)]
    trace = _narrowing_trace(target, [(0x40, a), (0x41, b)], noise, n=n)
    ctx = _build_context(trace)
    # Fallback set (all changing cells) is much larger than the narrowed set.
    fallback = set(int(x) for x in ctx.stateseq.addrs)
    assert {0x40, 0x41} <= fallback
    assert any(0x20 + k in fallback for k in range(6))
    cand = ctx.candidates(0xD418)
    assert cand is not None  # read log present -> narrowing active
    assert {0x40, 0x41} <= cand
    assert not any((0x20 + k) in cand for k in range(6))  # noise excluded
    assert len(ctx.candidate_cols(0xD418)) < len(fallback)


def test_candidates_fallback_without_readlog():
    rng = np.random.default_rng(8)
    n = 60
    a = np.array([_LOWENT[int(rng.integers(0, len(_LOWENT)))] for _ in range(n)])
    target = (a + 0x1F) & 0xFF
    noise = [(0x20 + k, (rng.integers(0, 200, n)).astype(np.int64)) for k in range(4)]
    trace = _narrowing_trace(target, [(0x40, a)], noise, n=n, with_reads=False)
    ctx = _build_context(trace)
    assert ctx.reads_near is None
    assert ctx.candidates(0xD418) is None  # no narrowing -> None
    # candidate_cols falls back to every changing-cell column.
    assert list(ctx.candidate_cols(0xD418)) == list(range(ctx.stateseq.grid.shape[1]))


def test_narrowing_preserves_recovery():
    # The same fold must be recovered with and without the read log (narrowing
    # never drops a cell the store genuinely reads).
    rng = np.random.default_rng(9)
    n = 72
    a = np.array([_LOWENT[int(rng.integers(0, len(_LOWENT)))] for _ in range(n)])
    b = np.array([(0x0C, 0x14, 0x1C)[int(rng.integers(0, 3))] for _ in range(n)])
    target = (a + b) & 0xFF
    noise = [(0x20 + k, (rng.integers(0, 200, n)).astype(np.int64)) for k in range(5)]
    feeders = [(0x40, a), (0x41, b)]
    narrowed = analyze(_narrowing_trace(target, feeders, noise, n=n, with_reads=True))
    fallback = analyze(_narrowing_trace(target, feeders, noise, n=n, with_reads=False))
    assert narrowed[0xD418]["type"] == fallback[0xD418]["type"] == "ADD"
    assert {narrowed[0xD418]["cell_a"], narrowed[0xD418]["cell_b"]} == {0x40, 0x41}
    for tr in (
        _narrowing_trace(target, feeders, noise, n=n, with_reads=True),
        _narrowing_trace(target, feeders, noise, n=n, with_reads=False),
    ):
        assert round_trip(tr)[0xD418] == 1.0


def test_narrowing_binop_speedup():
    # Measurable speedup: with many unrelated changing cells the fold search is
    # O(cells^2); narrowing collapses the candidate set, so the narrowed proposer
    # is faster and considers far fewer cells.
    import time  # noqa: PLC0415

    rng = np.random.default_rng(11)
    n = 200
    a = np.array([_LOWENT[int(rng.integers(0, len(_LOWENT)))] for _ in range(n)])
    b = np.array([(0x0C, 0x14, 0x1C)[int(rng.integers(0, 3))] for _ in range(n)])
    target = (a + b) & 0xFF
    noise = [(0x20 + k, (rng.integers(0, 24, n)).astype(np.int64)) for k in range(120)]
    feeders = [(0x40, a), (0x41, b)]
    tr_reads = _narrowing_trace(target, feeders, noise, n=n, with_reads=True)
    tr_plain = _narrowing_trace(target, feeders, noise, n=n, with_reads=False)
    ctx_n = _build_context(tr_reads)
    ctx_f = _build_context(tr_plain)
    n_narrow = len(ctx_n.candidate_cols(0xD418))
    n_full = len(list(ctx_f.candidate_cols(0xD418)))
    assert n_narrow <= 4 < n_full  # ~2 feeders vs ~122 changing cells

    def _clock(ctx):
        best = min(_timeit(_propose_binop, 0xD418, target, ctx) for _ in range(3))
        return best

    def _timeit(fn, *args):
        t0 = time.perf_counter()
        fn(*args)
        return time.perf_counter() - t0

    t_narrow = _clock(ctx_n)
    t_full = _clock(ctx_f)
    print(
        f"\n_propose_binop: full={t_full*1e3:.2f}ms ({n_full} cells) "
        f"narrowed={t_narrow*1e3:.2f}ms ({n_narrow} cells) speedup={t_full/max(t_narrow,1e-9):.1f}x"
    )
    assert t_narrow < t_full  # narrowing is a strict win at this cell count


# -- Phase 5: shared per-voice latent state -------------------------------

from preframr_playroutine.recover import (  # noqa: E402
    _build_context,
    _build_latents,
    _synth_tick,
    propose_index_sum,
)


def _shared_cursor_trace():
    """Trace whose voice-0 PW-lo and PW-hi are two table walks over ONE cursor.

    The cursor cell ($0040) resets to 0 at each note-on and steps +1; the voice-0
    CTRL gate provides the note-ons; PW-lo and PW-hi each read a DIFFERENT table at
    the SAME cursor (byte-even tables so the SID gate-mask search is lossless). The
    recovery rule: recover the cursor once, both registers reference the shared
    latent.
    """
    note_len = 8
    n = 96
    tab_lo = np.array([0x30, 0x80, 0x10, 0x60, 0x20, 0x90, 0x40, 0x70], dtype=np.uint8)
    tab_hi = np.array([0x12, 0x34, 0x56, 0x78, 0x2A, 0x4C, 0x6E, 0x22], dtype=np.uint8)
    ram = np.zeros(65536, dtype=np.uint8)
    lo_base, hi_base = 0x2000, 0x2400
    ram[lo_base : lo_base + len(tab_lo)] = tab_lo
    ram[hi_base : hi_base + len(tab_hi)] = tab_hi
    cursor_cell = 0x0040
    recs, ramwr = [], []
    for i in range(n):
        tick = _frame_cycle(i)
        cur = i % note_len
        gate = 0x40 if cur == note_len - 1 else 0x41  # note-on at cursor 0
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ))
        recs.append(_ev(tick + 10, SID_WRITE, reg=4, value=gate, addr=0xD404, aux=0x1500))
        recs.append(
            _ev(tick + 12, SID_WRITE, reg=2, value=int(tab_lo[cur]), addr=0xD402, aux=0x1510)
        )
        recs.append(
            _ev(tick + 14, SID_WRITE, reg=3, value=int(tab_hi[cur]), addr=0xD403, aux=0x1520)
        )
        ramwr.append(_ra(tick + 6, cursor_cell, cur))
    trace = _build_trace(recs, ram_writes=ramwr, ram=ram)
    return trace


def test_synth_tick_resets_and_steps():
    tick = _synth_tick([0, 4, 9], 12)
    assert list(tick) == [0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2]


def test_latents_group_cursor_and_prefer_captured_tick():
    trace = _shared_cursor_trace()
    ctx = _build_context(trace)
    lat = ctx.latents
    assert 0 in lat and "global" in lat
    # The $0040 cursor resets at voice-0 note-ons -> grouped to voice 0.
    cursor_addrs = [a for a, _s in lat[0]["cursors"]]
    assert 0x0040 in cursor_addrs
    # It matches the synthesized wavetable tick, so it is the preferred note cell.
    assert lat[0]["note_cell"] == 0x0040


def test_shared_cursor_two_registers_same_latent():
    """The core Phase 5 proof: two registers, ONE recovered cursor latent."""
    trace = _shared_cursor_trace()
    ctx = _build_context(trace)
    pw_lo = classify_register(trace, 0xD402, ctx=ctx)
    pw_hi = classify_register(trace, 0xD403, ctx=ctx)
    assert pw_lo["type"] == "TABLE_WALK"
    assert pw_hi["type"] == "TABLE_WALK"
    # Both walk the SAME cursor cell, tagged with the SAME shared latent id --
    # not two independently-derived cursors.
    assert pw_lo["cursor_addr"] == pw_hi["cursor_addr"] == 0x0040
    assert "index_latent" in pw_lo and "index_latent" in pw_hi
    assert pw_lo["index_latent"] == pw_hi["index_latent"]
    voice, kind, addr = pw_lo["index_latent"]
    assert (voice, kind, addr) == (0, "cursor", 0x0040)
    # The id resolves into the voice's shared latent set.
    assert addr in [a for a, _s in ctx.latents[voice]["cursors"]]
    # Reconstruction stays perfect through the shared latent (round_trip builds a
    # latent-less sampler, so this also exercises the eof-cell fallback).
    rt = round_trip(trace)
    assert rt[0xD402] == 1.0
    assert rt[0xD403] == 1.0


def test_propose_index_sum_recovers_additive_index():
    """The generalized index builder recovers a table index as a cell sum."""
    n = 80
    rng = np.random.default_rng(7)
    base = rng.integers(0, 20, n).astype(np.int64)
    offset = rng.integers(0, 6, n).astype(np.int64)
    idx = base + offset
    recs, ramwr = [], []
    for i in range(n):
        tick = _frame_cycle(i)
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ))
        recs.append(_ev(tick + 12, SID_WRITE, reg=3, value=int(idx[i]) & 0xFF, addr=0xD403))
        ramwr.append(_ra(tick + 4, 0x0050, int(base[i])))
        ramwr.append(_ra(tick + 5, 0x0051, int(offset[i])))
    trace = _build_trace(recs, ram_writes=ramwr)
    ctx = _build_context(trace)
    cells, running = propose_index_sum(idx, ctx, sid_addr=0xD403, clip_hi=int(idx.max()))
    assert set(cells) == {0x0050, 0x0051}
    assert np.array_equal(running, idx)
