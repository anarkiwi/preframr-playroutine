"""Unit tests for the Phase-7 emit-slice lifter + dynamic witness (lift.py).

These build synthetic traces directly (no sidtrace binary, no HVSC): a hand-laid
6502 routine in a RAM image, its executed-PC coverage bitmap, and the per-frame
cell / SID write streams the routine would have produced. The lifter disassembles
the image within coverage, symbolically slices the store, and its grammar tree is
verified frame-exact against the intended register series. The end-to-end
``classify_register`` cases prove the lifter is adopted (Tier 2) / the witness
backstop fires (Tier 3) only when it strictly improves an imperfect register.
"""

import numpy as np

from preframr_playroutine import Trace
from preframr_playroutine import ir
from preframr_playroutine import lift
from preframr_playroutine.recover import _build_context, classify_register
from preframr_playroutine.trace import EVENT_DTYPE, RAMACCESS_DTYPE, CPU_VECTOR, SID_WRITE, VEC_IRQ


def _cov(pcs):
    cov = np.zeros(8192, dtype=np.uint8)
    for pc in pcs:
        cov[pc >> 3] |= np.uint8(1 << (pc & 7))
    return cov


def _vector(cycle):
    rec = np.zeros(1, dtype=EVENT_DTYPE)[0]
    rec["cycle"] = np.uint64(cycle)
    rec["etype"] = CPU_VECTOR
    rec["value"] = VEC_IRQ
    return rec


def _sid(cycle, addr, value, pc):
    rec = np.zeros(1, dtype=EVENT_DTYPE)[0]
    rec["cycle"] = np.uint64(cycle)
    rec["etype"] = SID_WRITE
    rec["addr"] = addr
    rec["value"] = value & 0xFF
    rec["aux"] = pc
    return rec


def _rw(cycle, addr, value):
    rec = np.zeros(1, dtype=RAMACCESS_DTYPE)[0]
    rec["cycle"] = np.uint64(cycle)
    rec["addr"] = addr
    rec["value"] = value & 0xFF
    rec["kind"] = 0
    return rec


def build_trace(image, covered, frames, sid_addr, store_pc, extra_sid=None):
    """Assemble a synthetic Trace from per-frame (value, cells) tuples.

    ``frames`` is a list of ``(sid_value, {cell_addr: value})``; each frame writes
    its cells then the SID register (so the sampler's at-write instant sees them).
    ``extra_sid`` optionally maps other SID addrs to a constant, so a full voice is
    present. Returns ``(trace, context, series)``.
    """
    events, rams = [], []
    series = []
    for f, (val, cells) in enumerate(frames):
        base = f * 1000
        events.append(_vector(base + 1))
        for addr, cval in cells.items():
            rams.append(_rw(base + 5, addr, cval))
        events.append(_sid(base + 10, sid_addr, val, store_pc))
        for addr, cval in (extra_sid or {}).items():
            events.append(_sid(base + 9, addr, cval, 0x2000))
        series.append(val & 0xFF)
    trace = Trace.from_events(
        np.array(events, dtype=EVENT_DTYPE),
        {"effective_model": "PAL"},
        ramwr=np.array(rams, dtype=RAMACCESS_DTYPE),
        coverage=_cov(covered),
        ram=image,
    )
    ctx = _build_context(trace)
    return trace, ctx, np.array(series, dtype=np.int64)


def _lift_recon(image, covered, store_pc, sid_addr, ctx):
    is_smc = lift._smc_predicate(ctx)
    expr = lift.lift_store(image, frozenset(covered), store_pc, sid_addr, is_smc)
    assert expr is not None
    tree = ir._post(expr, {"addr": sid_addr, "sid": sid_addr}, width_mask=0xFF)
    return expr, ir.evaluate(tree, ctx.n_frames, ctx.sampler)


# -- decoder ------------------------------------------------------------------


def test_decode_basic_opcodes():
    img = np.zeros(65536, dtype=np.uint8)
    img[0x1000:0x1006] = [0xA5, 0x10, 0x8D, 0x00, 0xD4, 0xEA]
    lda = lift.decode(img, 0x1000)
    assert (lda.name, lda.mode, lda.length, lda.operand) == ("LDA", "zp", 2, 0x10)
    sta = lift.decode(img, 0x1002)
    assert (sta.name, sta.mode, sta.length, sta.operand) == ("STA", "abs", 3, 0xD400)
    assert lift.decode(img, 0x1005).name == "NOP"
    img[0x2000] = 0x02  # undocumented -> not decoded
    assert lift.decode(img, 0x2000) is None


def test_block_start_alignment():
    img = np.zeros(65536, dtype=np.uint8)
    # LDA $10 ; CLC ; ADC #$05 ; STA $D400
    prog = [0xA5, 0x10, 0x18, 0x69, 0x05, 0x8D, 0x00, 0xD4]
    img[0x1000 : 0x1000 + len(prog)] = prog
    covered = [0x1000, 0x1002, 0x1003, 0x1005]
    assert lift.block_start(img, frozenset(covered), 0x1005) == 0x1000


# -- Tier 2 slices ------------------------------------------------------------


def test_lift_plain_feeder():
    img = np.zeros(65536, dtype=np.uint8)
    prog = [0xA5, 0x10, 0x8D, 0x00, 0xD4]  # LDA $10 ; STA $D400
    img[0x1000 : 0x1000 + len(prog)] = prog
    frames = [((i * 7) & 0xFF, {0x10: (i * 7) & 0xFF}) for i in range(40)]
    _tr, ctx, series = build_trace(img, [0x1000, 0x1002], frames, 0xD400, 0x1002)
    expr, recon = _lift_recon(img, [0x1000, 0x1002], 0x1002, 0xD400, ctx)
    assert expr["op"] == "cell" and expr["addr"] == 0x10
    assert np.array_equal(recon, series)


def test_lift_add_fold():
    img = np.zeros(65536, dtype=np.uint8)
    # LDA $10 ; CLC ; ADC $11 ; STA $D400   -> (a+b) & 0xFF
    prog = [0xA5, 0x10, 0x18, 0x65, 0x11, 0x8D, 0x00, 0xD4]
    img[0x1000 : 0x1000 + len(prog)] = prog
    covered = [0x1000, 0x1002, 0x1003, 0x1005]
    frames = []
    for i in range(40):
        a, b = (i * 5) & 0xFF, (i * 3 + 9) & 0xFF
        frames.append(((a + b) & 0xFF, {0x10: a, 0x11: b}))
    _tr, ctx, series = build_trace(img, covered, frames, 0xD400, 0x1005)
    expr, recon = _lift_recon(img, covered, 0x1005, 0xD400, ctx)
    assert expr["op"] == "binop" and expr["fn"] == "add"
    assert np.array_equal(recon, series)


def test_lift_and_fold():
    img = np.zeros(65536, dtype=np.uint8)
    # LDA $10 ; AND $11 ; STA $D404
    prog = [0xA5, 0x10, 0x25, 0x11, 0x8D, 0x04, 0xD4]
    img[0x1000 : 0x1000 + len(prog)] = prog
    covered = [0x1000, 0x1002, 0x1004]
    frames = []
    for i in range(40):
        a, b = (i * 9) & 0xFF, 0xFE if i % 3 else 0xFF
        frames.append((a & b, {0x10: a, 0x11: b}))
    _tr, ctx, series = build_trace(img, covered, frames, 0xD404, 0x1004)
    expr, recon = _lift_recon(img, covered, 0x1004, 0xD404, ctx)
    assert expr["fn"] == "and"
    assert np.array_equal(recon, series)


def test_lift_smc_immediate_scaled():
    # SMC accumulator read as an immediate, then ASL A (scale x2): a plain feeder
    # cell cannot match the register (it is 2*cell), but the lift does.
    img = np.zeros(65536, dtype=np.uint8)
    # LDA #$00(smc) ; ASL A ; STA $D415     immediate operand byte = 0x1001
    prog = [0xA9, 0x00, 0x0A, 0x8D, 0x15, 0xD4]
    img[0x1000 : 0x1000 + len(prog)] = prog
    covered = [0x1000, 0x1002, 0x1003]
    smc = 0x1001
    frames = []
    for i in range(40):
        acc = (i * 3) & 0x7F
        frames.append(((acc * 2) & 0xFF, {smc: acc}))
    _tr, ctx, series = build_trace(img, covered, frames, 0xD415, 0x1003)
    assert smc in ctx.written_cells  # the code byte is a self-modified cell
    expr, recon = _lift_recon(img, covered, 0x1003, 0xD415, ctx)
    # ASL A lifts to add(cell, cell); the immediate grounds to the SMC cell.
    assert expr["op"] == "binop" and expr["fn"] == "add"
    assert expr["a"]["op"] == "cell" and expr["a"]["addr"] == smc
    assert np.array_equal(recon, series)


def test_lift_clamp_branch():
    img = np.zeros(65536, dtype=np.uint8)
    # LDA $10 ; CMP #$40 ; BCC +2 ; LDA #$40 ; STA $D400   -> min(a, 0x40)
    #  1000 A5 10       LDA $10
    #  1002 C9 40       CMP #$40
    #  1004 90 02       BCC $1008
    #  1006 A9 40       LDA #$40
    #  1008 8D 00 D4    STA $D400
    prog = [0xA5, 0x10, 0xC9, 0x40, 0x90, 0x02, 0xA9, 0x40, 0x8D, 0x00, 0xD4]
    img[0x1000 : 0x1000 + len(prog)] = prog
    covered = [0x1000, 0x1002, 0x1004, 0x1006, 0x1008]
    frames = [((min(i * 7 % 256, 0x40)) & 0xFF, {0x10: (i * 7) & 0xFF}) for i in range(60)]
    _tr, ctx, series = build_trace(img, covered, frames, 0xD400, 0x1008)
    expr, recon = _lift_recon(img, covered, 0x1008, 0xD400, ctx)
    assert expr["op"] == "cmpsel" and expr["cmp"] == "ult"
    assert np.array_equal(recon, series)


def test_lift_indexed_table_read():
    img = np.zeros(65536, dtype=np.uint8)
    table = np.array([0x41, 0x11, 0x21, 0x81, 0x09, 0x93], dtype=np.uint8)
    img[0x1800 : 0x1800 + len(table)] = table
    # LDX $10 ; LDA $1800,X ; STA $D404
    prog = [0xA6, 0x10, 0xBD, 0x00, 0x18, 0x8D, 0x04, 0xD4]
    img[0x1000 : 0x1000 + len(prog)] = prog
    covered = [0x1000, 0x1002, 0x1005]
    frames = [(int(table[i % len(table)]), {0x10: i % len(table)}) for i in range(40)]
    _tr, ctx, series = build_trace(img, covered, frames, 0xD404, 0x1005)
    expr, recon = _lift_recon(img, covered, 0x1005, 0xD404, ctx)
    assert expr["op"] == "table" and expr["index"]["addr"] == 0x10
    assert np.array_equal(recon, series)


def test_lift_fails_on_call():
    img = np.zeros(65536, dtype=np.uint8)
    # LDA $10 ; JSR $2000 ; STA $D400 -> a call crosses the slice; lift bails.
    prog = [0xA5, 0x10, 0x20, 0x00, 0x20, 0x8D, 0x00, 0xD4]
    img[0x1000 : 0x1000 + len(prog)] = prog
    covered = frozenset([0x1000, 0x1002, 0x1005])
    assert lift.lift_store(img, covered, 0x1005, 0xD400, lambda _a: False) is None


def test_lift_fails_outside_coverage():
    img = np.zeros(65536, dtype=np.uint8)
    prog = [0xA5, 0x10, 0x8D, 0x00, 0xD4]
    img[0x1000 : 0x1000 + len(prog)] = prog
    # Coverage omits the LDA, so the slice cannot enter executed code.
    assert lift.lift_store(img, frozenset([0x1002]), 0x1002, 0xD400, lambda _a: False) is None


def test_lift_sub_fold():
    img = np.zeros(65536, dtype=np.uint8)
    # LDA $10 ; SEC ; SBC $11 ; STA $D400   -> (a-b) & 0xFF
    prog = [0xA5, 0x10, 0x38, 0xE5, 0x11, 0x8D, 0x00, 0xD4]
    img[0x1000 : 0x1000 + len(prog)] = prog
    covered = [0x1000, 0x1002, 0x1003, 0x1005]
    frames = [
        (((i * 5) - (i * 2 + 3)) & 0xFF, {0x10: (i * 5) & 0xFF, 0x11: (i * 2 + 3) & 0xFF})
        for i in range(40)
    ]
    _tr, ctx, series = build_trace(img, covered, frames, 0xD400, 0x1005)
    expr, recon = _lift_recon(img, covered, 0x1005, 0xD400, ctx)
    assert expr["op"] == "binop" and expr["fn"] == "sub"
    assert np.array_equal(recon, series)


def test_lift_pha_pla_and_transfer():
    img = np.zeros(65536, dtype=np.uint8)
    # LDA $10 ; PHA ; TAX ; PLA ; STA $D400  -> value passes through stack, = $10
    prog = [0xA5, 0x10, 0x48, 0xAA, 0x68, 0x8D, 0x00, 0xD4]
    img[0x1000 : 0x1000 + len(prog)] = prog
    covered = [0x1000, 0x1002, 0x1003, 0x1004, 0x1005]
    frames = [((i * 3) & 0xFF, {0x10: (i * 3) & 0xFF}) for i in range(30)]
    _tr, ctx, series = build_trace(img, covered, frames, 0xD400, 0x1005)
    expr, recon = _lift_recon(img, covered, 0x1005, 0xD400, ctx)
    assert expr["op"] == "cell" and expr["addr"] == 0x10
    assert np.array_equal(recon, series)


def test_lift_fails_symbolic_carry_chain():
    img = np.zeros(65536, dtype=np.uint8)
    # LDA $10 ; CLC ; ADC $11 ; ADC $12 ; STA -> second ADC reuses an unknown carry.
    prog = [0xA5, 0x10, 0x18, 0x65, 0x11, 0x65, 0x12, 0x8D, 0x00, 0xD4]
    img[0x1000 : 0x1000 + len(prog)] = prog
    covered = frozenset([0x1000, 0x1002, 0x1003, 0x1005, 0x1007])
    assert lift.lift_store(img, covered, 0x1007, 0xD400, lambda _a: False) is None


# -- Tier 3 witness -----------------------------------------------------------


def test_classify_adopts_witness_when_slice_fails():
    # LSR A is outside the static value envelope, so the slice bails -- but the
    # input cone ($10) determines the register, so the Tier-3 witness is exact and
    # the arbiter adopts it (never XSTATE).
    img = np.zeros(65536, dtype=np.uint8)
    prog = [0xA5, 0x10, 0x4A, 0x8D, 0x00, 0xD4]  # LDA $10 ; LSR A ; STA $D400
    img[0x1000 : 0x1000 + len(prog)] = prog
    covered = [0x1000, 0x1002, 0x1003]
    rng = np.random.default_rng(5)
    cells = rng.integers(0, 256, size=100)
    frames = [(int(c) >> 1, {0x10: int(c)}) for c in cells]
    trace, _ctx, series = build_trace(img, covered, frames, 0xD400, 0x1003)
    assert lift.lift_store(img, frozenset(covered), 0x1003, 0xD400, lambda _a: False) is None
    res = classify_register(trace, 0xD400)
    assert res["type"] == "WITNESS", res["type"]
    recon = ir.evaluate(ir.to_ir(res), len(series), _build_context(trace).sampler)
    assert np.array_equal(recon, series)


def test_witness_inexact_on_collision():
    # A cone that does NOT determine the output (a hidden random addend) collides:
    # the witness is not exact, so it must not be adopted -- the register keeps its
    # value-stream (feeder) descriptor.
    img = np.zeros(65536, dtype=np.uint8)
    rng = np.random.default_rng(7)
    n = 60
    cells = rng.integers(0, 64, size=n)
    hidden = rng.integers(0, 2, size=n)
    series = (cells + hidden).astype(np.int64)
    specs = [(0x10, "write")]
    frames = [(int(series[i]), {0x10: int(cells[i])}) for i in range(n)]
    _tr, ctx, ser = build_trace(img, [], frames, 0xD400, 0x2000)
    wit = lift._witness_from_specs(ser, specs, 0xD400, ctx)
    recon = ir.evaluate(ir.to_ir(wit), ctx.n_frames, ctx.sampler)
    assert float(np.mean(recon == ser)) < 1.0


def test_witness_exact_and_replayable():
    img = np.zeros(65536, dtype=np.uint8)
    # A within-call structure the static value-slice will not model, but whose
    # input cone (two cells) determines the output: LDA $10 ; ORA $11 ; STA.
    prog = [0xA5, 0x10, 0x05, 0x11, 0x8D, 0x00, 0xD4]
    img[0x1000 : 0x1000 + len(prog)] = prog
    covered = [0x1000, 0x1002, 0x1004]
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(50):
        a, b = int(rng.integers(0, 256)), int(rng.integers(0, 256))
        frames.append((a | b, {0x10: a, 0x11: b}))
    _tr, ctx, series = build_trace(img, covered, frames, 0xD400, 0x1004)
    cone = lift.slice_cone(img, frozenset(covered), 0x1004, lambda _a: False)
    assert (0x10, "write") in cone and (0x11, "write") in cone
    wit = lift._witness_from_specs(series, cone, 0xD400, ctx)
    recon = ir.evaluate(ir.to_ir(wit), ctx.n_frames, ctx.sampler)
    assert np.array_equal(recon, series)


def test_cmpsel_node_signed_and_flag():
    n = 8
    a = {"op": "literal", "data": np.array([0, 1, 0x40, 0x7F, 0x80, 0xC0, 0xFF, 0x10])}
    node = {
        "op": "cmpsel",
        "cmp": "neg",
        "a": a,
        "b": None,
        "then": {"op": "const", "value": 1},
        "else": {"op": "const", "value": 0},
    }
    out = ir.evaluate(node, n, None)
    assert out.tolist() == [0, 0, 0, 0, 1, 1, 1, 0]
    node["cmp"] = "sge"
    node["b"] = {"op": "const", "value": 0}
    out = ir.evaluate(node, n, None)
    assert out.tolist() == [1, 1, 1, 1, 0, 0, 0, 1]


# -- end-to-end arbitration ---------------------------------------------------


def test_classify_adopts_lift_over_imperfect():
    # A register that no value-stream proposer models exactly (min(cell, k)); the
    # lifter recovers it and the arbiter adopts the Tier-2 tree.
    img = np.zeros(65536, dtype=np.uint8)
    prog = [0xA5, 0x10, 0xC9, 0x40, 0x90, 0x02, 0xA9, 0x40, 0x8D, 0x00, 0xD4]
    img[0x1000 : 0x1000 + len(prog)] = prog
    covered = [0x1000, 0x1002, 0x1004, 0x1006, 0x1008]
    rng = np.random.default_rng(3)
    cells = rng.integers(0, 256, size=120)
    frames = [(min(int(c), 0x40), {0x10: int(c)}) for c in cells]
    trace, _ctx, series = build_trace(img, covered, frames, 0xD400, 0x1008)
    res = classify_register(trace, 0xD400)
    assert res["type"] == "LIFT", res["type"]
    recon = ir.evaluate(ir.to_ir(res), len(series), _build_context(trace).sampler)
    assert np.array_equal(recon, series)
