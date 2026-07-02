"""Golden parity test for the reconstruction evaluator.

Loads ``tests/fixtures/ir_golden.npz`` (captured from the unmodified code) and
asserts that ``reconstruct_register`` still reproduces every descriptor's output
exactly, including ``None`` results. Also pins two sampler-mode properties.
"""

import os

import numpy as np
import pytest

from preframr_playroutine.recover import _CellSampler, reconstruct_register

from _ir_fixture import build_descriptors, build_fake_trace, build_ticks

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "ir_golden.npz")


@pytest.fixture(scope="module")
def golden():
    if not os.path.exists(GOLDEN):
        pytest.skip("ir_golden.npz missing -- run tests/make_ir_golden.py")
    with np.load(GOLDEN) as data:
        return {k: data[k].copy() for k in data.files}


def _fresh_sampler():
    return _CellSampler(build_fake_trace(), build_ticks())


def test_golden_parity(golden):
    ticks = build_ticks()
    descs = build_descriptors()
    for name, desc in descs.items():
        recon = reconstruct_register(desc, ticks, sampler=_fresh_sampler())
        if f"{name}__none" in golden:
            assert recon is None, (name, recon)
            continue
        assert name in golden, name
        assert recon is not None, name
        assert np.array_equal(np.asarray(recon), golden[name]), name


def test_sampler_modes():
    smp = _fresh_sampler()
    aw = smp.at_write(0x11, 0xD402)
    for f in range(1, 16):
        assert int(aw[f]) == ((f - 1) * 3) & 0xFF, (f, int(aw[f]))
    op = smp.operand(0x12, 0xD402)
    for f in range(16):
        assert int(op[f]) == (f * 5) & 0xFF, (f, int(op[f]))


# -- Phase 5: index-source resolution (frame / latent / recur index) ------

from preframr_playroutine import ir  # noqa: E402


class _LatentSampler:
    """Minimal sampler exposing latents + an eof cell for index resolution."""

    def __init__(self, latents, eof_cells=None):
        self.latents = latents
        self._eof = {int(a): np.asarray(s, dtype=np.int64) for a, s in (eof_cells or {}).items()}

    def eof(self, addr):
        return self._eof[int(addr)]


def test_evaluate_frame_index():
    table = np.arange(10, 20, dtype=np.int64)
    node = {"op": "table", "data": table, "index": "frame", "stride": 1, "offset": 0}
    out = ir.evaluate(node, 6, None)
    assert list(out) == [10, 11, 12, 13, 14, 15]


def test_evaluate_latent_tick_index():
    tick = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    table = np.array([100, 101, 102], dtype=np.int64)
    sampler = _LatentSampler({0: {"tick": tick, "cursors": []}})
    node = {"op": "table", "data": table, "index": "tick", "voice": 0, "stride": 1, "offset": 0}
    out = ir.evaluate(node, 6, sampler)
    assert list(out) == [100, 101, 102, 100, 101, 102]


def test_evaluate_latent_cursor_and_int_index_match():
    cur = np.array([0, 1, 2, 3, 0, 1], dtype=np.int64)
    table = np.array([5, 6, 7, 8], dtype=np.int64)
    sampler = _LatentSampler({1: {"tick": None, "cursors": [(0x40, cur)]}}, eof_cells={0x40: cur})
    latent_node = {
        "op": "table",
        "data": table,
        "index": {"op": "latent", "kind": "cursor", "voice": 1, "addr": 0x40},
        "stride": 1,
        "offset": 0,
    }
    int_node = {"op": "table", "data": table, "index": 0x40, "voice": 1, "stride": 1, "offset": 0}
    a = ir.evaluate(latent_node, 6, sampler)
    b = ir.evaluate(int_node, 6, sampler)
    assert np.array_equal(a, b)
    assert list(a) == [5, 6, 7, 8, 5, 6]


def test_evaluate_cursor_falls_back_to_eof_without_latents():
    cur = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    table = np.array([9, 8, 7], dtype=np.int64)
    sampler = _LatentSampler({}, eof_cells={0x77: cur})
    node = {
        "op": "table",
        "data": table,
        "index": {"op": "latent", "kind": "cursor", "voice": 0, "addr": 0x77},
        "stride": 1,
        "offset": 0,
    }
    out = ir.evaluate(node, 5, sampler)
    assert list(out) == [9, 8, 7, 8, 9]


def test_evaluate_select_first_match_and_default():
    # sel cell: 0,1,2,1,0 -> arm A on sel bit0 set, arm B on sel==2, else default.
    sel = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    sampler = _LatentSampler({}, eof_cells={0x30: sel})
    a = {"op": "const", "value": 10}
    b = {"op": "const", "value": 20}
    default = {"op": "const", "value": 99}
    node = {
        "op": "select",
        "arms": [
            ([{"kind": "bit", "cell": 0x30, "mask": 0x01, "value": 0x01}], a),
            ([{"kind": "eq", "cell": 0x30, "value": 2}], b),
        ],
        "default": default,
    }
    out = ir.evaluate(node, 5, sampler)
    # frame2 sel==2: bit0 clear so arm A misses, arm B (eq 2) fires -> 20.
    assert list(out) == [99, 10, 20, 10, 99]


def test_select_complexity_charges_predicate_terms():
    a = {"op": "const", "value": 1}
    default = {"op": "const", "value": 0}
    one_term = {
        "op": "select",
        "arms": [([{"kind": "eq", "cell": 0x30, "value": 1}], a)],
        "default": default,
    }
    two_term = {
        "op": "select",
        "arms": [
            (
                [
                    {"kind": "eq", "cell": 0x30, "value": 1},
                    {"kind": "bit", "cell": 0x31, "mask": 0x02, "value": 0x02},
                ],
                a,
            )
        ],
        "default": default,
    }
    # The extra predicate term must raise the MDL cost by exactly 1.
    assert ir.complexity(two_term) == ir.complexity(one_term) + 1.0


def test_recur_global_frame_index_drives_stride():
    # A tick-indexed reflecting sweep whose stride is read from a GLOBAL frame
    # counter (no note reset), verified against an independent simulation.
    rate = np.array([1, 1, 2, 2, 3, 3, 1, 1], dtype=np.int64)
    n = 8
    desc = {
        "op": "recur",
        "mode": "tickband",
        "lo": 0,
        "hi": 20,
        "resets": [0],
        "seeds": [0],
        "directions": [1],
        "rate_tables": [rate],
        "seg_tables": [0],
        "index": "frame",
    }
    out = ir.evaluate(desc, n, None)
    # Reference: single reflecting run stepping by rate[frame].
    v, d, ref = 0, 1, []
    for i in range(n):
        ref.append(v)
        st = int(rate[i])
        nv = v + d * st
        if nv > 20:
            d = -d
            nv = 20 - (nv - 20)
        elif nv < 0:
            d = -d
            nv = -nv
        v = nv
    assert list(out) == ref
