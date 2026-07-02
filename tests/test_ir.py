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
