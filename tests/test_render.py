"""Tests for the WAV renderer (matrix builder pure; audio path skipped sans pyresidfp)."""

import wave

import numpy as np
import pytest

from preframr_playroutine.trace import CPU_VECTOR
from preframr_playroutine.render import (
    recovered_register_matrix,
    render_wav,
    sid_model_from_file,
)

from test_recover import VEC_IRQ, SID_WRITE, _build_trace, _ev, _frame_cycle

# Constant per-register values -> CONST descriptors -> perfect recovery.
_CONST_REGS = {0x04: 0x41, 0x05: 0x09, 0x06: 0xF0, 0x18: 0x0F}


def _multi_register_trace(n=64):
    """Synthetic trace writing a few SID registers (constant) per frame."""
    recs = []
    for i in range(n):
        tick = _frame_cycle(i)
        recs.append(_ev(tick + 2, CPU_VECTOR, value=VEC_IRQ, addr=0x1003))
        for off, val in _CONST_REGS.items():
            recs.append(
                _ev(tick + 10 + off, SID_WRITE, reg=off, value=val, addr=0xD400 + off, aux=0x1100)
            )
    return _build_trace(recs)


def test_recovered_register_matrix_shape_and_oracle():
    trace = _multi_register_trace()
    ticks, matrix = recovered_register_matrix(trace)
    oracle = trace.register_frames()[1][:, :25]
    assert matrix.shape == (len(ticks), 25)
    assert matrix.dtype == np.uint8
    # Each written register matches the oracle exactly for a perfect CONST trace.
    for off, val in _CONST_REGS.items():
        assert np.array_equal(matrix[:, off], oracle[:, off])
        assert matrix[-1, off] == val
    assert np.count_nonzero(matrix != oracle) == 0


def test_sid_model_from_file(tmp_path):
    def write_sid(model_bits):
        data = bytearray(0x78)
        flags = model_bits << 4
        data[0x76] = (flags >> 8) & 0xFF
        data[0x77] = flags & 0xFF
        path = tmp_path / f"m{model_bits}.sid"
        path.write_bytes(data)
        return str(path)

    assert sid_model_from_file(write_sid(2)) == "8580"
    assert sid_model_from_file(write_sid(1)) == "6581"
    assert sid_model_from_file(write_sid(3)) == "6581"
    short = tmp_path / "short.sid"
    short.write_bytes(b"\x00")
    assert sid_model_from_file(str(short)) == "6581"


def test_render_wav_smoke(tmp_path):
    pytest.importorskip("pyresidfp")
    trace = _multi_register_trace()
    out = tmp_path / "out.wav"
    stats = render_wav(trace, str(out), model="8580")
    assert out.exists()
    assert stats["register_mismatches"] == 0
    assert stats["model"] == "8580"
    with wave.open(str(out), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 44100
        assert wav.getnframes() == stats["samples"]
