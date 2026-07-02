"""Shared deterministic fixture for the reconstruction-parity golden tests.

Builds one fake trace (no sidtrace binary, no HVSC, no randomness) and the full
descriptor set exercised by ``reconstruct_register``. Imported by both
``tests/make_ir_golden.py`` (which captures the golden outputs) and
``tests/test_ir.py`` (which replays them against the current evaluator).
"""

import numpy as np

from preframr_playroutine import EVENT_DTYPE
from preframr_playroutine.trace import RAMACCESS_DTYPE
from preframr_playroutine.recover import _CUTOFF_OFFS

SID_ADDRS = (0xD402, 0xD404, 0xD416, 0xD418, 0xD400, 0xD401)


class FakeTrace:
    """Minimal duck-type for _CellSampler + _default_until_first_write."""

    def __init__(self, ram_writes, sid_writes):
        self._rw, self._sw = ram_writes, sid_writes

    def ram_writes(self, kind=None):
        del kind
        return self._rw

    def sid_writes(self, chip=None):
        del chip
        return self._sw


def build_ticks():
    return np.arange(16, dtype=np.uint64) * 1000


def _sid_event(cycle, addr):
    rec = np.zeros(1, dtype=EVENT_DTYPE)[0]
    rec["cycle"] = np.uint64(cycle)
    rec["etype"] = 0
    rec["chip"] = 0
    rec["reg"] = 0
    rec["value"] = 0
    rec["addr"] = addr
    rec["aux"] = 0x1200
    return rec


def _ram_write(cycle, addr, value):
    rec = np.zeros(1, dtype=RAMACCESS_DTYPE)[0]
    rec["cycle"] = np.uint64(cycle)
    rec["pc"] = 0x1200
    rec["addr"] = addr
    rec["value"] = value & 0xFF
    rec["kind"] = 0
    rec["pad"] = 0
    return rec


def build_fake_trace():
    ticks = build_ticks()
    sids, rams = [], []
    for f in range(16):
        base = int(ticks[f])
        for addr in SID_ADDRS:
            if addr == 0xD418 and f < 3:
                continue
            sids.append(_sid_event(base + 10, addr))
        rams.append(_ram_write(base + 5, 0x10, f & 0xFF))
        rams.append(_ram_write(base + 50, 0x11, (f * 3) & 0xFF))
        rams.append(_ram_write(base + 4, 0x12, (f * 5) & 0xFF))
        rams.append(_ram_write(base + 6, 0x12, (f * 5 + 1) & 0xFF))
        rams.append(_ram_write(base + 5, 0x13, 0xFF if f < 8 else 0xFE))
        rams.append(_ram_write(base + 5, 0x14, f // 4))
        rams.append(_ram_write(base + 5, 0x15, 0x40 if f % 2 == 0 else 0x41))
        if f >= 6:
            rams.append(_ram_write(base + 5, 0x16, 0x30 + f))
        # CUTOFF SMC operand/opcode cells at 0x1200 + off.
        cut = {
            "op_lo": 0x69,
            "op_hi": 0x69,
            "slo": 2,
            "shi": 0,
            "lo": (f * 2) & 0xFF,
            "hi": f & 0xFF,
            "imm": 5,
            "base": 2,
            "scale": 0x0A,
        }
        for name, off in _CUTOFF_OFFS.items():
            rams.append(_ram_write(base + 5, 0x1200 + off, cut[name]))
    return FakeTrace(np.array(rams, dtype=RAMACCESS_DTYPE), np.array(sids, dtype=EVENT_DTYPE))


def build_ram():
    ram = np.zeros(65536, np.uint8)
    ram[0x2000:0x2010] = [
        0x41, 0x11, 0x21, 0x81, 0x09, 0x41, 0x11, 0x21,
        0x81, 0x09, 0x41, 0x11, 0x21, 0x81, 0x09, 0x41,
    ]
    ram[0x3000:0x3010] = np.arange(16)
    ram[0x3100:0x3110] = np.arange(16) + 1
    return ram


def build_descriptors():
    ram = build_ram()
    cutoff_cells = {name: 0x1200 + off for name, off in _CUTOFF_OFFS.items()}
    return {
        "const_v": {"type": "CONST", "value": 7, "addr": 0xD402},
        "const_none": {"type": "CONST", "value": None, "addr": 0xD402},
        "seq": {
            "type": "SEQ",
            "latch_frames": [0, 4, 9],
            "latch_values": [3, 8, 2],
            "addr": 0xD402,
        },
        "bacc_saw8": {
            "type": "BACC", "mode": "saw", "step": 3, "lo": 0, "hi": 12,
            "phase": 0, "modulus": 15, "width": 8, "byte_role": "full",
            "resets": [0], "seeds": [0], "addr": 0xD402,
        },
        "bacc_reflect16_lo": {
            "type": "BACC", "mode": "reflect", "step": 40, "lo": 256, "hi": 900,
            "width": 16, "byte_role": "lo", "resets": [0, 6], "seeds": [300, 700],
            "segmented": True, "addr": 0xD402,
        },
        "bacc_reflect16_hi": {
            "type": "BACC", "mode": "reflect", "step": 40, "lo": 256, "hi": 900,
            "width": 16, "byte_role": "hi", "resets": [0, 6], "seeds": [300, 700],
            "segmented": True, "addr": 0xD402,
        },
        "bacc_pingpong": {
            "type": "BACC", "mode": "pingpong", "step": 40, "lo": 256, "hi": 900,
            "width": 16, "byte_role": "lo", "resets": [0, 6], "seeds": [300, 700],
            "segmented": True, "down_step": 25, "clamp_lo": 256, "clamp_hi": 900,
            "steps": [40, 30], "down_steps": [25, 20], "directions": [1, -1],
            "addr": 0xD402,
        },
        "bacc_tickband": {
            "type": "BACC", "mode": "tickband", "step": 2, "lo": 100, "hi": 400,
            "segmented": True, "resets": [0, 8], "seeds": [120, 150],
            "directions": [1, 1],
            "rate_tables": [np.array([2, 2, 4, 4, 8], dtype=np.int64)],
            "seg_tables": [0, 0], "n_segments": 2, "addr": 0xD402,
        },
        "bacc_cellfed": {
            "type": "BACC", "mode": "saw", "step": 1, "lo": 0, "hi": 255,
            "cell": 0x16, "sid": 0xD402, "prelude_end": 6, "prelude_frames": [0, 2],
            "prelude_values": [9, 11], "resets": [0], "seeds": [0],
            "byte_role": "full", "width": 8, "addr": 0xD402,
        },
        "walk_mask": {
            "type": "TABLE_WALK", "base": 0x2000, "stride": 1, "length": 16,
            "loop": 0, "table": ram[0x2000:0x2010].copy(), "mask": 0xFE,
            "cursor_addr": 0x14, "cursor_offset": 1, "addr": 0xD404,
        },
        "walk_gate_override": {
            "type": "TABLE_WALK", "base": 0x2000, "stride": 1, "length": 16,
            "loop": 0, "table": ram[0x2000:0x2010].copy(), "mask": 0xFF,
            "cursor_addr": 0x14, "cursor_offset": 1, "gate_addr": 0x13,
            "overrides": [{"predicate": [(0x15, 0xFF, 0x41)], "force": 0x08}],
            "addr": 0xD404,
        },
        "composite16": {
            "type": "COMPOSITE", "byte_role": "lo", "width_mask": 0xFFFF,
            "base": {"lo": (0x10, 0xD400), "hi": (0x11, 0xD401)},
            "mod": {"lo": (0x12, 0xD400), "hi": (0x14, 0xD401)},
            "overrides": [{"predicate": [(0x13, 0x01, 0x00)], "force": 0xFF}],
            "addr": 0xD400,
        },
        "composite8": {
            "type": "COMPOSITE", "byte_role": "full", "width_mask": 0xFF,
            "base": {"cell": 0x10, "sid": 0xD402}, "mod": None,
            "overrides": [{"predicate": [(0x15, "in", (0x40,))], "force": 0x77}],
            "addr": 0xD402,
        },
        "pitchwalk_lo": {
            "type": "PITCHWALK", "byte_role": "lo", "lo_base": 0x3000,
            "hi_base": 0x3100, "lo_table": ram[0x3000:0x3010].copy(),
            "hi_table": ram[0x3100:0x3110].copy(), "index_cells": [0x14, 0x10],
            "overrides": [], "addr": 0xD400,
        },
        "pitchwalk_hi": {
            "type": "PITCHWALK", "byte_role": "hi", "lo_base": 0x3000,
            "hi_base": 0x3100, "lo_table": ram[0x3000:0x3010].copy(),
            "hi_table": ram[0x3100:0x3110].copy(), "index_cells": [0x14, 0x10],
            "overrides": [], "addr": 0xD400,
        },
        "feeder": {"type": "FEEDER", "cell": 0x10, "sid": 0xD416, "addr": 0xD416},
        "xor": {
            "type": "XOR", "cell_a": 0x10, "cell_b": 0x15, "sid": 0xD404,
            "addr": 0xD404,
        },
        "and_ov": {
            "type": "AND", "cell_a": 0x10, "cell_b": 0x13, "sid": 0xD404,
            "overrides": [{"predicate": [(0x14, 0xFF, 0x02)], "force": 0x81}],
            "addr": 0xD404,
        },
        "or_cells": {
            "type": "OR", "cell_a": 0x10, "cell_b": 0x15, "sid": 0xD418,
            "addr": 0xD418,
        },
        "or_const_prelude": {
            "type": "OR", "cell_a": 0x16, "const": 0x0F, "sid": 0xD418,
            "prelude_end": 6, "prelude_frames": [0, 3], "prelude_values": [0x1F, 0x2F],
            "addr": 0xD418,
        },
        "xstate_cell": {"type": "XSTATE", "cell": 0x11, "sid": 0xD402, "addr": 0xD402},
        "xstate_bare": {"type": "XSTATE", "addr": 0xD402},
        "cutoff": {
            "type": "CUTOFF", "addr": 0xD416, "sid": 0xD416, "cells": cutoff_cells,
            "base": 2, "imm": 5, "scale": 2,
        },
    }
