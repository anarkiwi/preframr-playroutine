"""Emit-slice lifter (Tier 2) + dynamic input witness (Tier 3).

The totality backstop of the recovery pipeline (GENERIC_RECOVERY.md section 3.5
and the Phase 7 subsection of section 4). For any register the value-stream
proposers leave imperfect, this module lifts the emitting 6502 code into the
expression grammar automatically:

* **Tier 2 -- emit-slice lifter.** A small tabular 6502 decoder plus a symbolic
  backward slice of the A/X/Y def-use chain from the ``STA $D4xx``, bounded to a
  few dozen executed instructions (the executed-PC coverage bitmap gates every
  decoded address, so only real code is lifted). Operands are grounded to grammar
  leaves: absolute -> ``cell@write``; an immediate whose code byte is itself a
  written RAM cell -> a self-modifying ``cell@write`` (how ``CUTOFF`` reads its
  step); indexed ``abs,X`` -> ``table[index]``. Conditional branches become
  ``cmpsel`` (value-comparison mux) arms -- clamps and min/max patterns. The
  result is a grammar tree verified with ``ir.evaluate`` against the oracle and
  handed to the arbiter like any proposer: NO opcode-signature allowlists, the
  only gate is verified reconstruction fidelity.

* **Tier 3 -- dynamic input witness.** When the static slice fails or does not
  reconstruct, the ordered input cells of the emitting window (the slice's
  grounded leaves, else the read/write-narrowed candidates) are sampled per frame
  and the observed ``inputs -> value`` mapping is memoised into a ``witness``
  node -- exact and replayable through the sampler. Heavily MDL-charged so
  Tiers 1-2 win wherever they fit.

No new heavy dependency; the decoder/slicer is naturally scalar python.
"""

from __future__ import annotations

import numpy as np

from . import ir

# Shared witness encoders (the sampler-column packer + per-spec sampler), reused so
# the lifter and the evaluator agree on the witness key layout exactly.
_encode_rows = ir._encode_rows  # pylint: disable=protected-access
_witness_col = ir._witness_col  # pylint: disable=protected-access

# -- 6502 decoder (tabular; ~60 opcodes) --------------------------------------
#
# Addressing modes and their instruction lengths (opcode byte included).
_MODE_LEN = {
    "imp": 1,
    "acc": 1,
    "imm": 2,
    "zp": 2,
    "zpx": 2,
    "zpy": 2,
    "indx": 2,
    "indy": 2,
    "rel": 2,
    "abs": 3,
    "abx": 3,
    "aby": 3,
    "ind": 3,
}


def _t(rows):
    """Expand ``{mnemonic: {opcode: mode}}`` rows into an ``opcode -> (name, mode)`` map."""
    out = {}
    for name, entries in rows.items():
        for opcode, mode in entries.items():
            out[opcode] = (name, mode)
    return out


OPCODES = _t(
    {
        "LDA": {
            0xA9: "imm",
            0xA5: "zp",
            0xB5: "zpx",
            0xAD: "abs",
            0xBD: "abx",
            0xB9: "aby",
            0xA1: "indx",
            0xB1: "indy",
        },
        "LDX": {0xA2: "imm", 0xA6: "zp", 0xB6: "zpy", 0xAE: "abs", 0xBE: "aby"},
        "LDY": {0xA0: "imm", 0xA4: "zp", 0xB4: "zpx", 0xAC: "abs", 0xBC: "abx"},
        "STA": {
            0x85: "zp",
            0x95: "zpx",
            0x8D: "abs",
            0x9D: "abx",
            0x99: "aby",
            0x81: "indx",
            0x91: "indy",
        },
        "STX": {0x86: "zp", 0x96: "zpy", 0x8E: "abs"},
        "STY": {0x84: "zp", 0x94: "zpx", 0x8C: "abs"},
        "ADC": {0x69: "imm", 0x65: "zp", 0x75: "zpx", 0x6D: "abs", 0x7D: "abx", 0x79: "aby"},
        "SBC": {0xE9: "imm", 0xE5: "zp", 0xF5: "zpx", 0xED: "abs", 0xFD: "abx", 0xF9: "aby"},
        "AND": {0x29: "imm", 0x25: "zp", 0x35: "zpx", 0x2D: "abs", 0x3D: "abx", 0x39: "aby"},
        "ORA": {0x09: "imm", 0x05: "zp", 0x15: "zpx", 0x0D: "abs", 0x1D: "abx", 0x19: "aby"},
        "EOR": {0x49: "imm", 0x45: "zp", 0x55: "zpx", 0x4D: "abs", 0x5D: "abx", 0x59: "aby"},
        "ASL": {0x0A: "acc", 0x06: "zp", 0x16: "zpx", 0x0E: "abs", 0x1E: "abx"},
        "LSR": {0x4A: "acc", 0x46: "zp", 0x56: "zpx", 0x4E: "abs", 0x5E: "abx"},
        "ROL": {0x2A: "acc", 0x26: "zp", 0x36: "zpx", 0x2E: "abs", 0x3E: "abx"},
        "ROR": {0x6A: "acc", 0x66: "zp", 0x76: "zpx", 0x6E: "abs", 0x7E: "abx"},
        "CMP": {0xC9: "imm", 0xC5: "zp", 0xD5: "zpx", 0xCD: "abs", 0xDD: "abx", 0xD9: "aby"},
        "CPX": {0xE0: "imm", 0xE4: "zp", 0xEC: "abs"},
        "CPY": {0xC0: "imm", 0xC4: "zp", 0xCC: "abs"},
        "INC": {0xE6: "zp", 0xF6: "zpx", 0xEE: "abs", 0xFE: "abx"},
        "DEC": {0xC6: "zp", 0xD6: "zpx", 0xCE: "abs", 0xDE: "abx"},
        "BIT": {0x24: "zp", 0x2C: "abs"},
        "CLC": {0x18: "imp"},
        "SEC": {0x38: "imp"},
        "CLD": {0xD8: "imp"},
        "SED": {0xF8: "imp"},
        "CLI": {0x58: "imp"},
        "SEI": {0x78: "imp"},
        "CLV": {0xB8: "imp"},
        "TAX": {0xAA: "imp"},
        "TXA": {0x8A: "imp"},
        "TAY": {0xA8: "imp"},
        "TYA": {0x98: "imp"},
        "TSX": {0xBA: "imp"},
        "TXS": {0x9A: "imp"},
        "INX": {0xE8: "imp"},
        "INY": {0xC8: "imp"},
        "DEX": {0xCA: "imp"},
        "DEY": {0x88: "imp"},
        "PHA": {0x48: "imp"},
        "PLA": {0x68: "imp"},
        "PHP": {0x08: "imp"},
        "PLP": {0x28: "imp"},
        "NOP": {0xEA: "imp"},
        "RTS": {0x60: "imp"},
        "RTI": {0x40: "imp"},
        "JSR": {0x20: "abs"},
        "JMP": {0x4C: "abs", 0x6C: "ind"},
        "BPL": {0x10: "rel"},
        "BMI": {0x30: "rel"},
        "BVC": {0x50: "rel"},
        "BVS": {0x70: "rel"},
        "BCC": {0x90: "rel"},
        "BCS": {0xB0: "rel"},
        "BNE": {0xD0: "rel"},
        "BEQ": {0xF0: "rel"},
    }
)

# Conditional branch -> (cmp kind, uses N/Z-source vs the CMP two-operand form).
# "flag" branches (BMI/BPL/BEQ/BNE) test the N/Z source; carry branches (BCC/BCS)
# test the recorded unsigned comparison.
_BRANCH = {
    "BCC": ("ult", "cmp"),
    "BCS": ("uge", "cmp"),
    "BEQ": ("eq", "nz"),
    "BNE": ("ne", "nz"),
    "BMI": ("neg", "nz"),
    "BPL": ("pos", "nz"),
}

_STORE_REG = {"STA": "A", "STX": "X", "STY": "Y"}
_LOAD_REG = {"LDA": "A", "LDX": "X", "LDY": "Y"}

_MAX_INSTRS = 24  # bounded backward slice / per-path budget
_MAX_STEPS = 512  # shared total-step budget across all forked branch paths
_TABLE_SPAN = 256

# Opcodes whose operand reads a value into the dataflow cone (Tier-3 witness).
_CONE_IMM = frozenset({"LDA", "LDX", "LDY", "ADC", "SBC", "AND", "ORA", "EOR"})
_CONE_ABS = _CONE_IMM | frozenset({"CMP", "CPX", "CPY"})


class Instr:
    """A decoded 6502 instruction: mnemonic, addressing mode, length, operand."""

    __slots__ = ("name", "mode", "length", "operand", "pc")

    def __init__(self, name, mode, length, operand, pc):
        self.name = name
        self.mode = mode
        self.length = length
        self.operand = operand
        self.pc = pc


def decode(image, pc) -> Instr | None:
    """Decode one instruction at ``pc`` from a RAM image, or ``None`` if unknown."""
    if pc < 0 or pc >= len(image):
        return None
    op = int(image[pc])
    entry = OPCODES.get(op)
    if entry is None:
        return None
    name, mode = entry
    length = _MODE_LEN[mode]
    if pc + length > len(image):
        return None
    operand = None
    if length == 2:
        operand = int(image[pc + 1])
    elif length == 3:
        operand = int(image[pc + 1]) | (int(image[pc + 2]) << 8)
    return Instr(name, mode, length, operand, pc)


def _prev_instr_start(image, covered, pc):
    """The executed instruction start immediately before ``pc`` (or ``None``)."""
    for back in (1, 2, 3):
        q = pc - back
        if q < 0 or q not in covered:
            continue
        ins = decode(image, q)
        if ins is not None and ins.length == back:
            return q
    return None


# A call/return terminates the backward walk: the slice cannot cross it (the
# slicer fails closed on calls), and the register the store reads is either
# redefined after the boundary (so earlier code is dead) or produced by the call
# (statically unrecoverable). Stopping here keeps the cone the store's own emit
# instructions -- crucially the SMC-immediate feeder of an ``LDA #imm ; STA $D4xx``
# unrolled writer -- instead of the whole compute routine before the JSR/RTS.
_SLICE_BOUNDARY = frozenset({"JSR", "RTS", "RTI"})


def block_start(image, covered, store_pc, max_instrs=_MAX_INSTRS):
    """Walk back over executed instructions to the slice's entry PC.

    Stops at a call/return boundary (see ``_SLICE_BOUNDARY``): the entry PC is the
    first executed instruction after the most recent JSR/RTS/RTI before the store.
    """
    cur = store_pc
    for _ in range(max_instrs):
        prev = _prev_instr_start(image, covered, cur)
        if prev is None:
            break
        pins = decode(image, prev)
        if pins is not None and pins.name in _SLICE_BOUNDARY:
            break
        cur = prev
    return cur


# -- symbolic grammar-leaf constructors ---------------------------------------


def _const(value):
    return {"op": "const", "value": int(value) & 0xFF}


def _cell(addr, sid, sample="write"):
    return {"op": "cell", "addr": int(addr), "sample": sample, "sid": sid}


def _binop(fn, a, b):
    return {"op": "binop", "fn": fn, "a": a, "b": b}


def _is_io(addr):
    return (int(addr) & 0xF000) == 0xD000 or (int(addr) & 0xFF00) in (0xDC00, 0xDD00)


class _Fail(Exception):
    """Raised to abort a slice that leaves the grammar's static envelope."""


class _Slicer:
    """Symbolic executor over an executed straight-line/diamond code region.

    Tracks the A/X/Y registers, the carry flag and the branch flag sources as
    grammar-expression nodes, resolving each ``STA/STX/STY $D4xx`` to the node
    feeding it. Conditional branches fork the execution and reconverge as a
    ``cmpsel`` mux (the clamp/min-max shape). ``store_pc`` terminates a path.
    """

    def __init__(self, image, covered, sid, is_smc):
        self.image = image
        self.covered = covered
        self.sid = sid
        self.is_smc = is_smc
        self._steps = [_MAX_STEPS]

    # -- operand grounding -------------------------------------------------

    def _operand_node(self, ins, state):
        """Ground a read operand of ``ins`` to a grammar node (raise ``_Fail``)."""
        mode = ins.mode
        if mode == "imm":
            opaddr = ins.pc + 1
            if self.is_smc(opaddr):
                return _cell(opaddr, self.sid, "write")
            return _const(ins.operand)
        if mode in ("zp", "abs"):
            if _is_io(ins.operand):
                # A live chip-state read (osc3/env3, CIA timer): the player's own
                # LDA $D41B idiom -> a chip source node driven by the I/O-read log.
                return {"op": "chip", "addr": int(ins.operand), "sid": self.sid}
            return _cell(ins.operand, self.sid, "write")
        if mode in ("abx", "aby", "zpx", "zpy"):
            idx = state["X"] if mode.endswith("x") else state["Y"]
            if idx is None:
                raise _Fail("indexed by unknown register")
            base = ins.operand
            data = np.asarray(self.image[base : base + _TABLE_SPAN], dtype=np.uint8)
            if len(data) == 0:
                raise _Fail("table out of range")
            return {"op": "table", "data": data, "index": idx, "stride": 1, "offset": 0}
        raise _Fail(f"unsupported read mode {mode}")

    def _set_nz(self, state, node):
        state["nz"] = node
        state["cmp"] = None

    def _apply(self, ins, state):  # pylint: disable=too-many-branches,too-many-statements
        """Apply one non-control-flow instruction to ``state`` (in place)."""
        name = ins.name
        if name in _LOAD_REG:
            node = self._operand_node(ins, state)
            state[_LOAD_REG[name]] = node
            self._set_nz(state, node)
        elif name in ("AND", "ORA", "EOR"):
            fn = {"AND": "and", "ORA": "or", "EOR": "xor"}[name]
            if state["A"] is None:
                raise _Fail("acc undefined")
            state["A"] = _binop(fn, state["A"], self._operand_node(ins, state))
            self._set_nz(state, state["A"])
        elif name in ("ADC", "SBC"):
            if state["A"] is None or state["C"] is None:
                raise _Fail("acc/carry undefined")
            if state["C"] not in (0, 1):
                raise _Fail("symbolic carry chain")  # deferred (multi-byte carry)
            b = self._operand_node(ins, state)
            if name == "ADC":
                expr = _binop("add", state["A"], b)
                if state["C"] == 1:
                    expr = _binop("add", expr, _const(1))
            else:
                expr = _binop("sub", state["A"], b)
                if state["C"] == 0:
                    expr = _binop("sub", expr, _const(1))
            state["A"] = expr
            state["C"] = None
            self._set_nz(state, expr)
        elif name == "ASL" and ins.mode == "acc":
            if state["A"] is None:
                raise _Fail("acc undefined")
            state["A"] = _binop("add", state["A"], state["A"])
            self._set_nz(state, state["A"])
        elif name in ("CMP", "CPX", "CPY"):
            reg = {"CMP": "A", "CPX": "X", "CPY": "Y"}[name]
            if state[reg] is None:
                raise _Fail("compare of undefined register")
            operand = self._operand_node(ins, state)
            state["cmp"] = (state[reg], operand)
            state["nz"] = _binop("sub", state[reg], operand)
        elif name == "CLC":
            state["C"] = 0
        elif name == "SEC":
            state["C"] = 1
        elif name in ("TAX", "TXA", "TAY", "TYA"):
            src, dst = {"TAX": ("A", "X"), "TXA": ("X", "A"), "TAY": ("A", "Y"), "TYA": ("Y", "A")}[
                name
            ]
            state[dst] = state[src]
            self._set_nz(state, state[dst])
        elif name in ("INX", "INY", "DEX", "DEY"):
            reg = "X" if "X" in name else "Y"
            if state[reg] is None:
                raise _Fail("inc/dec of undefined register")
            state[reg] = _binop("add", state[reg], _const(1 if name[0] == "I" else 0xFF))
            self._set_nz(state, state[reg])
        elif name in ("INC", "DEC"):
            # A read-modify-write on a RAM cell: the store back updates the cell,
            # so a later LDA of it reads the post-update value through the sampler.
            # Drop any stale symbolic value; the value is not needed by the store's
            # A/X/Y chain (an INC'd counter is read via its cell).
            if ins.mode in ("zp", "abs"):
                state["mem"].pop(int(ins.operand), None)
            state["nz"] = None
            state["cmp"] = None
        elif name == "PHA":
            state["stack"].append(state["A"])
        elif name == "PLA":
            if not state["stack"]:
                raise _Fail("PLA underflow")
            state["A"] = state["stack"].pop()
            self._set_nz(state, state["A"])
        elif name in ("STA", "STX", "STY"):
            self._store(ins, state)
        elif name in ("NOP", "PHP", "PLP", "CLD", "SED", "CLI", "SEI", "CLV", "TSX", "TXS", "BIT"):
            pass
        else:
            raise _Fail(f"unsupported opcode {name}")

    def _store(self, ins, state):
        """A non-terminal store: record symbolic RAM for a later same-call load."""
        if ins.mode in ("zp", "abs"):
            state["mem"][int(ins.operand)] = state[_STORE_REG[ins.name]]
        # indexed / indirect stores leave no statically-resolvable cell -> ignore.

    def _branch_pred(self, ins, state):
        """The ``cmpsel`` predicate ``(cmp, a, b)`` a conditional branch lifts to."""
        cmp, kind = _BRANCH[ins.name]
        if kind == "cmp":
            if state.get("cmp") is None:
                raise _Fail("carry branch without a compare")
            a, b = state["cmp"]
            return cmp, a, b
        nz = state.get("nz")
        if nz is None:
            raise _Fail("flag branch without a flag source")
        if cmp in ("eq", "ne"):  # tested against zero
            return cmp, nz, _const(0)
        return cmp, nz, None

    def run(self, start_pc, store_pc):
        """Symbolically execute from ``start_pc`` to the terminal store."""
        state = {
            "A": None,
            "X": None,
            "Y": None,
            "C": None,
            "nz": None,
            "cmp": None,
            "stack": [],
            "mem": {},
        }
        # A shared total-step budget across all forked branch paths bounds the worst
        # case (2^branches) so complex real-player code fails closed fast, keeping
        # the per-tune analyze well under the CPU budget.
        self._steps = [_MAX_STEPS]
        return self._exec(start_pc, store_pc, state, _MAX_INSTRS)

    def _load_mem(self, ins, state):
        """Prefer a symbolic RAM value written earlier in the same call."""
        if ins.mode in ("zp", "abs") and int(ins.operand) in state["mem"]:
            node = state["mem"][int(ins.operand)]
            if node is not None:
                return node
        return None

    def _exec(self, pc, store_pc, state, budget):  # pylint: disable=too-many-branches
        while budget > 0:
            budget -= 1
            self._steps[0] -= 1
            if self._steps[0] <= 0:
                raise _Fail("total step budget exhausted")
            if pc == store_pc:
                ins = decode(self.image, pc)
                if ins is None or ins.name not in _STORE_REG:
                    raise _Fail("terminal is not a store")
                node = state[_STORE_REG[ins.name]]
                if node is None:
                    raise _Fail("stored register undefined")
                return node
            if pc not in self.covered:
                raise _Fail("left executed coverage")
            ins = decode(self.image, pc)
            if ins is None:
                raise _Fail("undecodable")
            if ins.name in _BRANCH:
                cmp, a, b = self._branch_pred(ins, state)
                target = (pc + ins.length + _signed8(ins.operand)) & 0xFFFF
                taken = self._exec(target, store_pc, _copy_state(state), budget)
                fall = self._exec(pc + ins.length, store_pc, _copy_state(state), budget)
                return {"op": "cmpsel", "cmp": cmp, "a": a, "b": b, "then": taken, "else": fall}
            if ins.name == "JMP" and ins.mode == "abs":
                pc = ins.operand
                continue
            if ins.name in ("RTS", "RTI", "JSR") or (ins.name == "JMP" and ins.mode == "ind"):
                raise _Fail("call/return/indirect jump crosses the slice")
            if ins.name in _LOAD_REG:
                mem = self._load_mem(ins, state)
                if mem is not None:
                    state[_LOAD_REG[ins.name]] = mem
                    self._set_nz(state, mem)
                    pc += ins.length
                    continue
            self._apply(ins, state)
            pc += ins.length
        raise _Fail("instruction budget exhausted")


def _signed8(v):
    return ((int(v) & 0xFF) ^ 0x80) - 0x80


def _copy_state(state):
    new = dict(state)
    new["stack"] = list(state["stack"])
    new["mem"] = dict(state["mem"])
    return new


# -- public lifter entry points -----------------------------------------------


def lift_store(image, covered, store_pc, sid_addr, is_smc):
    """Lift one ``STA/STX/STY $D4xx`` store site to a grammar ``expr`` node.

    Returns the expression feeding the store (an ``ir`` ``expr`` node, without the
    outer ``post`` wrapper), or ``None`` when the static slice leaves the grammar
    envelope (I/O reads, calls, loops, symbolic carry chains, budget exhaustion).
    """
    if image is None:
        return None
    ins = decode(image, store_pc)
    if ins is None or ins.name not in _STORE_REG:
        return None
    start = block_start(image, covered, store_pc)
    slicer = _Slicer(image, covered, sid_addr, is_smc)
    try:
        return slicer.run(start, store_pc)
    except (_Fail, RecursionError):
        return None


_CONE_IDX = ("abx", "aby", "zpx", "zpy", "indx", "indy")


def slice_cone(image, covered, store_pc, is_smc, max_instrs=_MAX_INSTRS):
    """Ordered input cone of the code that emits ``store_pc``'s value.

    Decodes the executed instructions from the slice's entry to the store and
    collects every input the value reads as an ordered ``(id, sample)`` spec:
    absolute/zero-page RAM operands and self-modifying immediates -> ``(addr,
    "write")``; live I/O reads (osc3/env3, CIA) -> ``(addr, "chip")``; indexed
    reads whose effective address varies per frame -> ``(read_pc, "readpc")`` (the
    logged per-frame read value is the input, whatever the address). Unlike a full
    symbolic slice this succeeds even when the value function itself resists static
    lifting (within-call loops, unusual dataflow), giving Tier 3 the dataflow cone
    to memoise over (GENERIC_RECOVERY.md 3.5). Returns an ordered, de-duplicated
    list (a code-derived cone, never "every changing cell").
    """
    if image is None:
        return []
    ins = decode(image, store_pc)
    if ins is None or ins.name not in _STORE_REG:
        return []
    pc = block_start(image, covered, store_pc, max_instrs)
    seen, cone = set(), []
    for _ in range(max_instrs + 1):
        cur = decode(image, pc)
        if cur is None:
            break
        spec = None
        if cur.mode == "imm" and cur.name in _CONE_IMM and is_smc(cur.pc + 1):
            spec = (int(cur.pc + 1), "write")
        elif cur.mode in ("zp", "abs") and cur.name in _CONE_ABS:
            spec = (int(cur.operand), "chip" if _is_io(cur.operand) else "write")
        elif cur.mode in _CONE_IDX and cur.name in _CONE_ABS:
            spec = (int(cur.pc), "readpc")
        if spec is not None and spec not in seen:
            seen.add(spec)
            cone.append(spec)
        if pc == store_pc:
            break
        pc += cur.length
    return cone


# A structured key packs arbitrarily many byte columns, so the witness is not
# capped at 7 inputs -- a 16-bit SMC accumulator or an unrolled multi-site writer
# (Blackout/Commando FREQ) needs a wider determining set. The picks cap bounds the
# greedy cost; the pool bounds the candidate set considered.
_WITNESS_CAP = 16
_WITNESS_POOL = 40


def _pure_count(enc, target) -> int:
    """Frames in groups (by input tuple ``enc``) whose output is constant.

    Adding an input column only splits groups, so this is monotone -- the greedy
    objective for selecting the determining input subset. Equals ``n`` iff the
    columns determine the register (an exact witness).
    """
    _uniq, inv = np.unique(enc, return_inverse=True)
    inv = np.asarray(inv).ravel()
    k = int(inv.max()) + 1 if len(inv) else 0
    gmin = np.full(k, np.iinfo(np.int64).max, dtype=np.int64)
    gmax = np.full(k, np.iinfo(np.int64).min, dtype=np.int64)
    np.minimum.at(gmin, inv, target)
    np.maximum.at(gmax, inv, target)
    return int((gmin == gmax)[inv].sum())


def _select_determining(cols, target, cap=_WITNESS_CAP):
    """Greedily pick up to ``cap`` columns maximising output-determining frames.

    Returns the chosen column indices (cone order, ties to earliest). Only the
    inputs that actually reduce collisions are kept, so noise cells never dilute
    the witness key -- the memoised mapping is exact whenever the cone determines
    the output (GENERIC_RECOVERY.md 3.5 premises 1-3), else maximally pure.
    """
    n = len(target)
    chosen, chosen_cols, best = [], [], -1
    while len(chosen) < cap:
        add, add_score = None, best
        for i, col in enumerate(cols):
            if i in chosen:
                continue
            score = _pure_count(_encode_rows(chosen_cols + [col]), target)
            if score > add_score:
                add, add_score = i, score
        if add is None:
            break
        chosen.append(add)
        chosen_cols.append(cols[add])
        best = add_score
        if best >= n:
            break
    return chosen


def _witness_from_specs(series, specs, sid_addr, ctx):
    """Build a Tier-3 witness descriptor from a code-derived input cone.

    Samples each input spec per frame (RAM cell / SMC immediate / live chip read /
    per-frame read-PC value), greedily keeps the determining subset (<= 7 -- one
    int64 key), then memoises the observed ``inputs -> value`` mapping. Exact when
    the cone determines the register (the totality guarantee); a residual collision
    leaves that key at its first observed value (scored below 1.0, so a fitting
    tree still wins).
    """
    sampler = ctx.sampler
    n = ctx.n_frames
    if sampler is None or n == 0:
        return None
    seen, uniq = set(), []
    for spec in specs:
        key = (int(spec[0]), spec[1])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(key)
        if len(uniq) >= _WITNESS_POOL:
            break
    if not uniq:
        return None
    target = np.asarray(series, dtype=np.int64)[:n]
    cols = [
        np.asarray(_witness_col(spec, int(sid_addr), sampler), dtype=np.int64)[:n] & 0xFF
        for spec in uniq
    ]
    keep = _select_determining(cols, target)
    if not keep:
        return None
    specs_kept = [uniq[i] for i in keep]
    sel = [np.asarray(cols[i], dtype=np.uint8) for i in keep]
    _keys, first = np.unique(_encode_rows(sel), return_index=True)
    stacked = np.ascontiguousarray(np.stack(sel, axis=1))  # (n, k) byte rows
    return {
        "type": "WITNESS",
        "addr": int(sid_addr),
        "sid": int(sid_addr),
        "specs": specs_kept,
        "krows": stacked[first].astype(np.uint8),
        "values": target[first].astype(np.int64),
    }


def propose_lift(series, sid_addr, store_pcs, ctx):
    """Tier-2 lift + Tier-3 witness candidates for a sub-1.0 register.

    Runs last and only for registers the value-stream proposers leave imperfect
    (bounded, fail-closed). Returns zero or more descriptors: a ``LIFT`` per store
    site whose static slice grounds to a grammar tree, plus one ``WITNESS`` backstop
    memoising the store's code-derived input cone (when the code exposes one).
    """
    image = getattr(ctx, "ram", None)
    covered = getattr(ctx, "covered_pcs", None)
    if image is None or covered is None or not store_pcs:
        return []
    is_smc = _smc_predicate(ctx)
    out = []
    cone = []
    seen = set()
    for pc in store_pcs:
        expr = lift_store(image, covered, int(pc), int(sid_addr), is_smc)
        if expr is not None:
            tree = ir._post(  # pylint: disable=protected-access
                expr, {"addr": int(sid_addr), "sid": int(sid_addr)}, width_mask=0xFF
            )
            out.append(
                {
                    "type": "LIFT",
                    "addr": int(sid_addr),
                    "sid": int(sid_addr),
                    "tree": tree,
                    "tier": 2,
                    "store_pc": int(pc),
                }
            )
        for spec in slice_cone(image, covered, int(pc), is_smc):
            if spec not in seen:
                seen.add(spec)
                cone.append(spec)
    # Supplement the static (PC-decoded) cone with the log-based read-PC cone: reads
    # inside called subroutines a bounded static slice cannot reach. The greedy
    # subset selection keeps only the determining inputs (GENERIC_RECOVERY.md 3.5).
    pcs_near = getattr(ctx, "read_pcs_near", None)
    if pcs_near is not None:
        for rpc in sorted(pcs_near.get(int(sid_addr), ())):
            spec = (int(rpc), "readpc")
            if spec not in seen:
                seen.add(spec)
                cone.append(spec)
    wit = _witness_from_specs(series, cone, sid_addr, ctx)
    if wit is not None:
        out.append(wit)
    return out


def _smc_predicate(ctx):
    """A predicate: is a code byte address itself a written RAM cell (self-modifying)?"""
    written = getattr(ctx, "written_cells", None)
    if written is None:
        return lambda _addr: False
    return lambda addr: int(addr) in written
