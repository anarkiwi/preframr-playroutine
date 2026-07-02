"""Typed expression IR for register reconstruction.

``to_ir`` maps a recovered descriptor (the dicts produced by
:func:`recover.classify_register`) to a node tree; ``evaluate`` runs one
recursive interpreter over it. The per-op arithmetic is ported verbatim from the
former ``recover._recon_*`` bodies -- int64 dtypes, ``np.clip`` bounds and
``& 0xFF`` placements are load-bearing and must not be "improved". Byte-for-byte
parity with the old path is pinned by ``tests/test_ir.py``.

Node schemas (plain dicts; numpy arrays allowed as values)::

    {"op":"post", "expr":node, "byte_role":"full"|"lo"|"hi",
     "width_mask":int|None, "overrides":[...], "prelude":dict|None,
     "addr":int, "sid":int|None}
    {"op":"const", "value":int}
    {"op":"literal", "data":array}
    {"op":"seq", "frames":[...], "values":[...]}
    {"op":"cell", "addr":int, "sample":"write"|"eof"|"operand", "sid":int}
    {"op":"lohi", "lo":node, "hi":node}
    {"op":"table", "data":uint8[], "index":node, "stride":int, "offset":int}
    {"op":"binop", "fn":"or"|"and"|"xor"|"add"|"sub", "a":node, "b":node}
    {"op":"recur", ...}    # every BACC field; ports _recon_bacc_full (incl tickband)
    {"op":"cutoff", ...}   # every CUTOFF field; ports _recon_cutoff
"""

import numpy as np

# NOTE: the low-level recurrence kernels and ``combine_lohi`` live in
# ``recover`` and are imported lazily inside the functions that use them --
# ``recover`` imports this module, so a module-level import would be circular.

# -- bounded-accumulator recurrence (ported from recover._recon_bacc_full) ----


def _run_recurrence(
    mode,
    step,
    lo,
    hi,
    seed,
    length,
    modulus,
    down_step=None,
    clamp_lo=None,
    clamp_hi=None,
    direction=None,
) -> np.ndarray:
    """Regenerate one bounded-accumulator segment from a seed."""
    from . import recover  # pylint: disable=import-outside-toplevel,cyclic-import

    if length <= 0:
        return np.empty(0, dtype=np.int64)
    if mode == "pingpong" and hi > lo:
        d = (1 if seed <= (lo + hi) // 2 else -1) if direction is None else direction
        clo = lo if clamp_lo is None else clamp_lo
        chi = hi if clamp_hi is None else clamp_hi
        ds = step if down_step is None else down_step
        series, _ = recover._simulate_pingpong(  # pylint: disable=protected-access
            lo, hi, clo, chi, step, ds, int(seed), d, length
        )
        return series
    if mode == "reflect" and hi > lo:
        d = (1 if seed <= (lo + hi) // 2 else -1) if direction is None else direction
        series, _ = recover._simulate_reflect(  # pylint: disable=protected-access
            lo, hi, step, int(seed), d, length
        )
        return series
    mod = modulus if modulus else (hi - lo + step)
    if mod <= 0:
        return np.full(length, int(seed), dtype=np.int64)
    return lo + ((int(seed) - lo) + step * np.arange(length, dtype=np.int64)) % mod


def _recon_tickband(desc, n) -> np.ndarray:
    """Regenerate a tick-banded reflecting sweep from its per-segment rate tables."""
    from . import recover  # pylint: disable=import-outside-toplevel,cyclic-import

    resets = list(desc.get("resets", [0]))
    seeds = list(desc.get("seeds", []))
    dirs = list(desc.get("directions", []))
    seg_tables = list(desc.get("seg_tables", []))
    tables = desc.get("rate_tables", [])
    lo, hi = desc["lo"], desc["hi"]
    if not resets or resets[0] != 0:
        resets = [0] + resets
        seeds = [seeds[0] if seeds else lo] + seeds
        dirs = [dirs[0] if dirs else 1] + dirs
        seg_tables = [seg_tables[0] if seg_tables else 0] + seg_tables
    bounds = resets + [n]
    out = np.zeros(n, dtype=np.int64)
    for i, start in enumerate(resets):
        length = bounds[i + 1] - start
        if length <= 0:
            continue
        seed = seeds[i] if i < len(seeds) else lo
        d = dirs[i] if i < len(dirs) else 1
        ti = seg_tables[i] if i < len(seg_tables) else 0
        rate = tables[ti] if ti < len(tables) else np.zeros(0, dtype=np.int64)
        out[start : start + length] = (
            recover._simulate_tickreflect(  # pylint: disable=protected-access
                lo, hi, rate, int(seed), int(d), length
            )
        )
    return out


def _recon_product(desc, n) -> np.ndarray:
    """Regenerate a segmented step x boundary product recurrence.

    The general per-frame kernel (``recover._simulate_recur``) covers every cell
    of ``{const, updown, table} x {wrap, saw, reflect, clampflip}`` from one
    reseeded descriptor: ``step_kind``/``boundary`` name the axis values,
    per-segment ``seeds``/``directions``/``steps``/``down_steps`` carry the local
    state, and ``rate_tables``/``seg_tables`` hold the shared tick-rate programs
    (``table`` step). The scalar-mode kernels above are the fixture-exercised
    subset; this path evaluates the axis cells they cannot spell.
    """
    from . import recover  # pylint: disable=import-outside-toplevel,cyclic-import

    resets = list(desc.get("resets", [0]))
    seeds = list(desc.get("seeds", []))
    dirs = list(desc.get("directions", []))
    steps = list(desc.get("steps", []))
    down_steps = list(desc.get("down_steps", []))
    seg_tables = list(desc.get("seg_tables", []))
    tables = desc.get("rate_tables", [])
    lo, hi = int(desc["lo"]), int(desc["hi"])
    step_kind = desc.get("step_kind", "const")
    boundary = desc.get("boundary", "reflect")
    modulus = desc.get("modulus")
    if not resets or resets[0] != 0:
        resets = [0] + resets
        seeds = [seeds[0] if seeds else lo] + seeds
    bounds = resets + [n]
    out = np.zeros(n, dtype=np.int64)
    for i, start in enumerate(resets):
        length = bounds[i + 1] - start
        if length <= 0:
            continue
        seed = seeds[i] if i < len(seeds) else lo
        d = dirs[i] if i < len(dirs) else 1
        up = steps[i] if i < len(steps) else int(desc.get("step", 1))
        down = down_steps[i] if i < len(down_steps) else int(desc.get("down_step", up))
        rate = None
        if step_kind == "table":
            ti = seg_tables[i] if i < len(seg_tables) else 0
            rate = tables[ti] if ti < len(tables) else np.zeros(0, dtype=np.int64)
        out[start : start + length] = recover._simulate_recur(  # pylint: disable=protected-access
            lo,
            hi,
            int(seed),
            int(d),
            length,
            boundary,
            step_kind,
            int(up),
            int(down),
            rate,
            modulus,
        )
    return out


def _recon_bacc_full(desc, n) -> np.ndarray:
    """Regenerate the full (8- or 16-bit) accumulator series from its descriptor."""
    if desc.get("mode") == "tickband":
        return _recon_tickband(desc, n)
    if desc.get("mode") == "product":
        return _recon_product(desc, n)
    resets = list(desc.get("resets", [0]))
    seeds = list(desc.get("seeds", [desc.get("phase", desc.get("lo", 0))]))
    if not resets or resets[0] != 0:
        resets = [0] + resets
        seeds = [seeds[0] if seeds else desc.get("lo", 0)] + seeds
    bounds = resets + [n]
    out = np.zeros(n, dtype=np.int64)
    mode = desc["mode"]
    step = desc["step"]
    lo = desc["lo"]
    hi = desc["hi"]
    modulus = desc.get("modulus")
    down_step = desc.get("down_step")
    clamp_lo = desc.get("clamp_lo")
    clamp_hi = desc.get("clamp_hi")
    steps = desc.get("steps")
    down_steps = desc.get("down_steps")
    directions = desc.get("directions")
    for i, start in enumerate(resets):
        length = bounds[i + 1] - start
        seed = seeds[i] if i < len(seeds) else lo
        st = steps[i] if steps and i < len(steps) else step
        ds = down_steps[i] if down_steps and i < len(down_steps) else down_step
        di = directions[i] if directions and i < len(directions) else None
        out[start : start + length] = _run_recurrence(
            mode, st, lo, hi, seed, length, modulus, ds, clamp_lo, clamp_hi, di
        )
    return out


# -- defMON filter-cutoff SMC routine (ported from recover._recon_cutoff) -----


def _recon_cutoff(desc, n, sampler) -> np.ndarray:
    """Regenerate a defMON filter-cutoff register from its SMC micro-routine cells."""
    del n
    if sampler is None:
        return np.zeros(0, dtype=np.int64)
    c = desc["cells"]
    sid = int(desc["sid"])

    def s(addr):
        return sampler.at_write(int(addr), sid).astype(np.int64) & 0xFF

    def pre(addr):
        return sampler.operand(int(addr), sid).astype(np.int64) & 0xFF

    hi, lo = s(c["hi"]), s(c["lo"])
    slo, shi = s(c["slo"]), s(c["shi"])
    op_lo, op_hi = s(c["op_lo"]), s(c["op_hi"])
    imm = s(c["imm"])
    ihi, ilo = pre(c["hi"]), pre(c["lo"])
    add_lo = op_lo == 0x69
    carry_lo = np.where(add_lo, (ilo + slo) > 0xFF, (ilo - slo - 1) >= 0).astype(np.int64)
    add_hi = op_hi == 0x69
    hi_sum = np.where(add_hi, ihi + shi + carry_lo, ihi - shi - (1 - carry_lo))
    carry_rec = np.where(add_hi, hi_sum > 0xFF, hi_sum >= 0).astype(np.int64)
    carry_dir = ((hi * 256 + lo) < (ihi * 256 + ilo)).astype(np.int64)
    valid = np.isin(op_lo, (0x69, 0xE9)) & np.isin(op_hi, (0x69, 0xE9))
    carry_hi = np.where(valid, carry_rec, carry_dir)
    base = int(desc["base"])
    scale = int(desc.get("scale", 1))
    a = (hi + imm + carry_hi) & 0xFF
    emit = np.where((a < base) | (a >= 0x80), base, a)
    return ((emit * scale) & 0xFF).astype(np.int64)


# -- overrides (moved from recover; accepts legacy tuples and typed dicts) -----


def _predicate_terms(predicate):
    """Normalise each override predicate term to ``("mask", cell, mask, value)``
    or ``("in", cell, values)``, accepting both legacy tuples and typed dicts.

    Legacy tuples: ``(cell, int_mask, value)`` and ``(cell, "in", values)``.
    Typed dicts: ``{"kind":"eq"|"bit"|"mask", "cell", "mask"?, "value"}`` and
    ``{"kind":"in", "cell", "values"}``. ``eq`` is mask ``0xFF``, ``bit`` a
    single-bit mask -- all three go through one mask/value path.
    """
    for term in predicate:
        if isinstance(term, dict):
            kind = term["kind"]
            cell = int(term["cell"])
            if kind == "in":
                yield ("in", cell, tuple(int(x) for x in term["values"]))
            elif kind == "eq":
                yield ("mask", cell, 0xFF, int(term["value"]))
            else:  # "bit" or "mask" -- both are a mask/value equality test
                yield ("mask", cell, int(term["mask"]), int(term["value"]))
        else:
            cell, cmask, cval = term
            if cmask == "in":
                yield ("in", int(cell), tuple(int(x) for x in cval))
            else:
                yield ("mask", int(cell), int(cmask), int(cval))


def _apply_overrides(out, overrides, sampler) -> np.ndarray:
    """Force values where each override's cell-predicate conjunction holds."""
    if not overrides or sampler is None:
        return out
    n = len(out)
    for ov in overrides:
        sel = np.ones(n, dtype=bool)
        for term in _predicate_terms(ov.get("predicate", [])):
            col = sampler.eof(term[1])
            if term[0] == "in":
                sel &= np.isin(col, np.asarray(term[2], dtype=col.dtype))
            else:
                sel &= (col & term[2]) == term[3]
        out = np.where(sel, int(ov["force"]), out)
    return out


def _default_until_first_write(recon, descriptor, sampler):
    """Hold the power-on default (``0``) on frames before a register's first write.

    The oracle carries ``0`` until the player's first store to a register, but a
    reconstruction back-fills its recovered value to frame 0, mismatching those
    leading frames. Zero them to mirror the oracle. Strictly non-worsening: those
    frames are ``0`` in the oracle by construction.
    """
    if recon is None or sampler is None:
        return recon
    addr = descriptor.get("addr")
    if addr is None:
        return recon
    written = sampler.written_mask(addr)
    if written.all():
        return recon
    recon = np.array(recon, dtype=np.int64, copy=True)
    recon[~written] = 0
    return recon


# -- recursive evaluator ------------------------------------------------------


def _ev_const(node, n, _sampler):
    return np.full(n, int(node["value"]), dtype=np.int64)


def _ev_literal(node, _n, _sampler):
    return np.asarray(node["data"], dtype=np.int64)


def _ev_seq(node, n, _sampler):
    frames = np.asarray(node["frames"], dtype=np.int64)
    values = np.asarray(node["values"], dtype=np.int64)
    idx = np.clip(np.searchsorted(frames, np.arange(n), side="right") - 1, 0, len(values) - 1)
    return values[idx]


def _ev_cell(node, n, sampler):
    if sampler is None:
        return np.zeros(n, dtype=np.int64)
    sample = node["sample"]
    if sample == "eof":
        return sampler.eof(node["addr"])
    if sample == "operand":
        return sampler.operand(node["addr"], node["sid"])
    return sampler.at_write(node["addr"], node["sid"])


def _ev_lohi(node, n, sampler):
    from . import recover  # pylint: disable=import-outside-toplevel,cyclic-import

    return recover.combine_lohi(evaluate(node["lo"], n, sampler), evaluate(node["hi"], n, sampler))


def _ev_table(node, n, sampler):
    table = np.asarray(node["data"], dtype=np.int64)
    index = np.asarray(evaluate(node["index"], n, sampler), dtype=np.int64)
    idx = np.clip(index * int(node["stride"]) + int(node["offset"]), 0, len(table) - 1)
    return table[idx]


def _ev_binop(node, n, sampler):
    a = np.asarray(evaluate(node["a"], n, sampler), dtype=np.int64)
    b = np.asarray(evaluate(node["b"], n, sampler), dtype=np.int64)
    fn = node["fn"]
    if fn == "or":
        return a | b
    if fn == "and":
        return a & b
    if fn == "xor":
        return a ^ b
    if fn == "sub":
        return a - b
    return a + b


def _ev_recur(node, n, _sampler):
    return _recon_bacc_full(node, n)


def _ev_cutoff(node, n, sampler):
    if sampler is None:
        return np.zeros(n, dtype=np.int64)
    return _recon_cutoff(node, n, sampler)


_EVAL = {
    "const": _ev_const,
    "literal": _ev_literal,
    "seq": _ev_seq,
    "cell": _ev_cell,
    "lohi": _ev_lohi,
    "table": _ev_table,
    "binop": _ev_binop,
    "recur": _ev_recur,
    "cutoff": _ev_cutoff,
}


def evaluate(node, n, sampler):
    """Evaluate a node tree to a per-frame int64 array (or ``None``)."""
    if node is None:
        return None
    op = node["op"]
    if op == "post":
        return _eval_post(node, n, sampler)
    return _EVAL[op](node, n, sampler)


# -- MDL description-length cost (Phase 2 arbiter) -----------------------------
#
# ``complexity(tree)`` is the description length the arbiter trades against
# fidelity: node count (~1 each), override/predicate terms (~1 each), and the
# *captured-state* cost of every value-replay cell -- ``CAPTURED_W *
# changed_frames(cell) / n_frames``. Charging per *change* (not per frame held)
# is deliberate: event-latched song data (SEQ latches, note streams, generative
# melody output) is captured by design and stays cheap, while a feeder replaying
# per-frame modulation is expensive. Structured generators (``recur``,
# ``cutoff``) and derivable table indices/cursors carry only a small parameter
# cost, so on equal fidelity the arbiter prefers a closed form over raw replay --
# reproducing the old cascade's structured-first ordering. ``CAPTURED_W`` is
# calibrated (up from the nominal 0.5) so a fully-modulated replay cell outweighs
# a structured tree's node count; the calibration test pins this on the perfect
# set.
CAPTURED_W = 8.0
INDEX_COST = 0.2  # a derivable cursor/index cell (not replayed modulation)
PARAM_COST = 0.1  # a closed-form generator parameter (recur seed, cutoff cell)
# A SEQ/prelude latch is captured event data charged *per latch* (absolute, not
# per frame held) so event-latched song data stays cheap, yet a structured
# generator that actually fits (recur/table walk) still wins the arbiter over a
# latch list replaying its output. Modest, so genuine sparse latches stay cheaper
# than a per-frame feeder.
LATCH_COST = 2.0


def _changed_frames(series) -> int:
    """Number of value changes (+1) in a sampled cell series -- its latch count."""
    s = np.asarray(series, dtype=np.int64).ravel()
    if len(s) == 0:
        return 0
    return int(np.count_nonzero(np.diff(s) != 0)) + 1


def _cell_changed(node, sampler, n) -> int:
    """Changed-frame (latch) count of a ``cell`` node's sampled series."""
    if sampler is None or not n:
        return 0
    return _changed_frames(_ev_cell(node, n, sampler))


def _index_cost(node) -> float:
    """Cost of a derivable table index/cursor subtree (nodes only, no capture)."""
    if node is None:
        return 0.0
    op = node["op"]
    if op == "cell":
        return 1.0 + INDEX_COST
    if op == "binop":
        return 1.0 + _index_cost(node["a"]) + _index_cost(node["b"])
    if op == "lohi":
        return 1.0 + _index_cost(node["lo"]) + _index_cost(node["hi"])
    if op == "table":
        return 1.0 + _index_cost(node["index"])
    return 1.0


def _cc_post(node, sampler, n):
    cost, cap = _cost_captured(node["expr"], sampler, n)
    cost += 1.0
    for ov in node.get("overrides", []):
        cost += 1.0 + sum(1 for _ in _predicate_terms(ov.get("predicate", [])))
    prelude = node.get("prelude")
    if prelude and prelude.get("end"):
        cost += LATCH_COST * len(prelude.get("values", [0]))
    return cost, cap


def _cc_seq(node, _sampler, _n):
    return 1.0 + LATCH_COST * len(node.get("values", [])), 0


def _cc_literal(node, _sampler, n):
    cf = _changed_frames(node["data"])
    return 1.0 + CAPTURED_W * cf / max(1, n or len(node["data"]) or 1), cf


def _cc_cell(node, sampler, n):
    cf = _cell_changed(node, sampler, n)
    return 1.0 + CAPTURED_W * cf / max(1, n or 1), cf


def _cc_lohi(node, sampler, n):
    cl, capl = _cost_captured(node["lo"], sampler, n)
    ch, caph = _cost_captured(node["hi"], sampler, n)
    return 1.0 + cl + ch, capl + caph


def _cc_binop(node, sampler, n):
    ca, capa = _cost_captured(node["a"], sampler, n)
    cb, capb = _cost_captured(node["b"], sampler, n)
    return 1.0 + ca + cb, capa + capb


def _cc_table(node, _sampler, _n):
    return 1.0 + _index_cost(node["index"]), 0


def _cc_recur(node, _sampler, n):
    # Closed-form parameters (seeds/resets) are cheap; a tick-rate table is
    # captured per-frame stride data, so it is charged like a replay cell
    # (CAPTURED_W * captured-strides / n). This prices the old tickband anti-theft
    # heuristic ("few shared tables") into MDL: many distinct rate tables => high
    # cost, so a per-frame stride replay never out-scores a genuine generator.
    params = len(node.get("seeds", [])) + len(node.get("resets", []))
    rate = int(sum(len(t) for t in node.get("rate_tables", [])))
    cost = 1.0 + PARAM_COST * params + CAPTURED_W * rate / max(1, n or 1)
    return cost, rate


def _cc_cutoff(node, _sampler, _n):
    return 1.0 + PARAM_COST * len(node.get("cells", {})), 0


_COST = {
    "post": _cc_post,
    "const": lambda node, _s, _n: (1.0, 0),
    "seq": _cc_seq,
    "literal": _cc_literal,
    "cell": _cc_cell,
    "lohi": _cc_lohi,
    "binop": _cc_binop,
    "table": _cc_table,
    "recur": _cc_recur,
    "cutoff": _cc_cutoff,
}


def _cost_captured(node, sampler, n):
    """(description-length cost, captured-frame total) of a node tree."""
    if node is None:
        return 0.0, 0
    handler = _COST.get(node["op"])
    return handler(node, sampler, n) if handler else (1.0, 0)


def cost_captured(node, sampler=None, n_frames=None):
    """``(complexity, captured_frames)`` of a node tree in a single pass."""
    return _cost_captured(node, sampler, n_frames)


def complexity(node, sampler=None, n_frames=None) -> float:
    """MDL description-length cost of a node tree (see module note)."""
    return _cost_captured(node, sampler, n_frames)[0]


def captured_frames(node, sampler=None, n_frames=None) -> int:
    """Total captured (replayed) latch-frames referenced by a node tree."""
    return _cost_captured(node, sampler, n_frames)[1]


def _eval_post(node, n, sampler):
    """Apply the fixed post-pipeline: mask, byte extract, overrides, prelude, default."""
    v = evaluate(node["expr"], n, sampler)
    if v is None:
        return None
    width_mask = node.get("width_mask")
    if width_mask is not None:
        v = v & int(width_mask)
    role = node.get("byte_role", "full")
    if role == "lo":
        v = v & 0xFF
    elif role == "hi":
        v = (v >> 8) & 0xFF
    v = _apply_overrides(v, node.get("overrides", []), sampler)
    prelude = node.get("prelude")
    if prelude and prelude.get("end"):
        pre = evaluate(
            {
                "op": "seq",
                "frames": prelude.get("frames", [0]),
                "values": prelude.get("values", [0]),
            },
            n,
            sampler,
        )
        v = np.where(np.arange(n) < int(prelude["end"]), pre & 0xFF, v)
    return _default_until_first_write(v, {"addr": node.get("addr")}, sampler)


# -- descriptor -> node tree --------------------------------------------------


def _prelude_of(desc):
    """Build a post prelude dict from a descriptor's held-seed fields (or None)."""
    end = desc.get("prelude_end")
    if not end:
        return None
    return {
        "end": int(end),
        "frames": desc.get("prelude_frames", [0]),
        "values": desc.get("prelude_values", [0]),
    }


def _post(expr, desc, **kw):
    """Wrap ``expr`` in a post node, defaulting addr/sid/overrides from ``desc``."""
    node = {
        "op": "post",
        "expr": expr,
        "byte_role": kw.get("byte_role", "full"),
        "width_mask": kw.get("width_mask"),
        "overrides": kw.get("overrides", []),
        "prelude": kw.get("prelude"),
        "addr": desc.get("addr"),
        "sid": desc.get("sid"),
    }
    return node


def _const_ir(d):
    return _post({"op": "const", "value": d.get("value") or 0}, d)


def _seq_ir(d):
    return _post(
        {"op": "seq", "frames": d.get("latch_frames", [0]), "values": d.get("latch_values", [0])},
        d,
    )


def _feeder_ir(d):
    if d.get("cell") is None:  # bare XSTATE has no captured feeder cell
        return None
    return _post(
        {"op": "cell", "addr": d["cell"], "sample": "write", "sid": d["sid"]}, d, width_mask=0xFF
    )


def _bacc_ir(d):
    if d.get("cell") is not None and d.get("sid") is not None:
        return _post(
            {"op": "cell", "addr": d["cell"], "sample": "write", "sid": d["sid"]},
            d,
            width_mask=0xFF,
            prelude=_prelude_of(d),
        )
    recur = dict(d)
    recur["op"] = "recur"
    return _post(recur, d, byte_role=d.get("byte_role") or "full")


def _cutoff_ir(d):
    cut = dict(d)
    cut["op"] = "cutoff"
    return _post(cut, d)


def _table_walk_ir(d):
    if d.get("cursor") is not None:  # a literal cursor series (test fixtures)
        idx = {"op": "literal", "data": d["cursor"]}
    elif d.get("cursor_addr") is not None:
        idx = {"op": "cell", "addr": d["cursor_addr"], "sample": "eof", "sid": d.get("addr")}
    else:
        return _post({"op": "const", "value": 0}, d)
    t = {
        "op": "table",
        "data": d["table"],
        "index": idx,
        "stride": int(d.get("stride", 1)),
        "offset": int(d.get("cursor_offset", 0)),
    }
    if d.get("gate_addr") is not None:
        second = {"op": "cell", "addr": d["gate_addr"], "sample": "eof", "sid": d.get("addr")}
    else:
        second = {"op": "const", "value": int(d.get("mask", 0xFF))}
    expr = {"op": "binop", "fn": "and", "a": t, "b": second}
    return _post(expr, d, overrides=d.get("overrides", []))


def _comp_part_ir(part):
    """A composite part -> node (16-bit lo/hi pair, 8-bit cell, or literal series)."""
    if part is None:
        return None
    if "series" in part:
        return {"op": "literal", "data": part["series"]}
    if "lo" in part:
        return {
            "op": "lohi",
            "lo": {"op": "cell", "addr": part["lo"][0], "sample": "write", "sid": part["lo"][1]},
            "hi": {"op": "cell", "addr": part["hi"][0], "sample": "write", "sid": part["hi"][1]},
        }
    return {"op": "cell", "addr": part["cell"], "sample": "write", "sid": part["sid"]}


def part_value(part, n, sampler):
    """Per-frame value of a single composite part (``None`` -> zeros).

    Exposes the part-level reconstruction (16-bit lo/hi pair, 8-bit cell, or
    literal series) that ``_composite_ir`` folds into a COMPOSITE tree, for
    recover.py's composite scoring.
    """
    node = _comp_part_ir(part)
    return np.zeros(n, dtype=np.int64) if node is None else evaluate(node, n, sampler)


def _composite_ir(d):
    base = _comp_part_ir(d.get("base"))
    mod = _comp_part_ir(d.get("mod"))
    if base is None:
        base = {"op": "const", "value": 0}
    expr = base if mod is None else {"op": "binop", "fn": "add", "a": base, "b": mod}
    return _post(
        expr,
        d,
        width_mask=d.get("width_mask", 0xFF),
        byte_role=d.get("byte_role", "full"),
        overrides=d.get("overrides", []),
    )


def _pitchwalk_ir(d):
    lo_table = np.asarray(d["lo_table"])
    hi_table = np.asarray(d["hi_table"])
    assert len(lo_table) == len(hi_table), "pitchwalk lo/hi tables must be equal length"
    cells = list(d.get("index_cells", []))
    idx = {"op": "cell", "addr": cells[0], "sample": "eof", "sid": d.get("addr")}
    for cell in cells[1:]:
        idx = {
            "op": "binop",
            "fn": "add",
            "a": idx,
            "b": {"op": "cell", "addr": cell, "sample": "eof", "sid": d.get("addr")},
        }
    expr = {
        "op": "lohi",
        "lo": {"op": "table", "data": lo_table, "index": idx, "stride": 1, "offset": 0},
        "hi": {"op": "table", "data": hi_table, "index": idx, "stride": 1, "offset": 0},
    }
    return _post(expr, d, byte_role=d.get("byte_role", "full"), overrides=d.get("overrides", []))


_BINOP_FN = {"XOR": "xor", "AND": "and", "OR": "or", "ADD": "add", "SUB": "sub"}


def _binop_ir(kind, d):
    """A ``{or,and,xor,add,sub}`` fold of two cells / a cell and a constant.

    The 8-bit form (``cell_a`` op ``cell_b``/``const``) covers the CTRL/MODE-VOL
    idioms; the 16-bit form (``base``/``mod`` lo-hi parts) is the ``add`` over a
    freq/PW lo-hi pair. Overrides and the held-seed prelude are applied uniformly
    when the descriptor carries them (absent for the plain folds)."""
    fn = _BINOP_FN[kind]
    if d.get("base") is not None:  # 16-bit lo/hi pair fold
        mod = _comp_part_ir(d.get("mod"))
        expr = {
            "op": "binop",
            "fn": fn,
            "a": _comp_part_ir(d["base"]),
            "b": mod if mod is not None else {"op": "const", "value": 0},
        }
        return _post(
            expr,
            d,
            width_mask=d.get("width_mask", 0xFFFF),
            byte_role=d.get("byte_role", "full"),
            overrides=d.get("overrides", []),
            prelude=_prelude_of(d),
        )
    a = {"op": "cell", "addr": d["cell_a"], "sample": "write", "sid": d["sid"]}
    if d.get("cell_b") is not None:
        second = {"op": "cell", "addr": d["cell_b"], "sample": "write", "sid": d["sid"]}
    else:
        second = {"op": "const", "value": int(d.get("const", 0))}
    expr = {"op": "binop", "fn": fn, "a": a, "b": second}
    return _post(expr, d, width_mask=0xFF, overrides=d.get("overrides", []), prelude=_prelude_of(d))


_TO_IR = {
    "CONST": _const_ir,
    "SEQ": _seq_ir,
    "FEEDER": _feeder_ir,
    "XSTATE": _feeder_ir,
    "BACC": _bacc_ir,
    "TABLE_WALK": _table_walk_ir,
    "COMPOSITE": _composite_ir,
    "PITCHWALK": _pitchwalk_ir,
    "CUTOFF": _cutoff_ir,
    "XOR": lambda d: _binop_ir("XOR", d),
    "AND": lambda d: _binop_ir("AND", d),
    "OR": lambda d: _binop_ir("OR", d),
    "ADD": lambda d: _binop_ir("ADD", d),
    "SUB": lambda d: _binop_ir("SUB", d),
}


def to_ir(descriptor):
    """Map a recovered descriptor to a node tree (or ``None`` for no model)."""
    if descriptor is None:
        return None
    handler = _TO_IR.get(descriptor.get("type"))
    return handler(descriptor) if handler else None
