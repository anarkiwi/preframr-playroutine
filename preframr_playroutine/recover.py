"""Recover per-register generators from a sidtrace v2 oracle.

Given the SID register oracle plus the v2 internal-state signals (per-frame RAM
write log, store-site PC of each SID write, and the static RAM image), recover
the generator that produces each SID register's per-frame value, classified in
the BACC / TABLE-WALK / SEQ / XSTATE taxonomy of
``/scratch/anarkiwi/cbm/re-trackers``:

- ``BACC``       bounded accumulator (per-frame add/sub with bound + saw / wrap /
                 reflect behaviour).
- ``TABLE_WALK`` a cursor cell stepping a static table, with loop-back.
- ``SEQ``        event-latched (sparse) writes from the note/pattern sequencer.
- ``XSTATE``     depends on persistent cross-function state (the fallback).
- ``CONST``      a single constant value.

Everything is per-frame: writes are binned into frames by the chosen interrupt
cadence (``Trace.tick_cycles``) and carried forward, the same technique as
``Trace.register_frames``. Per-frame stateful recurrences (the BACC fit) are
inherently sequential and use short python loops over the frame count; the bulk
binning / table search is vectorised numpy.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

from .trace import WIN_IRQ, WIN_NMI

# Per-voice CTRL (gate) register addresses.
CTRL_ADDRS = {0: 0xD404, 1: 0xD40B, 2: 0xD412}

_U64_MAX = np.uint64(np.iinfo(np.uint64).max)

StateSequence = namedtuple("StateSequence", ["ticks", "addrs", "grid"])


# -- per-frame binning ----------------------------------------------------


def _frame_bounds(ticks: np.ndarray) -> np.ndarray:
    """Upper cycle bound (exclusive) of each frame; last frame is open."""
    return np.append(ticks[1:], _U64_MAX)


def _carry_series(cycles: np.ndarray, values: np.ndarray, ticks: np.ndarray) -> np.ndarray:
    """Carry-forward value at the end of each frame for one ordered cell.

    ``cycles``/``values`` are the writes to a single cell (any order); returns
    an int64 series of length ``len(ticks)`` holding the last value written
    before each frame boundary (0 before the first write).
    """
    out = np.zeros(len(ticks), dtype=np.int64)
    if len(ticks) == 0 or len(cycles) == 0:
        return out
    order = np.argsort(cycles, kind="stable")
    wc = cycles[order]
    wv = values[order].astype(np.int64)
    pos = np.searchsorted(wc, _frame_bounds(ticks), side="left")
    taken = pos > 0
    idx = np.clip(pos - 1, 0, len(wv) - 1)
    out[taken] = wv[idx][taken]
    return out


def _changing_cells(wr: np.ndarray) -> np.ndarray:
    """Sorted uint16 addresses whose written value is not constant."""
    if len(wr) == 0:
        return np.empty(0, dtype=np.uint16)
    addr = wr["addr"]
    val = wr["value"]
    order = np.argsort(addr, kind="stable")
    addr_s = addr[order]
    val_s = val[order]
    uniq, starts = np.unique(addr_s, return_index=True)
    bounds = np.append(starts, len(addr_s))
    keep = []
    for i, a in enumerate(uniq):
        seg = val_s[bounds[i] : bounds[i + 1]]
        if seg.min() != seg.max():
            keep.append(int(a))
    return np.array(keep, dtype=np.uint16)


def _window_kind(kind: str) -> int | None:
    """Map a state-sequence kind to a RAM-write window filter (None = all)."""
    return {"irq": WIN_IRQ, "nmi": WIN_NMI}.get(kind)


def state_sequence(trace, kind: str = "auto", addrs=None) -> StateSequence:
    """Per-frame carry-forward state grid of every changing RAM cell.

    ``ticks`` are the per-frame boundary cycles from the chosen interrupt source
    (``kind`` in {'irq','nmi','both','auto'}); ``addrs`` are the cells that ever
    change (or the supplied subset); ``grid`` is ``uint8 [n_frames, n_addrs]``,
    the carry-forward value of each cell at the end of each frame.
    """
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
    return StateSequence(ticks=ticks, addrs=addrs, grid=grid)


# -- bounded accumulator (BACC) -------------------------------------------


def combine_lohi(lo_series, hi_series) -> np.ndarray:
    """Combine two 8-bit per-frame cells into a 16-bit int64 series."""
    lo = np.asarray(lo_series, dtype=np.uint16) & 0xFF
    hi = np.asarray(hi_series, dtype=np.uint16) & 0xFF
    return (lo | (hi << 8)).astype(np.int64)


def _dominant_abs_step(diffs: np.ndarray) -> int:
    """Most common nonzero absolute step among per-frame diffs."""
    a = np.abs(diffs[diffs != 0])
    if len(a) == 0:
        return 0
    vals, counts = np.unique(a, return_counts=True)
    return int(vals[counts.argmax()])


def _simulate_reflect(lo, hi, step, start, direction, n):
    """Simulate a reflecting (triangle) accumulator; return (series, flips)."""
    out = np.empty(n, dtype=np.int64)
    v = start
    d = direction
    flips = []
    for i in range(n):
        out[i] = v
        nv = v + d * step
        if nv > hi:
            d = -d
            nv = hi - (nv - hi)
            flips.append(i)
        elif nv < lo:
            d = -d
            nv = lo + (lo - nv)
            flips.append(i)
        v = nv
    return out, flips


def _fit_reflect(series, lo, hi, step):
    """Fit a reflecting triangle accumulator (vectorised residual via sim)."""
    n = len(series)
    nz = np.diff(series)
    nz = nz[nz != 0]
    direction = 1 if (len(nz) == 0 or nz[0] > 0) else -1
    pred, flips = _simulate_reflect(lo, hi, step, int(series[0]), direction, n)
    residual = float(np.mean(pred == series))
    if len(flips) >= 2:
        period = int(np.median(np.diff(flips)))
    else:
        period = int((hi - lo) // step) if step else None
    return {
        "mode": "reflect",
        "step": int(step),
        "lo": int(lo),
        "hi": int(hi),
        "phase": int(series[0]),
        "direction": int(direction),
        "period": period,
        "residual": residual,
    }


def _fit_linear(series, lo, hi, step):
    """Fit a wrapping/sawtooth accumulator: value = lo + (start-lo + step*i) % M.

    The modulus M is recovered from the dominant reset (negative) jump, which is
    robust when the observed max does not reach M-1.
    """
    diffs = np.diff(series)
    neg = diffs[diffs < 0]
    if len(neg):
        vals, counts = np.unique(neg, return_counts=True)
        reset = int(vals[counts.argmax()])
        modulus = step - reset
    else:
        modulus = hi - lo + step
    if modulus <= 0:
        return None
    start = int(series[0])
    idx = np.arange(len(series), dtype=np.int64)
    pred = lo + ((start - lo) + step * idx) % modulus
    residual = float(np.mean(pred == series))
    # A reset that lands exactly on lo is a true sawtooth; otherwise modular.
    after_reset = series[1:][diffs < 0]
    mode = "saw" if len(after_reset) and np.all(after_reset == lo) else "wrap"
    return {
        "mode": mode,
        "step": int(step),
        "lo": int(lo),
        "hi": int(hi),
        "phase": start,
        "direction": 1,
        "period": int(modulus // step) if step else None,
        "residual": residual,
        "modulus": int(modulus),
    }


def fit_bacc(series, min_residual: float = 0.8):
    """Recover a bounded accumulator from a 1-D per-frame integer series.

    Returns a dict with ``mode`` in {'saw','reflect','wrap'}, ``step``, ``lo``,
    ``hi``, ``phase``, ``period`` (reflect flip period / wrap cycle length), and
    ``residual`` (the fraction of frames matching the recovered recurrence).
    Returns None if the series is constant or no mode fits at least
    ``min_residual``.
    """
    series = np.asarray(series, dtype=np.int64).ravel()
    if len(series) < 4:
        return None
    diffs = np.diff(series)
    if not np.any(diffs):
        return None
    step = _dominant_abs_step(diffs)
    lo, hi = int(series.min()), int(series.max())
    if step == 0 or hi == lo:
        return None
    best = None
    best_res = -1.0
    for cand in (_fit_reflect(series, lo, hi, step), _fit_linear(series, lo, hi, step)):
        if cand is not None and cand["residual"] > best_res:
            best = cand
            best_res = cand["residual"]
    if best is None or best_res < min_residual:
        return None
    best["type"] = "BACC"
    return best


# -- table walk -----------------------------------------------------------


def _find_table_base(cursor, values, ram, stride, max_candidates=2048):
    """Best (base, residual) so that ram[base + stride*cursor] == values."""
    c0 = int(cursor[0])
    cand = np.nonzero(ram == values[0])[0].astype(np.int64) - stride * c0
    cand = cand[cand >= 0]
    maxc = int(cursor.max())
    best_base, best_res = None, 0.0
    for base in cand[:max_candidates]:
        if base + stride * maxc >= len(ram):
            continue
        res = float(np.mean(ram[base + stride * cursor] == values))
        if res > best_res:
            best_res, best_base = res, int(base)
            if res == 1.0:
                break
    return best_base, best_res


def detect_table_walk(cursor_series, ram_image, value_series=None, stride: int = 1):
    """Recover a table walk: a cursor cell stepping a static table.

    Recovers ``base`` (table start in the RAM image), ``stride`` (default 1),
    ``length`` (cursor span), ``loop`` (the cursor value it loops back to), and
    ``table`` bytes. When ``value_series`` is supplied it is verified that
    ``table[base + stride*cursor]`` reproduces it (``residual``). Returns None if
    the cursor does not vary, or a value series cannot be matched to the image.
    """
    cursor = np.asarray(cursor_series, dtype=np.int64).ravel()
    if len(cursor) == 0 or cursor.min() == cursor.max():
        return None
    cmin, cmax = int(cursor.min()), int(cursor.max())
    length = cmax - cmin + 1
    drops = np.nonzero(np.diff(cursor) < 0)[0]
    loop = int(cursor[drops[0] + 1]) if len(drops) else cmin

    result = {
        "type": "TABLE_WALK",
        "base": None,
        "stride": int(stride),
        "length": int(length),
        "loop": loop,
        "table": None,
        "residual": 0.0,
    }
    if value_series is None or ram_image is None:
        return result
    values = np.asarray(value_series, dtype=np.uint8).ravel()
    ram = np.asarray(ram_image, dtype=np.uint8)
    if len(values) != len(cursor):
        return None
    base, residual = _find_table_base(cursor, values, ram, stride)
    if base is None:
        return None
    top = base + stride * cmax + 1
    result["base"] = int(base)
    result["table"] = ram[base:top:stride].copy()
    result["residual"] = residual
    return result


# -- per-register classification ------------------------------------------


def _register_series(trace, sid_addr, kind="auto"):
    """(ticks, per-frame int64 value series, write records) for a SID register."""
    ticks = trace.tick_cycles(kind).astype(np.uint64)
    writes = trace.sid_writes()
    wr = writes[writes["addr"] == sid_addr] if len(writes) else writes
    series = (
        _carry_series(wr["cycle"], wr["value"], ticks)
        if len(wr)
        else np.zeros(len(ticks), dtype=np.int64)
    )
    return ticks, series, wr


def _cursor_like(cursor: np.ndarray) -> bool:
    """Cheap pre-filter: small range and mostly non-decreasing (a cursor)."""
    if cursor.min() == cursor.max() or cursor.max() - cursor.min() > 255:
        return False
    d = np.diff(cursor)
    return float(np.mean((d >= 0) & (d <= 4))) > 0.5


def classify_register(trace, sid_addr, kind="auto", stateseq=None, ram=None) -> dict:
    """Classify the generator producing a SID register's per-frame value.

    Returns a dict with ``type`` in {'BACC','TABLE_WALK','SEQ','XSTATE','CONST'},
    the recovered parameters, and ``store_pcs`` (the store-site PCs that wrote
    this register). Strategy: CONST, then BACC, then TABLE_WALK (over candidate
    cursor cells from the state sequence + RAM image), else SEQ if the writes are
    sparse (event-latched at note boundaries), else XSTATE.
    """
    ticks, series, wr = _register_series(trace, sid_addr, kind)
    pcs = sorted({int(p) for p in wr["aux"]}) if len(wr) else []
    result = {"addr": int(sid_addr), "store_pcs": pcs, "n_writes": int(len(wr))}

    if len(ticks) == 0 or len(wr) == 0:
        result.update(type="CONST", value=None)
        return result
    if len(np.unique(wr["value"])) == 1:
        result.update(type="CONST", value=int(wr["value"][0]))
        return result

    bacc = fit_bacc(series)
    if bacc is not None:
        result.update(bacc)
        return result

    if ram is None:
        ram = trace.ram_image()
    if stateseq is None:
        stateseq = state_sequence(trace, kind)
    if ram is not None:
        for j, cell in enumerate(stateseq.addrs):
            cursor = stateseq.grid[:, j].astype(np.int64)
            if not _cursor_like(cursor):
                continue
            tw = detect_table_walk(cursor, ram, value_series=series)
            if tw is not None and tw["base"] is not None and tw["residual"] >= 0.8:
                result.update(tw)
                result["cursor_addr"] = int(cell)
                return result

    if len(wr) / max(1, len(ticks)) < 0.5:
        result.update(type="SEQ")
    else:
        result.update(type="XSTATE")
    return result


# -- event/state correlation ----------------------------------------------


def _discontinuities(series: np.ndarray) -> np.ndarray:
    """Frame indices (post-jump) where a cell changes anomalously fast."""
    d = np.diff(series.astype(np.int64))
    ad = np.abs(d)
    pos = ad[ad > 0]
    if len(pos) == 0:
        return np.empty(0, dtype=np.int64)
    thresh = max(2 * float(np.median(pos)), 1.0)
    return (np.nonzero(ad > thresh)[0] + 1).astype(np.int64)


def correlate_event_reset(trace, trigger_pred, cell_addr, kind="auto", max_lag=2) -> dict:
    """Quantify whether an event co-occurs with a reset of a state cell.

    ``trigger_pred(event) -> bool`` selects trigger events (e.g. a CTRL write);
    a reset is a discontinuity of ``cell_addr``'s per-frame series. Returns
    ``{'correlation', 'lag_frames', 'n_triggers'}`` where ``correlation`` is the
    fraction of triggers accompanied by a discontinuity at the best lag.
    """
    ss = state_sequence(trace, kind, addrs=[cell_addr])
    ticks = ss.ticks
    series = ss.grid[:, 0].astype(np.int64) if ss.grid.shape[1] else np.zeros(len(ticks))
    disc = set(int(x) for x in _discontinuities(series))

    trig_frames = []
    for ev in trace.events:
        if trigger_pred(ev):
            frame = int(np.searchsorted(ticks, ev["cycle"], side="right")) - 1
            if 0 <= frame < len(ticks):
                trig_frames.append(frame)
    n_trig = len(trig_frames)
    if n_trig == 0:
        return {"correlation": 0.0, "lag_frames": 0, "n_triggers": 0}

    best_corr, best_lag = 0.0, 0
    for lag in range(0, max_lag + 1):
        hits = sum((f + lag) in disc for f in trig_frames)
        corr = hits / n_trig
        if corr > best_corr:
            best_corr, best_lag = corr, lag
    return {"correlation": float(best_corr), "lag_frames": int(best_lag), "n_triggers": n_trig}


def voice_events(trace, kind="auto") -> dict:
    """Per-voice note events from CTRL gate-bit transitions + cell resets.

    Returns ``{voice: [event, ...]}`` where each event is a gate change
    (``note_on``/``note_off``) with its frame, cycle, CTRL value, and the list of
    state cells that reset on that frame.
    """
    ss = state_sequence(trace, kind)
    ticks = ss.ticks
    disc_map: dict[int, list] = {}
    for j, cell in enumerate(ss.addrs):
        for frame in _discontinuities(ss.grid[:, j].astype(np.int64)):
            disc_map.setdefault(int(frame), []).append(int(cell))

    out: dict[int, list] = {0: [], 1: [], 2: []}
    for voice, addr in CTRL_ADDRS.items():
        _, ctrl, _wr = _register_series(trace, addr, kind)
        if len(ctrl) == 0:
            continue
        gate = (ctrl & 1).astype(np.int64)
        for i in np.nonzero(np.diff(gate) != 0)[0]:
            frame = int(i) + 1
            out[voice].append(
                {
                    "frame": frame,
                    "cycle": int(ticks[frame]),
                    "type": "note_on" if gate[frame] else "note_off",
                    "ctrl": int(ctrl[frame]),
                    "resets": disc_map.get(frame, []),
                }
            )
    return out


def analyze(trace, kind="auto") -> dict:
    """Full per-register generator map (the headline entry point).

    Classifies every written SID register address, returning a dict keyed by
    address plus a ``'summary'`` of generator-type counts.
    """
    writes = trace.sid_writes()
    addrs = np.unique(writes["addr"]) if len(writes) else np.empty(0, dtype=np.uint16)
    stateseq = state_sequence(trace, kind)
    ram = trace.ram_image()
    out: dict = {}
    summary: dict = {}
    for a in addrs:
        res = classify_register(trace, int(a), kind, stateseq=stateseq, ram=ram)
        out[int(a)] = res
        summary[res["type"]] = summary.get(res["type"], 0) + 1
    out["summary"] = summary
    return out
