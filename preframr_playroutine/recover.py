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


# -- feeder cells, segmentation, and the recovery context -----------------

# Per-voice SID register layout: offset within the 7-register voice block.
_FREQ_LO, _FREQ_HI, _PW_LO, _PW_HI, _CTRL, _AD, _SR = range(7)
_LOHI_PARTNER = {_FREQ_LO: 1, _FREQ_HI: -1, _PW_LO: 1, _PW_HI: -1}

RecoverContext = namedtuple(
    "RecoverContext",
    ["kind", "stateseq", "ram", "tables", "cursor_cols", "note_on", "all_on", "n_frames"],
)


def _voice_of(sid_addr: int):
    """(voice, register-offset) for a per-voice SID register, else (None, None)."""
    off = int(sid_addr) - 0xD400
    if 0 <= off < 21:
        return off // 7, off % 7
    return None, None


def _bits_set(mask: int) -> int:
    """Population count of an 8-bit mask."""
    return bin(int(mask) & 0xFF).count("1")


def _varying_bits(series: np.ndarray) -> int:
    """Bitmask of the bit positions that are not constant across ``series``."""
    s = np.asarray(series, dtype=np.int64)
    out = 0
    for b in range(8):
        col = (s >> b) & 1
        if col.min() != col.max():
            out |= 1 << b
    return out


def _note_on_frames(trace, voice, ctx_kind="auto") -> list:
    """Frames where ``voice``'s CTRL gate bit transitions 0->1 (note-on)."""
    if voice is None or voice not in CTRL_ADDRS:
        return []
    ctrl = _register_series(trace, CTRL_ADDRS[voice], ctx_kind)[1]
    if len(ctrl) == 0:
        return []
    gate = (ctrl & 1).astype(np.int64)
    return (np.nonzero(np.diff(gate) > 0)[0] + 1).tolist()


def _best_feeder(series: np.ndarray, stateseq) -> tuple:
    """Best state cell reproducing ``series`` directly (8-bit).

    Returns ``(cell_addr, fraction)`` for the grid column whose value best
    matches ``series``; ``(None, 0.0)`` if no cell is available. Prefilters by
    value range so only plausible cells are scored.
    """
    s = np.asarray(series, dtype=np.int64)
    if stateseq is None or stateseq.grid.shape[1] == 0 or len(s) == 0:
        return None, 0.0
    grid = stateseq.grid.astype(np.int64)
    smin, smax = int(s.min()), int(s.max())
    keep = (grid.max(axis=0) >= smin) & (grid.min(axis=0) <= smax)
    if not np.any(keep):
        return None, 0.0
    cols = np.nonzero(keep)[0]
    match = (grid[:, cols] == s[:, None]).mean(axis=0)
    j = int(match.argmax())
    return int(stateseq.addrs[cols[j]]), float(match[j])


def _strip_holds(seg: np.ndarray) -> np.ndarray:
    """Trim leading and trailing constant runs from a segment."""
    i = 0
    n = len(seg)
    while i + 1 < n and seg[i] == seg[i + 1]:
        i += 1
    j = n
    while j - 1 > i and seg[j - 1] == seg[j - 2]:
        j -= 1
    return seg[i:j]


def segmented_bacc(series, reset_frames, min_residual: float = 0.6, min_segments: int = 3):
    """Recover a note-reseeded bounded accumulator from a per-frame series.

    The series is segmented at note-on ``reset_frames`` and at internal
    discontinuities (the seed jumps), each segment is trimmed of its constant
    hold runs and fitted with :func:`fit_bacc`. BACC is recovered when a
    consistent step is found across a majority of fittable segments. Returns a
    BACC dict (``step``/``mode``/``lo``/``hi``/``residual`` plus ``n_segments``/
    ``n_fit``/``segmented=True``) or ``None``.
    """
    series = np.asarray(series, dtype=np.int64).ravel()
    if len(series) < 8:
        return None
    cuts = set(int(x) for x in _discontinuities(series))
    cuts.update(int(x) for x in reset_frames if 0 < int(x) < len(series))
    bounds = [0] + sorted(cuts) + [len(series)]
    fits = []
    n_try = 0
    for k in range(len(bounds) - 1):
        seg = _strip_holds(series[bounds[k] : bounds[k + 1]])
        if len(seg) < 4:
            continue
        n_try += 1
        fit = fit_bacc(seg, min_residual=min_residual)
        if fit is not None:
            fits.append(fit)
    if n_try < min_segments or len(fits) < min_segments:
        return None
    steps = {}
    for fit in fits:
        steps[fit["step"]] = steps.get(fit["step"], 0) + 1
    step = max(steps, key=steps.get)
    cnt = steps[step]
    if cnt / len(fits) < 0.5 or len(fits) / n_try < 0.5:
        return None
    modal = [f for f in fits if f["step"] == step]
    modes = {}
    for fit in modal:
        modes[fit["mode"]] = modes.get(fit["mode"], 0) + 1
    return {
        "type": "BACC",
        "mode": max(modes, key=modes.get),
        "step": int(step),
        "lo": int(min(f["lo"] for f in modal)),
        "hi": int(max(f["hi"] for f in modal)),
        "residual": float(cnt / len(fits)),
        "n_segments": int(n_try),
        "n_fit": int(len(fits)),
        "segmented": True,
    }


def _read_log_tables(trace, ram, max_span: int = 256, cap: int = 96) -> list:
    """Candidate static tables from the RAM read log: ``(base, top, valueset)``.

    Each read-PC that reads a contiguous, bounded address range is a likely
    ``LDA table,X`` table walk; ``base`` is the lowest address read, ``top`` the
    highest, and ``valueset`` the distinct bytes of that region in the image.
    """
    if ram is None:
        return []
    rd = trace.ram_reads()
    if len(rd) == 0:
        return []
    pcs, inv = np.unique(rd["pc"], return_inverse=True)
    out = []
    for i in range(len(pcs)):
        addrs = rd["addr"][inv == i]
        if len(addrs) < 8:
            continue
        lo, hi = int(addrs.min()), int(addrs.max())
        span = hi - lo + 1
        if span < 2 or span > max_span or hi >= len(ram):
            continue
        out.append((lo, hi, frozenset(int(x) for x in np.unique(ram[lo : hi + 1]))))
    # De-duplicate by (base, top); cap to bound the per-register search.
    uniq = {(lo, hi): vs for lo, hi, vs in out}
    return [(lo, hi, vs) for (lo, hi), vs in sorted(uniq.items())][:cap]


def _cursor_columns(stateseq, cap: int = 128) -> list:
    """Grid column indices whose per-frame series looks like a table cursor."""
    if stateseq is None or stateseq.grid.shape[1] == 0:
        return []
    grid = stateseq.grid.astype(np.int64)
    cols = []
    for j in range(grid.shape[1]):
        if _cursor_like(grid[:, j]):
            cols.append(j)
    return cols[:cap]


def _score_cursor(series, cur, ram, lo, hi, mask, n):
    """Best (residual, offset) reproducing ``series`` from ``ram[lo:hi]`` via cur.

    Searches a small index offset (the cursor is read post-increment / with
    loop markers, so the carry-forward value can lead/lag by a frame or two).
    """
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


def _table_walk_search(series, ctx, min_res: float = 0.82):
    """Recover a (masked) table walk for ``series`` using the read-log tables.

    For each candidate table whose byte set contains the register's values
    (under an SID gate mask) and each cursor cell, search a small index offset
    and pick the (base, cursor, offset, mask) with the highest reproduction
    fraction. Returns a TABLE_WALK dict or ``None``.
    """
    series = np.asarray(series, dtype=np.int64).ravel()
    ram = ctx.ram
    if ram is None or not ctx.tables or not ctx.cursor_cols:
        return None
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
        svals = {int(x) & mask for x in np.unique(series)}
        for lo, hi, vset in ctx.tables:
            if len(svals - {x & mask for x in vset}) > 1:
                continue
            for j in ctx.cursor_cols:
                res, off = _score_cursor(series, grid[:, j], ram, lo, hi, mask, n)
                if res >= best_res:
                    best_res = res
                    best = (res, lo, hi, int(ctx.stateseq.addrs[j]), off, mask)
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


def _anchor_positions(series, ramm, mask, max_pos: int = 256):
    """(anchor_frame, image positions) for the rarest masked register value.

    Picks the masked value of ``series`` with the fewest occurrences in the
    masked image ``ramm`` (so the value-scan yields few candidate bases), and
    returns the first frame carrying it plus the image positions matching it.
    """
    sm = np.asarray(series, dtype=np.int64) & mask
    uniq = np.unique(sm)
    best_pos = None
    best_frame = 0
    best_count = max_pos + 1
    for v in uniq:
        pos = np.nonzero(ramm == int(v))[0]
        if 0 < len(pos) < best_count:
            best_count = len(pos)
            best_pos = pos
            best_frame = int(np.nonzero(sm == int(v))[0][0])
    if best_pos is None or best_count > max_pos:
        return None
    return best_frame, best_pos.astype(np.int64)


def _table_walk_scan(series, ctx, min_res: float = 0.8, max_bases: int = 96):
    """No-read-log table walk: cursor state cell + static ``ram_image``.

    Recovers ``CTRL[frame] == ram_image[base + cursor[frame]] & mask`` by
    value-scanning the image for candidate bases (anchored on a rare register
    value) for each cursor cell, under SID gate masks ``{0xFF, 0xFE}`` and the
    same small index-offset search as the read-log path. This mirrors the
    read-log recovery so a whole-song trace rendered without ``--reads`` still
    recovers e.g. the DMC waveform table walk (``$18AD[$177A] & 0xFE``).
    """
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
                res, off = _score_cursor(series, cur, ram, int(base), int(base) + cmax, mask, n)
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


def _seq_correlation(series, on_frames, n_frames, lag: int = 2) -> tuple:
    """(fraction, n_changes): how strongly value changes align with note-ons."""
    s = np.asarray(series, dtype=np.int64)
    changes = np.nonzero(np.diff(s) != 0)[0] + 1
    if len(changes) == 0:
        return 1.0, 0
    onset = np.zeros(n_frames, dtype=bool)
    for f in on_frames:
        for shift in range(-1, lag + 1):
            g = f + shift
            if 0 <= g < n_frames:
                onset[g] = True
    changes = changes[changes < n_frames]
    if len(changes) == 0:
        return 1.0, 0
    return float(np.mean(onset[changes])), int(len(changes))


def _build_context(trace, kind="auto", stateseq=None, ram=None) -> RecoverContext:
    """Precompute the per-trace recovery context once (shared by analyze)."""
    if stateseq is None:
        stateseq = state_sequence(trace, kind)
    if ram is None:
        ram = trace.ram_image()
    tables = _read_log_tables(trace, ram)
    cursor_cols = _cursor_columns(stateseq)
    note_on = {v: _note_on_frames(trace, v, kind) for v in CTRL_ADDRS}
    all_on = sorted({f for frames in note_on.values() for f in frames})
    return RecoverContext(
        kind=kind,
        stateseq=stateseq,
        ram=ram,
        tables=tables,
        cursor_cols=cursor_cols,
        note_on=note_on,
        all_on=all_on,
        n_frames=len(stateseq.ticks),
    )


def _bacc_candidate(trace, sid_addr, series, voice, reg_off, ctx):
    """Recover a BACC on the register or its 16-bit lo/hi partner.

    Tries, in order: a global :func:`fit_bacc` (a non-reseeded accumulator), then
    a note-segmented BACC (re-seeded per note); each on the register's own 8-bit
    series and on the 16-bit lo/hi combination when the register is a freq/PW
    byte. The 16-bit form is preferred when both fit.
    """
    resets = ctx.note_on.get(voice, []) if voice is not None else ctx.all_on
    partner_off = _LOHI_PARTNER.get(reg_off)
    combined = None
    if partner_off is not None:
        partner = _register_series(trace, sid_addr + partner_off, ctx.kind)[1]
        if len(np.unique(partner)) > 1:
            lo, hi = (series, partner) if partner_off == 1 else (partner, series)
            combined = combine_lohi(lo, hi)

    if combined is not None:
        glob = fit_bacc(combined)
        if glob is not None:
            glob["width"] = 16
            return glob
    glob = fit_bacc(series)
    if glob is not None:
        return glob

    best = segmented_bacc(series, resets)
    if combined is not None:
        cb = segmented_bacc(combined, resets)
        if cb is not None and (best is None or cb["n_fit"] >= best["n_fit"]):
            cb["width"] = 16
            best = cb
    return best


def classify_register(trace, sid_addr, kind="auto", stateseq=None, ram=None, ctx=None) -> dict:
    """Classify the generator producing a SID register's per-frame value.

    Returns a dict with ``type`` in {'BACC','TABLE_WALK','SEQ','XSTATE','CONST'},
    the recovered parameters, ``store_pcs`` (the store-site PCs that wrote this
    register), and (where recovered) the feeder ``cell_addr``, table walk
    ``cursor_addr``/``table``/``mask``, and BACC segment info. Strategy, in
    order: CONST -> (note-segmented, feeder-cell) BACC -> (read-log/image)
    TABLE_WALK -> SEQ (note-gated event latch) -> XSTATE.

    ``ctx`` is a precomputed :class:`RecoverContext` (built once by
    :func:`analyze`); ``stateseq``/``ram`` are accepted for backwards
    compatibility and seed a context when ``ctx`` is not given.
    """
    if ctx is None:
        ctx = _build_context(trace, kind, stateseq, ram)
    ticks, series, wr = _register_series(trace, sid_addr, ctx.kind)
    pcs = sorted({int(p) for p in wr["aux"]}) if len(wr) else []
    result = {"addr": int(sid_addr), "store_pcs": pcs, "n_writes": int(len(wr))}
    voice, reg_off = _voice_of(sid_addr)

    if len(ticks) == 0 or len(wr) == 0:
        result.update(type="CONST", value=None)
        return result
    if len(np.unique(wr["value"])) == 1:
        result.update(type="CONST", value=int(wr["value"][0]))
        return result

    feeder, ffrac = _best_feeder(series, ctx.stateseq)
    if feeder is not None and ffrac >= 0.5:
        result["cell_addr"] = feeder
        result["cell_frac"] = round(ffrac, 4)

    on_frames = ctx.note_on.get(voice, []) if voice is not None else ctx.all_on
    # AD / SR are envelope registers: every player writes them once per note (or
    # re-blits a per-note shadow). They are event-latched SEQ, never a per-frame
    # accumulator or table walk -- short-circuit so a per-frame re-blit cannot
    # over-fit a spurious BACC/TABLE_WALK.
    if reg_off in (_AD, _SR):
        seq_frac, n_changes = _seq_correlation(series, on_frames, ctx.n_frames)
        result.update(type="SEQ", seq_frac=round(seq_frac, 4), n_changes=int(n_changes))
        return result

    bacc = _bacc_candidate(trace, sid_addr, series, voice, reg_off, ctx)
    if bacc is not None:
        result.update(bacc)
        return result

    return _classify_walk_or_seq(series, wr, on_frames, ctx, result)


def _classify_walk_or_seq(series, wr, on_frames, ctx, result) -> dict:
    """Resolve a per-frame register as TABLE_WALK, else SEQ / XSTATE."""
    seq_frac, n_changes = _seq_correlation(series, on_frames, ctx.n_frames)
    result["seq_frac"] = round(seq_frac, 4)
    result["n_changes"] = int(n_changes)
    note_gated = n_changes <= 1 or (seq_frac >= 0.85 and n_changes <= 2.5 * max(1, len(on_frames)))
    sparse = n_changes <= max(8, 0.15 * ctx.n_frames)
    # A register written on (almost) every frame is a per-frame generator even
    # when its value is held; one written only once per note is an event-latched
    # SEQ write. Gate the per-frame table-walk search on the write density (not
    # the change density), so a held-waveform CTRL is still recovered as a
    # TABLE_WALK while per-note AD/SR/RES/VOL stay SEQ.
    per_frame = len(wr) >= 0.5 * max(1, ctx.n_frames)
    if per_frame:
        tw = _table_walk_search(series, ctx)
        if tw is None and not ctx.tables:
            tw = _table_walk_scan(series, ctx)
        if tw is not None:
            result.pop("seq_frac", None)
            result.pop("n_changes", None)
            result.update(tw)
            return result

    result.update(type="SEQ" if (note_gated or sparse or not per_frame) else "XSTATE")
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
    ctx = _build_context(trace, kind)
    out: dict = {}
    summary: dict = {}
    for a in addrs:
        res = classify_register(trace, int(a), kind, ctx=ctx)
        out[int(a)] = res
        summary[res["type"]] = summary.get(res["type"], 0) + 1
    out["summary"] = summary
    return out
