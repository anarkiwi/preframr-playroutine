"""Recover per-register generators from a sidtrace v2 oracle.

Given the SID register oracle plus the v2 internal-state signals (per-frame RAM
write log, store-site PC of each SID write, and the static RAM image), recover
the generator that produces each SID register's per-frame value, classified in
the BACC / TABLE-WALK / SEQ / COMPOSITE / XSTATE taxonomy of
``/scratch/anarkiwi/cbm/re-trackers``:

- ``BACC``       bounded accumulator (per-frame add/sub with bound + saw / wrap /
                 reflect behaviour).
- ``TABLE_WALK`` a cursor cell stepping a static table, with loop-back.
- ``SEQ``        event-latched (sparse) writes from the note/pattern sequencer.
- ``COMPOSITE``  base + modulation + override: a sequencer-indexed base table,
                 an additive modulation accumulator, and value-forcing overrides.
- ``XSTATE``     a cross-function dependency not yet modelled by a single
                 generator. The emulator logs every bit, so the output is always
                 a function of observable state; ``XSTATE`` marks a register whose
                 dependency we have not yet decomposed, and :func:`round_trip`
                 fidelity quantifies exactly how large that gap is (and where).

Everything is per-frame: writes are binned into frames by the chosen interrupt
cadence (``Trace.tick_cycles``) and carried forward, the same technique as
``Trace.register_frames``. Per-frame stateful recurrences (the BACC fit) are
inherently sequential and use short python loops over the frame count; the bulk
binning / table search is vectorised numpy. Recovered descriptors are
executable: :func:`reconstruct_register` regenerates a register's per-frame
output from its descriptor, and :func:`round_trip` scores that regeneration
against the oracle.
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


def _simulate_pingpong(lo, hi, clamp_lo, clamp_hi, up_step, down_step, start, direction, n):
    """Simulate a clamp-and-flip (ping-pong) accumulator; return (series, flips).

    Unlike the mirror reflect, an overshoot saturates at a fixed clamp value and
    reverses direction (defMON PW: ceiling -> ``clamp_hi``, floor -> ``clamp_lo``).
    The step magnitude may differ by direction (``up_step``/``down_step``).
    """
    out = np.empty(n, dtype=np.int64)
    v = start
    d = direction
    flips = []
    for i in range(n):
        out[i] = v
        step = up_step if d > 0 else down_step
        nv = v + d * step
        if nv > hi:
            d = -d
            nv = clamp_hi
            flips.append(i)
        elif nv < lo:
            d = -d
            nv = clamp_lo
            flips.append(i)
        v = nv
    return out, flips


def _simulate_tickreflect(lo, hi, rate, start, direction, n):
    """Simulate a reflecting accumulator whose per-frame stride is tick-indexed.

    Unlike :func:`_simulate_reflect` (a scalar step), the stride is read from the
    ``rate`` vector at the per-segment tick (the frame offset since the note-on
    that seeded the segment); past the end of ``rate`` the last entry is held.
    This is the Future Composer PW sweep: ``step = rate_table[tick]`` with the
    direction flipping (mirror reflect) at the recovered PW bounds.
    """
    out = np.empty(n, dtype=np.int64)
    v = start
    d = direction
    m = len(rate)
    for i in range(n):
        out[i] = v
        if m == 0:
            continue
        st = int(rate[i]) if i < m else int(rate[m - 1])
        nv = v + d * st
        if nv > hi:
            d = -d
            nv = hi - (nv - hi)
        elif nv < lo:
            d = -d
            nv = lo + (lo - nv)
        v = nv
    return out


def _dominant_signed_step(diffs: np.ndarray, positive: bool) -> int:
    """Most common positive (or |negative|) per-frame step."""
    arr = diffs[diffs > 0] if positive else -diffs[diffs < 0]
    if len(arr) == 0:
        return 0
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[counts.argmax()])


def _fit_pingpong(series, lo, hi):
    """Fit a clamp-and-flip (ping-pong) accumulator (defMON PW sweep).

    The value saturates at ``lo``/``hi`` and reverses (rather than mirroring the
    overshoot like ``_fit_reflect``); up and down step magnitudes are recovered
    independently. Requires both directions to be present.
    """
    diffs = np.diff(series)
    up = _dominant_signed_step(diffs, True)
    down = _dominant_signed_step(diffs, False)
    if up == 0 or down == 0:
        return None
    n = len(series)
    nz = diffs[diffs != 0]
    # A ping-pong sweep ramps for many frames between reversals; a fast oscillation
    # (a short table walk read as ±step) is not one. Require long monotonic runs.
    signs = np.sign(nz)
    n_rev = int(np.count_nonzero(np.diff(signs))) if len(signs) > 1 else 0
    if n_rev and n / (n_rev + 1) < 3.0:
        return None
    direction = 1 if nz[0] > 0 else -1
    pred, flips = _simulate_pingpong(lo, hi, lo, hi, up, down, int(series[0]), direction, n)
    residual = float(np.mean(pred == series))
    period = int(np.median(np.diff(flips))) if len(flips) >= 2 else None
    return {
        "mode": "pingpong",
        "step": int(up),
        "down_step": int(down),
        "lo": int(lo),
        "hi": int(hi),
        "clamp_lo": int(lo),
        "clamp_hi": int(hi),
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
    candidates = (
        _fit_reflect(series, lo, hi, step),
        _fit_linear(series, lo, hi, step),
        _fit_pingpong(series, lo, hi),
    )
    for cand in candidates:
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
    [
        "kind",
        "stateseq",
        "ram",
        "tables",
        "cursor_cols",
        "note_on",
        "all_on",
        "n_frames",
        "sampler",
    ],
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
    resets = [int(b) for b in bounds[:-1]]
    seeds = [int(series[b]) for b in resets]
    base = {
        "type": "BACC",
        "step": int(step),
        "lo": int(min(f["lo"] for f in modal)),
        "hi": int(max(f["hi"] for f in modal)),
        "residual": float(cnt / len(fits)),
        "n_segments": int(n_try),
        "n_fit": int(len(fits)),
        "segmented": True,
        "resets": resets,
        "seeds": seeds,
    }
    out = dict(base, mode=max(modes, key=modes.get))
    # A clamp-and-flip (ping-pong) sweep degenerates to a plain ramp away from its
    # bounds, so non-reversing note segments fit as reflect/saw and outvote it; the
    # only structural difference is the boundary. When any segment fit ping-pong,
    # compare both modes by how well each regenerates the whole series and keep the
    # better -- the boundary is the only place they disagree.
    if "pingpong" in modes:
        pp = [f for f in modal if f["mode"] == "pingpong"]
        downs = {}
        for f in pp:
            downs[f["down_step"]] = downs.get(f["down_step"], 0) + 1
        ping = dict(
            base,
            mode="pingpong",
            down_step=int(max(downs, key=downs.get)),
            clamp_lo=int(min(f["clamp_lo"] for f in pp)),
            clamp_hi=int(max(f["clamp_hi"] for f in pp)),
        )
        if _seg_fidelity(ping, series) > _seg_fidelity(out, series):
            out = ping
    return out


def _seg_fidelity(desc, series) -> float:
    """Fraction of frames a segmented BACC descriptor regenerates from its seeds."""
    recon = _recon_bacc_full(desc, len(series))
    return float(np.mean(recon == np.asarray(series, dtype=np.int64)))


def segmented_pingpong(series, reset_frames, min_residual: float = 0.6, min_reversing: int = 2):
    """Recover a per-note clamp-and-flip (ping-pong) sweep (defMON PW).

    Each note seeds the accumulator and sets its own signed rate, so step and
    direction vary per segment while the floor/ceiling clamp is shared. Unlike
    :func:`segmented_bacc`, no single modal step is required: every segment keeps
    its own step/direction. The shared bounds/clamps are taken from the segments
    that actually reverse (where the clamp is observed). Returns a ``pingpong``
    BACC descriptor with per-segment ``steps``/``down_steps``/``directions`` or
    ``None`` when too few segments reverse.
    """
    series = np.asarray(series, dtype=np.int64).ravel()
    if len(series) < 8:
        return None
    cuts = set(int(x) for x in _discontinuities(series))
    cuts.update(int(x) for x in reset_frames if 0 < int(x) < len(series))
    bounds = [0] + sorted(cuts) + [len(series)]
    resets, seeds, steps, downs, dirs = [], [], [], [], []
    reversing = []
    for k in range(len(bounds) - 1):
        seg = series[bounds[k] : bounds[k + 1]]
        d = np.diff(seg)
        nz = d[d != 0]
        up = _dominant_signed_step(d, True)
        down = _dominant_signed_step(d, False)
        resets.append(int(bounds[k]))
        seeds.append(int(seg[0]))
        steps.append(int(up or down or 1))
        downs.append(int(down or up or 1))
        dirs.append(1 if (len(nz) == 0 or nz[0] > 0) else -1)
        strip = _strip_holds(seg)
        if len(strip) >= 4 and up and down:
            fit = _fit_pingpong(strip, int(strip.min()), int(strip.max()))
            if fit is not None and fit["residual"] >= min_residual:
                reversing.append(fit)
    if len(reversing) < min_reversing:
        return None
    lo = int(min(f["lo"] for f in reversing))
    hi = int(max(f["hi"] for f in reversing))
    rate = {}
    for s in steps:
        rate[s] = rate.get(s, 0) + 1
    return {
        "type": "BACC",
        "mode": "pingpong",
        "step": int(max(rate, key=rate.get)),
        "lo": lo,
        "hi": hi,
        "clamp_lo": lo,
        "clamp_hi": hi,
        "segmented": True,
        "resets": resets,
        "seeds": seeds,
        "steps": steps,
        "down_steps": downs,
        "directions": dirs,
        "n_segments": int(len(resets)),
        "n_fit": int(len(reversing)),
        "residual": float(np.mean([f["residual"] for f in reversing])),
    }


def segmented_tickband(series, reset_frames, min_segments: int = 6):
    """Recover a tick-banded reflecting sweep (Future Composer PW LO/HI).

    The accumulator is reseeded at every note-on; within a note its per-frame
    stride is a step-function of the per-voice tick (frames since that note-on) --
    ``step = rate_table[tick]`` -- and the direction mirror-reflects at the
    recovered PW bounds. The tick is synthesized directly from the note-on
    ``reset_frames`` (a sawtooth reset at each note), so no RAM cursor cell is
    needed. Each note segment yields its own tick-rate vector (the absolute
    per-frame diffs) and start direction; the vectors are de-duplicated into a
    small table set shared across notes -- a real player has only a handful of
    rate programs, so a tick-banded sweep collapses to a few tables while noise
    does not. Returns a ``tickband`` BACC descriptor or ``None``.

    Guarded to never steal a constant-step fit: the stride must genuinely vary
    with the tick on a majority of segments, and the shared table set must stay
    small relative to the segment count (else the per-segment vectors are not a
    reused program and the candidate is rejected).
    """
    series = np.asarray(series, dtype=np.int64).ravel()
    n = len(series)
    if n < 8:
        return None
    resets = sorted({int(x) for x in reset_frames if 0 < int(x) < n})
    bounds = [0] + resets + [n]
    n_seg = len(bounds) - 1
    if n_seg < min_segments:
        return None
    lo16, hi16 = int(series.min()), int(series.max())
    if hi16 <= lo16:
        return None
    seg_resets, seeds, dirs, rate_vecs = [], [], [], []
    varying = 0
    for k in range(n_seg):
        a, b = bounds[k], bounds[k + 1]
        seg = series[a:b]
        d = np.diff(seg)
        nz = d[d != 0]
        rate = np.abs(d).astype(np.int64)
        seg_resets.append(int(a))
        seeds.append(int(seg[0]))
        dirs.append(1 if (len(nz) == 0 or nz[0] > 0) else -1)
        rate_vecs.append(rate)
        if len(np.unique(rate[rate != 0])) > 1:
            varying += 1
    # A tick-banded stride genuinely varies within a note; a constant per-note
    # stride is a plain saw/reflect (handled by the scalar modes). Require the
    # within-note variation on a majority of segments so this never steals a
    # clean constant-step fit.
    if varying < max(3, 0.3 * n_seg):
        return None
    tables, index, seg_tables = [], {}, []
    for rate in rate_vecs:
        key = rate.tobytes()
        if key not in index:
            index[key] = len(tables)
            tables.append(rate)
        seg_tables.append(index[key])
    # The rate vectors must be a reused program (few tables, many notes); noise
    # gives a distinct vector per note and is rejected here.
    if len(tables) > max(8, 0.5 * n_seg):
        return None
    desc = {
        "type": "BACC",
        "mode": "tickband",
        "step": int(tables[0][0]) if len(tables[0]) else 0,
        "lo": lo16,
        "hi": hi16,
        "segmented": True,
        "resets": seg_resets,
        "seeds": seeds,
        "directions": dirs,
        "rate_tables": tables,
        "seg_tables": seg_tables,
        "n_segments": int(n_seg),
    }
    recon = _recon_tickband(desc, n)
    desc["residual"] = float(np.mean(recon == series))
    return desc


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


def _score_cursor_bases(smask, cur, ramm, bases, cmax, n):
    """Vectorized :func:`_score_cursor` over many bases sharing one cursor cell.

    ``ramm`` is the image already masked (``ram & mask``, ``uint8``) and
    ``smask`` is the series already masked (``uint8``). For a fixed cursor cell
    ``lo = base`` and ``hi = base + cmax``, so ``span = cmax + 1`` is constant
    across bases and the per-offset ``ok`` mask and table index are
    base-independent. Per offset it builds one ``(span, 256)`` joint histogram of
    (table-index, masked-series) over the kept frames, so every base's residual
    is a small ``sum_c hist[c, tab[b, c]]`` gather instead of an ``(n_bases, n)``
    compare. Returns ``(res, off)`` arrays whose ``k``-th entry equals
    ``_score_cursor(series, cur, ram, bases[k], bases[k]+cmax, mask, n)`` (same
    strict ``>`` offset tie-break: first offset attaining the max wins). The
    ``cur.max() >= span + 2`` guard is ``cmax >= cmax + 3`` here, always false,
    so it never excludes a base.
    """
    nb = len(bases)
    best_res = np.full(nb, -1.0)
    best_off = np.zeros(nb, dtype=np.int64)
    if nb == 0:
        return best_res, best_off
    thresh = n * 0.8
    span = cmax + 1
    cols = np.arange(span)
    # Candidate table per base: tab[b] == ramm[bases[b] : bases[b] + span].
    tab = ramm[bases[:, None] + cols[None, :]]
    for off in (-2, -1, 0, 1, 2):
        idx = cur + off
        ok = (idx >= 0) & (idx <= cmax)
        n_ok = int(ok.sum())
        if n_ok < thresh:
            continue
        # ``ok`` already constrains ``idx`` to ``[0, cmax]``, so the original
        # ``clip(idx, 0, cmax)`` is the identity on the kept frames.
        idx_ok = idx[ok]
        # Joint (table-index, masked-series) histogram, shared across all bases.
        hist = np.bincount(idx_ok * 256 + smask[ok], minlength=span * 256).reshape(span, 256)
        # matches[b] == #frames where ramm[base+idx] == series&mask
        #            == sum_c hist[c, tab[b, c]].
        matches = hist[cols[None, :], tab].sum(axis=1)
        res = matches / n_ok
        present = np.nonzero(hist.sum(axis=1))[0]
        vals = tab[:, present]
        uniq_ok = vals.min(axis=1) != vals.max(axis=1)
        upd = uniq_ok & (res > best_res)
        best_off = np.where(upd, off, best_off)
        best_res = np.where(upd, res, best_res)
    return best_res, best_off


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
        ramm = ram & mask
        anchor = _anchor_positions(series, ramm, mask)
        if anchor is None:
            continue
        anchor_frame, positions = anchor
        smask = (series & mask).astype(ramm.dtype)
        for j in ctx.cursor_cols:
            cur = grid[:, j]
            cmax = int(cur.max())
            bases = positions - int(cur[anchor_frame])
            bases = bases[(bases >= 0) & (bases + cmax < len(ram))][:max_bases]
            if len(bases) == 0:
                continue
            res_vec, off_vec = _score_cursor_bases(smask, cur, ramm, bases, cmax, n)
            addr = int(ctx.stateseq.addrs[j])
            for base, res_b, off_b in zip(bases, res_vec, off_vec):
                res = float(res_b)
                if res >= best_res:
                    best_res = res
                    best = (res, int(base), int(base) + cmax, addr, int(off_b), mask)
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
        sampler=_CellSampler(trace, stateseq.ticks),
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
    role = "lo" if reg_off in (_FREQ_LO, _PW_LO) else "hi"
    combined = None
    if partner_off is not None:
        partner = _register_series(trace, sid_addr + partner_off, ctx.kind)[1]
        if len(np.unique(partner)) > 1:
            lo, hi = (series, partner) if partner_off == 1 else (partner, series)
            combined = combine_lohi(lo, hi)

    series = np.asarray(series, dtype=np.int64)
    # Candidate accumulators, 16-bit (lo/hi combined) first so it wins ties: the
    # combined form is the true generator domain (a fast-wrapping 8-bit byte fits
    # spuriously but reconstructs poorly). Pick the one that best regenerates the
    # register's own series rather than the one with the most fittable segments.
    cands = []
    if combined is not None:
        cands.append((fit_bacc(combined), combined, 16, role))
        cands.append((segmented_pingpong(combined, resets), combined, 16, role))
        cands.append((segmented_bacc(combined, resets), combined, 16, role))
        # A tick-banded reflecting sweep (FC PW): per-frame stride = rate[tick].
        # Appended after the scalar-step modes so it loses ties (never steals a
        # clean constant-step fit) and only wins on strictly higher fidelity.
        cands.append((segmented_tickband(combined, resets), combined, 16, role))
    cands.append((fit_bacc(series), series, 8, "full"))
    cands.append((segmented_bacc(series, resets), series, 8, "full"))
    best = None
    best_fid = -1.0
    for fit, src, width, crole in cands:
        if fit is None:
            continue
        finished = _bacc_finish(fit, src, width, crole)
        fid = _candidate_fidelity(finished, series, crole)
        if fid > best_fid:
            best, best_fid = finished, fid
    return best


def _candidate_fidelity(desc, series, role) -> float:
    """Fraction of frames a BACC descriptor regenerates the 8-bit register series."""
    full = _recon_bacc_full(desc, len(series))
    if role == "lo":
        full = full & 0xFF
    elif role == "hi":
        full = (full >> 8) & 0xFF
    return float(np.mean(full == series))


def _bacc_finish(fit, src, width, role):
    """Attach reconstruction fields (width/byte role + per-segment seeds)."""
    fit["width"] = width
    fit["byte_role"] = role
    if "resets" not in fit:
        fit["resets"] = [0]
        fit["seeds"] = [int(np.asarray(src, dtype=np.int64)[0])]
    return fit


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
        return _finish_seq(result, series)

    # FREQ is base (note->freq table, sequencer-indexed) + modulation + hard-
    # restart override: a composite, not a single accumulator. Choose between the
    # composite decomposition and an accumulator fit by which actually
    # reconstructs the register; if neither does (e.g. a tick-driven pitch-table
    # player not yet modelled) record the best-effort feeder.
    if reg_off in (_FREQ_LO, _FREQ_HI):
        return _classify_freq(trace, sid_addr, series, voice, reg_off, ctx, result)

    bacc = _bacc_candidate(trace, sid_addr, series, voice, reg_off, ctx)
    if bacc is not None:
        result.update(bacc)
        cell, _recon, frac = _best_feeder_at_write(series, sid_addr, ctx)
        # The feeder cell captures dwell/hold the bare recurrence can miss, so it
        # is used at reconstruction when attached (:func:`_recon_bacc`). The feeder
        # plus its held-seed prelude (the captured cell drives every live frame, the
        # prelude fills the pre-modulation leading hold the cell has not yet been
        # written on) is attached only when it strictly beats the recurrence's own
        # fidelity, so a model that already regenerates the register (e.g. the
        # tick-banded sweep) is never displaced by a lower-fidelity captured cell.
        recurrence_fid = _candidate_fidelity(bacc, series, bacc.get("byte_role", "full"))
        if cell is not None and frac >= 0.5:
            trial = dict(result)
            trial["cell"] = cell
            trial["sid"] = int(sid_addr)
            trial["cell_frac"] = round(float(frac), 4)
            _attach_seed_prelude(trial, series, ctx)
            if _descriptor_fidelity(trial, series, ctx) > recurrence_fid:
                result = trial
        return result

    result = _classify_walk_or_seq(trace, sid_addr, series, wr, on_frames, ctx, result)
    if reg_off == _CTRL:
        result = _maybe_xor_ctrl(sid_addr, series, ctx, result)
        result = _maybe_and_ctrl(sid_addr, series, ctx, result)
    result = _maybe_feeder_upgrade(result, series, sid_addr, ctx)
    return result


def _descriptor_fidelity(desc, series, ctx):
    """Fraction of frames a descriptor's reconstruction matches ``series``."""
    recon = reconstruct_register(desc, ctx.stateseq.ticks, sampler=ctx.sampler)
    if recon is None:
        return 0.0
    return float(np.mean(recon == np.asarray(series, dtype=np.int64)))


def _bacc_with_feeder(bacc, series, sid_addr, ctx):
    """Attach the captured accumulator feeder cell to a BACC descriptor."""
    if bacc is None:
        return None
    cell, _recon, frac = _best_feeder_at_write(series, sid_addr, ctx)
    if cell is not None and frac >= 0.5:
        bacc["cell"] = cell
        bacc["sid"] = int(sid_addr)
        bacc["cell_frac"] = round(float(frac), 4)
    return bacc


def _attach_seed_prelude(result, series, ctx, max_latches: int = 6):
    """Fill the pre-modulation held-seed prelude of a cell-fed accumulator.

    Before its feeder cell's first RAM write the register holds its note-on seed
    (the instrument PW seed loaded once and held until modulation starts), but the
    captured cell still reads its power-on default, so the cell replay is wrong on
    those leading frames. Capture that bounded held prelude as SEQ latches; the
    cell drives every frame it is actually written. Recorded only when the hold is
    a few latches (a genuine seed prelude, not arbitrary modulation), so it never
    displaces the captured cell where the cell is live.
    """
    sampler = ctx.sampler
    cell = result.get("cell")
    if sampler is None or cell is None:
        return
    sid = result.get("sid", result.get("addr"))
    end = sampler.first_live_frame(cell, sid)
    if not 0 < end < ctx.n_frames:
        return
    frames, values = _seq_latches(np.asarray(series, dtype=np.int64)[:end])
    if len(frames) > max_latches:
        return
    result["prelude_end"] = int(end)
    result["prelude_frames"] = frames
    result["prelude_values"] = values


def _classify_freq(trace, sid_addr, series, voice, reg_off, ctx, result):
    """Pick the FREQ model (composite vs accumulator) with the best round-trip."""
    candidates = [
        _pitch_walk(trace, sid_addr, reg_off, ctx),
        _composite(trace, sid_addr, series, ctx, reg_off),
        _bacc_with_feeder(
            _bacc_candidate(trace, sid_addr, series, voice, reg_off, ctx), series, sid_addr, ctx
        ),
    ]
    best, best_fid = None, -1.0
    for cand in candidates:
        if cand is None:
            continue
        fid = _descriptor_fidelity(cand, series, ctx)
        if fid > best_fid:
            best, best_fid = cand, fid
    if best is not None and best_fid >= 0.6:
        result.update(best)
        return result
    return _xstate_with_feeder(result, series, sid_addr, ctx)


def _finish_seq(result, series):
    """Attach the event latch points an SEQ register reconstructs from."""
    frames, values = _seq_latches(series)
    result["latch_frames"] = frames
    result["latch_values"] = values
    return result


def _classify_walk_or_seq(trace, sid_addr, series, wr, on_frames, ctx, result) -> dict:
    """Resolve a per-frame register as TABLE_WALK, COMPOSITE, else SEQ / XSTATE."""
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
            _recover_gate(tw, series, ctx)
            _recover_walk_overrides(tw, series, ctx)
            result.update(tw)
            return result

    if note_gated or sparse or not per_frame:
        result.update(type="SEQ")
        return _finish_seq(result, series)

    # No single generator fits: decompose into base + modulation + overrides
    # before declaring the dependency not-yet-modelled (XSTATE).
    _voice, reg_off = _voice_of(sid_addr)
    composite = _composite(trace, sid_addr, series, ctx, reg_off)
    if composite is not None and composite["residual"] >= 0.6:
        result.update(composite)
        return result
    # Not yet decomposed into a single generator: record the closest observable
    # feeder so reconstruction (and round_trip) reports the actual gap rather
    # than nothing -- the dependency is modellable, just not yet modelled.
    return _xstate_with_feeder(result, series, sid_addr, ctx)


def _xstate_with_feeder(result, series, sid_addr, ctx):
    """Mark XSTATE and attach the closest observable feeder for best-effort recon."""
    result.update(type="XSTATE")
    cell, _recon, frac = _best_feeder_at_write(series, sid_addr, ctx)
    if cell is not None and frac > 0.0:
        result["cell"] = cell
        result["sid"] = int(sid_addr)
        result["cell_frac"] = round(float(frac), 4)
    return result


def _maybe_feeder_upgrade(result, series, sid_addr, ctx, min_feeder: float = 0.999):
    """Recover a per-frame register as a FEEDER (exact captured-cell copy).

    The player often computes a register's value into a RAM cell and then stores
    that cell to SID; a register that is a near-exact latched copy of such a
    captured feeder cell is a real per-frame primitive (``FEEDER``), not an
    unmodelled dependency (``XSTATE``) nor an over-fit table/composite. This
    generalizes the original filter-only relabel ($D415-$D418) to every per-frame
    register, in two cases:

    1. an ``XSTATE`` result whose attached feeder cell is an exact copy
       (``cell_frac >= min_feeder``) -- e.g. a wave-table CTRL written through a
       gated shadow cell; relabel it ``FEEDER``;
    2. an imperfect ``TABLE_WALK``/``COMPOSITE`` that a captured feeder cell,
       sampled at the write instant, reconstructs exactly (``>= min_feeder``) and
       strictly better than the chosen generator -- the register IS that cell
       copy; replace the descriptor.

    Strictly non-worsening: case 2 only fires when the feeder beats the current
    generator, and case 1 never changes the reconstruction (both replay the same
    cell). Generators that already reconstruct perfectly are left untouched.
    """
    rtype = result.get("type")
    if rtype == "XSTATE" and result.get("cell_frac", 0.0) >= min_feeder:
        result["type"] = "FEEDER"
        return result
    if rtype in ("TABLE_WALK", "COMPOSITE"):
        cur = _descriptor_fidelity(result, series, ctx)
        if cur >= 1.0:
            return result
        cell, _recon, frac = _best_feeder_at_write(series, sid_addr, ctx)
        if cell is not None and frac >= min_feeder and frac > cur:
            keep = {k: result[k] for k in ("addr", "store_pcs", "n_writes") if k in result}
            keep.update(
                type="FEEDER",
                cell=int(cell),
                sid=int(sid_addr),
                cell_frac=round(float(frac), 4),
            )
            return keep
    return result


def _recover_gate(tw, series, ctx):
    """Find a gate-mask cell so ``CTRL = table[cursor] & gate`` reconstructs better.

    A waveform table walk is gated by a per-voice mask cell (``$FF`` pass /
    ``$FE`` force gate off). Reconstructing with the captured gate cell beats a
    constant mask whenever the gate toggles, so record it as ``gate_addr``.
    """
    sampler = ctx.sampler
    if sampler is None or tw.get("cursor_addr") is None:
        return
    table = np.asarray(tw["table"], dtype=np.int64)
    cursor = sampler.eof(tw["cursor_addr"])
    idx = np.clip(cursor * tw["stride"] + tw["cursor_offset"], 0, len(table) - 1)
    tv = table[idx]
    base_fid = float(np.mean(series == (tv & tw["mask"])))
    grid = ctx.stateseq.grid.astype(np.int64)
    best_fid, best_addr = base_fid, None
    for j in range(grid.shape[1]):
        col = grid[:, j]
        if int(col.max()) < 0xF0 or len(np.unique(col)) > 4:
            continue
        fid = float(np.mean(series == (tv & col)))
        if fid > best_fid + 0.05:
            best_fid, best_addr = fid, int(ctx.stateseq.addrs[j])
    if best_addr is not None:
        tw["gate_addr"] = best_addr
        tw["residual"] = best_fid


def _recover_walk_overrides(tw, series, ctx, max_overrides: int = 3):
    """Recover value-forcing overrides (e.g. CTRL ``$08``/``$81``) over a walk.

    The table can hold non-waveform command bytes (e.g. a ``$0A`` wavetable
    control entry) that the player never emits, forcing the real waveform at that
    cursor cell instead; each such force is one override. Overrides are taken
    greedily and an override is kept only when it strictly raises reproduction, so
    raising the cap never regresses a register that needed fewer.
    """
    if ctx.sampler is None:
        return
    series = np.asarray(series, dtype=np.int64)
    work = _recon_table(tw, ctx.n_frames, ctx.sampler)
    best_fid = float(np.mean(work == series))
    overrides = []
    for _ in range(max_overrides):
        forced = np.where(series == work, -1, series).astype(np.int64)
        ov = _override_descriptor(forced, ctx)
        if ov is None or ov in overrides:
            break
        cand = _apply_overrides(work, [ov], ctx.sampler)
        cand_fid = float(np.mean(cand == series))
        if cand_fid <= best_fid:
            break
        overrides.append(ov)
        work, best_fid = cand, cand_fid
    if overrides:
        tw["overrides"] = overrides
        tw["residual"] = best_fid


def _best_feeder_at_write(series, sid_addr, ctx):
    """Best feeder cell reproducing ``series`` when sampled at the write instant.

    Prefilters by end-of-frame match, then re-scores the top candidates with the
    cell value sampled at the register's SID-write cycle (where the player reads
    it), which captures any modulation already folded into the feeder.

    The prefilter also ranks each cell by its match to the register series shifted
    one frame -- an output-then-compute player (defMON) writes the SID from a
    self-modified operand then computes the NEXT value into its feeder cell, so the
    cell's end-of-frame value leads the register by one call; that feeder still
    reconstructs exactly when sampled at the write instant, so include the
    one-call-latency shift so it survives the prefilter.
    """
    ss = ctx.stateseq
    if ss.grid.shape[1] == 0 or ctx.sampler is None:
        return None, None, 0.0
    grid = ss.grid.astype(np.int64)
    s = np.asarray(series, dtype=np.int64)
    eof_match = (grid == s[:, None]).mean(axis=0)
    lead = (grid == np.roll(s, -1)[:, None]).mean(axis=0)
    rank = np.maximum(eof_match, lead)
    order = np.argsort(rank)[::-1][:8]
    best_addr, best_recon, best_frac = None, None, -1.0
    for j in order:
        cell = int(ss.addrs[j])
        recon = ctx.sampler.at_write(cell, sid_addr)
        frac = float(np.mean(recon == s))
        if frac > best_frac:
            best_addr, best_recon, best_frac = cell, recon, frac
    return best_addr, best_recon, best_frac


def _find_override(forced, ctx, max_terms: int = 3):
    """A conjunction of cell predicates that fires exactly on ``forced`` frames.

    Candidate predicates (cell value-equality / single-bit tests / small
    value-membership) are restricted to those true on every forced frame (recall
    1.0); a greedy intersection then drives precision to 1.0. A membership term
    ``cell in {v0,..}`` (recovered set of <= ``max_set`` distinct cell values, e.g.
    the per-voice waveform shadow whose value selects the instruments that
    hard-restart) captures a force gated by a handful of states no single
    equality/bit can express. Returns ``[(cell, mask, value) | (cell, "in",
    (v0,..)), ...]`` or ``None``.
    """
    if int(forced.sum()) == 0:
        return None
    grid = ctx.stateseq.grid.astype(np.int64)
    addrs = ctx.stateseq.addrs
    n_forced = int(forced.sum())
    max_set = 6
    cands = []
    for j in range(grid.shape[1]):
        col = grid[:, j]
        fc = col[forced]
        uniq = np.unique(fc)
        if len(uniq) == 1:
            cands.append((col == int(uniq[0]), (int(addrs[j]), 0xFF, int(uniq[0]))))
        elif 2 <= len(uniq) <= max_set and len(uniq) < len(np.unique(col)):
            # A force gated by the cell holding one of a few states (recall 1.0 by
            # construction); the greedy precision/strict-improvement gating below
            # rejects it unless it genuinely tightens the selection.
            cands.append((np.isin(col, uniq), (int(addrs[j]), "in", tuple(int(x) for x in uniq))))
        for b in range(8):
            bit = (fc >> b) & 1
            if bit.min() == bit.max():
                val = int(bit[0]) << b
                cands.append(((col & (1 << b)) == val, (int(addrs[j]), 1 << b, val)))
    sel = np.ones(len(forced), dtype=bool)
    terms = []
    precision = n_forced / max(1, int(sel.sum()))
    for _ in range(max_terms):
        best_prec, best_sel, best_term = precision, None, None
        for pred, term in cands:
            cand_sel = sel & pred
            kept = int(np.sum(cand_sel & forced))
            total = int(cand_sel.sum())
            if kept < n_forced or total == 0 or term in terms:
                continue
            prec = kept / total
            if prec > best_prec:
                best_prec, best_sel, best_term = prec, cand_sel, term
        if best_term is None:
            break
        precision, sel = best_prec, best_sel
        terms.append(best_term)
        if precision >= 0.999:
            break
    return terms if terms and precision >= 0.95 else None


def _feeder_cell(target, sid_addr, ctx):
    """Best single feeder cell (sampled at the write instant) for an 8-bit target."""
    ss = ctx.stateseq
    if ss.grid.shape[1] == 0:
        return None
    grid = ss.grid.astype(np.int64)
    t = np.asarray(target, dtype=np.int64) & 0xFF
    order = np.argsort((grid == t[:, None]).mean(axis=0))[::-1][:8]
    best_addr, best_frac = None, -1.0
    for j in order:
        cell = int(ss.addrs[j])
        frac = float(np.mean((ctx.sampler.at_write(cell, sid_addr) & 0xFF) == t))
        if frac > best_frac:
            best_addr, best_frac = cell, frac
    return best_addr


def _override_descriptor(forced_byte, ctx):
    """A value-forcing override from the dominant residual byte (or ``None``).

    ``forced_byte`` is the register byte on frames the base failed to explain
    and ``-1`` elsewhere; the dominant value becomes a forced override gated by a
    recovered cell predicate.
    """
    seen = forced_byte[forced_byte >= 0]
    if len(seen) == 0:
        return None
    vals, counts = np.unique(seen, return_counts=True)
    if counts.max() / counts.sum() < 0.5:
        return None
    force_val = int(vals[counts.argmax()])
    terms = _find_override(forced_byte == force_val, ctx)
    return None if terms is None else {"predicate": terms, "force": force_val}


def _anchor_bases(cur_s, val_s, cmax, ram, ramhist, n_anchors=16, cap=256):
    """Candidate table bases so ``ram[base + cur_s] == val_s`` could hold.

    Each anchor frame ``f`` votes for every base ``pos - cur_s[f]`` where
    ``ram[pos] == val_s[f]``; the true base collects a vote from every in-table
    anchor while spurious bases get one or two, so the top-voted bases survive
    even when the values are low-entropy and ubiquitous in the image (the FREQ hi
    bytes). The rarest-valued anchor frames are used first to keep the vote pools
    small. Returns the most-voted candidate bases.
    """
    order = np.argsort(ramhist[val_s], kind="stable")[:n_anchors]
    pools = []
    for f in order:
        pos = np.nonzero(ram == int(val_s[f]))[0].astype(np.int64) - int(cur_s[f])
        pools.append(pos[(pos >= 0) & (pos + cmax < len(ram))])
    if not pools:
        return np.empty(0, dtype=np.int64)
    allb = np.concatenate(pools)
    if len(allb) == 0:
        return np.empty(0, dtype=np.int64)
    ub, cnt = np.unique(allb, return_counts=True)
    return ub[np.argsort(cnt, kind="stable")[::-1][:cap]]


def _pitch_table_for_cell(cursor, lo_o, hi_o, cmax, ram, ramhist, sample):
    """(base_lo, base_hi, combined_fraction) of a 16-bit pitch table for ``cursor``.

    Finds, in the static RAM image, contiguous lo/hi byte sub-tables such that
    ``ram[base_lo + cursor]`` and ``ram[base_hi + cursor]`` reproduce the emitted
    FREQ lo/hi bytes. The (high-entropy) lo base is fixed first; the hi base is
    then chosen to maximise the *combined* lo&hi match, so the low-entropy hi
    bytes (values 1..8) cannot anchor a spurious base. Scored on the subsample,
    so the cost is independent of trace length.
    """
    cur_s = cursor[sample]
    lo_s = lo_o[sample]
    hi_s = hi_o[sample]
    lo_bases = _anchor_bases(cur_s, lo_s, cmax, ram, ramhist)
    if len(lo_bases) == 0:
        return None
    lo_scores = np.array([np.mean(ram[b + cur_s] == lo_s) for b in lo_bases])
    base_lo = int(lo_bases[int(lo_scores.argmax())])
    lo_ok = ram[base_lo + cur_s] == lo_s
    hi_bases = _anchor_bases(cur_s, hi_s, cmax, ram, ramhist)
    if len(hi_bases) == 0:
        return None
    combined = np.array([np.mean(lo_ok & (ram[b + cur_s] == hi_s)) for b in hi_bases])
    k = int(combined.argmax())
    return base_lo, int(hi_bases[k]), float(combined[k])


def _pitch_index_cells(idx, inb, grid, addrs, tlen, primary, max_extra=2):
    """Index cells whose end-of-frame sum reproduces a pitch-table index.

    ``idx`` is the per-frame table index inverted from the oracle (``-1`` off
    table, ``inb`` marks the in-table frames). Starting from the ``primary``
    column (the note cell), greedily add cells whose value explains the residual
    ``idx - running``, keeping each only while it raises the in-table match. The
    sum models ``note + transpose + arp/wavetable offset`` of the FC pitch walk.
    """
    chosen = [int(addrs[primary])]
    running = grid[:, primary].copy()
    used = {primary}

    def score(run):
        return float(np.mean((np.clip(run, 0, tlen - 1) == idx)[inb]))

    cur_score = score(running)
    for _ in range(max_extra):
        resid = idx - running
        match = (grid == resid[:, None])[inb].mean(axis=0)
        match[list(used)] = -1.0
        j = int(match.argmax())
        if j in used:
            break
        cand = running + grid[:, j]
        if score(cand) <= cur_score + 1e-9:
            break
        chosen.append(int(addrs[j]))
        used.add(j)
        running = cand
        cur_score = score(cand)
    return chosen, running


def _pitch_walk(trace, sid_addr, reg_off, ctx):
    """Recover FC-style FREQ: a 16-bit pitch-table walk indexed by note + offset.

    FREQ is ``pitchtable[idx]`` with ``idx = note (+ transpose + arp/wavetable
    offset)``, all paced by the per-voice note sequencer -- a two-level table
    walk, not a single accumulator. The pitch table (lo/hi byte sub-tables) is
    located in the static RAM image via the note state cell; the index is then
    recovered as a sum of observable cells (:func:`_pitch_index_cells`). Off-table
    frames (vibrato/portamento, computed in registers and never stored to RAM)
    are left to a value-forcing override pass. Returns a ``PITCHWALK`` descriptor
    or ``None``.
    """
    ram = ctx.ram
    partner = _LOHI_PARTNER.get(reg_off)
    if ram is None or partner is None or ctx.sampler is None:
        return None
    role = "lo" if reg_off == _FREQ_LO else "hi"
    lo_addr = sid_addr if role == "lo" else sid_addr + partner
    hi_addr = sid_addr + partner if role == "lo" else sid_addr
    lo_o = _register_series(trace, lo_addr, ctx.kind)[1]
    hi_o = _register_series(trace, hi_addr, ctx.kind)[1]
    ramu = np.asarray(ram, dtype=np.uint8)
    grid = ctx.stateseq.grid.astype(np.int64)
    addrs = ctx.stateseq.addrs
    n = grid.shape[0]
    if n == 0:
        return None
    # An index (note) cell has a small range and few distinct values; that prefilter
    # plus a subsampled, anchor-based base search keeps the cost trace-length
    # independent (the table walk reconstructs over all frames either way).
    cols = [
        j
        for j in range(grid.shape[1])
        if grid[:, j].max() <= 95
        and grid[:, j].min() != grid[:, j].max()
        and len(np.unique(grid[:, j])) <= 64
    ][:48]
    sample = np.unique(np.linspace(0, n - 1, min(n, 256)).astype(np.int64))
    ramhist = np.bincount(ramu, minlength=256)
    found = []
    for j in cols:
        cur = grid[:, j]
        res = _pitch_table_for_cell(cur, lo_o, hi_o, int(cur.max()), ramu, ramhist, sample)
        if res is not None and res[2] >= 0.2:
            found.append((res[2], j, res[0], res[1]))
    if not found:
        return None
    # The base of a single (note) cell only matches frames whose note offset is
    # zero, so try the strongest few candidate tables and keep whichever, after
    # recovering its additive index, actually reconstructs FREQ best.
    found.sort(reverse=True)
    best_desc, best_fid = None, -1.0
    for _frac, primary, base_lo, base_hi in found[:4]:
        desc = _build_pitchwalk(role, base_lo, base_hi, primary, lo_o, hi_o, grid, addrs, ramu, ctx)
        if desc is None:
            continue
        if desc["residual"] > best_fid:
            best_desc, best_fid = desc, desc["residual"]
    return best_desc


def _build_pitchwalk(role, base_lo, base_hi, primary, lo_o, hi_o, grid, addrs, ramu, ctx):
    """Assemble (and score) a PITCHWALK descriptor for one candidate pitch table."""
    span = abs(base_hi - base_lo) if base_hi != base_lo else 96
    tlen = int(
        min(max(span, int(grid[:, primary].max()) + 1), 256, 65536 - base_lo, 65536 - base_hi)
    )
    lotab = ramu[base_lo : base_lo + tlen].astype(np.int64)
    hitab = ramu[base_hi : base_hi + tlen].astype(np.int64)
    tab16 = lotab | (hitab << 8)
    tgt = (lo_o | (hi_o << 8)).astype(np.int64)
    idx = np.full(len(tgt), -1, dtype=np.int64)
    for k in range(tlen):
        idx[tgt == int(tab16[k])] = k
    inb = idx >= 0
    if not np.any(inb):
        return None
    index_cells, _run = _pitch_index_cells(idx, inb, grid, addrs, tlen, primary)
    desc = {
        "type": "PITCHWALK",
        "byte_role": role,
        "lo_base": int(base_lo),
        "hi_base": int(base_hi),
        "lo_table": lotab.astype(np.uint8),
        "hi_table": hitab.astype(np.uint8),
        "index_cells": index_cells,
        "overrides": [],
    }
    out = _recon_pitchwalk(desc, ctx.n_frames, ctx.sampler)
    oracle = lo_o if role == "lo" else hi_o
    forced = np.where(out == oracle, -1, oracle).astype(np.int64)
    ov = _override_descriptor(forced, ctx)
    if ov is not None:
        desc["overrides"].append(ov)
        out = _recon_pitchwalk(desc, ctx.n_frames, ctx.sampler)
    desc["residual"] = float(np.mean(out == oracle))
    return desc


def _composite(trace, sid_addr, series, ctx, reg_off=None):
    """Decompose a register into base + additive modulation + value overrides.

    The base is a sequencer-indexed feeder cell sampled at the write instant
    (16-bit for a freq/PW lo-hi pair); the modulation is the residual carried in
    a second captured accumulator cell (its recurrence shape is recoverable via
    :func:`fit_bacc`); overrides force outlier values (e.g. the hard-restart
    ``$FFFF``) where a cell predicate holds. Emits a COMPOSITE descriptor whose
    reconstruction reproduces the register, or ``None`` when no base dominates.
    """
    if ctx.sampler is None:
        return None
    if _LOHI_PARTNER.get(reg_off) is not None:
        return _composite16(trace, sid_addr, reg_off, ctx)
    return _composite8(sid_addr, series, ctx)


def _composite16(trace, sid_addr, reg_off, ctx):
    """16-bit base + accumulator + override for a freq/PW lo or hi register."""
    n = ctx.n_frames
    partner_off = _LOHI_PARTNER[reg_off]
    role = "lo" if reg_off in (_FREQ_LO, _PW_LO) else "hi"
    lo_addr = sid_addr if role == "lo" else sid_addr + partner_off
    hi_addr = sid_addr + partner_off if role == "lo" else sid_addr
    lo_oracle = _register_series(trace, lo_addr, ctx.kind)[1]
    hi_oracle = _register_series(trace, hi_addr, ctx.kind)[1]
    target = combine_lohi(lo_oracle, hi_oracle)
    base_lo = _feeder_cell(lo_oracle, lo_addr, ctx)
    base_hi = _feeder_cell(hi_oracle, hi_addr, ctx)
    if base_lo is None or base_hi is None:
        return None
    base = {"lo": (base_lo, int(lo_addr)), "hi": (base_hi, int(hi_addr))}
    base_val = _comp_part(base, n, ctx.sampler)
    out_byte = np.asarray(lo_oracle if role == "lo" else hi_oracle, dtype=np.int64)
    target_byte = out_byte & 0xFF

    # An additive modulation term is only real when it improves the *whole*
    # reconstruction. With output-then-compute players (defMON) the base operand
    # cell already carries the whole value (one call late), so its residual is a
    # phase artefact a feeder search can spuriously "explain" with a noise cell,
    # lowering fidelity; with a genuine accumulator the mod is needed. Build both
    # the base-only and base+mod descriptors -- each with its own override pass,
    # since the 16-bit base+mod sum is what isolates the override frames -- and
    # keep whichever reconstructs the register best (strict-improvement, cf. the
    # table-walk override guard).
    mods = [None]
    residual = (target - base_val) & 0xFFFF
    if np.any(residual != 0):
        mod_lo = _feeder_cell(residual & 0xFF, lo_addr, ctx)
        mod_hi = _feeder_cell((residual >> 8) & 0xFF, hi_addr, ctx)
        if mod_lo is not None and mod_hi is not None:
            mods.append({"lo": (mod_lo, int(lo_addr)), "hi": (mod_hi, int(hi_addr))})

    best_fid, best_desc = -1.0, None
    for mod in mods:
        desc = {
            "type": "COMPOSITE",
            "byte_role": role,
            "width_mask": 0xFFFF,
            "base": base,
            "mod": mod,
            "overrides": [],
        }
        modelled = (base_val + _comp_part(mod, n, ctx.sampler)) & 0xFFFF
        _best_composite_override(desc, modelled, target, out_byte, target_byte, n, ctx)
        fid = float(np.mean(_recon_composite(desc, n, ctx.sampler) == target_byte))
        desc["residual"] = fid
        if fid > best_fid:
            best_fid, best_desc = fid, desc
    return best_desc


def _best_composite_override(desc, modelled, target, out_byte, target_byte, n, ctx):
    """Attach the per-byte residual override only when it reconstructs strictly
    better; none otherwise.

    The forced value is the register byte on the frames the base+mod model fails
    (a note-onset reset / hard-restart), gated by a recovered cell predicate (an
    equality, single-bit, or small value-membership test -- see
    :func:`_find_override`). Strictly non-worsening: an override that does not
    raise reconstruction (a spurious membership predicate, say) is dropped, so a
    register an override already nails is never displaced.
    """
    base_fid = float(np.mean(_recon_composite(desc, n, ctx.sampler) == target_byte))
    candidates = []
    ov_b = _override_descriptor(np.where(modelled == target, -1, out_byte).astype(np.int64), ctx)
    if ov_b is not None:
        candidates.append(ov_b)
    best_fid, best_ov = base_fid, None
    for ov in candidates:
        desc["overrides"] = [ov]
        fid = float(np.mean(_recon_composite(desc, n, ctx.sampler) == target_byte))
        if fid > best_fid:
            best_fid, best_ov = fid, ov
    desc["overrides"] = [best_ov] if best_ov is not None else []


def _composite8(sid_addr, series, ctx):
    """8-bit base feeder + override composite."""
    base_cell, base_recon, frac = _best_feeder_at_write(series, sid_addr, ctx)
    if base_cell is None or frac < 0.5 or frac >= 0.999:
        return None
    desc = {
        "type": "COMPOSITE",
        "byte_role": "full",
        "width_mask": 0xFF,
        "base": {"cell": int(base_cell), "sid": int(sid_addr)},
        "mod": None,
        "overrides": [],
    }
    forced = np.where(series == base_recon, -1, series).astype(np.int64)
    ov = _override_descriptor(forced, ctx)
    if ov is not None:
        desc["overrides"].append(ov)
    recon = _recon_composite(desc, ctx.n_frames, ctx.sampler)
    desc["residual"] = float(np.mean(recon == series))
    return desc


def _xor_pair(sid_addr, series, ctx, min_fid: float = 0.999):
    """Recover a CTRL register written as ``cellA XOR cellB`` (or ``None``).

    A common gate/test/waveform idiom (defMON, and any player that flips control
    bits with an eor mask rather than rewriting the whole byte) computes
    ``CTRL = base XOR eor`` from two captured RAM cells. Neither cell alone
    reproduces CTRL, so the per-frame machinery (table-walk / feeder / SEQ) can
    only partly fit it; the exact-XOR of the right pair reproduces it byte-for-
    byte. Searches low-entropy (control-like) cells, sampled at the SID-write
    instant, over a frame subsample, then verifies the best pair on all frames.
    Returns an ``XOR`` descriptor only when it reproduces ``>= min_fid``.
    """
    ss = ctx.stateseq
    if ss.grid.shape[1] == 0 or ctx.sampler is None:
        return None
    s = np.asarray(series, dtype=np.int64) & 0xFF
    n = len(s)
    grid = ss.grid
    addrs = ss.addrs
    # Control bytes toggle among a handful of values; that prefilter bounds the
    # pair search to O(m^2 * subsample), independent of trace length.
    cols = [j for j in range(grid.shape[1]) if len(np.unique(grid[:, j])) <= 24]
    if len(cols) < 2:
        return None
    mat_full = np.stack(
        [ctx.sampler.at_write(int(addrs[j]), sid_addr).astype(np.int64) & 0xFF for j in cols],
        axis=1,
    )
    sample = np.unique(np.linspace(0, n - 1, min(n, 512)).astype(np.int64))
    mat = mat_full[sample]
    ss_s = s[sample]
    best = (-1.0, None, None)
    for ai in range(len(cols)):
        f = (mat[:, ai][:, None] ^ mat == ss_s[:, None]).mean(axis=0)
        k = int(f.argmax())
        if f[k] > best[0]:
            best = (float(f[k]), ai, k)
    _frac, ai, bi = best
    if ai is None or ai == bi:
        return None
    recon = (mat_full[:, ai] ^ mat_full[:, bi]) & 0xFF
    fid = float(np.mean(recon == s))
    if fid < min_fid:
        return None
    return {
        "type": "XOR",
        "cell_a": int(addrs[cols[ai]]),
        "cell_b": int(addrs[cols[bi]]),
        "sid": int(sid_addr),
        "residual": fid,
    }


def _maybe_xor_ctrl(sid_addr, series, ctx, result, min_fid: float = 0.999):
    """Upgrade a CTRL register to the ``cellA XOR cellB`` model when it wins.

    Tried only after the generic per-frame classification, and adopted only when
    the XOR reproduces ``>= min_fid`` AND strictly beats the current descriptor
    (so a clean table-walk/SEQ CTRL that already reconstructs perfectly is never
    displaced, and no spurious pair can fire below near-exact fidelity).
    """
    xor = _xor_pair(sid_addr, series, ctx, min_fid)
    if xor is None:
        return result
    if xor["residual"] > _descriptor_fidelity(result, series, ctx):
        keep = {k: result[k] for k in ("addr", "store_pcs", "n_writes") if k in result}
        keep.update(xor)
        return keep
    return result


def _and_pair(sid_addr, series, ctx, min_fid: float = 0.999):
    """Recover a CTRL register written as ``waveCell AND gateCell`` (or ``None``).

    The GoatTracker2 idiom ``CTRL = chnwave AND chngate`` masks a waveform shadow
    cell with a gate cell holding ``$FF`` (pass) or ``$FE`` (force gate-off). The
    gate cell is near-binary while the waveform cell carries many distinct bytes,
    so the search is asymmetric: each low-entropy gate-like cell (``<= 4`` values)
    is AND-ed against every control-like candidate cell (``<= 32`` values),
    sampled at the SID-write instant over a frame subsample, then the best pair
    is verified on all frames. Returns an ``AND`` descriptor only when it
    reproduces ``>= min_fid``.
    """
    ss = ctx.stateseq
    if ss.grid.shape[1] == 0 or ctx.sampler is None:
        return None
    s = np.asarray(series, dtype=np.int64) & 0xFF
    n = len(s)
    grid = ss.grid
    addrs = ss.addrs
    uniq = [len(np.unique(grid[:, j])) for j in range(grid.shape[1])]
    cols = [j for j in range(grid.shape[1]) if uniq[j] <= 32]
    gates = [j for j in range(grid.shape[1]) if uniq[j] <= 4]
    if not cols or not gates:
        return None
    full = {
        j: ctx.sampler.at_write(int(addrs[j]), sid_addr).astype(np.int64) & 0xFF
        for j in set(cols) | set(gates)
    }
    sample = np.unique(np.linspace(0, n - 1, min(n, 512)).astype(np.int64))
    ss_s = s[sample]
    cand = np.stack([full[j][sample] for j in cols], axis=1)
    best = (-1.0, None, None)
    for g in gates:
        gv = full[g][sample]
        f = ((gv[:, None] & cand) == ss_s[:, None]).mean(axis=0)
        k = int(f.argmax())
        if f[k] > best[0]:
            best = (float(f[k]), g, cols[k])
    _frac, gj, cj = best
    if gj is None or gj == cj:
        return None
    desc = {
        "type": "AND",
        "cell_a": int(addrs[cj]),
        "cell_b": int(addrs[gj]),
        "sid": int(sid_addr),
        "overrides": [],
    }
    work = _and_recon_masked(desc, ctx)
    desc["residual"] = float(np.mean(work == s))
    # The steady CTRL is wave AND gate; the note-onset / hard-restart frames force
    # a control byte the shadow never carries. Recover those as value-forcing
    # overrides (same greedy, strictly-improving pass as the table-walk path) so a
    # gated wave-table CTRL written through a shadow cell recovers byte-exact.
    _recover_pair_overrides(desc, s, ctx)
    if desc["residual"] < min_fid:
        return None
    return desc


def _and_recon_masked(desc, ctx) -> np.ndarray:
    """``_recon_and`` with the pre-first-write power-on default applied (for scoring)."""
    base = _recon_and(desc, ctx.n_frames, ctx.sampler)
    written = ctx.sampler.written_mask(desc["sid"])
    return np.where(written, base, 0)


def _recover_pair_overrides(desc, series, ctx, max_overrides: int = 4):
    """Recover value-forcing overrides over an ``AND`` pair recon (CTRL onsets).

    Mirrors :func:`_recover_walk_overrides`: each onset/hard-restart byte the
    wave-AND-gate base fails to explain is forced where a recovered cell predicate
    holds, taken greedily and kept only when it strictly raises reproduction (so it
    never regresses a pair that already reconstructs exactly).
    """
    sampler = ctx.sampler
    if sampler is None:
        return
    series = np.asarray(series, dtype=np.int64)
    written = sampler.written_mask(desc["sid"])
    cur = _recon_and(desc, ctx.n_frames, sampler)
    work = np.where(written, cur, 0)
    best_fid = float(np.mean(work == series))
    overrides = []
    for _ in range(max_overrides):
        forced = np.where(series == work, -1, series).astype(np.int64)
        ov = _override_descriptor(forced, ctx)
        if ov is None or ov in overrides:
            break
        cur2 = _apply_overrides(cur, [ov], sampler)
        work2 = np.where(written, cur2, 0)
        fid2 = float(np.mean(work2 == series))
        if fid2 <= best_fid:
            break
        overrides.append(ov)
        cur, work, best_fid = cur2, work2, fid2
    if overrides:
        desc["overrides"] = overrides
        desc["residual"] = best_fid


def _maybe_and_ctrl(sid_addr, series, ctx, result, min_fid: float = 0.999):
    """Upgrade a CTRL register to the ``waveCell AND gateCell`` model when it wins.

    Mirrors :func:`_maybe_xor_ctrl`: adopted only when the AND reproduces
    ``>= min_fid`` AND strictly beats the current descriptor (so a CTRL that
    already reconstructs perfectly -- including a freshly adopted XOR -- is never
    displaced).
    """
    and_d = _and_pair(sid_addr, series, ctx, min_fid)
    if and_d is None:
        return result
    if and_d["residual"] > _descriptor_fidelity(result, series, ctx):
        keep = {k: result[k] for k in ("addr", "store_pcs", "n_writes") if k in result}
        keep.update(and_d)
        return keep
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


# -- per-frame cell sampling (for reconstruction) -------------------------


def _carry_at(cycles, values, sample_cycles) -> np.ndarray:
    """Value of a cell (last write strictly before each sample cycle)."""
    out = np.zeros(len(sample_cycles), dtype=np.int64)
    if len(cycles) == 0:
        return out
    order = np.argsort(cycles, kind="stable")
    wc = cycles[order]
    wv = values[order].astype(np.int64)
    pos = np.searchsorted(wc, sample_cycles, side="right")
    taken = pos > 0
    idx = np.clip(pos - 1, 0, len(wv) - 1)
    out[taken] = wv[idx][taken]
    return out


class _CellSampler:
    """Per-frame RAM-cell value sampling at frame end or at a SID-write instant.

    A register's output is the value of its feeder cell at the cycle the player
    stores it to the SID -- not necessarily the end-of-frame value -- so
    reconstruction samples feeder cells at the register's write cycle, while
    event flags (gate masks, hard-restart counters) are read at frame end.
    """

    def __init__(self, trace, ticks):
        self.trace = trace
        self.ticks = np.asarray(ticks, dtype=np.uint64)
        self.n = len(self.ticks)
        self._rw = trace.ram_writes()
        self._eof: dict = {}
        self._writecyc: dict = {}
        self._atwrite: dict = {}
        self._written: dict = {}

    def _cell_writes(self, addr):
        sel = self._rw[self._rw["addr"] == addr]
        return sel["cycle"], sel["value"]

    def eof(self, addr) -> np.ndarray:
        """End-of-frame carried value of a cell."""
        addr = int(addr)
        if addr not in self._eof:
            cyc, val = self._cell_writes(addr)
            self._eof[addr] = _carry_series(cyc, val, self.ticks)
        return self._eof[addr]

    def write_cycles(self, sid_addr) -> np.ndarray:
        """Per-frame cycle of the last SID write to ``sid_addr`` (else frame start)."""
        sid_addr = int(sid_addr)
        if sid_addr not in self._writecyc:
            wr = self.trace.sid_writes()
            sel = wr[wr["addr"] == sid_addr]
            wc = self.ticks.copy()
            if len(sel):
                cyc = np.sort(sel["cycle"])
                pos = np.searchsorted(cyc, _frame_bounds(self.ticks), side="left")
                taken = pos > 0
                idx = np.clip(pos - 1, 0, len(cyc) - 1)
                wc[taken] = cyc[idx][taken]
            self._writecyc[sid_addr] = wc
        return self._writecyc[sid_addr]

    def at_write(self, cell_addr, sid_addr) -> np.ndarray:
        """Value of ``cell_addr`` sampled just after ``sid_addr``'s write each frame."""
        key = (int(cell_addr), int(sid_addr))
        if key not in self._atwrite:
            sample = self.write_cycles(sid_addr) + np.uint64(2)
            cyc, val = self._cell_writes(int(cell_addr))
            self._atwrite[key] = _carry_at(cyc, val, sample)
        return self._atwrite[key]

    def cell_first_frame(self, cell_addr) -> int:
        """Frame index of a RAM cell's first write (``0`` if never written)."""
        cyc, _ = self._cell_writes(int(cell_addr))
        if len(cyc) == 0:
            return 0
        return int(np.searchsorted(self.ticks, np.uint64(int(cyc.min())), side="right")) - 1

    def first_live_frame(self, cell_addr, sid_addr) -> int:
        """First frame whose at-write sample of ``cell_addr`` reflects a real write.

        A feeder cell may be written later in its first frame than the register's
        store, so :meth:`at_write` still reads the power-on default on that frame
        even though :meth:`cell_first_frame` counts it written. The held-seed
        prelude must cover up to here -- the first frame the captured cell actually
        drives the register.
        """
        cyc, _ = self._cell_writes(int(cell_addr))
        if len(cyc) == 0:
            return self.n
        sample = self.write_cycles(sid_addr) + np.uint64(2)
        live = np.nonzero(sample >= np.uint64(int(cyc.min())))[0]
        return int(live[0]) if len(live) else self.n

    def written_mask(self, sid_addr) -> np.ndarray:
        """Per-frame bool: True once ``sid_addr`` has been written by frame end.

        Before its first write a register holds the power-on default (``0``, the
        oracle's carry value), so reconstruction zeros these leading frames.
        """
        sid_addr = int(sid_addr)
        if sid_addr not in self._written:
            wr = self.trace.sid_writes()
            sel = wr[wr["addr"] == sid_addr]
            mask = np.zeros(self.n, dtype=bool)
            if len(sel):
                cyc = np.sort(sel["cycle"])
                pos = np.searchsorted(cyc, _frame_bounds(self.ticks), side="left")
                mask = pos > 0
            self._written[sid_addr] = mask
        return self._written[sid_addr]


def _sampler_for(ticks, trace, sampler):
    """Return the supplied sampler, or build one from a trace (or None)."""
    if sampler is not None:
        return sampler
    if trace is not None:
        return _CellSampler(trace, ticks)
    return None


# -- reconstruction (regenerate a register from its descriptor) -----------


def _seq_latches(series) -> tuple:
    """(change frames incl. 0, latched values) describing an event-latched series."""
    s = np.asarray(series, dtype=np.int64)
    changes = (np.nonzero(np.diff(s) != 0)[0] + 1).tolist()
    frames = [0] + changes
    values = [int(s[f]) for f in frames]
    return frames, values


def _recon_seq(desc, n) -> np.ndarray:
    frames = np.asarray(desc.get("latch_frames", [0]), dtype=np.int64)
    values = np.asarray(desc.get("latch_values", [0]), dtype=np.int64)
    idx = np.clip(np.searchsorted(frames, np.arange(n), side="right") - 1, 0, len(values) - 1)
    return values[idx]


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
    if length <= 0:
        return np.empty(0, dtype=np.int64)
    if mode == "pingpong" and hi > lo:
        d = (1 if seed <= (lo + hi) // 2 else -1) if direction is None else direction
        clo = lo if clamp_lo is None else clamp_lo
        chi = hi if clamp_hi is None else clamp_hi
        ds = step if down_step is None else down_step
        series, _ = _simulate_pingpong(lo, hi, clo, chi, step, ds, int(seed), d, length)
        return series
    if mode == "reflect" and hi > lo:
        d = (1 if seed <= (lo + hi) // 2 else -1) if direction is None else direction
        series, _ = _simulate_reflect(lo, hi, step, int(seed), d, length)
        return series
    mod = modulus if modulus else (hi - lo + step)
    if mod <= 0:
        return np.full(length, int(seed), dtype=np.int64)
    return lo + ((int(seed) - lo) + step * np.arange(length, dtype=np.int64)) % mod


def _recon_tickband(desc, n) -> np.ndarray:
    """Regenerate a tick-banded reflecting sweep from its per-segment rate tables.

    Each note segment reseeds the accumulator and replays its shared tick-rate
    table (``rate_tables[seg_tables[i]]``) via :func:`_simulate_tickreflect`,
    mirror-reflecting at the recovered bounds.
    """
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
        out[start : start + length] = _simulate_tickreflect(lo, hi, rate, int(seed), int(d), length)
    return out


def _recon_bacc_full(desc, n) -> np.ndarray:
    """Regenerate the full (8- or 16-bit) accumulator series from its descriptor."""
    if desc.get("mode") == "tickband":
        return _recon_tickband(desc, n)
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


def _recon_bacc(desc, n, sampler=None) -> np.ndarray:
    # The recurrence regenerates the accumulator, but a note-reseeded accumulator
    # also carries sequencer-driven per-note seeds, dwell/hold frames and a start
    # phase (the dwell counter, e.g. DMC pw_stepctr $1762) that the bare
    # recurrence does not reproduce. Where the recovered accumulator cell was
    # captured, regenerate from it (same "else use the captured cell" fallback as
    # the table-walk cursor); otherwise run the pure recurrence.
    if sampler is not None and desc.get("cell") is not None and desc.get("sid") is not None:
        out = sampler.at_write(desc["cell"], desc["sid"]) & 0xFF
        end = desc.get("prelude_end")
        if end:
            prelude = _recon_seq(
                {
                    "latch_frames": desc.get("prelude_frames", [0]),
                    "latch_values": desc.get("prelude_values", [0]),
                },
                n,
            )
            out = np.where(np.arange(n) < int(end), prelude & 0xFF, out)
        return out
    full = _recon_bacc_full(desc, n)
    role = desc.get("byte_role", "full")
    if role == "lo":
        return full & 0xFF
    if role == "hi":
        return (full >> 8) & 0xFF
    return full


def _recon_table(desc, n, sampler) -> np.ndarray:
    table = np.asarray(desc["table"], dtype=np.int64)
    mask = int(desc.get("mask", 0xFF))
    stride = int(desc.get("stride", 1))
    off = int(desc.get("cursor_offset", 0))
    cursor = desc.get("cursor")
    if cursor is None and sampler is not None and desc.get("cursor_addr") is not None:
        cursor = sampler.eof(desc["cursor_addr"])
    if cursor is None:
        return np.zeros(n, dtype=np.int64)
    idx = np.clip(np.asarray(cursor, dtype=np.int64) * stride + off, 0, len(table) - 1)
    out = table[idx]
    gate_addr = desc.get("gate_addr")
    if gate_addr is not None and sampler is not None:
        out = out & sampler.eof(gate_addr)
    else:
        out = out & mask
    return _apply_overrides(out, desc.get("overrides", []), sampler)


def _apply_overrides(out, overrides, sampler) -> np.ndarray:
    """Force values where each override's cell-predicate conjunction holds."""
    if not overrides or sampler is None:
        return out
    n = len(out)
    for ov in overrides:
        sel = np.ones(n, dtype=bool)
        for cell, cmask, cval in ov.get("predicate", []):
            col = sampler.eof(cell)
            # ``cmask == "in"`` marks a value-membership term (cval is the value
            # tuple); an integer ``cmask`` is the historical bit/equality test.
            if cmask == "in":
                sel &= np.isin(col, np.asarray(cval, dtype=col.dtype))
            else:
                sel &= (col & cmask) == cval
        out = np.where(sel, int(ov["force"]), out)
    return out


def _comp_part(part, n, sampler) -> np.ndarray:
    """Per-frame value of a composite part (8-bit cell or 16-bit lo/hi cell pair)."""
    if part is None:
        return np.zeros(n, dtype=np.int64)
    if "series" in part:
        return np.asarray(part["series"], dtype=np.int64)
    if sampler is None:
        return np.zeros(n, dtype=np.int64)
    if "lo" in part:
        lo = sampler.at_write(part["lo"][0], part["lo"][1])
        hi = sampler.at_write(part["hi"][0], part["hi"][1])
        return combine_lohi(lo, hi)
    return sampler.at_write(part["cell"], part["sid"])


def _recon_composite(desc, n, sampler) -> np.ndarray:
    total = _comp_part(desc.get("base"), n, sampler) + _comp_part(desc.get("mod"), n, sampler)
    total = total & int(desc.get("width_mask", 0xFF))
    role = desc.get("byte_role", "full")
    if role == "lo":
        out = total & 0xFF
    elif role == "hi":
        out = (total >> 8) & 0xFF
    else:
        out = total
    return _apply_overrides(out, desc.get("overrides", []), sampler)


def _recon_pitchwalk(desc, n, sampler) -> np.ndarray:
    """Regenerate an FC pitch-table walk: ``pitchtable[sum(index_cells)]``."""
    lotab = np.asarray(desc["lo_table"], dtype=np.int64)
    hitab = np.asarray(desc["hi_table"], dtype=np.int64)
    length = len(lotab)
    if sampler is None or length == 0:
        return np.zeros(n, dtype=np.int64)
    isum = np.zeros(n, dtype=np.int64)
    for cell in desc.get("index_cells", []):
        isum = isum + sampler.eof(cell)
    idx = np.clip(isum, 0, length - 1)
    val16 = lotab[idx] | (hitab[idx] << 8)
    out = val16 & 0xFF if desc.get("byte_role") == "lo" else (val16 >> 8) & 0xFF
    return _apply_overrides(out, desc.get("overrides", []), sampler)


def _recon_feeder(desc, n, sampler) -> np.ndarray:
    """Regenerate a cell-latched filter register: the captured feeder cell.

    The player computes the value into a RAM cell then stores it to SID; replay
    is that captured cell sampled at the register's write instant.
    """
    if sampler is None:
        return np.zeros(n, dtype=np.int64)
    return sampler.at_write(desc["cell"], desc["sid"]) & 0xFF


def _recon_xor(desc, n, sampler) -> np.ndarray:
    """Regenerate a ``cellA XOR cellB`` CTRL register from its two captured cells."""
    if sampler is None:
        return np.zeros(n, dtype=np.int64)
    a = sampler.at_write(desc["cell_a"], desc["sid"]).astype(np.int64)
    b = sampler.at_write(desc["cell_b"], desc["sid"]).astype(np.int64)
    return (a ^ b) & 0xFF


def _recon_and(desc, n, sampler) -> np.ndarray:
    """Regenerate a ``cellA AND cellB`` CTRL register from its two captured cells.

    The waveform-shadow cell AND the gate-mask cell reproduce the steady CTRL; the
    note-onset / hard-restart frames force a control byte (e.g. ``$08``/``$09``/
    ``$81``) the shadow never carries, recovered as value-forcing overrides.
    """
    if sampler is None:
        return np.zeros(n, dtype=np.int64)
    a = sampler.at_write(desc["cell_a"], desc["sid"]).astype(np.int64)
    b = sampler.at_write(desc["cell_b"], desc["sid"]).astype(np.int64)
    out = (a & b) & 0xFF
    return _apply_overrides(out, desc.get("overrides", []), sampler)


def reconstruct_register(descriptor, ticks, trace=None, sampler=None) -> np.ndarray:
    """Regenerate a register's per-frame output from its recovered descriptor.

    Executes the descriptor produced by :func:`classify_register`:
    ``CONST`` -> a constant; ``SEQ`` -> latched values held between note events;
    ``BACC`` -> the recurrence re-run (reset at the recovered note-on seeds);
    ``TABLE_WALK`` -> ``table[base + cursor*stride] & mask`` (gate-masked by a
    captured cell when one was recovered); ``COMPOSITE`` -> base + modulation +
    overrides; ``FEEDER`` -> a global filter register's captured RAM feeder cell
    sampled at the write instant; ``XOR`` -> a CTRL register's ``cellA XOR cellB``
    (base/eor gate idiom). Cell-referencing descriptors read those cells
    from ``trace`` (or a shared ``sampler``); ``XSTATE`` has no single-generator
    model yet and returns ``None``.
    """
    n = len(ticks)
    kind = descriptor.get("type")
    smp = _sampler_for(ticks, trace, sampler)
    if kind == "CONST":
        value = descriptor.get("value")
        recon = np.full(n, 0 if value is None else int(value), dtype=np.int64)
    elif kind == "SEQ":
        recon = _recon_seq(descriptor, n)
    else:
        builders = {
            "BACC": _recon_bacc,
            "TABLE_WALK": _recon_table,
            "COMPOSITE": _recon_composite,
            "PITCHWALK": _recon_pitchwalk,
            "FEEDER": _recon_feeder,
            "XOR": _recon_xor,
            "AND": _recon_and,
        }
        if kind in builders:
            recon = builders[kind](descriptor, n, smp)
        # XSTATE: not yet modelled -- best effort is the closest observable feeder
        # cell sampled at the write instant (or nothing if none was recorded).
        elif kind == "XSTATE" and smp is not None and descriptor.get("cell") is not None:
            recon = smp.at_write(descriptor["cell"], descriptor["sid"]) & 0xFF
        else:
            return None  # no model yet for this descriptor
    return _default_until_first_write(recon, descriptor, smp)


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


def round_trip(trace, kind="auto") -> dict:
    """Score every recovered register descriptor against the oracle.

    Returns ``{addr: fidelity, ..., 'overall': fidelity, 'unmodeled': [...]}``
    where ``fidelity`` is the fraction of frames whose reconstruction equals the
    oracle's per-frame register value, ``overall`` is frame-weighted across all
    written registers, and ``unmodeled`` lists registers below 1.0 (with an
    example mismatch frame range) so the remaining gap is actionable.
    """
    result = analyze(trace, kind)
    ticks = trace.tick_cycles(kind)
    n = len(ticks)
    sampler = _CellSampler(trace, ticks)
    fid: dict = {}
    matched_total = 0
    frames_total = 0
    unmodeled = []
    for addr, desc in result.items():
        if not isinstance(addr, int):
            continue
        oracle = _register_series(trace, addr, kind)[1]
        recon = reconstruct_register(desc, ticks, sampler=sampler)
        if recon is None:
            fidelity = 0.0
            mism = np.arange(n)
        else:
            eq = recon == oracle
            fidelity = float(np.mean(eq)) if n else 1.0
            mism = np.nonzero(~eq)[0]
        fid[addr] = fidelity
        matched_total += int(round(fidelity * n))
        frames_total += n
        if fidelity < 1.0:
            example = [int(mism[0]), int(mism[-1])] if len(mism) else []
            unmodeled.append(
                {
                    "addr": addr,
                    "type": desc.get("type"),
                    "fidelity": round(fidelity, 4),
                    "example_frames": example,
                }
            )
    fid["overall"] = (matched_total / frames_total) if frames_total else 1.0
    fid["unmodeled"] = sorted(unmodeled, key=lambda d: d["fidelity"])
    return fid


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
    note_tables = [
        (res["lo_table"], res["hi_table"])
        for res in out.values()
        if isinstance(res, dict) and res.get("type") == "PITCHWALK"
    ]
    out["tuning"] = recover_tuning(trace, kind, note_tables=note_tables)
    out["detune"] = voice_detune(trace, kind)
    return out


# -- global tuning + detune (the note-space / pitch layer) ----------------


def _sustained_freqs(trace, kind="auto") -> np.ndarray:
    """Held (unchanged from the previous frame), gate-on per-voice 16-bit
    frequencies -- the sustained note pitches, excluding vibrato/slide/arp
    frames which move every frame.
    """
    _, fr = trace.register_frames(0, kind)
    if len(fr) < 2:
        return np.empty(0, dtype=np.int64)
    out = []
    for voice in range(3):
        base = voice * 7
        freq = fr[:, base].astype(np.int64) | (fr[:, base + 1].astype(np.int64) << 8)
        gate = (fr[:, base + 4] & 1).astype(bool)
        held = np.zeros(len(freq), dtype=bool)
        held[1:] = freq[1:] == freq[:-1]
        sel = gate & held & (freq > 0)
        if np.any(sel):
            out.append(freq[sel])
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def _fit_a4(f_hz):
    """Grid-search a +/-50 cent window around A440 for the reference A4 that best
    aligns ``f_hz`` to a 12-TET semitone grid. Returns ``(ref_hz, residual_cents)``.

    The grid-alignment cost is periodic every semitone, so frequency alone fixes
    the reference only modulo 100 cents (which note is "A" needs the note table).
    The single +/-50 cent window yields the unique A440-referenced sub-semitone
    offset.
    """
    a4 = np.linspace(440.0 * 2.0 ** (-50.0 / 1200.0), 440.0 * 2.0 ** (50.0 / 1200.0), 1001)
    midi = 69.0 + 12.0 * np.log2(f_hz[None, :] / a4[:, None])
    cents = np.abs(midi - np.round(midi)) * 100.0
    err = np.median(cents, axis=1)
    best = int(np.argmin(err))
    return float(a4[best]), float(err[best])


def _table_freqs(note_tables, hz):
    """Chromatic frequency ladder(s) from PITCHWALK note tables.

    Each ``(lo, hi)`` uint8 pair becomes 16-bit entries -> Hz; entries are kept in
    (30, 8000) Hz and made strictly increasing (zeros/dupes dropped). Returns the
    concatenated, sorted, unique musical frequencies.
    """
    out = []
    for lo, hi in note_tables:
        lo_i = np.asarray(lo).astype(np.int64)
        hi_i = np.asarray(hi).astype(np.int64)
        tab16 = lo_i | (hi_i << 8)
        tf = tab16.astype(np.float64) * hz / float(1 << 24)
        tf = tf[(tf > 30.0) & (tf < 8000.0)]
        out.append(np.unique(tf))
    if not out:
        return np.empty(0, dtype=np.float64)
    return np.unique(np.concatenate(out))


def recover_tuning(trace, kind="auto", note_tables=None):
    """Recover a song's global tuning (offset from A440 / 12-TET).

    SID frequency maps to pitch as ``f_Hz = sidfreq * cpu_hz / 2**24``. A
    reference A4 is fit to the sustained (held, gate-on) note frequencies by
    minimising the median absolute deviation to a 12-TET semitone grid. When
    ``note_tables`` (a list of ``(lo_table, hi_table)`` uint8 arrays from PITCHWALK
    FREQ descriptors) supplies a denser chromatic ladder, the reference is fit over
    those table frequencies instead (``source == "note_table"``); the fit falls
    back to the live frequencies (``source == "live_freq"``) when a table is not
    chromatic (worse residual) or too short. Returns ``{a4_hz, cents_from_a440,
    residual_cents, temperament, n_samples, source, note_numbers, note_range}`` or
    None when there is too little sustained pitch data. This is offline-only: the
    per-song note->frequency table already bakes the tuning in for replay, so it
    costs nothing at runtime but makes the IR's note numbers absolute pitch,
    comparable across tunes.

    Frequency alone fixes tuning only modulo one semitone (which grid point is
    "A" needs the note table), so ``cents_from_a440`` is the sub-semitone offset
    in (-50, +50]: tunes that share an offset have aligning notes; those that
    differ do not.
    """
    hz = float(trace.meta.get("cpu_hz", 985248.444))
    freqs = _sustained_freqs(trace, kind)
    f_hz = freqs.astype(np.float64) * hz / float(1 << 24)
    f_hz = f_hz[(f_hz > 30.0) & (f_hz < 8000.0)]
    if len(f_hz) < 16:
        return None
    ref, residual = _fit_a4(f_hz)
    source = "live_freq"
    if note_tables:
        tab_f = _table_freqs(note_tables, hz)
        if len(tab_f) >= 16:
            t_ref, t_res = _fit_a4(tab_f)
            if t_res <= residual:
                ref, residual, source = t_ref, t_res, "note_table"
    # Absolute MIDI note numbers of the sounded notes under the winning reference.
    sounded = np.round(69.0 + 12.0 * np.log2(f_hz / ref)).astype(int)
    note_numbers = sorted({int(m) for m in sounded})
    note_range = [note_numbers[0], note_numbers[-1]] if note_numbers else None
    return {
        "a4_hz": ref,
        "cents_from_a440": float(np.log2(ref / 440.0) * 1200.0),
        "residual_cents": residual,
        "temperament": "12-TET" if residual < 5.0 else "non-TET",
        "n_samples": int(len(f_hz)),
        "source": source,
        "note_numbers": note_numbers,
        "note_range": note_range,
    }


def voice_detune(trace, kind="auto") -> dict:
    """Recover per-voice detune: when two voices sound the same note, the small
    constant frequency-space offset that makes them beat/chorus.

    Returns ``{detuned, median_cents, pairs, n_frames}``. The offset is reported
    relative to notes (cents) but is a frequency-space delta -- reconstruction
    never quantises freq to a note, so round-trip stays exact across it.
    """
    hz = float(trace.meta.get("cpu_hz", 985248.444))
    _, fr = trace.register_frames(0, kind)
    empty = {"detuned": False, "median_cents": 0.0, "pairs": {}, "n_frames": int(len(fr))}
    if len(fr) < 2:
        return empty
    vfreq = []
    vgate = []
    for voice in range(3):
        base = voice * 7
        freq = fr[:, base].astype(np.int64) | (fr[:, base + 1].astype(np.int64) << 8)
        vfreq.append(freq.astype(np.float64) * hz / float(1 << 24))
        vgate.append((fr[:, base + 4] & 1).astype(bool))
    pairs = {}
    all_offsets = []
    for i in range(3):
        for j in range(i + 1, 3):
            both = vgate[i] & vgate[j] & (vfreq[i] > 30.0) & (vfreq[j] > 30.0)
            if not np.any(both):
                continue
            cents = 1200.0 * np.log2(vfreq[i][both] / vfreq[j][both])
            same_note = cents[np.abs(cents) < 50.0]
            offset = same_note[np.abs(same_note) > 0.5]
            if len(offset):
                pairs[f"{i}-{j}"] = float(np.median(offset))
                all_offsets.extend(np.abs(offset).tolist())
    median = float(np.median(all_offsets)) if all_offsets else 0.0
    return {
        "detuned": bool(median > 1.0),
        "median_cents": median,
        "pairs": pairs,
        "n_frames": int(len(fr)),
    }
