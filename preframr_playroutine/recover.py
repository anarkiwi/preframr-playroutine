"""Recover per-register generators from a sidtrace v2 oracle.

Given the SID register oracle plus the v2 internal-state signals (per-frame RAM
write log, store-site PC of each SID write, and the static RAM image), recover
the generator that produces each SID register's per-frame value, classified in
the BACC / TABLE-WALK / SEQ / COMPOSITE taxonomy of
``/scratch/anarkiwi/cbm/re-trackers``:

- ``BACC``       bounded accumulator (per-frame add/sub with bound + saw / wrap /
                 reflect behaviour).
- ``TABLE_WALK`` a cursor cell stepping a static table, with loop-back.
- ``SEQ``        event-latched (sparse) writes from the note/pattern sequencer.
- ``COMPOSITE``  base + modulation + override: a sequencer-indexed base table,
                 an additive modulation accumulator, and value-forcing overrides.
- ``LIFT``/``WITNESS`` the Phase-7 emit-slice lifter (:mod:`preframr_playroutine.lift`)
                 and its Tier-3 dynamic-input backstop. Every inter-frame value is a
                 function of observable state (RAM + logged I/O reads), so the
                 witness memoises the exact code-derived input->output mapping --
                 retiring ``XSTATE`` as a terminal category (GENERIC_RECOVERY.md 3.5).
                 ``FEEDER`` is the raw closest-cell fallback the witness upgrades.

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
from typing import NamedTuple

import numpy as np

from . import ir
from . import lift
from .trace import RAMACCESS_DTYPE, WIN_IRQ, WIN_NMI

# Per-voice CTRL (gate) register addresses.
CTRL_ADDRS = {0: 0xD404, 1: 0xD40B, 2: 0xD412}

_U64_MAX = np.uint64(np.iinfo(np.uint64).max)

StateSequence = namedtuple("StateSequence", ["ticks", "addrs", "grid"])

# Backward-slice window (CPU cycles) for read-log dataflow narrowing: the cells a
# store's emitting code touched sit within its local instruction slice, well under
# one play call. Capped here so the window never spans a whole call's per-frame
# writes (which would readmit every changing cell).
_SLICE_CYCLES = 512


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
        # One stable argsort by address groups every cell's writes into a
        # contiguous slice (cycle order preserved within a group), replacing the
        # per-address boolean scan of the whole write log.
        order = np.argsort(wr["addr"], kind="stable")
        swr = wr[order]
        uniq, starts = np.unique(swr["addr"], return_index=True)
        bounds = np.append(starts, len(swr))
        pos = np.searchsorted(uniq, addrs)
        for j, p in enumerate(pos):
            if p < len(uniq) and uniq[p] == addrs[j]:
                sel = swr[bounds[p] : bounds[p + 1]]
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


def _simulate_recur(  # pylint: disable=too-many-branches,too-many-locals,too-many-arguments,too-many-statements
    lo,
    hi,
    seed,
    direction,
    length,
    boundary,
    step_kind,
    up=1,
    down=1,
    rate=None,
    modulus=None,
    divide=1,
    up_n=0,
    down_n=0,
    target=None,
):
    """One segment of the general ``step x boundary`` bounded-accumulator product.

    ``step_kind`` selects the per-frame stride: ``const`` (``up``), ``updown``
    (``up`` rising / ``down`` falling), or ``table`` (``rate[tick]``, holding the
    last entry past its end). ``boundary`` selects the bound behaviour: ``wrap``
    (modular), ``saw`` (reset to ``lo``), ``reflect`` (mirror the overshoot and
    flip), ``clampflip`` (saturate at the bound and flip), ``countflip`` (flip
    direction after ``up_n``/``down_n`` dwell frames, independent of the value
    bounds), or ``target`` (glide toward ``target`` and latch on arrival). The
    ``divide`` modifier applies the step only every ``divide``-th frame (half-rate
    / parity). This single kernel spans the whole product; the scalar-mode
    simulators above are its fixture-exercised subset and stay byte-for-byte.
    """
    out = np.empty(length, dtype=np.int64)
    v = int(seed)
    d = int(direction)
    m = len(rate) if rate is not None else 0
    span = hi - lo
    divide = int(divide) if divide else 1
    tgt = None if target is None else int(target)
    dc = 0  # divider counter: the step fires only when it reaches ``divide``
    fc = 0  # countflip dwell counter (applied steps in the current direction)
    tk = 0  # table cursor: one rate entry consumed per APPLIED step (not per frame)
    for i in range(length):
        out[i] = v
        if step_kind == "table":
            if m == 0:
                continue
            st = int(rate[tk]) if tk < m else int(rate[m - 1])
        elif step_kind == "updown":
            st = up if d > 0 else down
        else:
            st = up
        dc += 1
        if dc < divide:
            continue
        dc = 0
        if step_kind == "table":
            tk += 1
        if boundary == "target":
            step = st if st else 1
            if tgt is not None:
                if v < tgt:
                    nv = min(v + step, tgt)
                elif v > tgt:
                    nv = max(v - step, tgt)
                else:
                    nv = v
                v = nv
            continue
        nv = v + d * st
        if boundary == "wrap":
            mod = int(modulus) if modulus else (span + st if span > 0 else 1)
            if mod > 0:
                nv = lo + ((nv - lo) % mod)
        elif boundary == "saw":
            if nv > hi or nv < lo:
                nv = lo
        elif boundary == "reflect":
            if nv > hi:
                d = -d
                nv = hi - (nv - hi)
            elif nv < lo:
                d = -d
                nv = lo + (lo - nv)
        elif boundary == "countflip":
            fc += 1
            limit = up_n if d > 0 else down_n
            if limit and fc >= limit:
                d = -d
                fc = 0
        else:  # clampflip
            if nv > hi:
                d = -d
                nv = hi
            elif nv < lo:
                d = -d
                nv = lo
        v = nv
    return out


def _modal_int(values) -> int:
    """Most common value in a list of ints (1 if empty)."""
    if len(values) == 0:
        return 1
    vals, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    return int(vals[counts.argmax()])


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


class RecoverContext(NamedTuple):
    """Precomputed per-trace recovery state, shared by every proposer."""

    kind: str
    stateseq: object
    ram: object
    tables: list
    cursor_cols: list
    note_on: dict
    all_on: list
    n_frames: int
    sampler: object
    # dict[sid_addr] -> frozenset of RAM cells the emitting code read/wrote near a
    # store (read-log dataflow narrowing); None when no read log -> fall back to
    # every changing cell.
    reads_near: object = None
    # Shared per-voice latent state (recovered once, referenced by every
    # proposer): ``latents[voice] = {"tick", "cursors", "note_cell"}`` plus a
    # song-level ``latents["global"] = {"counter_addr", "counter_step"}``.
    latents: object = None
    # Executed-PC coverage set and the written-RAM-cell set, used by the Phase-7
    # emit-slice lifter (:mod:`preframr_playroutine.lift`) to disassemble only real
    # code and to recognise self-modifying (SMC) immediate operands.
    covered_pcs: object = None
    written_cells: object = None
    # dict[sid_addr] -> frozenset of read-PCs whose RAM/I/O read fed a store to the
    # address (the Tier-3 witness's log-based dynamic input cone); None when no read
    # log is present.
    read_pcs_near: object = None

    def latent_cursor(self, voice, addr):
        """The (voice, "cursor", addr) latent id if ``addr`` is a grouped cursor."""
        if self.latents is None:
            return None
        lv = self.latents.get(voice)
        if lv is None:
            return None
        for a, _series in lv.get("cursors", []):
            if int(a) == int(addr):
                return (int(voice), "cursor", int(addr))
        return None

    def candidates(self, sid_addr):
        """Cells the emitting code read/wrote near a store to ``sid_addr``.

        Returns a frozenset of RAM addresses when the read log narrowed the
        search, or ``None`` (no narrowing -> consider every changing cell) in the
        fallback path. An *empty* narrowed set (the emitting code read no tracked
        cell near the store -- e.g. an ``LDA #imm ; STA`` immediate/SMC writer)
        means "could not localize", NOT "no cell matters", so it also falls back to
        the global set -- otherwise a register whose feeder the window missed would
        lose its only candidate (regressing a perfect register to a bare fallback).
        """
        if self.reads_near is None or sid_addr is None:
            return None
        cand = self.reads_near.get(int(sid_addr), frozenset())
        return cand if cand else None

    def candidate_cols(self, sid_addr):
        """Column indices into ``stateseq.grid`` for the candidate cells.

        The fallback returns every column; with a read log it returns only the
        columns whose cell the emitting code actually touched near the store.
        """
        cand = self.candidates(sid_addr)
        n = self.stateseq.grid.shape[1]
        if cand is None:
            return list(range(n))
        addrs = self.stateseq.addrs
        return [j for j in range(n) if int(addrs[j]) in cand]


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


def _segment_bounds(series, resets):
    """Segment boundaries: internal discontinuities plus note-on reseeds.

    Segmentation keys on BOTH the seed-jump discontinuities and the note-on
    ``resets`` (gate edges): a legato / tie note reseeds the accumulator without a
    gate edge, so gates alone miss it, and a mid-note reseed has no gate edge at
    all. This one segmentation feeds every step x boundary hypothesis below.
    """
    n = len(series)
    cuts = set(int(x) for x in _discontinuities(series))
    cuts.update(int(x) for x in resets if 0 < int(x) < n)
    return [0] + sorted(cuts) + [n]


def _seg_state(series, bounds):
    """Per-segment (reset frame, seed, start direction, up step, |down| step)."""
    resets, seeds, dirs, ups, downs = [], [], [], [], []
    for k in range(len(bounds) - 1):
        seg = series[bounds[k] : bounds[k + 1]]
        d = np.diff(seg)
        nz = d[d != 0]
        up = _dominant_signed_step(d, True)
        down = _dominant_signed_step(d, False)
        resets.append(int(bounds[k]))
        seeds.append(int(seg[0]))
        dirs.append(1 if (len(nz) == 0 or nz[0] > 0) else -1)
        ups.append(int(up or down or 1))
        downs.append(int(down or up or 1))
    return resets, seeds, dirs, ups, downs


def _recur_const(series, bounds, min_residual: float = 0.6, min_segments: int = 3):
    """Const-step boundary vote: {saw, wrap, reflect, clampflip} at one shared step."""
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


def _recur_updown(series, bounds, min_residual: float = 0.6, min_reversing: int = 2):
    """Up/down-step clamp-and-flip: per-note signed rate, shared floor/ceiling."""
    resets, seeds, dirs, steps, downs = _seg_state(series, bounds)
    reversing = []
    for k in range(len(bounds) - 1):
        seg = series[bounds[k] : bounds[k + 1]]
        strip = _strip_holds(seg)
        d = np.diff(seg)
        if len(strip) >= 4 and _dominant_signed_step(d, True) and _dominant_signed_step(d, False):
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


def _rate_tables(series, bounds, div: int = 1):
    """Per-segment abs-stride vectors deduplicated into a shared program set.

    With ``div > 1`` the stride table is the applied-step subsequence (every
    ``div``-th diff): a ``table`` step under the ``divide`` modifier consumes one
    rate entry per applied step, so its compact program is the sub-sampled diffs.
    """
    seeds, dirs, rate_vecs = [], [], []
    varying = 0
    for k in range(len(bounds) - 1):
        seg = series[bounds[k] : bounds[k + 1]]
        d = np.diff(seg)
        nz = d[d != 0]
        rate = np.abs(d).astype(np.int64)[div - 1 :: div] if div > 1 else np.abs(d).astype(np.int64)
        seeds.append(int(seg[0]))
        dirs.append(1 if (len(nz) == 0 or nz[0] > 0) else -1)
        rate_vecs.append(rate)
        if len(np.unique(rate[rate != 0])) > 1:
            varying += 1
    tables, index, seg_tables = [], {}, []
    for rate in rate_vecs:
        key = rate.tobytes()
        if key not in index:
            index[key] = len(tables)
            tables.append(rate)
        seg_tables.append(index[key])
    return seeds, dirs, tables, seg_tables, varying


def _recur_table(series, bounds, min_segments: int = 6):
    """Tick-indexed stride (``step = rate_table[tick]``) reflecting sweep.

    The tick-banded generator (Future Composer PW) reseeds per segment and reads
    its stride from a small set of shared rate tables. The old anti-theft gates
    (a majority of segments must genuinely vary within the note; the shared table
    set must stay small) are kept as a cheap prefilter, and the shared-table cost
    is ALSO charged by ``ir.complexity`` (many distinct rate tables = high cost),
    so the arbiter, not the gate, is the final adopter.
    """
    n = len(series)
    n_seg = len(bounds) - 1
    if n_seg < min_segments:
        return None
    lo16, hi16 = int(series.min()), int(series.max())
    if hi16 <= lo16:
        return None
    seeds, dirs, tables, seg_tables, varying = _rate_tables(series, bounds)
    if varying < max(3, 0.3 * n_seg):
        return None
    if len(tables) > max(8, 0.5 * n_seg):
        return None
    desc = {
        "type": "BACC",
        "mode": "tickband",
        "step": int(tables[0][0]) if len(tables[0]) else 0,
        "lo": lo16,
        "hi": hi16,
        "segmented": True,
        "resets": [int(b) for b in bounds[:-1]],
        "seeds": seeds,
        "directions": dirs,
        "rate_tables": tables,
        "seg_tables": seg_tables,
        "n_segments": int(n_seg),
    }
    desc["residual"] = float(np.mean(_recon_tickband(desc, n) == series))
    return desc


_STEP_KINDS = ("const", "updown", "table")
_BOUNDARIES = ("wrap", "saw", "reflect", "clampflip")


def _estimate_divide(series, bounds, cap: int = 8) -> int:
    """Modal hold-period between value changes within segments (parity/half-rate).

    A ``divide`` step applies only every n-th frame, so the register dwells on each
    value for n frames. Recover n as the modal gap between successive changes; 1
    (every-frame stepping) is the default cell and needs no separate hypothesis.
    """
    gaps = []
    for k in range(len(bounds) - 1):
        seg = np.asarray(series[bounds[k] : bounds[k + 1]], dtype=np.int64)
        chg = np.nonzero(np.diff(seg) != 0)[0]
        if len(chg) >= 2:
            gaps.extend(int(g) for g in np.diff(chg))
    if not gaps:
        return 1
    period = _modal_int(gaps)
    return period if 1 < period <= cap else 1


def _estimate_countflip(series, bounds):
    """Modal (up_n, down_n) dwell counts: applied steps before each direction flip.

    ``countflip`` flips direction after a fixed dwell count, independent of the
    value bounds. Run-length the signed steps per segment and take the modal
    rising / falling run lengths.
    """
    ups, downs = [], []
    for k in range(len(bounds) - 1):
        seg = np.asarray(series[bounds[k] : bounds[k + 1]], dtype=np.int64)
        d = np.diff(seg)
        d = d[d != 0]
        if len(d) == 0:
            continue
        sign = np.sign(d)
        run, cur = 1, sign[0]
        for s in sign[1:]:
            if s == cur:
                run += 1
            else:
                (ups if cur > 0 else downs).append(run)
                run, cur = 1, s
        (ups if cur > 0 else downs).append(run)
    up_n = _modal_int(ups) if ups else 0
    down_n = _modal_int(downs) if downs else 0
    return int(up_n), int(down_n)


def _segment_targets(series, bounds):
    """Per-segment latch value: the value a glide settles to (segment's last)."""
    return [int(np.asarray(series[bounds[k] : bounds[k + 1]])[-1]) for k in range(len(bounds) - 1)]


def _recur_product(  # pylint: disable=too-many-branches
    series, bounds, min_segments: int = 3, min_residual: float = 0.95
):
    """Best cell of the full ``step x boundary`` product as one ``product`` BACC.

    Covers the axis cells the scalar/tickband modes above cannot spell (e.g.
    ``updown x wrap``, ``table x clampflip``). Every hypothesis shares the single
    segmentation; per-segment seeds/directions/steps and deduplicated rate tables
    are extracted once, then each of the 12 (step_kind, boundary) cells is
    simulated and scored, and the best-reconstructing cell is emitted.

    Cheap prefilters keep this from degenerating into a universal per-frame data
    replayer: a genuine reseeded generator has several segments and reconstructs
    almost exactly, and a ``table`` step is adopted only when its rate program is
    reused across segments and genuinely varies within a note (the former
    tickband anti-theft guards, now applied per hypothesis). The rate-table
    capture is additionally MDL-charged, so a scalar closed form still wins where
    both fit."""
    n = len(series)
    lo, hi = int(series.min()), int(series.max())
    n_seg = len(bounds) - 1
    if hi <= lo or n_seg < min_segments:
        return None
    # Guard against a fragmented series (a bounce over-segmented at every
    # discontinuity): if the segments are tiny, a shared step + per-segment seeds
    # merely replays the captured seed values rather than fitting a generator.
    if n / n_seg < 6:
        return None
    resets, seeds, dirs, ups, downs = _seg_state(series, bounds)
    _s, _d, tables, seg_tables, varying = _rate_tables(series, bounds)
    # A reused, within-note-varying rate program (few shared tables) is a genuine
    # tick-indexed stride; a distinct dense vector per segment is a replay -- allow
    # the ``table`` step only when it is the former.
    table_ok = varying >= max(3, 0.3 * n_seg) and len(tables) <= max(8, 0.5 * n_seg)
    # The step model is SHARED across segments (a single modal up/down), only the
    # reseed state (seed, direction) is per-segment. This is the anti-overfit
    # constraint: capturing a distinct step per segment would let the product
    # replay any bouncing series; a genuine reseeded generator reuses one rate.
    up = _modal_int(ups)
    down = _modal_int(downs)
    common = {
        "type": "BACC",
        "mode": "product",
        "lo": lo,
        "hi": hi,
        "segmented": True,
        "resets": resets,
        "seeds": seeds,
        "directions": dirs,
        "rate_tables": tables,
        "seg_tables": seg_tables,
        "n_segments": int(n_seg),
        "step": up,
        "down_step": down,
    }
    # ``divide`` (parity/half-rate) and the count/target boundaries are extra axes;
    # their parameters are derived from the series (never fixture magic values) and
    # tried only when the data indicates them, so the product stays cost-bounded.
    divides = [1]
    dtables, dseg_tables = None, None
    est_div = _estimate_divide(series, bounds)
    if est_div > 1:
        divides.append(est_div)
        # A ``table`` step under ``divide`` consumes one rate entry per APPLIED
        # step, so its program is the sub-sampled diffs (not the zero-interleaved
        # per-frame vector) -- recover the compact table for the divided cell.
        _s, _d, dtables, dseg_tables, _v = _rate_tables(series, bounds, est_div)
    up_n, down_n = _estimate_countflip(series, bounds)
    targets = _segment_targets(series, bounds)
    best_res, best_desc = -1.0, None
    axes = []
    for step_kind in _STEP_KINDS:
        if step_kind == "table" and not table_ok:
            continue
        for boundary in _BOUNDARIES:
            axes.append((step_kind, boundary, {}))
    # Count-dwell flip and clamp-to-target glide: const/updown steps only (a
    # tick-table stride under these boundaries has no fixture and only adds cost).
    for step_kind in ("const", "updown"):
        if up_n > 0 or down_n > 0:
            axes.append((step_kind, "countflip", {"up_n": up_n, "down_n": down_n}))
        axes.append((step_kind, "target", {"targets": targets}))
    for step_kind, boundary, extra in axes:
        # A ``table`` step under ``divide`` and its zero-interleaved ``divide=1``
        # spelling reconstruct identically; prefer the compact (larger-``divide``)
        # form so the MDL-cheaper program wins the tie (fewer captured strides).
        div_order = sorted(divides, reverse=True) if step_kind == "table" else divides
        for divide in div_order:
            desc = dict(common, step_kind=step_kind, boundary=boundary, divide=divide, **extra)
            if step_kind == "table" and divide > 1:
                desc["rate_tables"] = dtables
                desc["seg_tables"] = dseg_tables
            recon = ir._recon_product(desc, n)  # pylint: disable=protected-access
            res = float(np.mean(recon == series))
            if res > best_res:
                best_res, best_desc = res, desc
    if best_desc is None or best_res < min_residual:
        return None
    best_desc["residual"] = best_res
    return best_desc


def _segmented_recur(series, resets):
    """Unified segmented recurrence fitter: ONE segmentation
    (:func:`_segment_bounds`) feeds the whole step x boundary product, sharing the
    vote/consistency logic. Returns candidate BACC descriptors -- the const-step
    boundary vote, the up/down clamp-flip, and the tick-indexed reflect -- for the
    arbiter to score (the general product cell is added by :func:`_bacc_candidate`
    only when these scalar modes leave the series imperfect)."""
    series = np.asarray(series, dtype=np.int64).ravel()
    if len(series) < 8:
        return []
    bounds = _segment_bounds(series, resets)
    out = []
    for desc in (
        _recur_const(series, bounds),
        _recur_updown(series, bounds),
        _recur_table(series, bounds),
    ):
        if desc is not None:
            out.append(desc)
    return out


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


def _window_cells(store_cycles, acc_cycles, acc_addrs, window) -> set:
    """Cells whose read/write falls within ``window`` cycles before any store.

    Vectorized over the whole access log: for each access find the first store
    strictly after it (``searchsorted`` right), then keep the access when that
    store is within ``window`` cycles. No per-store loop.
    """
    if len(store_cycles) == 0 or len(acc_cycles) == 0:
        return set()
    pos = np.searchsorted(store_cycles, acc_cycles, side="right")
    inb = pos < len(store_cycles)
    cov = np.zeros(len(acc_cycles), dtype=bool)
    nxt = store_cycles[np.clip(pos, 0, len(store_cycles) - 1)]
    cov[inb] = (nxt[inb] - acc_cycles[inb]) <= window
    return {int(a) for a in np.unique(acc_addrs[cov])}


def _reads_near_store(trace, ticks, kind):
    """Map every stored SID address to the cells read/written just before it.

    Returns ``None`` when the trace carries no read log (the CI default), so the
    context falls back to the global changing-cells set. The window brackets the
    store's local dataflow slice: at most one play call (the median tick period),
    but capped at ``_SLICE_CYCLES`` -- a whole play call spans every per-frame
    write, so an un-capped window would readmit every changing cell and narrow
    nothing, whereas a store's backward slice (the LDA/SMC feeding it) sits within
    a few dozen instructions.
    """
    rd = trace.ram_reads(_window_kind(kind))
    if len(rd) == 0:
        return None
    # I/O reads (osc3/env3, CIA timers) join the narrowed cell set so a chip/CIA
    # input is a candidate like any read cell (GENERIC_RECOVERY.md 3.5). The None
    # gate stays keyed on the RAM read log, so a no-``--reads`` trace still falls
    # back to the global changing-cells set (unchanged Phase-4 behaviour).
    rd = np.concatenate([rd, trace.io_reads(_window_kind(kind))])
    window = _call_window(trace, ticks)
    wr = trace.ram_writes(_window_kind(kind))
    ro = np.argsort(rd["cycle"], kind="stable")
    rc, ra = rd["cycle"][ro].astype(np.int64), rd["addr"][ro]
    wo = np.argsort(wr["cycle"], kind="stable")
    wc, wa = wr["cycle"][wo].astype(np.int64), wr["addr"][wo]
    out = {}
    sw = trace.sid_writes()
    for addr in np.unique(sw["addr"]):
        sc = np.sort(sw[sw["addr"] == addr]["cycle"].astype(np.int64))
        cells = _window_cells(sc, rc, ra, window) | _window_cells(sc, wc, wa, window)
        out[int(addr)] = frozenset(cells)
    return out


def _call_window(trace, ticks) -> int:
    """One play-call span in cycles (median tick period), capped at ``_SLICE_CYCLES``."""
    if len(ticks) >= 2:
        call = int(np.median(np.diff(ticks.astype(np.int64))))
    else:
        call = int(trace.frame_cycles)
    return min(call, _SLICE_CYCLES)


def _read_pcs_near_store(trace, ticks, kind):
    """Map each stored SID address to the read-PCs that fed it (the log-based cone).

    The Tier-3 witness's dynamic input cone: every PC whose RAM/I/O read fell within
    one play call before a store to the address -- reads inside called subroutines a
    bounded static slice cannot reach (GENERIC_RECOVERY.md 3.5). Always available
    (I/O reads are logged unconditionally), so a chip read in a shared subroutine is
    still witnessable without ``--reads``.
    """
    reads = np.concatenate(
        [trace.ram_reads(_window_kind(kind)), trace.io_reads(_window_kind(kind))]
    )
    if len(reads) == 0:
        return None
    window = _call_window(trace, ticks)
    ro = np.argsort(reads["cycle"], kind="stable")
    rc, rp = reads["cycle"][ro].astype(np.int64), reads["pc"][ro]
    out = {}
    sw = trace.sid_writes()
    for addr in np.unique(sw["addr"]):
        sc = np.sort(sw[sw["addr"] == addr]["cycle"].astype(np.int64))
        out[int(addr)] = frozenset(_window_cells(sc, rc, rp, window))
    return out


def _synth_tick(on_frames, n) -> np.ndarray:
    """A per-note wavetable tick: 0 at each note-on, +1 every following frame."""
    tick = np.zeros(n, dtype=np.int64)
    bounds = [0] + [int(f) for f in sorted(on_frames) if 0 < int(f) < n] + [n]
    for k in range(len(bounds) - 1):
        start, stop = bounds[k], bounds[k + 1]
        tick[start:stop] = np.arange(stop - start, dtype=np.int64)
    return tick


def _cursor_resets_at(col, on_frames, max_lag: int = 2, min_frac: float = 0.6) -> bool:
    """Whether ``col`` resets (a downward discontinuity) at this voice's note-ons."""
    if not on_frames:
        return False
    disc = set(int(x) for x in _discontinuities(col))
    hits = sum(any((int(f) + lag) in disc for lag in range(max_lag + 1)) for f in on_frames)
    return hits / len(on_frames) >= min_frac


def _global_counter(grid, addrs, n) -> dict:
    """A cell advancing by a constant +k every frame without note resets.

    The Commando-class global phase source: pick the column whose forward diff is
    a single positive step on almost every frame (wraps allowed). Falls back to a
    synthesized frame index (``counter_addr`` None, step 1).
    """
    out = {"counter_addr": None, "counter_step": 1}
    if grid.shape[1] == 0 or n < 4:
        return out
    for j in range(grid.shape[1]):
        col = grid[:, j]
        d = np.diff(col)
        pos = d[d > 0]
        if len(pos) == 0:
            continue
        step = _modal_int(pos.tolist())
        if step <= 0:
            continue
        # nearly every frame either advances by the modal step or wraps (large drop)
        ok = (d == step) | (d < -step)
        if float(np.mean(ok)) >= 0.9:
            out = {"counter_addr": int(addrs[j]), "counter_step": int(step)}
            break
    return out


def _build_latents(stateseq, note_on, n_frames) -> dict:
    """Recover the shared per-voice latent state once (see :class:`RecoverContext`).

    For each voice: the wavetable *tick* (synthesized from note-ons, then replaced
    by a captured cell that matches it -- the captured cursor survives retrigger
    nuances), the table *cursors* (``_cursor_columns`` grouped to the voice by
    note-on reset correlation), and the ``note_cell`` holding the tick. A
    song-level global counter (:func:`_global_counter`) is stored under
    ``"global"``.
    """
    latents: dict = {}
    if stateseq is None or stateseq.grid.shape[1] == 0:
        grid = np.zeros((n_frames, 0), dtype=np.int64)
        addrs = np.zeros(0, dtype=np.int64)
    else:
        grid = stateseq.grid.astype(np.int64)
        addrs = stateseq.addrs
    cursor_cols = _cursor_columns(stateseq)
    for voice in CTRL_ADDRS:
        ons = note_on.get(voice, [])
        tick = _synth_tick(ons, n_frames)
        note_cell = None
        best_j, best_score = None, 0.99
        for j in cursor_cols:
            score = float(np.mean(grid[:, j] == tick))
            if score > best_score:
                best_score, best_j = score, j
        if best_j is not None:
            note_cell = int(addrs[best_j])
            tick = grid[:, best_j].copy()
        cursors = [
            (int(addrs[j]), grid[:, j].copy())
            for j in cursor_cols
            if _cursor_resets_at(grid[:, j], ons)
        ]
        latents[voice] = {"tick": tick, "cursors": cursors, "note_cell": note_cell}
    latents["global"] = _global_counter(grid, addrs, n_frames)
    return latents


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
    n_frames = len(stateseq.ticks)
    latents = _build_latents(stateseq, note_on, n_frames)
    sampler = _CellSampler(trace, stateseq.ticks)
    sampler.latents = latents
    cov = trace.coverage_pcs()
    covered = frozenset(int(p) for p in cov.tolist()) if len(cov) else frozenset()
    rw = trace.ram_writes(_window_kind(kind))
    written = frozenset(int(a) for a in np.unique(rw["addr"]).tolist()) if len(rw) else frozenset()
    return RecoverContext(
        kind=kind,
        stateseq=stateseq,
        ram=ram,
        tables=tables,
        cursor_cols=cursor_cols,
        note_on=note_on,
        all_on=all_on,
        n_frames=n_frames,
        sampler=sampler,
        reads_near=_reads_near_store(trace, stateseq.ticks, kind),
        latents=latents,
        covered_pcs=covered,
        written_cells=written,
        read_pcs_near=_read_pcs_near_store(trace, stateseq.ticks, kind),
    )


def _bacc_candidate(
    trace, sid_addr, series, voice, reg_off, ctx
):  # pylint: disable=too-many-branches
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
    # spuriously but reconstructs poorly). The segmented candidates share one
    # segmentation (:func:`_segment_bounds`) and cover the step x boundary product;
    # the up/down clamp-flip is tried before the const vote and tick-indexed
    # reflect (the 16-bit domain), while the 8-bit series only offers the const
    # vote -- preserving the pre-unification candidate set so a richer model never
    # steals a register that the scalar/bitwise paths already own.
    cands = []
    if combined is not None:
        cb = np.asarray(combined, dtype=np.int64)
        cands.append((fit_bacc(cb), cb, 16, role))
        if len(cb) >= 8:
            b16 = _segment_bounds(cb, resets)
            cands.append((_recur_updown(cb, b16), cb, 16, role))
            cands.append((_recur_const(cb, b16), cb, 16, role))
            cands.append((_recur_table(cb, b16), cb, 16, role))
    cands.append((fit_bacc(series), series, 8, "full"))
    if len(series) >= 8:
        cands.append((_recur_const(series, _segment_bounds(series, resets)), series, 8, "full"))
    best = None
    best_fid = -1.0
    for fit, src, width, crole in cands:
        if fit is None:
            continue
        finished = _bacc_finish(fit, src, width, crole)
        fid = _candidate_fidelity(finished, series, crole)
        if fid > best_fid:
            best, best_fid = finished, fid
    # General step x boundary product cell: only when the scalar/tickband modes
    # leave the series imperfect (bounds the per-frame simulate cost and keeps the
    # perfect set on its unchanged scalar descriptors). It only replaces ``best``
    # on strictly higher fidelity, so it never regresses a register.
    if best_fid < 0.999:
        for src, width, crole in ((combined, 16, role), (series, 8, "full")):
            if src is None:
                continue
            arr = np.asarray(src, dtype=np.int64)
            if len(arr) < 8:
                continue
            prod = _recur_product(arr, _segment_bounds(arr, resets))
            if prod is None:
                continue
            finished = _bacc_finish(prod, arr, width, crole)
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


# defMON global-filter cutoff micro-routine. A self-modifying 16-bit accumulator
# is stepped (ADC/SBC operand cells) and emitted as clamp(hi + imm + carry) * scale
# to $D415-$D418. Operand cells sit at fixed offsets before the store PC; the store
# and the surrounding opcodes form a signature specific enough that no other player
# matches it, so recovery is gated on both the signature and reconstruction.
_CUTOFF_OFFS = {
    "lo": -31,
    "op_lo": -29,
    "slo": -28,
    "hi": -23,
    "op_hi": -22,
    "shi": -21,
    "imm": -11,
    "base": -7,
    "scale": -1,
}
# store-PC-relative byte -> allowed opcode bytes (routine signature).
_CUTOFF_SIG = {
    -32: (0xA9,),  # lda #lo
    -30: (0x18,),  # clc
    -29: (0x69, 0xE9),  # adc/sbc #steplo
    -27: (0x8D,),  # sta lo
    -24: (0xA9,),  # lda #hi
    -22: (0x69, 0xE9),  # adc/sbc #stephi
    -20: (0x10,),  # bpl (hi bound)
    -15: (0x8D,),  # sta hi
    -12: (0x69,),  # adc #imm
    -10: (0x30,),  # bmi (clamp)
    -8: (0xC9,),  # cmp #base
    -1: (0xEA, 0x0A),  # nop / asl (SID-model scale)
    0: (0x8D,),  # sta $d4xx
}


def _defmon_cutoff(trace, sid_addr, series, store_pcs, ctx):
    """Recover the defMON filter-cutoff SMC micro-routine, or ``None``.

    Verifies a store PC carries the routine's opcode signature (so it never fires
    on another player -- a legitimate precondition, kept inside the proposer) and
    reads the operand-cell addresses from their fixed offsets. Returns the
    ``CUTOFF`` descriptor whenever the signature matches; the arbiter scores it
    against the other proposals (no per-proposer fidelity gate).
    """
    del trace
    if ctx.sampler is None or ctx.ram is None or not store_pcs:
        return None
    img = ctx.ram
    reg_byte = int(sid_addr) & 0xFF
    for pc in store_pcs:
        p = int(pc)
        if not all(0 <= p + off < len(img) for off in (-32, 2)):
            continue
        if img[p + 1] != reg_byte or img[p + 2] != 0xD4:
            continue
        if not all(img[p + off] in allowed for off, allowed in _CUTOFF_SIG.items()):
            continue
        cells = {name: p + off for name, off in _CUTOFF_OFFS.items()}
        desc = {
            "type": "CUTOFF",
            "addr": int(sid_addr),
            "sid": int(sid_addr),
            "cells": cells,
            "base": int(img[cells["base"]]),
            "imm": int(img[cells["imm"]]),
            "scale": 2 if img[cells["scale"]] == 0x0A else 1,
        }
        recon = ir.evaluate(ir.to_ir(desc), ctx.n_frames, ctx.sampler)
        fid = float(np.mean(recon == np.asarray(series, dtype=np.int64)))
        desc["residual"] = round(fid, 4)
        return desc
    return None


# -- MDL arbiter tuning (the single documented place) -------------------------
#
# Selection is fidelity-dominant lexicographic: (fidelity, -complexity,
# -priority). This equals ``argmax(fidelity - LAMBDA * complexity)`` in the
# LAMBDA -> 0+ limit (LAMBDA below the 1/n_frames fidelity granularity), so a
# higher-fidelity tree always wins -- keeping the perfect set at 1.0 -- and
# complexity only breaks exact-fidelity ties, then the documented proposer
# priority order. ``LAMBDA``/``ir.CAPTURED_W`` are the reported-score constants;
# the calibration test (tests/test_hvsc.py) pins that this reproduces the old
# cascade's winners on the perfect set.
_LAMBDA = 1e-3

# Proposer priority (earlier = preferred on an exact-fidelity tie), mirroring the
# old cascade order: structured generators first, then bitwise folds, then the
# SEQ latch list / raw-cell fallbacks last (a fitting generator is preferred over
# replaying its output). Also the descriptor-type -> priority rank.
_PRIORITY = (
    "CONST",
    "CUTOFF",
    "BACC",
    "PITCHWALK",
    "COMPOSITE",
    "TABLE_WALK",
    "XOR",
    "AND",
    "OR",
    "ADD",
    "SUB",
    "SELECT",
    "PROGRAM",
    "SEQ",
    "WITNESS",
    "FEEDER",
)


def _type_priority(desc_type) -> int:
    try:
        return _PRIORITY.index(desc_type)
    except ValueError:
        return len(_PRIORITY)


# -- stateless proposers (each returns >= 0 candidate descriptors) ------------


def _propose_cutoff(trace, sid_addr, series, pcs, ctx):
    """CUTOFF proposer (only $D415-$D418; the opcode-signature gate stays inside)."""
    if not 0xD415 <= int(sid_addr) <= 0xD418:
        return []
    cutoff = _defmon_cutoff(trace, sid_addr, series, pcs, ctx)
    return [cutoff] if cutoff is not None else []


def _propose_seq(series, sid_addr, reg_off, on_frames, wr, ctx):
    """SEQ proposer: envelope (AD/SR) always, else note-gated / sparse registers."""
    seq_frac, n_changes = _seq_correlation(series, on_frames, ctx.n_frames)
    if reg_off not in (_AD, _SR):
        per_frame = len(wr) >= 0.5 * max(1, ctx.n_frames)
        note_gated = n_changes <= 1 or (
            seq_frac >= 0.85 and n_changes <= 2.5 * max(1, len(on_frames))
        )
        sparse = n_changes <= max(8, 0.15 * ctx.n_frames)
        if not (note_gated or sparse or not per_frame):
            return []
    desc = {
        "type": "SEQ",
        "addr": int(sid_addr),
        "seq_frac": round(seq_frac, 4),
        "n_changes": int(n_changes),
    }
    return [_finish_seq(desc, series)]


def _phase_source(ctx):
    """``(counter_series, index_id)`` for the song's global frame counter, or None.

    The global counter latent (:func:`_global_counter`) is a cell advancing +k
    every frame with NO note resets; ``index_id`` is its address (or ``"frame"``
    for the synthesized frame index) -- the ``recur`` phase source for a
    Commando-class reflected triangle.
    """
    g = ctx.latents.get("global") if ctx.latents else None
    if not g:
        return None
    addr = g.get("counter_addr")
    if addr is None:
        return np.arange(ctx.n_frames, dtype=np.int64), "frame"
    if ctx.sampler is None:
        return None
    return ctx.sampler.eof(int(addr)).astype(np.int64), int(addr)


def _fit_phase(series, counter, index_id, max_step: int = 8):
    """Fit ``value = fold(seed + step*counter)`` phase-locked to a global counter.

    Searches boundary x step with an analytically-solved phase ``seed`` (both
    reflect branches), scoring against the exact global-counter series. Returns a
    ``phase``-mode BACC descriptor (index = the counter) or ``None``. The generator
    is the fold; the counter is a derivable global index (cheap in MDL), never
    replayed per-frame modulation.
    """
    s = np.asarray(series, dtype=np.int64).ravel()
    c = np.asarray(counter, dtype=np.int64).ravel()
    n = len(s)
    if n < 8 or len(c) != n:
        return None
    lo, hi = int(s.min()), int(s.max())
    span = hi - lo
    if span < 2 or len(np.unique(s)) < 3:
        return None
    best_res, best = -1.0, None
    for boundary in ("reflect", "saw", "wrap"):
        period = 2 * span if boundary == "reflect" else (span + 1)
        base = int(s[0] - lo)
        branches = [base] + ([period - base] if boundary == "reflect" else [])
        for step in range(1, max_step + 1):
            for br in branches:
                seed = int((br - step * int(c[0])) % period)
                recon = ir.phase_fold(lo, hi, step, seed, boundary, c)
                res = float(np.mean(recon == s))
                if res > best_res:
                    best_res = res
                    best = {
                        "type": "BACC",
                        "mode": "phase",
                        "lo": lo,
                        "hi": hi,
                        "step": int(step),
                        "seed": int(seed),
                        "seeds": [int(seed)],
                        "resets": [0],
                        "boundary": boundary,
                        "index": index_id,
                        "residual": res,
                    }
    if best is None or best_res < 0.95:
        return None
    return best


def _propose_recurrence(trace, sid_addr, series, voice, reg_off, ctx):
    """BACC proposer: the bare recurrence, plus a captured-cell (+prelude) variant."""
    out = []
    src = _phase_source(ctx)
    if src is not None:
        phase = _fit_phase(np.asarray(series, dtype=np.int64), src[0], src[1])
        if phase is not None:
            phase["addr"] = int(sid_addr)
            phase["width"] = 8
            phase["byte_role"] = "full"
            out.append(phase)
    bacc = _bacc_candidate(trace, sid_addr, series, voice, reg_off, ctx)
    if bacc is None:
        return out
    out.append(dict(bacc))
    cell, _recon, frac = _best_feeder_at_write(series, sid_addr, ctx)
    if cell is not None and frac >= 0.5:
        trial = dict(bacc)
        trial["cell"] = int(cell)
        trial["sid"] = int(sid_addr)
        trial["cell_frac"] = round(float(frac), 4)
        trial["addr"] = int(sid_addr)
        if reg_off not in (_FREQ_LO, _FREQ_HI):
            attach_prelude(trial, series, ctx, [trial.get("cell")])
        out.append(trial)
    return out


def _propose_pitchwalk(trace, sid_addr, reg_off, ctx):
    pw = _pitch_walk(trace, sid_addr, reg_off, ctx)
    return [pw] if pw is not None else []


def _propose_composite(trace, sid_addr, series, ctx, reg_off):
    comp = _composite(trace, sid_addr, series, ctx, reg_off)
    return [comp] if comp is not None else []


def _propose_table_walk(series, wr, ctx, sid_addr):
    """TABLE_WALK proposer (per-frame registers): read-log then image scan."""
    if len(wr) < 0.5 * max(1, ctx.n_frames):
        return []
    tw = _table_walk_search(series, ctx)
    if tw is None and not ctx.tables:
        tw = _table_walk_scan(series, ctx)
    if tw is None:
        return []
    # The masked search scores under its gate mask, so it can pick a mask that
    # clears a bit the register in fact always sets (harmless to the masked score,
    # fatal to the reconstruction). Emit an extra candidate that keeps every
    # constant-1 series bit in the mask; the arbiter's evaluated fidelity chooses,
    # so a genuine gated table (bit truly toggles -> no constant-1 bit -> only the
    # scanned mask) is untouched while a dropped constant bit is repaired.
    s = np.asarray(series, dtype=np.int64) & 0xFF
    const_one = int(np.bitwise_and.reduce(s)) if len(s) else 0
    base_mask = int(tw.get("mask", 0xFF))
    masks = [base_mask]
    if base_mask | const_one != base_mask:
        masks.append(base_mask | const_one)
    out = []
    for mask in masks:
        cand = dict(tw)
        cand["mask"] = int(mask)
        cand["addr"] = int(sid_addr)
        cand.pop("gate_addr", None)
        cand["overrides"] = []
        _annotate_cursor_latent(cand, ctx, sid_addr)
        _recover_gate(cand, series, ctx, sid_addr)
        recover_overrides(cand, series, ctx, sid_addr=sid_addr)
        out.append(cand)
    return out


def _pow2_mask(span: int) -> int:
    """Smallest ``2**k - 1`` mask able to index a ``span``-length table."""
    m = 1
    while m < span:
        m <<= 1
    return m - 1


def _find_table_in_image(ram, tab, gate, present=None):
    """First image offset where a contiguous window equals ``tab`` (under ``gate``).

    Grounds a masked-cursor lookup in a REAL RAM table rather than a table
    fabricated from the data -- the anti-overfit gate for :func:`_masked_table_lookup`.
    ``present`` (per-index bool) marks the table slots the data actually determined;
    unobserved slots are wildcards in the match.
    """
    span = len(tab)
    if ram is None or span == 0 or span > len(ram):
        return None
    r = ram.astype(np.int64) & int(gate)
    t = np.asarray(tab, dtype=np.int64) & int(gate)
    windows = np.lib.stride_tricks.sliding_window_view(r, span)
    eq = windows == t[None, :]
    if present is not None:
        eq = eq | ~np.asarray(present, dtype=bool)[None, :]
    pos = np.nonzero(np.all(eq, axis=1))[0]
    return int(pos[0]) if len(pos) else None


def _masked_table_lookup(series, ctx, sid_addr, masks=(3, 7, 15, 31), max_cols: int = 96):
    """Masked captured-cell table lookup: ``register = table[(cell & mask)]``.

    The AMIB bass idiom ``$F7[cell & 7]``: a captured cell (the LFSR / melody
    state -- CAPTURED DATA, replayed as an ordinary ``cell@eof`` input, NEVER fit
    as a recurrence) masked to a small power-of-two range indexes a real RAM table.
    Recovers the table from the (masked-cell -> register) mapping, then REQUIRES it
    to exist in the RAM image (grounding the lookup) before adopting; the lookup is
    the generator, the masked cell stays captured. Returns ``[TABLE_WALK]`` or ``[]``.
    """
    ss = ctx.stateseq
    ram = ctx.ram
    if ss.grid.shape[1] == 0 or ctx.sampler is None or ram is None:
        return []
    s = np.asarray(series, dtype=np.int64).ravel() & 0xFF
    n = len(s)
    if n < 8 or len(np.unique(s)) < 3:
        return []
    cols = list(ctx.candidate_cols(sid_addr))[:max_cols]
    grid = ss.grid.astype(np.int64)
    best, best_res = None, 0.98
    for j in cols:
        col = grid[:, j]
        if int(col.max()) < 4 or len(np.unique(col)) < 3:
            continue
        for mask in masks:
            # The mask must genuinely fold the cell (its range overflows the table);
            # a cell already within ``[0, mask]`` is a plain cursor handled by
            # ``_propose_table_walk`` -- masking it would only steal a legit walk.
            if int(col.max()) <= mask:
                continue
            mv = col & mask
            if len(np.unique(mv)) < 3:
                continue
            for gate in (0xFF, 0xFE):
                res, tab, present = _mask_table_from_data(mv, s & gate, mask + 1, n)
                if res <= best_res:
                    continue
                base = _find_table_in_image(ram, tab, gate, present)
                if base is None:
                    continue
                img = ram[base : base + mask + 1].astype(np.int64)
                fit = float(np.mean((img[np.clip(mv, 0, mask)] & gate) == (s & gate)))
                if fit > best_res:
                    best_res = fit
                    best = (int(ss.addrs[j]), int(mask), int(gate), int(base))
    if best is None:
        return []
    addr, mask, gate, base = best
    desc = {
        "type": "TABLE_WALK",
        "base": base,
        "stride": 1,
        "length": int(mask + 1),
        "loop": 0,
        "table": ram[base : base + mask + 1].copy(),
        "mask": int(gate),
        "cursor_addr": addr,
        "cursor_offset": 0,
        "index_mask": mask,
        "addr": int(sid_addr),
        "overrides": [],
        "residual": float(best_res),
    }
    _recover_gate(desc, series, ctx, sid_addr)
    recover_overrides(desc, series, ctx, sid_addr=sid_addr)
    return [desc]


def _mask_table_from_data(mv, sg, span, n):
    """(coverage, table, present) recovering ``table[m]`` = modal value per index ``m``.

    Coverage is the fraction of frames the deterministic ``(cell & mask) -> value``
    mapping explains; a genuine lookup is near-deterministic (coverage ~ 1), a
    coincidence is not -- so the caller's high floor rejects fabricated tables.
    Vectorized via one ``(span, 256)`` joint histogram (no per-index python loop):
    ``present`` marks the indices the data actually observed.
    """
    hist = np.bincount(mv * 256 + sg, minlength=span * 256).reshape(span, 256)
    tab = hist.argmax(axis=1).astype(np.int64)
    matched = int(hist.max(axis=1).sum())
    present = hist.sum(axis=1) > 0
    return matched / max(1, n), tab, present


def _annotate_cursor_latent(desc, ctx, sid_addr):
    """Tag a table-walk cursor with its shared per-voice latent id, when grouped.

    When the recovered ``cursor_addr`` is one of the voice's latent cursors, every
    register walked by that cursor reconstructs from the SAME latent (the JCH/FC
    recovery rule: recover the cursor once). The latent cursor series equals the
    cell's end-of-frame series, so the reconstruction is byte-identical -- only the
    index *identity* is now shared, not independently re-derived per register.
    """
    voice, _off = _voice_of(sid_addr)
    cursor = desc.get("cursor_addr")
    if voice is None or cursor is None:
        return
    latent = ctx.latent_cursor(voice, cursor)
    if latent is not None:
        desc["index_latent"] = latent


def _propose_binop(sid_addr, series, ctx):
    """Unified fold proposer: ``{or, and, xor, add, sub} x {cell, const}``.

    Replaces the three copy-paste pair searches. ONE entropy prefilter (cells
    with <= 24 distinct values -- the union of the former XOR/AND/OR caps),
    sampled at the SID-write instant and subsampled for the scan then verified on
    all frames. Generalizes ``_or_pair``'s per-bit-optimal constant to every
    bitwise op (``and``/``xor`` choose each constant bit independently) and
    searches a small residual-histogram constant set for ``add``/``sub``. Routing
    hints (matching the former ``_propose_pairs``): XOR/AND are CTRL idioms and
    are offered only for CTRL; OR/ADD/SUB apply to any per-frame register. The
    arbiter, not the proposer, adopts."""
    ss = ctx.stateseq
    if ss.grid.shape[1] == 0 or ctx.sampler is None:
        return []
    s = np.asarray(series, dtype=np.int64) & 0xFF
    n = len(s)
    grid = ss.grid
    addrs = ss.addrs
    cols = [
        j for j in ctx.candidate_cols(sid_addr) if len(np.unique(grid[:, j])) <= _BINOP_MAX_DISTINCT
    ]
    if not cols:
        return []
    mat_full = np.stack(
        [ctx.sampler.at_write(int(addrs[j]), sid_addr).astype(np.int64) & 0xFF for j in cols],
        axis=1,
    )
    sample = np.unique(np.linspace(0, n - 1, min(n, 512)).astype(np.int64))
    mat = mat_full[sample]
    ss_s = s[sample]
    bits = (1 << np.arange(8)).astype(np.int64)
    sb = (ss_s[:, None] & bits) > 0

    _voice, reg_off = _voice_of(sid_addr)
    ops = ["or", "add", "sub"] + (["xor", "and"] if reg_off == _CTRL else [])
    out = []
    for fn in ops:
        cand = _binop_best(fn, cols, addrs, mat, ss_s, sb, s, sid_addr, ctx)
        if cand is not None:
            out.append(cand)
    return out


def _propose_feeder(series, sid_addr, ctx):
    """Raw-cell fallback: the closest captured cell replayed at the write instant.

    Always a FEEDER (the closest captured cell); the Tier-3 dynamic witness -- not a
    terminal XSTATE category -- is the backstop that upgrades this raw replay to the
    exact code-derived input mapping (GENERIC_RECOVERY.md 3.5). A bare fallback with
    no captured cell reconstructs to None (fidelity 0), so any fitting proposal or
    the witness wins the arbiter.
    """
    desc = {"type": "FEEDER", "addr": int(sid_addr)}
    cell, _recon, frac = _best_feeder_at_write(series, sid_addr, ctx)
    if cell is not None and frac > 0.0:
        desc["cell"] = int(cell)
        desc["sid"] = int(sid_addr)
        desc["cell_frac"] = round(float(frac), 4)
    return [desc]


def _register_proposals(trace, sid_addr, series, wr, pcs, voice, reg_off, on_frames, ctx):
    """All candidate descriptors for a register, routed by the class hint table.

    The routing priors (XOR/AND only-CTRL, PITCHWALK only-FREQ, CUTOFF only
    $D415-18) are proposer *hints* here -- which proposers run for a register
    class -- not adoption rules; the arbiter scores whatever they emit.
    """
    cands = _propose_cutoff(trace, sid_addr, series, pcs, ctx)
    if reg_off in (_AD, _SR):
        cands += _propose_seq(series, sid_addr, reg_off, on_frames, wr, ctx)
    elif reg_off in (_FREQ_LO, _FREQ_HI):
        cands += _propose_pitchwalk(trace, sid_addr, reg_off, ctx)
        cands += _propose_composite(trace, sid_addr, series, ctx, reg_off)
        cands += _propose_recurrence(trace, sid_addr, series, voice, reg_off, ctx)
        cands += _masked_table_lookup(series, ctx, sid_addr)
        cands += _propose_feeder(series, sid_addr, ctx)
    else:
        cands += _propose_recurrence(trace, sid_addr, series, voice, reg_off, ctx)
        cands += _propose_table_walk(series, wr, ctx, sid_addr)
        cands += _masked_table_lookup(series, ctx, sid_addr)
        cands += _propose_composite(trace, sid_addr, series, ctx, reg_off)
        cands += _propose_binop(sid_addr, series, ctx)
        cands += _propose_seq(series, sid_addr, reg_off, on_frames, wr, ctx)
        cands += _propose_feeder(series, sid_addr, ctx)
    return cands


def _arbitrate(cands, series, sid_addr, ctx, base):
    """Score every candidate and pick the MDL winner, merged onto ``base``.

    ``score = fidelity - LAMBDA * complexity``; selection is fidelity-dominant
    lexicographic ``(fidelity, -complexity, -priority)`` (see ``_LAMBDA``). Each
    winner carries ``score``/``complexity``/``captured_frames`` for the report.
    """
    s = np.asarray(series, dtype=np.int64)
    scored = [_score_candidate(cand, s, sid_addr, ctx) for cand in cands]
    scored = [e for e in scored if e is not None]
    scored += [
        _score_candidate(prog, s, sid_addr, ctx)
        for prog in propose_program(scored, s, sid_addr, ctx)
    ]
    scored = [e for e in scored if e is not None]
    scored += [
        _score_candidate(sel, s, sid_addr, ctx) for sel in propose_select(scored, s, sid_addr, ctx)
    ]
    scored = [e for e in scored if e is not None]
    best = None
    best_key = None
    for cand, _tree, _recon, key in scored:
        if best_key is None or key > best_key:
            best, best_key = cand, key
    best = _maybe_lift(best, best_key, series, sid_addr, ctx, base)
    result = dict(base)
    if best is not None:
        result.update(best)
    return result


def _maybe_lift(best, best_key, series, sid_addr, ctx, base):
    """Run the Phase-7 lifter LAST, only for a still-imperfect register.

    The emit-slice lifter / dynamic witness is the most general and most expensive
    proposer, so it runs only when every value-stream proposer left the register
    below 1.0. It adopts ONLY on a STRICT fidelity improvement over the incumbent:
    this keeps the fidelity ratchet safe (a lift never displaces an equal-fidelity
    winner) and turns a previously-imperfect register perfect where the code lifts.
    """
    best_fid = best_key[0] if best_key else 0.0
    if best_fid >= 1.0 - 1e-9:
        return best
    pcs = base.get("store_pcs", [])
    s = np.asarray(series, dtype=np.int64)
    for cand in lift.propose_lift(series, sid_addr, pcs, ctx):
        scored = _score_candidate(cand, s, sid_addr, ctx)
        if scored is None:
            continue
        cand, _tree, _recon, key = scored
        # A Tier-2 lift adopts on any strict fidelity gain (a recovered generator).
        # The Tier-3 witness adopts when EXACT, or when the incumbent is only the
        # raw-cell FEEDER fallback -- there the witness (the code-derived input
        # mapping) is the totality descriptor that RETIRES the ex-XSTATE register,
        # so it takes any strict improvement. Against a structured winner it still
        # needs to be exact, so it never displaces a genuine generator on MDL.
        raw_incumbent = best is None or best.get("type") == "FEEDER"
        exact = key[0] >= 1.0 - 1e-9
        gain = key[0] > best_fid + 1e-9
        adopt = gain and (cand.get("type") != "WITNESS" or exact or raw_incumbent)
        if adopt and (best_key is None or key > best_key):
            best, best_key, best_fid = cand, key, key[0]
    return best


def _score_candidate(cand, s, sid_addr, ctx):
    """Score one candidate descriptor: ``(cand, tree, recon, sort_key)`` or ``None``.

    Annotates the descriptor with ``score``/``complexity``/``captured_frames`` and
    returns its reconstruction so the arbiter (and :func:`propose_select`) can
    reuse the fidelity/correct-frame sets without re-evaluating.
    """
    if cand is None:
        return None
    n = ctx.n_frames
    cand = dict(cand)
    cand["addr"] = int(sid_addr)
    tree = ir.to_ir(cand)
    recon = ir.evaluate(tree, n, ctx.sampler)
    fid = 1.0 if not n else (0.0 if recon is None else float(np.mean(recon == s)))
    cx, cap = ir.cost_captured(tree, ctx.sampler, n)
    cand["complexity"] = round(cx, 4)
    cand["captured_frames"] = int(cap)
    cand["score"] = round(fid - _LAMBDA * cx, 6)
    key = (fid, -cx, -_type_priority(cand.get("type", "")))
    return cand, tree, recon, key


def classify_register(trace, sid_addr, kind="auto", stateseq=None, ram=None, ctx=None) -> dict:
    """Classify the generator producing a SID register's per-frame value.

    Runs the class-routed stateless proposers (:func:`_register_proposals`) and
    picks the MDL winner with the global arbiter (:func:`_arbitrate`): the tree
    that maximises ``fidelity - LAMBDA * complexity``, ties broken by lower
    complexity then documented proposer priority. Returns a descriptor dict with
    ``type`` in {'CONST','CUTOFF','SEQ','BACC','PITCHWALK','COMPOSITE',
    'TABLE_WALK','XOR','AND','OR','SELECT','PROGRAM','LIFT','WITNESS','FEEDER'},
    its recovered parameters,
    ``store_pcs``, and the arbiter's ``score``/``complexity``/``captured_frames``.

    ``ctx`` is a precomputed :class:`RecoverContext` (built once by
    :func:`analyze`); ``stateseq``/``ram`` seed a context when ``ctx`` is absent.
    """
    if ctx is None:
        ctx = _build_context(trace, kind, stateseq, ram)
    ticks, series, wr = _register_series(trace, sid_addr, ctx.kind)
    pcs = sorted({int(p) for p in wr["aux"]}) if len(wr) else []
    result = {"addr": int(sid_addr), "store_pcs": pcs, "n_writes": int(len(wr))}
    voice, reg_off = _voice_of(sid_addr)

    if len(ticks) == 0 or len(wr) == 0 or len(np.unique(wr["value"])) == 1:
        empty = len(ticks) == 0 or len(wr) == 0
        result.update(type="CONST", value=None if empty else int(wr["value"][0]))
        return result

    # End-of-frame feeder hint (informational cell_addr/cell_frac), kept for
    # callers that inspect the closest captured cell regardless of the winner.
    feeder, ffrac = _best_feeder(series, ctx.stateseq)
    if feeder is not None and ffrac >= 0.5:
        result["cell_addr"] = feeder
        result["cell_frac"] = round(ffrac, 4)

    on_frames = ctx.note_on.get(voice, []) if voice is not None else ctx.all_on
    cands = _register_proposals(trace, sid_addr, series, wr, pcs, voice, reg_off, on_frames, ctx)
    if not cands:
        cands = _propose_feeder(series, sid_addr, ctx)
    return _arbitrate(cands, series, sid_addr, ctx, result)


def attach_prelude(node, series, ctx, cells, max_latches: int = 6):
    """Capture the pre-modulation held-seed prelude of a cell-fed register.

    Before its source cells' first RAM write the register holds its note-on seed
    (e.g. an instrument PW seed loaded once and held until modulation starts, or
    ``$D418`` holding ``mode|$0F`` until the volume cell is stored), but the
    captured cells still read their power-on default, so the cell replay is wrong
    on those leading frames. Record that bounded hold as SEQ latches; the cell(s)
    drive every frame they are live. The prelude ends at the latest first-live
    frame over all source cells (``first_live_frame``, not ``cell_first_frame``: a
    cell written later in its first frame than the register's store still reads its
    power-on default at the write instant). Recorded only when the hold is a few
    latches (a genuine seed prelude, not arbitrary modulation), so it never
    displaces the captured cell where the cell is live.
    """
    sampler = ctx.sampler
    if sampler is None:
        return
    cells = [c for c in cells if c is not None]
    if not cells:
        return
    sid = node.get("sid", node.get("addr"))
    ends = [sampler.first_live_frame(int(c), sid) for c in cells]
    end = max(ends) if ends else 0
    if not 0 < end < len(series):
        return
    # Only the frames the register is *written* but the cell is not yet live need
    # the held-seed prelude; frames before the register's own first write are
    # already carried by ``_default_until_first_write``, and an all-zero held seed
    # equals the cell's power-on default there too. Recording such an inert prelude
    # only inflates the descriptor's MDL cost (making an identically-reconstructing
    # raw feeder win the arbiter tie), so skip it -- it never changes the recon.
    written = sampler.written_mask(int(sid))
    first_written = int(np.argmax(written)) if written.any() else 0
    if end <= first_written:
        return
    frames, values = _seq_latches(np.asarray(series, dtype=np.int64)[:end])
    if len(frames) > max_latches or not any(values):
        return
    node["prelude_end"] = int(end)
    node["prelude_frames"] = frames
    node["prelude_values"] = values


def _finish_seq(result, series):
    """Attach the event latch points an SEQ register reconstructs from."""
    frames, values = _seq_latches(series)
    result["latch_frames"] = frames
    result["latch_values"] = values
    return result


def _recover_gate(tw, series, ctx, sid_addr=None):
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
    for j in ctx.candidate_cols(sid_addr):
        col = grid[:, j]
        if int(col.max()) < 0xF0 or len(np.unique(col)) > 4:
            continue
        fid = float(np.mean(series == (tv & col)))
        if fid > best_fid + 0.05:
            best_fid, best_addr = fid, int(ctx.stateseq.addrs[j])
    if best_addr is not None:
        tw["gate_addr"] = best_addr
        tw["residual"] = best_fid


def recover_overrides(
    node, series, ctx, max_overrides: int = 3, mask=None, explained=None, sid_addr=None
):
    """Greedy strictly-improving value-forcing override recovery over a base recon.

    A generator's table/pair/composite base can leave a handful of frames
    unexplained -- a note-onset reset, a hard-restart, a ``$08``/``$81`` control
    byte the wavetable never emits -- each forced where a recovered cell predicate
    holds (see :func:`_find_override`). Overrides are taken greedily and kept only
    when they strictly raise reproduction, so raising the cap never regresses a
    register that needed fewer.

    ``mask`` (cf. the AND pair's pre-first-write zero, ``_and_recon_masked``), when
    given, zeroes the reconstruction where the register was not yet written before
    scoring. ``explained`` overrides the default byte-wise ``series == work`` test
    of which frames the base already reproduces (the composite path passes its
    16-bit ``base+mod == target`` mask). Records ``node["overrides"]`` and
    ``node["residual"]`` when any override survives.
    """
    sampler = ctx.sampler
    if sampler is None:
        return
    series = np.asarray(series, dtype=np.int64)
    cur = ir.evaluate(ir.to_ir(node), ctx.n_frames, sampler)
    work = cur if mask is None else np.where(mask, cur, 0)
    best_fid = float(np.mean(work == series))
    overrides = []
    for _ in range(max_overrides):
        hit = (series == work) if explained is None else explained
        forced = np.where(hit, -1, series).astype(np.int64)
        ov = _override_descriptor(forced, ctx, sid_addr)
        if ov is None or ov in overrides:
            break
        cand = _apply_overrides(cur, [ov], sampler)
        cand_work = cand if mask is None else np.where(mask, cand, 0)
        cand_fid = float(np.mean(cand_work == series))
        if cand_fid <= best_fid:
            break
        overrides.append(ov)
        cur, work, best_fid = cand, cand_work, cand_fid
    if overrides:
        node["overrides"] = overrides
        node["residual"] = best_fid


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
    cols = np.asarray(ctx.candidate_cols(sid_addr), dtype=np.int64)
    if len(cols) == 0:
        return None, None, 0.0
    grid = ss.grid.astype(np.int64)
    s = np.asarray(series, dtype=np.int64)
    eof_match = (grid == s[:, None]).mean(axis=0)
    lead = (grid == np.roll(s, -1)[:, None]).mean(axis=0)
    rank = np.maximum(eof_match, lead)[cols]
    order = cols[np.argsort(rank)[::-1][:8]]
    best_addr, best_recon, best_frac = None, None, -1.0
    for j in order:
        cell = int(ss.addrs[j])
        recon = ctx.sampler.at_write(cell, sid_addr)
        frac = float(np.mean(recon == s))
        if frac > best_frac:
            best_addr, best_recon, best_frac = cell, recon, frac
    return best_addr, best_recon, best_frac


def _candidate_terms(forced, ctx, sid_addr=None, max_set: int = 6):
    """Typed cell-predicate terms true on every ``forced`` frame (recall 1.0).

    Returns ``[(pred_bool_array, term_dict), ...]`` over the read-log-narrowed
    candidate cells: full-byte equality (``eq``), single-bit tests (``bit``),
    small value-membership (``in``), and a multi-bit **mask-equality** term
    (``mask``) whose mask is the set of bits that stay constant across the forced
    frames (< 8 of them, i.e. a genuine multi-bit tap such as ``(ctr & $30) ==
    0``). The mask is read off the cell's distinct-value structure -- never a
    256-mask brute force. Shared by :func:`_find_override` and
    :func:`propose_select`.
    """
    grid = ctx.stateseq.grid.astype(np.int64)
    addrs = ctx.stateseq.addrs
    cands = []
    for j in ctx.candidate_cols(sid_addr):
        col = grid[:, j]
        fc = col[forced]
        if len(fc) == 0:
            continue
        cell = int(addrs[j])
        uniq = np.unique(fc)
        if len(uniq) == 1:
            cands.append((col == int(uniq[0]), {"kind": "eq", "cell": cell, "value": int(uniq[0])}))
        elif 2 <= len(uniq) <= max_set and len(uniq) < len(np.unique(col)):
            # A force gated by the cell holding one of a few states (recall 1.0 by
            # construction); the greedy precision/strict-improvement gating below
            # rejects it unless it genuinely tightens the selection.
            cands.append(
                (
                    np.isin(col, uniq),
                    {"kind": "in", "cell": cell, "values": tuple(int(x) for x in uniq)},
                )
            )
        const_mask = 0
        for b in range(8):
            bit = (fc >> b) & 1
            if bit.min() == bit.max():
                val = int(bit[0]) << b
                const_mask |= 1 << b
                cands.append(
                    (
                        (col & (1 << b)) == val,
                        {"kind": "bit", "cell": cell, "mask": 1 << b, "value": val},
                    )
                )
        if 2 <= _bits_set(const_mask) < 8:
            mval = int(fc[0]) & const_mask
            cands.append(
                (
                    (col & const_mask) == mval,
                    {"kind": "mask", "cell": cell, "mask": int(const_mask), "value": mval},
                )
            )
    return cands


def _find_override(forced, ctx, max_terms: int = 3, sid_addr=None):
    """A conjunction of cell predicates that fires exactly on ``forced`` frames.

    Candidate predicates (cell value-equality / single-bit tests / small
    value-membership) are restricted to those true on every forced frame (recall
    1.0); a greedy intersection then drives precision to 1.0. A membership term
    ``cell in {v0,..}`` (recovered set of <= ``max_set`` distinct cell values, e.g.
    the per-voice waveform shadow whose value selects the instruments that
    hard-restart) captures a force gated by a handful of states no single
    equality/bit can express. Returns a list of typed predicate dicts
    (``{"kind":"eq"|"bit"|"in", ...}``) or ``None``; :func:`ir._apply_overrides`
    still accepts the legacy tuple forms too.
    """
    if int(forced.sum()) == 0:
        return None
    n_forced = int(forced.sum())
    cands = _candidate_terms(forced, ctx, sid_addr)
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
    cols = np.asarray(ctx.candidate_cols(sid_addr), dtype=np.int64)
    if len(cols) == 0:
        return None
    grid = ss.grid.astype(np.int64)
    t = np.asarray(target, dtype=np.int64) & 0xFF
    match = (grid == t[:, None]).mean(axis=0)[cols]
    order = cols[np.argsort(match)[::-1][:8]]
    best_addr, best_frac = None, -1.0
    for j in order:
        cell = int(ss.addrs[j])
        frac = float(np.mean((ctx.sampler.at_write(cell, sid_addr) & 0xFF) == t))
        if frac > best_frac:
            best_addr, best_frac = cell, frac
    return best_addr


def _override_descriptor(forced_byte, ctx, sid_addr=None):
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
    terms = _find_override(forced_byte == force_val, ctx, sid_addr=sid_addr)
    return None if terms is None else {"predicate": terms, "force": force_val}


# -- program (table-program interpreter) proposer -----------------------------
#
# The program node (ir._recon_program) is the interpreter behind GT2 pulse+filter,
# JCH PW+filter, DMC filter and FC filter: a cursor walks a table of (duration,
# step-or-SET) records, integrating an accumulator, with $FF-style loop rows.
# ``propose_program`` fits it two ways, both grounded in the SERIES' constant-slope
# run structure (never fixture-specific magic values):
#
#  1. Grounded: when a candidate RAM table (read-log ``ctx.tables`` first) decodes
#     -- as interleaved (duration, value) records with a data-driven SET flag and a
#     loop marker -- into a program that reproduces the series, adopt its records
#     (recovering SET rows + the loop point that the series alone cannot spell).
#  2. Series-only: otherwise every constant-slope run becomes one STEP record
#     ``[len, slope, 0]`` (a jump is a length-1 record); exact by construction.
#
# It is offered only as an alternative to a FEEDER / imperfect incumbent (see
# ``propose_program``) so it never displaces a structured BACC/TABLE_WALK winner,
# and MDL (ir._cc_program: one record per run vs one replay value per changed
# frame) then adopts it over the feeder it supplants.
_PROGRAM_LOOP_MARKER = 0xFF  # a duration byte of $FF flags a loop row (value = target)
_PROGRAM_MIN_RUNS = 3  # a genuine program has several records, not a flat/CONST series
# A program is adopted only when its record count is meaningfully below the
# register's changed-frame count (coarse slope structure) -- so a per-frame-random
# feeder / mux (records ~= changed frames) is never re-encoded record-per-frame and
# stolen from the FEEDER/SELECT it belongs to. Filter/pulse sweeps sit well under.
_PROGRAM_RUN_FRAC = 0.85


def _program_runs(series) -> np.ndarray:
    """Constant-slope runs of a series as ``[length, slope]`` STEP records.

    Each maximal run of a constant first difference ``slope`` over ``length``
    frames becomes one record; a lone final frame (no outgoing slope) is emitted as
    a trailing ``[1, 0]`` hold so the record lengths sum to exactly ``len(series)``.
    Fed straight into :func:`_program_from_runs`.
    """
    s = np.asarray(series, dtype=np.int64).ravel()
    n = len(s)
    if n == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if n == 1:
        return np.array([[1, 0]], dtype=np.int64)
    d = np.diff(s)
    cuts = np.nonzero(np.diff(d) != 0)[0] + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(d)]))
    recs = [[int(e - b), int(d[b])] for b, e in zip(starts, ends)]
    recs.append([1, 0])  # the trailing value carried in the accumulator
    return np.array(recs, dtype=np.int64)


def _program_from_runs(series):
    """Series-only program descriptor: one STEP record per constant-slope run."""
    s = np.asarray(series, dtype=np.int64).ravel()
    runs = _program_runs(s)
    records = [[int(L), int(v), 0] for L, v in runs]
    return {
        "type": "PROGRAM",
        "records": records,
        "loop": None,
        "seed": int(s[0]) if len(s) else 0,
        "variant": "series",
        "width": 8,
    }


def _int8(v: int) -> int:
    """A table byte read as a signed step (``$80..$FF`` -> negative)."""
    v &= 0xFF
    return v - 256 if v >= 0x80 else v


def _decode_program_table(ram, base, top):
    """Decode a RAM byte region as interleaved ``(duration, value)`` records.

    Data-driven record variants (no per-engine paths): a duration byte of
    ``$FF`` is a loop row whose value byte is the target record index; a duration
    byte with bit7 set is a SET row (absolute ``acc := value``, held ``dur & $7F``
    frames); otherwise a STEP row (``acc += int8(value)`` for ``dur`` frames). A
    ``0`` duration terminates. Returns ``(records, loop)`` or ``None``.
    """
    records, loop = [], None
    idx = int(base)
    while idx + 1 <= int(top):
        dur = int(ram[idx])
        val = int(ram[idx + 1])
        idx += 2
        if dur == 0:
            break
        if dur == _PROGRAM_LOOP_MARKER:
            loop = val if 0 <= val < len(records) else 0
            break
        if dur & 0x80:
            records.append([dur & 0x7F, val, 1])
        else:
            records.append([dur, _int8(val), 0])
        if len(records) > 256:
            break
    if not records:
        return None
    return records, loop


def _ground_program(series, ctx):
    """A grounded program from a read-log candidate RAM table, or ``None``.

    Searches ``ctx.tables`` (the read-log ``(base, top, valueset)`` candidates)
    for a region whose decoded records reproduce ``series`` exactly (seeded at the
    first value, cursor at 0). This is what recovers SET rows and the loop point
    the series alone cannot distinguish.
    """
    if ctx.ram is None or not ctx.tables:
        return None
    s = np.asarray(series, dtype=np.int64).ravel()
    n = len(s)
    if n == 0:
        return None
    ram = np.asarray(ctx.ram, dtype=np.uint8)
    seed = int(s[0])
    for base, top, _vals in ctx.tables:
        decoded = _decode_program_table(ram, base, top)
        if decoded is None:
            continue
        records, loop = decoded
        desc = {
            "type": "PROGRAM",
            "records": records,
            "loop": loop,
            "seed": seed,
            "variant": "table",
            "width": 8,
            "time_base": int(base),
            "spd_base": int(base) + 1,
        }
        recon = ir._recon_program(desc, n)  # pylint: disable=protected-access
        if np.array_equal(recon & 0xFF, s & 0xFF):
            return desc
    return None


def _best_base(scored):
    """The highest-scoring already-scored candidate ``(cand, fid)`` or ``None``."""
    best_cand, best_fid = None, -1.0
    for cand, _tree, recon, _key in scored:
        if recon is None:
            continue
        fid = _key[0]
        if best_cand is None:
            best_cand, best_fid = cand, fid
            continue
        if fid > best_fid or (fid == best_fid and cand.get("score", 0) > best_cand.get("score", 0)):
            best_cand, best_fid = cand, fid
    return None if best_cand is None else (best_cand, best_fid)


_PROGRAM_INCUMBENTS = {"FEEDER"}


def propose_program(scored, series, sid_addr, ctx):
    """Fit a table-program, but only where a FEEDER / imperfect register would win.

    Gated on the scored base candidates so it never displaces a structured winner
    (a fid-1.0 BACC/TABLE_WALK/SEQ): a program is offered only when the best base
    candidate is a raw FEEDER fallback or is itself imperfect. It reproduces the series
    exactly (grounded in a read-log table when one decodes, else one STEP record
    per run), and MDL charges one record per run vs the feeder's one replay value
    per changed frame -- so the arbiter adopts it over the feeder it supplants.
    """
    n = ctx.n_frames
    if not n:
        return []
    s = np.asarray(series, dtype=np.int64).ravel()
    if len(s) != n:
        return []
    best = _best_base(scored)
    if best is not None:
        cand, fid = best
        if fid >= 1.0 - 1e-9 and cand.get("type") not in _PROGRAM_INCUMBENTS:
            return []
    desc = _ground_program(s, ctx) or _program_from_runs(s)
    records = desc.get("records", [])
    # A program pays only when it captures FAR LESS state than the feeder it
    # supplants: one record per constant-slope run vs one replay value per changed
    # frame. Require the record count to be well under the register's changed-frame
    # count (else a per-frame-random register -- rightly a feeder / mux -- would be
    # re-encoded record-per-frame at no MDL gain and steal it).
    changed = int(np.count_nonzero(np.diff(s) != 0)) + 1 if n > 1 else 1
    if len(records) < _PROGRAM_MIN_RUNS or len(records) >= min(n, _PROGRAM_RUN_FRAC * changed):
        return []
    desc["addr"] = int(sid_addr)
    desc["sid"] = int(sid_addr)
    recon = ir.evaluate(ir.to_ir(desc), n, ctx.sampler)
    if recon is None or float(np.mean(np.asarray(recon, dtype=np.int64) == s)) < 1.0 - 1e-9:
        return []
    return [desc]


# -- select (mux) proposer ----------------------------------------------------
#
# A ``select`` muxes two or three partial hypotheses per-frame by a recovered
# cell predicate (WEMUSIC A/B sets, DMC slide-vs-vibrato FREQ, FC's FREQ emit
# blocks, MusicAssembler PW mode-select). Arms come from the arbiter's scored
# runner-up trees, augmented with residual feeder cells for pure A/B/C feeder
# muxes. Three overfit guards (all required): each arm must explain >= a frame
# floor, its separating predicate is MDL-charged (see ``ir._cc_select``), and the
# mux is emitted only as one more arbiter candidate -- so it is adopted only when
# it BEATS the best single-arm tree on score, never on fidelity alone.
_SELECT_ARM_FLOOR = 0.05  # each arm must explain >= 5% of frames
_SELECT_MAX_ARMS = 2  # predicate arms beyond the default -> up to 3 mux branches


def _residual_feeder_arm(residual, s, sid_addr, ctx):
    """A feeder (captured-cell) arm best reproducing ``s`` on the ``residual`` frames.

    Returns ``(tree, correct_mask)`` for the best state cell (sampled at the write
    instant) over the residual frames, or ``None``. Lets a pure A/B/C feeder mux
    (each mode selecting a different captured cell) recover without any single
    proposer having surfaced the per-mode cells.
    """
    ss = ctx.stateseq
    if ss is None or ss.grid.shape[1] == 0 or ctx.sampler is None:
        return None
    cols = np.asarray(ctx.candidate_cols(sid_addr), dtype=np.int64)
    if len(cols) == 0 or int(residual.sum()) == 0:
        return None
    grid = ss.grid.astype(np.int64)
    t = np.asarray(s, dtype=np.int64) & 0xFF
    eof_match = (grid[residual] == t[residual, None]).mean(axis=0)
    order = cols[np.argsort(eof_match[cols])[::-1][:8]]
    best_addr, best_frac = None, -1.0
    for j in order:
        cell = int(ss.addrs[j])
        recon = ctx.sampler.at_write(cell, sid_addr).astype(np.int64) & 0xFF
        frac = float(np.mean(recon[residual] == t[residual]))
        if frac > best_frac:
            best_addr, best_frac = cell, frac
    if best_addr is None or best_frac <= 0.0:
        return None
    tree = ir.to_ir(
        {"type": "FEEDER", "addr": int(sid_addr), "sid": int(sid_addr), "cell": best_addr}
    )
    recon = ir.evaluate(tree, ctx.n_frames, ctx.sampler)
    if recon is None:
        return None
    return tree, (np.asarray(recon, dtype=np.int64) == np.asarray(s, dtype=np.int64))


def _try_select_arm(correct, assigned, covered, ctx, sid_addr):
    """Return ``(predicate, assigned, covered)`` if this arm is net-positive.

    ``gain`` is the currently-wrong frames the arm could still claim; a separating
    predicate is recovered with the shared :func:`_find_override` machinery. Only
    the frames the arm claims first (``hold & ~assigned``) take its value, mirroring
    the evaluator's first-match precedence. Rejected unless it clears the >=5% floor
    and strictly increases the covered-correct count.
    """
    n = ctx.n_frames
    floor = max(1, int(_SELECT_ARM_FLOOR * n))
    gain = correct & ~covered & ~assigned
    if int(gain.sum()) < floor:
        return None
    pred = _find_override(gain, ctx, sid_addr=sid_addr)
    if pred is None:
        return None
    hold = ir.predicate_mask(pred, ctx.sampler, n)
    new_covered = np.where(hold & ~assigned, correct, covered)
    if int(new_covered.sum()) <= int(covered.sum()):
        return None
    return pred, assigned | hold, new_covered


def _fill_select_arms(pool, arms, covered, assigned, s, sid_addr, ctx):
    """Greedily fill up to ``_SELECT_MAX_ARMS`` arms, runner-up trees then feeders.

    Mutates ``arms`` in place (appending ``(predicate, tree)``) and returns the
    updated ``(covered, assigned)`` first-match state.
    """
    n = ctx.n_frames
    floor = max(1, int(_SELECT_ARM_FLOOR * n))
    for tree, correct in pool:
        if len(arms) >= _SELECT_MAX_ARMS:
            return covered, assigned
        got = _try_select_arm(correct, assigned, covered, ctx, sid_addr)
        if got is not None:
            pred, assigned, covered = got
            arms.append((pred, tree))
    while len(arms) < _SELECT_MAX_ARMS and int((~covered).sum()) >= floor:
        arm = _residual_feeder_arm(~covered, s, sid_addr, ctx)
        if arm is None:
            break
        tree, correct = arm
        got = _try_select_arm(correct, assigned, covered, ctx, sid_addr)
        if got is None:
            break
        pred, assigned, covered = got
        arms.append((pred, tree))
    return covered, assigned


def propose_select(scored, series, sid_addr, ctx):
    """Mux the arbiter's runner-up trees (and residual feeders) into a ``select``.

    ``scored`` is the arbiter's ``[(cand, tree, recon, key), ...]``. The
    highest-fidelity tree is the default arm; further arms are the runner-up trees
    whose correct-frame set covers frames the default gets wrong, separated by a
    recovered cell predicate, then residual feeder cells for any frames still
    unexplained. Returns ``[]`` (no mux) when the default is already exact or no
    net-positive separated arm exists.
    """
    n = ctx.n_frames
    if not n or ctx.sampler is None or ctx.stateseq is None or ctx.stateseq.grid.shape[1] == 0:
        return []
    s = np.asarray(series, dtype=np.int64)
    entries = []
    for _cand, tree, recon, _key in scored:
        if recon is not None and len(recon) == n:
            correct = np.asarray(recon, dtype=np.int64) == s
            entries.append((float(correct.mean()), tree, correct))
    if not entries:
        return []
    entries.sort(key=lambda e: -e[0])
    base_fid, base_tree, base_correct = entries[0]
    if base_fid >= 1.0 - 1e-9:
        return []
    assigned = np.zeros(n, dtype=bool)
    covered = base_correct.copy()
    arms = []
    pool = [(tree, correct) for _fid, tree, correct in entries[1:] if id(tree) != id(base_tree)]
    covered, assigned = _fill_select_arms(pool, arms, covered, assigned, s, sid_addr, ctx)
    if not arms:
        return []
    desc = {
        "type": "SELECT",
        "addr": int(sid_addr),
        "sid": int(sid_addr),
        "arms": arms,
        "default_tree": base_tree,
    }
    # Third overfit guard: adopt only if the mux BEATS the best single-arm tree on
    # SCORE (fidelity - LAMBDA*complexity), not on fidelity alone -- the predicate
    # terms and both arm subtrees are MDL-charged, so a mux buying a few frames at
    # a large complexity cost is rejected here before it reaches the arbiter.
    sel_fid = float(covered.mean())
    sel_cx = ir.complexity(ir.to_ir(desc), ctx.sampler, n)
    sel_score = sel_fid - _LAMBDA * sel_cx
    best_single = max(float(c.get("score", 0.0)) for c, _t, _r, _k in scored)
    if sel_score <= best_single:
        return []
    return [desc]


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


def _greedy_index_sum(idx, inb, grid, addrs, clip_hi, primary, cols, max_extra):
    """Greedy additive index recovery: cells whose EOF sum reproduces ``idx``.

    Starting from the ``primary`` column, add cells (restricted to ``cols``) whose
    value explains the residual ``idx - running`` while they raise the in-table
    (``inb``) match. Shared core of :func:`_pitch_index_cells` (all columns, given
    primary) and :func:`propose_index_sum` (candidate columns, discovered
    primary). ``clip_hi`` bounds the index into the table.
    """
    cand_cols = np.asarray(sorted(set(int(c) for c in cols)), dtype=np.int64)
    chosen = [int(addrs[primary])]
    running = grid[:, primary].copy()
    used = {primary}

    def score(run):
        return float(np.mean((np.clip(run, 0, clip_hi) == idx)[inb]))

    cur_score = score(running)
    for _ in range(max_extra):
        resid = idx - running
        raw = (grid == resid[:, None])[inb].mean(axis=0)
        match = np.full(len(raw), -1.0)
        match[cand_cols] = raw[cand_cols]
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


def _pitch_index_cells(idx, inb, grid, addrs, tlen, primary, max_extra=2):
    """Index cells whose end-of-frame sum reproduces a pitch-table index.

    ``idx`` is the per-frame table index inverted from the oracle (``-1`` off
    table, ``inb`` marks the in-table frames). Starting from the ``primary``
    column (the note cell), greedily add cells whose value explains the residual
    ``idx - running``, keeping each only while it raises the in-table match. The
    sum models ``note + transpose + arp/wavetable offset`` of the FC pitch walk.
    """
    return _greedy_index_sum(
        idx, inb, grid, addrs, tlen - 1, primary, range(grid.shape[1]), max_extra
    )


def propose_index_sum(idx, ctx, sid_addr=None, inb=None, clip_hi=None, max_extra=2):
    """Recover a table index as a sum of observable state cells (any table node).

    The generalisation of :func:`_pitch_index_cells`: given a per-frame integer
    index ``idx`` (``-1`` / ``inb=False`` off table) and the table's upper clip
    bound ``clip_hi``, discover the best single index cell among the read-log
    narrowed candidate columns, then greedily add cells whose EOF sum reproduces
    ``idx``. Returns ``(index_cells, running)`` (``([], zeros)`` if no cell fits).
    Usable by CTRL/PW/filter table lanes, not just the FC pitch walk.
    """
    idx = np.asarray(idx, dtype=np.int64)
    n = len(idx)
    if inb is None:
        inb = idx >= 0
    if not np.any(inb):
        return [], np.zeros(n, dtype=np.int64)
    if clip_hi is None:
        clip_hi = int(idx[inb].max())
    ss = ctx.stateseq
    if ss is None or ss.grid.shape[1] == 0:
        return [], np.zeros(n, dtype=np.int64)
    grid = ss.grid.astype(np.int64)
    addrs = ss.addrs
    cols = ctx.candidate_cols(sid_addr)
    if not cols:
        return [], np.zeros(n, dtype=np.int64)
    match = (np.clip(grid[:, cols], 0, clip_hi) == idx[:, None])[inb].mean(axis=0)
    primary = int(cols[int(match.argmax())])
    return _greedy_index_sum(idx, inb, grid, addrs, clip_hi, primary, cols, max_extra)


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
        for j in ctx.candidate_cols(sid_addr)
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
        desc = _build_pitchwalk(
            role, base_lo, base_hi, primary, lo_o, hi_o, grid, addrs, ramu, ctx, sid_addr
        )
        if desc is None:
            continue
        if desc["residual"] > best_fid:
            best_desc, best_fid = desc, desc["residual"]
    return best_desc


def _build_pitchwalk(
    role, base_lo, base_hi, primary, lo_o, hi_o, grid, addrs, ramu, ctx, sid_addr=None
):
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
    out = ir.evaluate(ir.to_ir(desc), ctx.n_frames, ctx.sampler)
    oracle = lo_o if role == "lo" else hi_o
    forced = np.where(out == oracle, -1, oracle).astype(np.int64)
    ov = _override_descriptor(forced, ctx, sid_addr)
    if ov is not None:
        desc["overrides"].append(ov)
        out = ir.evaluate(ir.to_ir(desc), ctx.n_frames, ctx.sampler)
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
    base_val = ir.part_value(base, n, ctx.sampler)
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
        modelled = (base_val + ir.part_value(mod, n, ctx.sampler)) & 0xFFFF
        # The 16-bit base+mod sum is what isolates the override frames (a note
        # onset / hard-restart), so score explained-ness on the whole word, not the
        # emitted byte; ``out_byte`` (== ``target_byte``) is the forced value.
        recover_overrides(
            desc, target_byte, ctx, 1, explained=modelled == target, sid_addr=sid_addr
        )
        fid = float(np.mean(ir.evaluate(ir.to_ir(desc), n, ctx.sampler) == target_byte))
        desc["residual"] = fid
        if fid > best_fid:
            best_fid, best_desc = fid, desc
    return best_desc


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
    ov = _override_descriptor(forced, ctx, sid_addr)
    if ov is not None:
        desc["overrides"].append(ov)
    recon = ir.evaluate(ir.to_ir(desc), ctx.n_frames, ctx.sampler)
    desc["residual"] = float(np.mean(recon == series))
    return desc


_BINOP_MAX_DISTINCT = 24
_BINOP_TYPE = {"or": "OR", "and": "AND", "xor": "XOR", "add": "ADD", "sub": "SUB"}


def _binop_apply(fn, a, b):
    """Apply an 8-bit fold ``a fn b`` (broadcasting), wrapping to a byte."""
    if fn == "or":
        r = a | b
    elif fn == "and":
        r = a & b
    elif fn == "xor":
        r = a ^ b
    elif fn == "sub":
        r = a - b
    else:
        r = a + b
    return r & 0xFF


def _binop_bit_const(a_bits, t_bits, fn):
    """Per-bit optimal 8-bit constant for a bitwise ``fn(cell, const)``.

    Each constant bit is chosen independently to maximise agreement (as
    ``_or_pair`` did for OR): OR sets a bit when the target is 1 more often than
    the cell already agrees; AND keeps a bit when the cell agrees more often than
    forcing 0; XOR flips a bit when the complement agrees more often."""
    n = t_bits.shape[0]
    agree = (a_bits == t_bits).sum(axis=0)  # frames cell bit == target bit
    ones = t_bits.sum(axis=0)  # frames target bit == 1
    if fn == "or":
        kbit = ones >= agree
    elif fn == "and":
        kbit = agree >= (n - ones)  # keep (== cell) vs force 0 (agrees where target 0)
    else:  # xor
        kbit = (n - agree) > agree
    return int((kbit.astype(np.int64) * (1 << np.arange(8))).sum())


def _binop_add_consts(a, t, sub, cap: int = 4):
    """Small constant set for ``add``/``sub`` from the residual histogram.

    ``add``: ``t = cell + k`` -> ``k = t - cell``; ``sub``: ``t = cell - k`` ->
    ``k = cell - t``. Returns the most common residual bytes."""
    diff = (a - t) if sub else (t - a)
    resid = diff & 0xFF
    vals, counts = np.unique(resid, return_counts=True)
    order = np.argsort(counts)[::-1][:cap]
    return [int(v) for v in vals[order]]


def _binop_best(fn, cols, addrs, mat, ss_s, sb, s, sid_addr, ctx):
    """Best ``fn`` fold (cell op cell / cell op const) over the subsample, then
    verified on all frames and built into a descriptor (or ``None``)."""
    best = (-1.0, None, None, None)  # frac, ai, bi, const
    bit_pos = (1 << np.arange(8)).astype(np.int64)
    for ai in range(len(cols)):
        a = mat[:, ai]
        if fn in ("or", "and", "xor"):
            ab = (a[:, None] & bit_pos) > 0
            k = _binop_bit_const(ab, sb, fn)
            fk = float(np.mean(_binop_apply(fn, a, k) == ss_s))
        else:
            fk, k = -1.0, 0
            for c in _binop_add_consts(a, ss_s, fn == "sub"):
                fc = float(np.mean(_binop_apply(fn, a, c) == ss_s))
                if fc > fk:
                    fk, k = fc, c
        if fk > best[0]:
            best = (fk, ai, None, int(k))
        f = (_binop_apply(fn, a[:, None], mat) == ss_s[:, None]).mean(axis=0)
        bi = int(f.argmax())
        if f[bi] > best[0] and bi != ai:
            best = (float(f[bi]), ai, bi, None)
    _frac, ai, bi, k = best
    if ai is None:
        return None
    desc = {"type": _BINOP_TYPE[fn], "cell_a": int(addrs[cols[ai]]), "sid": int(sid_addr)}
    cells = [desc["cell_a"]]
    if bi is None:
        desc["const"] = int(k)
    else:
        desc["cell_b"] = int(addrs[cols[bi]])
        cells.append(desc["cell_b"])
    if fn == "and":
        desc["overrides"] = []
        desc["residual"] = float(np.mean(_and_recon_masked(desc, ctx) == s))
        recover_overrides(
            desc, s, ctx, 4, mask=ctx.sampler.written_mask(desc["sid"]), sid_addr=sid_addr
        )
    else:
        attach_prelude(desc, s, ctx, cells)
        recon = ir.evaluate(ir.to_ir(desc), len(s), ctx.sampler)
        desc["residual"] = float(np.mean(recon == s))
    return desc


def _and_recon_masked(desc, ctx) -> np.ndarray:
    """``_recon_and`` with the pre-first-write power-on default applied (for scoring)."""
    base = ir.evaluate(ir.to_ir(desc), ctx.n_frames, ctx.sampler)
    written = ctx.sampler.written_mask(desc["sid"])
    return np.where(written, base, 0)


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


def _opt_log(trace, name) -> np.ndarray:
    """A trace's RAM-access log by method ``name``, or an empty log if unsupported."""
    fn = getattr(trace, name, None)
    return fn() if callable(fn) else np.empty(0, dtype=RAMACCESS_DTYPE)


def _carry_at(cycles, values, sample_cycles, back: int = 1) -> np.ndarray:
    """Value of a cell ``back`` writes before each sample cycle (1 = the last).

    ``back=2`` returns the write immediately preceding the last one at/before the
    sample -- the pre-update operand a read-modify-write store leaves behind.
    """
    out = np.zeros(len(sample_cycles), dtype=np.int64)
    if len(cycles) == 0:
        return out
    order = np.argsort(cycles, kind="stable")
    wc = cycles[order]
    wv = values[order].astype(np.int64)
    pos = np.searchsorted(wc, sample_cycles, side="right")
    taken = pos >= back
    idx = np.clip(pos - back, 0, len(wv) - 1)
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
        self._operand: dict = {}
        self._written: dict = {}
        self._chip: dict = {}
        self._readpc: dict = {}
        # Logged I/O reads ($D000-$DFFF: SID osc3/env3, CIA timers) -- always
        # present on a real trace; the ``chip`` source node samples these. The
        # combined read log (RAM reads when rendered with ``--reads``, plus I/O
        # reads) is keyed by read-PC for the Tier-3 witness's per-frame code-derived
        # input cone. Absent on minimal duck-typed fakes -> empty logs.
        self._iord = _opt_log(trace, "io_reads")
        reads = np.concatenate([_opt_log(trace, "ram_reads"), self._iord])
        # Index the combined read log by PC once (sorted), so ``read_pc`` slices a
        # contiguous run instead of scanning the whole log per call -- the witness
        # samples many read-PCs on the longest fixtures (millions of reads).
        order = np.argsort(reads["pc"], kind="stable")
        self._rbp_pc = reads["pc"][order]
        self._rbp_cyc = reads["cycle"][order]
        self._rbp_val = reads["value"][order]
        # Shared per-voice latent state, attached by ``_build_context`` so the
        # evaluator can resolve ``latent``/``tick`` index nodes; ``None`` when the
        # sampler is built standalone (reconstruction falls back to eof cells).
        self.latents = None

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

    def operand(self, cell_addr, sid_addr) -> np.ndarray:
        """Value of ``cell_addr`` one write before ``sid_addr``'s write instant.

        A read-modify-write player reads an operand, updates it, then stores it
        back before the SID write, so :meth:`at_write` captures the post-update
        value; the pre-update operand it read is the write immediately preceding
        that store. A note reseed writes the cell earlier in the frame, so this is
        the correct input for the routine's recurrence (``roll(at_write)`` is not).
        """
        key = (int(cell_addr), int(sid_addr))
        if key not in self._operand:
            sample = self.write_cycles(sid_addr) + np.uint64(2)
            cyc, val = self._cell_writes(int(cell_addr))
            self._operand[key] = _carry_at(cyc, val, sample, back=2)
        return self._operand[key]

    def chip(self, addr, sid_addr) -> np.ndarray:
        """Per-frame logged I/O read value of ``addr`` at ``sid_addr``'s write instant.

        The player's own ``LDA $D41B`` (osc3/env3) or CIA-timer read: the value the
        code observed feeding the store, taken from the I/O-read log at (just after)
        the store cycle. During recovery these are data the trace observed, so they
        drive reconstruction with no circularity (GENERIC_RECOVERY.md 3.5).
        """
        key = (int(addr), int(sid_addr) if sid_addr is not None else -1)
        if key not in self._chip:
            sample = self.write_cycles(sid_addr) + np.uint64(2)
            sel = self._iord[self._iord["addr"] == int(addr)]
            self._chip[key] = _carry_at(sel["cycle"], sel["value"], sample)
        return self._chip[key]

    def read_pc(self, pc, sid_addr) -> np.ndarray:
        """Per-frame value read by store-PC ``pc`` at ``sid_addr``'s write instant.

        A read-PC's per-frame value is exactly the input the emitting code used --
        regardless of its (possibly per-frame varying) effective address -- so it is
        the Tier-3 witness's canonical dynamic input (indexed/subroutine reads
        included). Sampled from the combined RAM+I/O read log at the store cycle.
        """
        key = (int(pc), int(sid_addr) if sid_addr is not None else -1)
        if key not in self._readpc:
            sample = self.write_cycles(sid_addr) + np.uint64(2)
            lo = int(np.searchsorted(self._rbp_pc, int(pc), side="left"))
            hi = int(np.searchsorted(self._rbp_pc, int(pc), side="right"))
            self._readpc[key] = _carry_at(self._rbp_cyc[lo:hi], self._rbp_val[lo:hi], sample)
        return self._readpc[key]

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


def _apply_overrides(out, overrides, sampler) -> np.ndarray:
    """Force values where each override's cell-predicate conjunction holds.

    Delegates to the single implementation in :mod:`preframr_playroutine.ir`,
    which accepts both legacy predicate tuples and the typed-dict form.
    """
    return ir._apply_overrides(out, overrides, sampler)  # pylint: disable=protected-access


def _recon_bacc_full(desc, n) -> np.ndarray:
    """Full (8/16-bit) accumulator series; the port now lives in ir (segment fits)."""
    return ir._recon_bacc_full(desc, n)  # pylint: disable=protected-access


def _recon_tickband(desc, n) -> np.ndarray:
    """Tick-banded reflecting sweep; the port now lives in ir (segment fits)."""
    return ir._recon_tickband(desc, n)  # pylint: disable=protected-access


def reconstruct_register(descriptor, ticks, trace=None, sampler=None) -> np.ndarray:
    """Regenerate a register's per-frame output from its recovered descriptor.

    Executes the descriptor produced by :func:`classify_register`:
    ``CONST`` -> a constant; ``SEQ`` -> latched values held between note events;
    ``BACC`` -> the recurrence re-run (reset at the recovered note-on seeds);
    ``TABLE_WALK`` -> ``table[base + cursor*stride] & mask`` (gate-masked by a
    captured cell when one was recovered); ``COMPOSITE`` -> base + modulation +
    overrides; ``FEEDER`` -> a global filter register's captured RAM feeder cell
    sampled at the write instant; ``XOR`` -> a CTRL register's ``cellA XOR cellB``
    (base/eor gate idiom); ``OR`` -> a MODE/VOL-style ``cellA | cellB`` /
    ``cell | const`` blit. Cell-referencing descriptors read those cells
    from ``trace`` (or a shared ``sampler``); ``LIFT``/``WITNESS`` regenerate the
    lifted grammar tree / the memoised code-derived input mapping. A bare ``FEEDER``
    with no captured cell has no model and returns ``None``.
    """
    smp = _sampler_for(ticks, trace, sampler)
    return ir.evaluate(ir.to_ir(descriptor), len(ticks), smp)


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
