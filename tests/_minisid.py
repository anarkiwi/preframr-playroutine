"""Build tiny, original (non-copyrighted) PSID tunes for end-to-end tests.

The C64 player is a few bytes of hand-assembled 6502 that, each call,
increments a zero-page counter and writes it to voice-1 frequency low plus a
fixed master volume. With the PSID speed flag clear the libsidplayfp driver
calls it from a VIC raster IRQ (VBI); set, from a CIA timer IRQ.
"""

import struct

import numpy as np

LOAD = 0x1000
INIT = 0x1000
PLAY = 0x1003

# 6502 player, assembled at $1000:
#   1000 60           init: RTS
#   1001 EA EA        pad
#   1003 E6 FB        play: INC $FB        ; frame counter
#   1005 A5 FB              LDA $FB
#   1007 8D 00 D4           STA $D400      ; voice1 freq lo = counter
#   100A A9 0F              LDA #$0F
#   100C 8D 18 D4           STA $D418      ; master volume
#   100F 60                 RTS
CODE = bytes(
    [
        0x60,
        0xEA,
        0xEA,
        0xE6,
        0xFB,
        0xA5,
        0xFB,
        0x8D,
        0x00,
        0xD4,
        0xA9,
        0x0F,
        0x8D,
        0x18,
        0xD4,
        0x60,
    ]
)


# I/O-probe player, assembled at $1000: init sets voice 3 to max-frequency
# noise, play copies the osc3 readback ($D41B) and CIA1 timer A lo ($DC04)
# into voice-1 pulse width and INCs the VIC border colour (I/O read + write).
#   1000 A9 FF     init: LDA #$FF
#   1002 8D 0E D4        STA $D40E
#   1005 8D 0F D4        STA $D40F
#   1008 A9 80           LDA #$80
#   100A 8D 12 D4        STA $D412
#   100D 60              RTS
#   100E AD 1B D4  play: LDA $D41B
#   1011 8D 02 D4        STA $D402
#   1014 AD 04 DC        LDA $DC04
#   1017 8D 03 D4        STA $D403
#   101A A9 0F           LDA #$0F
#   101C 8D 18 D4        STA $D418
#   101F EE 20 D0        INC $D020
#   1022 60              RTS
IOPROBE_INIT = 0x1000
IOPROBE_PLAY = 0x100E
IOPROBE_CODE = bytes(
    [
        0xA9,
        0xFF,
        0x8D,
        0x0E,
        0xD4,
        0x8D,
        0x0F,
        0xD4,
        0xA9,
        0x80,
        0x8D,
        0x12,
        0xD4,
        0x60,
        0xAD,
        0x1B,
        0xD4,
        0x8D,
        0x02,
        0xD4,
        0xAD,
        0x04,
        0xDC,
        0x8D,
        0x03,
        0xD4,
        0xA9,
        0x0F,
        0x8D,
        0x18,
        0xD4,
        0xEE,
        0x20,
        0xD0,
        0x60,
    ]
)


def _build_psid(speed: int, init: int, play: int, code: bytes) -> bytes:
    """Assemble PSID v2 bytes for a player loaded at LOAD ($1000)."""
    magic = b"PSID"
    version = 2
    data_offset = 0x7C
    load_address = 0  # embedded in C64 data below
    songs = 1
    start_song = 1
    name = b"preframr test"
    author = b"preframr-playroutine"
    released = b"2026 test"
    # flags: clock PAL (bit2), sid model 6581 (bit4)
    flags = (1 << 2) | (1 << 4)

    header = magic
    header += struct.pack(">H", version)
    header += struct.pack(">H", data_offset)
    header += struct.pack(">H", load_address)
    header += struct.pack(">H", init)
    header += struct.pack(">H", play)
    header += struct.pack(">H", songs)
    header += struct.pack(">H", start_song)
    header += struct.pack(">I", speed)
    header += name.ljust(32, b"\x00")[:32]
    header += author.ljust(32, b"\x00")[:32]
    header += released.ljust(32, b"\x00")[:32]
    header += struct.pack(">H", flags)
    header += struct.pack(">B", 0)  # startPage
    header += struct.pack(">B", 0)  # pageLength
    header += struct.pack(">B", 0)  # secondSIDAddress
    header += struct.pack(">B", 0)  # thirdSIDAddress
    assert len(header) == data_offset, len(header)

    c64data = struct.pack("<H", LOAD) + code
    return header + c64data


def build_psid(speed: int = 0) -> bytes:
    """Return PSID v2 bytes. ``speed`` bit0: 0=VBI (raster), 1=CIA timer."""
    return _build_psid(speed, INIT, PLAY, CODE)


def build_ioprobe_psid() -> bytes:
    """Return the I/O-probe PSID (VBI-driven; exercises I/O reads + writes)."""
    return _build_psid(0, IOPROBE_INIT, IOPROBE_PLAY, IOPROBE_CODE)


def recur_segment(  # pylint: disable=too-many-branches,too-many-locals,too-many-arguments
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
    divide=1,
    up_n=0,
    down_n=0,
    target=None,
):
    """Reference simulator for one step x boundary product segment.

    A deliberately INDEPENDENT reimplementation of the recovery kernel
    (``recover._simulate_recur``): the product-fitter tests build ground truth
    here and recover it there, so a bug in one is not masked by the other.
    ``step_kind`` in {const, updown, table}; ``boundary`` in
    {wrap, saw, reflect, clampflip, countflip, target}. ``divide`` applies the
    step only every n-th frame; ``up_n``/``down_n`` are the countflip dwell
    counts; ``target`` is the clamp-to-target latch value."""
    rate = None if rate is None else [int(x) for x in rate]
    m = 0 if rate is None else len(rate)
    span = hi - lo
    divide = int(divide) if divide else 1
    tgt = None if target is None else int(target)
    out = []
    v, d = int(seed), int(direction)
    dc, fc, tk = 0, 0, 0
    for _ in range(length):
        out.append(v)
        if step_kind == "table":
            if m == 0:
                continue
            st = rate[tk] if tk < m else rate[m - 1]
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
            mod = (span + st) if span > 0 else 1
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
            if limit > 0 and fc >= limit:
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
    return np.array(out, dtype=np.int64)


def select_feeder_mux(rng, n, arm_cells, sel_cell, mode_values):
    """Ground truth for a k-arm feeder mux (a ``select`` over captured cells).

    Each entry of ``arm_cells`` is a RAM address whose per-frame value is drawn
    from a distinct byte band (so no single cell reproduces the whole register and
    the arms never collide); ``sel_cell`` holds one of ``mode_values`` each frame
    and picks which arm cell the register copies. Deterministic given ``rng``.

    Returns ``(reg, {cell_addr: series})`` where ``reg`` is the emitted register
    series and the dict carries every arm cell plus the selector cell -- the RAM
    write streams the trace must replay. A fitter that only latches magic byte
    values (rather than recovering the selector predicate) cannot reproduce this,
    since the arm data is randomized per seed.
    """
    k = len(arm_cells)
    assert len(mode_values) == k, "one mode value per arm"
    band = 256 // (k + 1)
    arm_data = [(rng.integers(1, band, size=n) + j * band).astype(np.int64) for j in range(k)]
    mode_idx = rng.integers(0, k, size=n)
    sel_series = np.array([int(mode_values[m]) for m in mode_idx], dtype=np.int64)
    reg = np.array([int(arm_data[mode_idx[i]][i]) for i in range(n)], dtype=np.int64)
    cells = {int(c): arm_data[j] for j, c in enumerate(arm_cells)}
    cells[int(sel_cell)] = sel_series
    return reg, cells


def program_table(rows):
    """Encode ``rows`` as an interleaved (duration, value) table byte sequence.

    Each row is ``("step", dur, step)`` / ``("set", hold, value)`` / ``("loop",
    target)``; a deliberately INDEPENDENT reimplementation of
    ``recover._decode_program_table`` (byte layout: STEP dur ``$01..$7F``, SET dur
    ``dur|$80``, LOOP dur ``$FF`` with the target index in the value byte). Returns
    the raw ``bytes``.
    """
    out = []
    for row in rows:
        if row[0] == "step":
            _, dur, step = row
            out += [dur & 0x7F, step & 0xFF]
        elif row[0] == "set":
            _, hold, value = row
            out += [(hold & 0x7F) | 0x80, value & 0xFF]
        else:  # loop
            out += [0xFF, row[1] & 0xFF]
    return bytes(out)


def program_records(rows):
    """The ``[length, value, is_set]`` records + loop index a table decodes to."""
    records, loop = [], None
    for row in rows:
        if row[0] == "step":
            step = row[2] & 0xFF
            records.append([row[1], step - 256 if step >= 0x80 else step, 0])
        elif row[0] == "set":
            records.append([row[1], row[2] & 0xFF, 1])
        else:  # loop
            loop = row[1]
            break
    return records, loop


def program_series(rows, seed, n):
    """Reference: run a program table to an ``n``-frame register series.

    An independent reimplementation of ``ir._recon_program`` so the synthetic test
    builds ground truth here and recovers it there (a bug in one is not masked by
    the other)."""
    records, loop = program_records(rows)
    out = np.zeros(n, dtype=np.int64)
    acc, f, cur, guard = int(seed), 0, 0, 0
    while f < n:
        if cur >= len(records):
            guard += 1
            if loop is None or not 0 <= loop < len(records) or guard > n + len(records) + 8:
                break
            cur = loop
            continue
        length, value, is_set = records[cur]
        cur += 1
        if length <= 0:
            continue
        take = min(length, n - f)
        if is_set:
            acc = value
            out[f : f + take] = acc
        else:
            out[f : f + take] = acc + value * np.arange(take, dtype=np.int64)
            acc += value * length
        f += take
    if f < n:
        out[f:] = acc
    return out


def phase_triangle_series(counter, lo, hi, step, seed, boundary="reflect"):
    """Reference for ``ir.phase_fold``: ``value = fold(seed + step*counter)``.

    A deliberately INDEPENDENT (scalar-loop) reimplementation of the phase-locked
    boundary fold so the global-frame-counter recur test builds ground truth here
    and recovers it in ``recover._fit_phase`` -- a bug in one is not masked by the
    other. ``boundary`` in {reflect, saw, wrap}; ``counter`` is the global
    +k/frame phase source (Commando-class reflected triangle)."""
    c = np.asarray(counter, dtype=np.int64)
    span = int(hi - lo)
    out = np.empty(len(c), dtype=np.int64)
    for i, cv in enumerate(c):
        phase = int(seed) + int(step) * int(cv)
        if boundary == "wrap":
            mod = span + int(step) if step > 0 else span + 1
            out[i] = lo + phase % max(1, mod)
        elif boundary == "saw":
            out[i] = lo + phase % (span + 1)
        else:
            period = 2 * span
            p = phase % period
            out[i] = lo + (p if p <= span else period - p)
    return out


def masked_lookup_series(cell_vals, table, mask):
    """Reference for a masked captured-cell table lookup: ``table[cell & mask]``.

    The AMIB bass idiom ``$F7[cell & 7]`` -- the captured cell (LFSR / melody
    state) masked to the table span. Independent of ``recover._masked_table_lookup``
    (which recovers this shape from the trace)."""
    c = np.asarray(cell_vals, dtype=np.int64) & int(mask)
    t = np.asarray(table, dtype=np.int64)
    return t[np.clip(c, 0, len(t) - 1)]


def recur_series(  # pylint: disable=too-many-arguments
    lo,
    hi,
    seeds,
    length,
    boundary,
    step_kind,
    up=1,
    down=1,
    rate=None,
    divide=1,
    up_n=0,
    down_n=0,
    targets=None,
):
    """A reseeded multi-note product series and its note-on reset frames.

    Each entry of ``seeds`` seeds one note segment of ``length`` frames; the
    return is ``(series, resets)`` where ``resets`` are the segment start frames
    (the synthesized note-ons the product fitter segments on). ``targets`` (one
    per seed) drives the clamp-to-target glide boundary; the other new axes
    (``divide``, ``up_n``, ``down_n``) are shared across segments."""
    segs, resets, acc = [], [], 0
    for k, seed in enumerate(seeds):
        resets.append(acc)
        tgt = None if targets is None else targets[k]
        seg = recur_segment(
            lo, hi, seed, 1, length, boundary, step_kind, up, down, rate, divide, up_n, down_n, tgt
        )
        segs.append(seg)
        acc += len(seg)
    return np.concatenate(segs), resets
