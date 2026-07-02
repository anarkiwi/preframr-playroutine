# sidtrace instrumentation contract (v3)

This is the authoritative interface contract between the C++ tracer
(`app/sidtrace.cpp` + `patches/instrument.patch`) and the Python tooling
(`preframr_playroutine/`). Both sides MUST agree on every dtype and field.

v2 extends v1 (SID register + IRQ/NMI oracle) with the internal program-state
signals needed to recover per-register generators (BACC / TABLE-WALK / SEQ /
XSTATE) in the style of `/scratch/anarkiwi/cbm/re-trackers`:

- the store-site **PC** of every SID write (attribute a write to its generator),
- a **RAM-write log** (accumulators, table cursors, counters, and
  self-modifying-code immediates — the raw material for a per-frame `stateseq`),
- an **executed-PC coverage** bitmap (the `EXEC_PLAY` PC set),
- a **RAM image** dump (relocated player code + static tables),
- an optional **RAM-read log** (direct table-walk capture),
- everything tagged with the **interrupt window kind** (IRQ/NMI) so Python can
  bin flexibly.

All logging is a read-only observer: determinism (see README) is preserved.

## Play-window scoping

High-frequency logs (RAM writes/reads, PC coverage) are recorded only while the
CPU is executing inside an interrupt handler — a "play window". A window opens
when the CPU vectors through an IRQ/NMI (the existing `IRQHiRequest` hook) and
closes when the handler returns (stack pointer rises back above the entry
frame). This captures the per-frame player work and excludes the one-shot
unpack/relocate/init that runs before the first interrupt. Each high-frequency
record carries the window kind (0 = IRQ, 1 = NMI), so a non-digi tune's player
work (IRQ) and any NMI work are separable, and Python can bin on either.

## Output artifacts (`<prefix>.*`)

| file | content |
| --- | --- |
| `<prefix>.bin` | event stream, `EVENT_DTYPE` (v1, 16 B) — SID writes + interrupts + CPU vectors |
| `<prefix>.ramwr.bin` | RAM write log, `RAMACCESS_DTYPE` (16 B) |
| `<prefix>.ramrd.bin` | RAM read log, `RAMACCESS_DTYPE` (only when `--reads`) |
| `<prefix>.iord.bin` | I/O-area ($D000-$DFFF) read log, `RAMACCESS_DTYPE` (always emitted) |
| `<prefix>.iowr.bin` | I/O-area ($D000-$DFFF) write log, `RAMACCESS_DTYPE` (always emitted) |
| `<prefix>.cov.bin` | 8192-byte executed-PC bitmap (bit `pc` set if fetched in a play window) |
| `<prefix>.ram` | 65536-byte C64 RAM image, dumped after init settle |
| `<prefix>.json` | metadata sidecar (extended) |

### EVENT_DTYPE (16 B, little-endian) — unchanged layout, one new meaning

```
('cycle','<u8'), ('etype','u1'), ('chip','u1'),
('reg','u1'), ('value','u1'), ('addr','<u2'), ('aux','<u2')
```
- `SID_WRITE` (0): `chip`=SID index, `reg`=0..0x1f, `value`, `addr`=full SID
  address, **`aux`=store-site PC** (NEW — was 0 in v1).
- `CIA_IRQ` (1): `chip`=1 CIA1(IRQ)/2 CIA2(NMI), `addr`=timerA latch, `aux`=timerB latch.
- `VIC_IRQ` (2): `addr`=raster compare line, `aux`=current raster line.
- `CPU_VECTOR` (3): `value`=vector kind (0xfe IRQ/0xfa NMI/0xfc RST), `addr`=handler PC.

### RAMACCESS_DTYPE (16 B, little-endian)

```
('cycle','<u8'), ('pc','<u2'), ('addr','<u2'), ('value','u1'), ('kind','u1'), ('pad','<u2')
```
- `cycle`: absolute PHI2 cycle of the access.
- `pc`: `instrStartPC` of the storing/reading instruction (the generator code site).
- `addr`: C64 address written/read.
- `value`: byte written/read.
- `kind`: play-window kind — 0 = IRQ, 1 = NMI.

All of `.ramwr.bin`, `.ramrd.bin`, `.iord.bin` and `.iowr.bin` use this dtype.
I/O-area ($D000-$DFFF) reads and writes are logged to the
`.iord.bin`/`.iowr.bin` sidecars (SID writes additionally appear in
`<prefix>.bin`). The stack page (`$0100-$01FF`) is excluded from the RAM logs
unless `--stack` is given, in which case `$01xx` accesses flow into
`.ramwr.bin`/`.ramrd.bin` with their natural addresses. RAM writes to the code
region ARE included (self-modifying code).

## CLI (additions to `app/sidtrace.cpp`)

- `--reads` — also emit the RAM read log (off by default; large).
- `--stack` — include stack-page (`$01xx`) accesses in the RAM logs (off by
  default).
- `--no-ramwrites` / `--no-coverage` / `--no-ram` — disable individual v2 logs.
- `--window irq|nmi|both` — which interrupt opens a play window (default `both`).
- `--ram-dump-seconds S` — emulated time at which the `.ram` image is captured
  (default: at the first play window, i.e. just after init).
- existing v1 flags unchanged (`--seconds`, `--song`, `--out`, `--model`,
  `--sid`, `--frequency`, `--power-on-delay`, ROM paths).

## JSON sidecar additions

Add: `"artifacts"` (object mapping logical name -> filename for each emitted
file), `"ramwr_dtype"`/`"ramacc_fields"` (documented field order), `"window"`
(string), `"reads_enabled"` (bool), `"ram_dump_cycle"` (int), `"num_ram_writes"`,
`"num_ram_reads"`, `"coverage_count"` (number of distinct executed PCs). Keep all
v1 fields.

v3 (`"schema_version": 3`, all v2 keys kept) adds: `"stack_enabled"` (bool),
`"num_io_reads"`, `"num_io_writes"`, and always-present `"iord"`/`"iowr"`
entries in `"artifacts"`.

## Python API (preframr_playroutine)

`trace.py`:
- `RAMACCESS_DTYPE`, plus window-kind constants `WIN_IRQ=0`, `WIN_NMI=1`.
- `Trace.load(prefix)` also loads the v2 sidecars when present (absent files ->
  empty arrays / None), exposed as:
  - `Trace.ram_writes(kind=None)` -> RAMACCESS array (optionally filtered by window kind)
  - `Trace.ram_reads(kind=None)` -> RAMACCESS array
  - `Trace.io_reads(kind=None)` -> RAMACCESS array (I/O-area read log)
  - `Trace.io_writes(kind=None)` -> RAMACCESS array (I/O-area write log)
  - `Trace.coverage_pcs()` -> np.ndarray of executed PCs (uint16, sorted)
  - `Trace.ram_image()` -> np.ndarray(65536, uint8) or None
  - `Trace.sid_write_pc()` -> convenience: PC column (aux) of SID writes
- v1 API unchanged.

`recover.py` (new module) — headline functions (final signatures at
implementer's discretion, but these names/roles must exist and be tested):
- `state_sequence(trace, kind='auto', addrs=None)` -> object with `ticks`
  (uint64 per-frame boundary cycles), `addrs` (uint16 list of cells that ever
  change), and `grid` (uint8 `[n_frames, n_addrs]` carry-forward value of each
  cell at each frame). Binnable by `kind` in {'irq','nmi','both','auto'}.
- `fit_bacc(series)` -> dict|None: recover a bounded accumulator — `step`,
  `lo`, `hi`, `mode` ('saw'/'reflect'/'wrap'), `period` (flip period in frames),
  `phase`, and a `residual`/fit-quality. Operates on a 1-D per-frame series
  (8- or 16-bit; helper to combine lo/hi cells).
- `detect_table_walk(cursor_series, ram_image, value_series=None)` -> dict|None:
  recover `base`, `stride`, `length`, `loop` (loop-back marker handling), and
  the table bytes, by matching a cursor cell against the static image and/or the
  written value series.
- `classify_register(trace, sid_addr)` -> dict: one of
  `{'BACC','TABLE_WALK','SEQ','XSTATE','CONST'}` plus recovered params, for a
  given SID register address (e.g. 0xD400). Uses store-site PC grouping +
  stateseq + ram_image.
- `correlate_event_reset(trace, trigger_pred, cell_addr)` -> dict: quantify
  whether an event (e.g. a CTRL write — `trigger_pred(event)->bool`) co-occurs
  with a reset/discontinuity of `cell_addr` (the "control write triggers BACC
  reset — or not" question). Return correlation strength + lag.
- `voice_events(trace)` -> per-voice list of note/gate/instrument events derived
  from CTRL ($D404/$D40B/$D412) gate-bit transitions and correlated cell resets.
- `analyze(trace)` -> dict: full per-register generator map (calls
  `classify_register` for every written SID register), the headline entry point.

## Test fixtures (`tests/fixtures/tunes.json`)

A committed JSON list of 21 entries (NO .sid binaries committed). Schema per
entry:
```
{ "path": "MUSICIANS/G/Gop/Doctagop.sid",   // HVSC C64Music-relative
  "family": "DMC",                            // tracker family label
  "subtune": 1,                               // 1-based song to trace
  "md5": "....",                              // md5 of the .sid file (for songlengths + cache key)
  "seconds": 123.4 }                          // full song length for the subtune
```
21 entries: 3 each for DMC, GoatTracker2, MusicAssembler, FutureComposer,
JCH_NewPlayer, Soundmonitor, plus 3 defMON. Complex, feature-rich, non-digi.

The fixture downloads each tune on demand from
`${HVSC_BASE_URL:-https://hvsc.c64.org/download/C64Music}/<path>` into a cache
dir (`${PREFRAMR_HVSC_CACHE:-<repo>/tests/.hvsc_cache}`), verifying the md5; if a
local mirror `${HVSC_ROOT:-/scratch/hvsc/C64Music}/<path>` exists it is copied
instead of downloaded. Cache dir is gitignored.

Real-tune tests render the WHOLE song (`seconds` from the entry) and run under
pytest-xdist (`-n auto`); they must be independent/parallel-safe.
