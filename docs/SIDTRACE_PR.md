# Tracer PR: complete the observability contract (all sidtrace work, one PR)

Standalone, mechanically-executable instructions for ONE pull request that
lands ALL tracer-side (libsidplayfp patch + `app/sidtrace.cpp` + `trace.py`
loader) work the recovery roadmap needs, so every later phase of
`docs/GENERIC_RECOVERY.md` is python-only in `preframr_playroutine/`. No
research or design decisions are required; every hook edit, record format,
flag, and test is specified here. Open a PR, watch CI, fix failures, merge on
green.

## Mission

Close the two observability gaps identified in `GENERIC_RECOVERY.md` §3.5
(the recovery-totality argument) without changing any existing artifact byte:

1. **I/O reads** (`$D000–$DFFF`) inside play windows — SID osc3/env3
   (`$D41B/1C`), CIA timers/ports, VIC registers — currently dropped by the
   read hook. Needed by the `chip` IR node and the Phase 7 lifter.
2. **I/O writes** (`$D000–$DFFF`) inside play windows — CIA reprogramming
   (cadence-as-state, e.g. Soundmonitor per-section timer reloads), VIC
   raster-compare changes — currently dropped by the write hook (only SID
   writes reach the oracle).
3. **Stack-page accesses** (`$0100–$01FF`) — opt-in via `--stack`, for
   dynamic dataflow through PHA/PLA if static tracking ever needs ground
   truth.

## Scope

IN: `patches/instrument.patch` (regenerated), `app/sidtrace.cpp`,
`preframr_playroutine/trace.py`, `docs/INSTRUMENTATION.md`, `tests/_minisid.py`,
`tests/test_oracle.py`, one determinism-comparison extension in
`tests/test_hvsc.py`.

OUT (do not do): any change to `recover.py`, `render.py`, or recovery
semantics; no new recovery features; no Dockerfile changes (the pinned refs
stay; the patch is regenerated against the same
`LIBSIDPLAYFP_REF=47766e4cef3f835a3d17dac574f44831088010d4`); no CI workflow
changes. If something outside this list seems needed, stop and report.

## Hard invariants

- **Existing artifacts are byte-identical by default.** `.bin`, `.ramwr.bin`,
  `.ramrd.bin`, `.cov.bin`, `.ram` produced by a default-flag run must be
  byte-for-byte what the current tracer produces (the new information goes to
  NEW files; stack records appear only under the new `--stack` flag).
- Determinism: two identical runs produce byte-identical output for EVERY
  file, including the new ones.
- `black`/`pylint`/`pytest -q` clean locally (`pip install -e ".[test]"`);
  both CI jobs green (`lint-and-unit`, and `docker-oracle` which builds the
  patched library and runs the full suite including whole-song HVSC tunes).
- All new records use the existing 16-byte `RAMACCESS_DTYPE` — no new dtypes.

## Step-by-step (commit after each step)

### Step 1 — regenerate the patch with routed hooks

Work in `/scratch/tmp/sidtrace-pr` (NOT /tmp — project convention):

```sh
mkdir -p /scratch/tmp/sidtrace-pr && cd /scratch/tmp/sidtrace-pr
git clone https://github.com/libsidplayfp/libsidplayfp
cd libsidplayfp
git checkout 47766e4cef3f835a3d17dac574f44831088010d4
git apply --whitespace=nowarn <repo>/patches/instrument.patch
# ... make the three edits below ...
git add -A
git diff --cached > <repo>/patches/instrument.patch
```

**Edit 1 — `src/instrument.h`:** add to `class InstrumentSink`, after
`cpuExec`:

```cpp
    /// A CPU data read from the I/O area ($D000-$DFFF) inside a play window
    /// (SID osc3/env3, CIA timers/ports, VIC registers). Always emitted when
    /// a sink is installed (not gated by wantReads(); volume is tiny).
    virtual void ioRead(int64_t cycle, uint16_t pc, uint16_t addr,
                        uint8_t value, uint8_t kind) {}

    /// A CPU write to the I/O area inside a play window. SID writes also
    /// appear as sidWrite events; consumers filter as needed.
    virtual void ioWrite(int64_t cycle, uint16_t pc, uint16_t addr,
                         uint8_t value, uint8_t kind) {}

    /// Whether stack-page ($0100-$01FF) accesses should flow into the RAM
    /// write/read logs (dynamic PHA/PLA dataflow ground truth).
    virtual bool wantStack() const { return false; }
```

**Edit 2 — `src/c64/CPU/mos6510.cpp`,** in `FetchEffAddrDataByte()`: replace
the existing instrumentation block (the one testing
`(Cycle_EffectiveAddress & 0xff00) != 0x0100 && (… & 0xf000) != 0xd000`) with
routing:

```cpp
    if (m_instrInPlay) UNLIKELY
    {
        InstrumentSink *sink = getInstrumentSink();
        if (sink != nullptr)
        {
            if ((Cycle_EffectiveAddress & 0xf000) == 0xd000)
                sink->ioRead(eventScheduler.getTime(EVENT_CLOCK_PHI2),
                             m_instrCurPC, Cycle_EffectiveAddress, Cycle_Data,
                             m_instrWindowKind);
            else if (((Cycle_EffectiveAddress & 0xff00) != 0x0100
                      || sink->wantStack()) && sink->wantReads())
                sink->ramRead(eventScheduler.getTime(EVENT_CLOCK_PHI2),
                              m_instrCurPC, Cycle_EffectiveAddress, Cycle_Data,
                              m_instrWindowKind);
        }
    }
```

**Edit 3 — `src/c64/CPU/mos6510.h`,** in `cpuWrite()`: replace the existing
exclusion block the same way:

```cpp
    inline void cpuWrite(uint_least16_t addr, uint8_t data)
    {
        if (m_instrInPlay) UNLIKELY
        {
            InstrumentSink *sink = getInstrumentSink();
            if (sink != nullptr)
            {
                if ((addr & 0xf000) == 0xd000)
                    sink->ioWrite(eventScheduler.getTime(EVENT_CLOCK_PHI2),
                                  m_instrCurPC, addr, data, m_instrWindowKind);
                else if ((addr & 0xff00) != 0x0100 || sink->wantStack())
                    sink->ramWrite(eventScheduler.getTime(EVENT_CLOCK_PHI2),
                                   m_instrCurPC, addr, data, m_instrWindowKind);
            }
        }
        dataBus.cpuWrite(addr, data);
    }
```

Note the hook SITES are unchanged — only the drop conditions become routing.
Semantics preserved exactly for RAM: non-stack non-I/O writes always logged;
non-stack non-I/O reads logged when `wantReads()`.

Commit (`tracer: route I/O and optional stack accesses to the sink`).

### Step 2 — `app/sidtrace.cpp`

- `Options`: add `bool stack = false;` and parse `--stack`. Add to `usage()`:
  `"  --stack             include stack-page ($01xx) accesses in the RAM logs\n"`.
- Output paths: `iordPath = prefix + ".iord.bin"`,
  `iowrPath = prefix + ".iowr.bin"`; open both ALWAYS (like `bin`).
- `TraceSink`: constructor gains `FILE *iord, FILE *iowr, bool wantStack`;
  add buffers `m_iordbuf`/`m_iowrbuf`, counters `m_nior`/`m_niow`, and:

```cpp
    void ioRead(int64_t cycle, uint16_t pc, uint16_t addr, uint8_t value, uint8_t kind) override {
        if (!windowWanted(kind) || m_iord == nullptr) return;
        m_iordbuf.push_back({static_cast<uint64_t>(cycle), pc, addr, value, kind, 0});
        ++m_nior;
        if (m_iordbuf.size() >= BUF) flushBuf(m_iordbuf, m_iord);
    }
    void ioWrite(int64_t cycle, uint16_t pc, uint16_t addr, uint8_t value, uint8_t kind) override {
        if (!windowWanted(kind) || m_iowr == nullptr) return;
        m_iowrbuf.push_back({static_cast<uint64_t>(cycle), pc, addr, value, kind, 0});
        ++m_niow;
        if (m_iowrbuf.size() >= BUF) flushBuf(m_iowrbuf, m_iowr);
    }
    bool wantStack() const override { return m_wantStack; }
```

  plus `flush()` flushes the two new buffers, and `iorCount()`/`iowCount()`
  accessors. Close both files with the others.
- JSON sidecar: bump `"schema_version"` to `3` (all v2 keys kept); add
  `"stack_enabled": true|false`, `"num_io_reads": N`, `"num_io_writes": N`;
  add `"iord"`/`"iowr"` entries to `"artifacts"` (always present).
- Header comment block: add the two new files to the artifact list.

Commit (`sidtrace: emit .iord.bin/.iowr.bin, --stack flag, schema v3`).

### Step 3 — `preframr_playroutine/trace.py`

- `Trace.__init__`: new optional params `iord=None`, `iowr=None`, stored as
  `self._iord`/`self._iowr` (empty `RAMACCESS_DTYPE` array when None) —
  mirror `_ramrd` exactly.
- `Trace.load`: also load `base + ".iord.bin"` / `base + ".iowr.bin"` via the
  existing `_load_ramacc` (absent → empty).
- `Trace.from_events`: accepts the new kwargs (already generic if it forwards
  `**kwargs`; verify).
- New accessors, mirroring `ram_reads` verbatim:

```python
    def io_reads(self, kind: int | None = None) -> np.ndarray:
        """I/O-area ($D000-$DFFF) read log, optionally filtered by window kind."""

    def io_writes(self, kind: int | None = None) -> np.ndarray:
        """I/O-area ($D000-$DFFF) write log, optionally filtered by window kind."""
```

Commit (`trace: load io_reads/io_writes artifacts`).

### Step 4 — `docs/INSTRUMENTATION.md`

- Files table: add `<prefix>.iord.bin` / `<prefix>.iowr.bin` rows
  (RAMACCESS_DTYPE, always emitted).
- Replace the exclusion sentences ("The stack page … is excluded … other IO
  ($D000-$DFFF) is excluded") with: I/O-area reads and writes are logged to
  the `.iord.bin`/`.iowr.bin` sidecars (SID writes additionally appear in
  `<prefix>.bin`); the stack page is excluded from the RAM logs unless
  `--stack` is given, in which case `$01xx` accesses flow into
  `.ramwr.bin`/`.ramrd.bin` with their natural addresses.
- CLI section: add `--stack`.
- JSON section: `schema_version` 3, `stack_enabled`, `num_io_reads`,
  `num_io_writes`, `iord`/`iowr` artifacts.
- Python API section: `Trace.io_reads(kind=None)` / `Trace.io_writes(kind=None)`.

Commit (`docs: instrumentation contract v3`).

### Step 5 — tests

**5a. `tests/_minisid.py` — I/O-probe player.** Add a second hand-assembled
player + builder `build_ioprobe_psid()` (same PSID header code path,
parameterize `INIT`/`PLAY`/`CODE`; keep `build_psid` byte-identical):

```
1000  A9 FF     init: LDA #$FF
1002  8D 0E D4        STA $D40E     ; voice 3 freq lo = $FF
1005  8D 0F D4        STA $D40F     ; voice 3 freq hi = $FF
1008  A9 80           LDA #$80
100A  8D 12 D4        STA $D412     ; voice 3 ctrl = noise
100D  60              RTS
100E  AD 1B D4  play: LDA $D41B     ; osc3 readback (I/O read)
1011  8D 02 D4        STA $D402     ; -> voice 1 PW lo
1014  AD 04 DC        LDA $DC04     ; CIA1 timer A lo (I/O read)
1017  8D 03 D4        STA $D403     ; -> voice 1 PW hi
101A  A9 0F           LDA #$0F
101C  8D 18 D4        STA $D418
101F  EE 20 D0        INC $D020     ; VIC border: I/O read + I/O write
1022  60              RTS
```

(`CODE = bytes([0xA9,0xFF,0x8D,0x0E,0xD4,0x8D,0x0F,0xD4,0xA9,0x80,0x8D,0x12,
0xD4,0x60,0xAD,0x1B,0xD4,0x8D,0x02,0xD4,0xAD,0x04,0xDC,0x8D,0x03,0xD4,0xA9,
0x0F,0x8D,0x18,0xD4,0xEE,0x20,0xD0,0x60])`, `INIT=0x1000`, `PLAY=0x100E`,
speed=0 VBI.)

**5b. `tests/test_oracle.py` additions** (same skip-if-no-sidtrace marker):

- `test_io_probe`: trace the ioprobe tune 2s; assert `trace.io_reads()`
  non-empty and its `addr` set contains `0xD41B`, `0xDC04`, `0xD020`; assert
  ≥ 2 distinct `value`s among the `0xD41B` reads (the noise oscillator moves);
  assert `trace.io_writes()["addr"]` contains `0xD020` and `0xD402`; assert
  the SID writes to `$D402` in the oracle equal the `$D41B` read value of the
  same frame's window (pair by nearest preceding `io_reads` cycle — this is
  the osc3→PW copy, the chip-node ground truth).
- `test_io_logs_deterministic`: run the ioprobe trace twice into different
  prefixes; byte-compare ALL emitted files (`.bin`, `.ramwr.bin`, `.cov.bin`,
  `.ram`, `.iord.bin`, `.iowr.bin`, and `.json` except the `"bin"`/path-bearing
  fields — simplest: compare the binary files byte-exact and skip the json).
- `test_ram_log_purity`: on the existing counter tune (default flags), assert
  `ram_writes()` contains NO records with `(addr & 0xf000) == 0xd000` and
  none with `(addr & 0xff00) == 0x0100` — the byte-identity contract.
- `test_stack_flag`: trace the counter tune once with `--stack` and once
  without; with the flag, `ram_writes()` contains `$01xx` records (the
  driver's JSR pushes inside the window); without, none. The no-flag run's
  `.ramwr.bin` must be byte-identical to a default run.
- Loader unit test (no binary needed, runs in `lint-and-unit`): write
  synthetic `RAMACCESS_DTYPE` bytes to `tmp_path / "x.iord.bin"` (plus
  minimal `.bin`), `Trace.load` → `io_reads()` returns them; absent files →
  empty arrays.

**5c. `tests/test_hvsc.py`:** in the existing determinism comparison inside
`test_real_tune_anchors` (which byte-compares `events` and `ram_writes`
between two renders), add the same `np.array_equal(...view(np.uint8)...)`
comparison for `io_reads()` and `io_writes()`. Touch nothing else in that
file.

Commit (`tests: I/O probe minisid, io-log determinism, purity + stack flag`).

### Step 6 — build, verify, PR

- Local verify without docker: `pip install -e ".[test]" && black --check
  preframr_playroutine tests && pylint preframr_playroutine && pytest -q`
  (oracle tests skip without the binary; loader unit test still runs).
- Full verify: `docker build -t preframr-playroutine:dev .` (this applies the
  regenerated patch against the pinned ref and compiles — the real gate for
  Step 1) then `docker run --rm -e
  HVSC_BASE_URL=https://hvsc.c64.org/download/C64Music
  preframr-playroutine:dev -q`. If the build fails on patch application,
  regenerate per Step 1 — do not hand-edit hunk headers.
- Branch, push, `gh pr create --fill`, watch `gh pr checks --watch`, fix, and
  merge on green (squash). Do not merge with any red check; do not skip or
  xfail a failing test to get green.

## Pitfalls

- The patch is consumed by `git apply` against the EXACT pinned ref; always
  regenerate via the Step 1 clone/apply/edit/`git diff --cached` flow. Never
  edit the `.patch` text directly.
- Keep the hook edits to the two existing sites only. Do not add new hook
  sites (other CPU cycle functions) — the read hook covers the data-fetch
  cycle the current read log already uses, which is what recovery consumes.
- `INC $D020` is a read-modify-write: it produces BOTH an ioRead and an
  ioWrite (and a dummy write) — the probe test relies on the read+write pair;
  do not "deduplicate".
- `windowWanted(kind)` gating applies to the new logs exactly as to the old
  ones (`--window` filters all logs consistently).
- Buffers: copy the `flushBuf` pattern; forgetting to flush the new buffers in
  `flush()` truncates the logs and fails the probe test only sometimes —
  flush both unconditionally.
- `tests/_minisid.py`: `build_psid` must stay byte-identical (the oracle tests
  and `test_determinism` depend on it); add the probe as a separate builder
  sharing the header code via a parameter, and `assert len(header) ==
  data_offset` still holds.
- Coverage ≥ 85% overall: the loader accessors are covered by the synthetic
  unit test; keep them tiny mirrors of `ram_reads`.
- If `tests/test_hvsc.py` conflicts with the IR-foundation PR
  (`docs/IR_FOUNDATION_PR.md` also touches it), rebase — the edits are in
  different functions; otherwise the two PRs touch disjoint files and may land
  in either order.

## Definition of done

PR merged to main with: regenerated patch (routed hooks, sink additions);
sidtrace emitting `.iord.bin`/`.iowr.bin` always and honoring `--stack`;
schema v3 sidecar; `Trace.io_reads()`/`io_writes()`; INSTRUMENTATION.md
updated; probe/determinism/purity/stack tests green in the docker job; all
pre-existing artifacts byte-identical under default flags. After this PR, all
remaining roadmap work (`GENERIC_RECOVERY.md` Phases 0–10) touches only
python in `preframr_playroutine/` and `tests/` — any further tracer change is
a bug fix, not planned work.
