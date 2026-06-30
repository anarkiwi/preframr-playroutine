# preframr-playroutine

A **universal tracker playroutine** for the Commodore 64 SID, built on
[libsidplayfp](https://github.com/libsidplayfp/libsidplayfp).

`.sid` files are complete, idiosyncratic 6502 machine-language programs — every
composer's player is different. This project plays each one exactly as
`sidplayfp` would (cycle-accurate CPU / CIA / VIC-II / SID emulation), observes
every bit of the running machine, and **decompiles the song into a common
form**: each SID register's per-frame behaviour expressed in a small set of
shared primitives — `CONST`, `SEQ` (sequencer-latched), `BACC` (bounded
accumulator: vibrato / sweep), `TABLE_WALK` (table + cursor), and `COMPOSITE`
(base + modulation + overrides) — plus the song's **periodic update structure**
(PAL/NTSC raster vs CIA-timer, single/multi-speed) and **global tuning**. A
single Python playroutine can then replay any song from that common IR using
the shared primitives, independent of the original player.

The correctness criterion is **lossless round-trip**: regenerate each register
from its recovered IR and compare byte-for-byte to the oracle. `round_trip == 1.0`
means the decompilation reproduces the original player exactly. Because the
emulator logs every bit, every output is a function of observable state — so a
register that doesn't yet round-trip is a *not-yet-modelled* primitive (a
fixable gap that `round_trip` localises), never an unrecoverable one.

The emulator also runs arbitrary code (unpacking, relocation, generative melody
code); that's the *song data*, not the *generators*. We recover the per-register
generator functions and replay the song data (note/sequencer events) through
them.

## Current status

Built and merged (PRs #2/#7/#8/#9/#10/#11/#12/#13/#15/#16/#18):

- **Oracle + instrumentation.** Deterministic, byte-exact tracer (`sidtrace`) on
  patched libsidplayfp: cycle-stamped SID writes (PC-tagged), CIA/VIC interrupts,
  CPU vectors, play-window-scoped RAM read/write logs, PC coverage, RAM image.
- **Recovery primitives.** `CONST`, `SEQ`, `BACC`, `TABLE_WALK`, `COMPOSITE`,
  `PITCHWALK`, `FEEDER` (a global filter register `$D415`–`$D418` recovered as an
  exact latched copy of its captured RAM feeder cell), `XOR` (a CTRL register
  recovered as `cellA XOR cellB` — the gate/test/waveform "base XOR eor" idiom
  defMON and similar players use to toggle control bits), plus the note/pitch
  layer (`recover_tuning`, `voice_detune`). The CTRL waveform table-walk also
  tolerates up to three suppressed off-table command bytes (strict-improvement
  guarded), and the 16-bit `COMPOSITE` keeps its additive-modulation term only
  when it strictly improves reconstruction — so an output-then-compute player's
  operand cell (which already carries the whole value, one call late) is
  recovered base-only instead of being polluted by a spurious phase-residual
  modulation cell. The `BACC` accumulator also has a **ping-pong** (clamp-and-flip)
  mode: unlike the mirror-reflect triangle, an overshoot saturates to a fixed
  boundary clamp and reverses (with independent up/down step magnitudes), the
  defMON PW sweep idiom (`$1474`: clamp + `eor #$80` flip) — recovered per-note
  (`segmented_pingpong`, each note seeding its own rate) and chosen by whichever
  candidate best reconstructs the register's own series. A **tick-banded** BACC
  mode (`segmented_tickband`) further generalizes the stride from a scalar to
  `step = rate_table[tick]`: the FutureComposer PW sweep, whose per-frame rate is
  a step-function of the per-voice tick (frames since note-on, the FC
  sawtooth-reset pinning invariant). The per-note rate vectors de-duplicate into
  a small shared table set (8 tables explain 1031 note segments) — proof it's a
  reused program, not memorization — and the candidate is gated to never steal a
  clean constant-step fit.
- **Global tuning / absolute notes.** `recover_tuning` emits absolute MIDI
  `note_numbers` / `note_range` (so the IR's notes are absolute pitch, comparable
  across tunes) and fits the reference A4 from the recovered note→frequency table
  when a `PITCHWALK` ladder is available (`source: note_table`, cleaner than the
  sparse live frequencies), falling back to the live fit (`source: live_freq`).
- **Lossless gate.** `reconstruct_register` + `round_trip` regenerate each
  register from its IR and score it byte-for-byte against the oracle.
  Reconstruction holds each register's **power-on default** (`0`) until its
  first write, mirroring the oracle's pre-first-write frames across every
  primitive.
- **26 HVSC fixtures** (3× the top-6 trackers + 4 defMON, incl. Goto80's
  *Automatas* — the canonical defMON reverse-engineering reference tune — + A
  Mind Is Born + Commando/Grid Runner/Cauldron II), whole-song, parallel, with an
  **xfail ratchet**: every fixture must round-trip perfectly with zero XSTATE; the
  not-yet-perfect ones are `xfail(strict=True)`, so the gap can only shrink (a
  fix XPASS-fails CI until its marker is removed).
- **Analysis performance.** `_table_walk_scan` (the no-read-log table-walk
  recovery, dominant on long JCH tunes) is vectorized ~2×, byte-for-byte
  identical (frozen-reference parity test), keeping CI time bounded as fixtures
  grow.

**Round-trip landscape** (whole-song): 6 perfect (Doctagop, Only_3, Denarius,
Tom_Tom, 24th_Amaranth, Hawkeye); most tunes 0.96–1.0; weakest are Commando 0.77
(Hubbard's hand-coded player) and the FutureComposer FREQ cases. The
defMON tunes rose to ~0.92–0.96 once the `XOR` CTRL primitive recovered the
`base XOR eor` gate sequencer (`$D404`/`$D40B`/`$D412`) and the `COMPOSITE`
mod-guard stopped a phase-residual cell from polluting the FREQ operands. The
filter-cutoff stragglers (`FEEDER`) and the pre-first-write default are also
recovered; the defMON PW **ping-pong** sweep (Vacuole 0.94→0.98) and the
FutureComposer **tick-banded** PW sweep (Hawkeye → perfect, its sole gap closed)
are now recovered `BACC` modes. The remaining gaps are the bespoke players, the
FutureComposer / defMON FREQ slide (vibrato/portamento), and the global
filter-cutoff BACC.

## Next steps

1. **Drive the rest of the xfail set to zero.** Cross-referenced against the
   byte-exact per-register RE (`re-trackers/*-generators.md`), the remaining gaps
   are **not** scattered one-offs: they cluster onto three BACC generalizations,
   each violating one of the current model's two structural assumptions —
   `_simulate_reflect` mirrors the overshoot (`hi-(nv-hi)`), and `fit_bacc` /
   `segmented_bacc` assume a single scalar step. In leverage order:
   - **defMON PW ping-pong → BACC clamp-and-flip mode** — ✅ **done** (#22): the
     `pingpong` mode (saturate to a fixed boundary clamp + reverse, independent
     up/down steps) recovers the defMON PW sweep, per-note-segmented
     (`segmented_pingpong`); Vacuole 0.94→0.98, no regression on the 5 perfect
     tunes. The other defMON FREQ-slide / filter-cutoff gaps below still apply.
   - **FutureComposer PW → BACC with table-indexed stride** — ✅ **done** (#24):
     `segmented_tickband` generalizes the stride to `step = rate_table[tick]`,
     the tick synthesized from note-on resets (the FC pinning invariant), the
     per-note rate vectors de-duplicated into a small shared table set. Recovered
     Hawkeye's PW `$D402` 0.856→1.0 (its sole gap → **new perfect tune**), MU PW
     0.96→0.997; no regression. The same tick foundation still needs extending to
     the FC **FREQ slide** (vibrato/portamento) and the CTRL-wave / filter-segment
     lanes (`step = table[tick]` paced by `$1942`), the dominant remaining FC gaps.
   - **defMON + FC `$D416` cutoff → segmented BACC with SEQ-directed sign +
     post-scale**. RE (`$10b6/$10be`, opcode at `$10b8`): direction is the
     `ADC`/`SBC` **opcode patched per instrument step** (observable in the
     SMC-immediate log), then an `ASL` (×2 on 6581 / `NOP` on 8580) before
     `$D416`. Let `segmented_bacc` carry a per-segment sign and `reconstruct_register`
     honor a scalar post-scale.

   Each gap names the exact accumulator/table/opcode cell, so a new primitive is
   **unit-testable against ground truth**, not just round-trip-scored. The
   residual bespoke cases come last, sharing no machinery: the FutureComposer
   vibrato/portamento (recover depth/rate from the already-logged SMC immediate
   writes — `$12F7` — before adding any observable) and Commando 0.77 (Hubbard's
   reflected triangle off the **global** frame counter plus an octave-toggle mux —
   a different reset axis).
2. **Absolute octave/semitone label.** `recover_tuning` now emits absolute note
   numbers and fits the reference from the recovered note table; the residual is
   the integer octave/semitone label, which is per-player-convention dependent
   (frequency pins tuning only modulo one semitone). Wire a per-player
   note-table base where the convention is known so A4 is fully absolute.
3. **6502 reference replayer** (assemble the IR with the in-build `xa65`, run it
   under the emulator, round-trip against the oracle) — proves on-platform
   practicality.

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for detail.

**Scope of generative tunes.** Some songs synthesise their melody
algorithmically (e.g. *A Mind Is Born* by lft). We deliberately do **not** model
the generating algorithm — only its **melodic output as notes**. The IR's note
layer is the realised note-event stream (captured data); the modulation /
generator layer is the recovered primitives. So a generative tune round-trips by
replaying its observed note stream through recovered generators, not by
re-deriving the melody.

## Global tuning

SID frequency registers are not notes: a note's pitch is `f_Hz = sidfreq ·
cpu_hz / 2^24`, and each player's note→frequency table is calibrated to its own
reference. Recovered across the fixture set, most tunes sit at A4 ≈ 440 Hz
(clean 12-TET, sub-cent residual), but a cluster (several DMC / Future Composer
/ Music Assembler tunes) is a consistent ~35 cents off the A440 grid — about a
third of a semitone — so their notes do **not** align with the A440 tunes
(likely an NTSC-tuned table played at the PAL clock). So the common form carries
a per-song tuning descriptor — the sub-semitone offset from A440 and the
temperament — so that a NOTE in the IR is an absolute pitch comparable across
every tune. (Frequency alone pins tuning only modulo one semitone; which grid
point is "A" comes from the note table.)

## Program-state instrumentation (generator recovery)

The register oracle says *what* a tune wrote; recovering the per-register
**generators** (bounded accumulators, sweeps, table-walks, and their
cross-dependencies — the BACC / TABLE-WALK / SEQ / XSTATE taxonomy) needs the
internal program state that produced those writes. While the CPU runs inside an
interrupt handler (a "play window"), `sidtrace` additionally captures, all
cycle-stamped and tagged with the window kind (IRQ/NMI) so frames can be binned
flexibly:

- the **store-site PC** of every SID write (which code wrote which register),
- a **RAM-write log** — accumulators, table cursors, counters, and
  self-modifying-code immediates (the raw per-frame `stateseq`),
- an **executed-PC coverage** bitmap (the player's code, for disassembly),
- a **64K RAM image** (relocated player code + static tables),
- optionally a **RAM-read log** (`--reads`) for direct table-walk capture.

The `preframr_playroutine.recover` module turns these into a per-frame state
sequence, derives voice/note events, and classifies each SID register's
generator (`fit_bacc`, `detect_table_walk`, `correlate_event_reset`,
`analyze`). The captured cells and store-site PCs match published manual
reverse-engineering of real players exactly (e.g. the DMC freq accumulator
`$1735/$1738`, PW `$1750`, wave cursor `$177a`, and the FREQ/PW/CTRL store sites
`$160d/$161c/$162b`). See [`docs/INSTRUMENTATION.md`](docs/INSTRUMENTATION.md)
for the full binary contract.

## Components

| Path | What |
| --- | --- |
| `patches/instrument.patch` | Hooks added to a pinned libsidplayfp: SID-register writes (PC-tagged), CIA/VIC interrupt-line assertions, CPU interrupt vectors, and play-window-scoped RAM access + PC coverage, all cycle-stamped via the event scheduler. |
| `app/sidtrace.cpp` | CLI that plays a tune and writes the oracle + program-state artifacts (`.bin`, `.ramwr.bin`, `.cov.bin`, `.ram`, `.json`). |
| `preframr_playroutine/` | numpy package: `trace` (load the artifacts) and `recover` (generator recovery). |
| `docs/INSTRUMENTATION.md` | The authoritative binary/format contract between the tracer and the python tooling. |
| `Dockerfile` | Multi-stage build: reSIDfp + instrumented libsidplayfp + `sidtrace`, then a python test image. |

The C++ tracer and the libsidplayfp patch are derivative of libsidplayfp and
are licensed **GPL-2.0-or-later**. The python tooling under
`preframr_playroutine/` is independent and licensed under Apache-2.0 (see
`LICENSE`).

## Build & test

```sh
docker build -t preframr-playroutine .
docker run --rm preframr-playroutine            # runs the full pytest suite
```

The suite includes an end-to-end oracle test that assembles tiny original PSID
tunes (one VBI/raster, one CIA-timer) and asserts the recovered structure.

## Trace a real tune

```sh
docker run --rm -v /path/to/hvsc:/sids -v "$PWD/out:/out" \
  --entrypoint sidtrace preframr-playroutine \
  --seconds 30 --out /out/song /sids/Some_Tune.sid
```

This writes `out/song.bin` and `out/song.json`.

## Oracle format

`song.bin` is a flat array of fixed 16-byte little-endian records, directly
loadable with numpy:

```python
import numpy as np
from preframr_playroutine import EVENT_DTYPE, Trace

trace = Trace.load("out/song")          # reads .bin + .json
print(trace.classify())                  # driver, speed, interrupt sources, ...
ticks, frames = trace.register_frames()  # (n, 32) per-play register snapshots
```

Record fields (`preframr_playroutine.EVENT_DTYPE`):

| field | meaning |
| --- | --- |
| `cycle` (u8) | absolute event-scheduler cycle (single monotonic axis) |
| `etype` (u1) | `0` SID write, `1` CIA IRQ/NMI, `2` VIC raster IRQ, `3` CPU vector |
| `chip` (u1) | SID index, or interrupt source (`1` CIA1/IRQ, `2` CIA2/NMI, `3` VIC) |
| `reg` (u1) | SID register `0..0x1f` (writes) |
| `value` (u1) | SID value (writes); CPU vector kind `0xfe`/`0xfa`/`0xfc` (IRQ/NMI/RST) |
| `addr` (u2) | full SID address (writes); CIA timer-A latch; VIC raster compare; handler PC |
| `aux` (u2) | CIA timer-B latch; VIC current raster line |

## Determinism

The oracle is byte-exact reproducible: tracing the same tune twice yields
identical `.bin` output (the `test_determinism` test enforces this). Three
things make a libsidplayfp run deterministic, and `sidtrace` pins all of them:

- **Power-on delay.** libsidplayfp's default `powerOnDelay`
  (`DEFAULT_POWER_ON_DELAY > MAX_POWER_ON_DELAY`) draws the warm-up delay from a
  **wall-clock-time-seeded** RNG (`sidrandom`, seeded `std::time(nullptr)`),
  shifting the whole cycle timeline each run. `sidtrace` forces a fixed
  `powerOnDelay` (default 0, `--power-on-delay N` to override), which skips the
  random path entirely — the only place that RNG is ever used.
- **SID noise LFSR.** The reSIDfp engine resets the noise shift register to a
  fixed `0x7fffff` in `WaveformGenerator::reset()` — no time/random seed — so
  noise and any osc3/env3 reads a player makes are reproducible.
- **Power-on RAM.** The C64 RAM pattern comes from libsidplayfp's committed
  `poweron.bin`, not a random fill.

We use the reSIDfp builder (built from `libresidfp`) rather than the bundled
`sidlite` fallback so the emulation matches the high-quality engine `sidplayfp`
uses by default.

## libsidplayfp pin

The patch is generated against libsidplayfp commit
`47766e4cef3f835a3d17dac574f44831088010d4` (see `Dockerfile`
`LIBSIDPLAYFP_REF`). To re-pin, update the ref and regenerate
`patches/instrument.patch`.
