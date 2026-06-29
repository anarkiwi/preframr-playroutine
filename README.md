# preframr-playroutine

Byte-exact, cycle-stamped SID instrumentation built on
[libsidplayfp](https://github.com/libsidplayfp/libsidplayfp).

`.sid` files are complete 6502 machine-language programs that perform music on
a Commodore 64. This project plays them exactly as `sidplayfp` would — same
cycle-accurate emulation of the CPU, CIA timers, VIC-II and SID — and emits a
deterministic **oracle**: every value written to a SID chip and every IRQ/NMI
event, each stamped with the absolute machine cycle. From that trace it
recovers the *periodic update structure* of a tune (PAL/NTSC raster-driven vs
CIA-timer driven, single- or multi-speed) and the per-frame register tables
that table / sweep / arpeggio (BACC-style) players step through.

The emulator runs arbitrary code (unpacking, relocation, generative melody
code); we don't try to decompile that. We capture what the program *does* to
the hardware, periodically, so the structure can be recovered downstream.

## Components

| Path | What |
| --- | --- |
| `patches/instrument.patch` | Minimal hooks added to a pinned libsidplayfp: SID register writes, CIA/VIC interrupt-line assertions, and CPU interrupt vector-throughs, all cycle-stamped via the event scheduler. |
| `app/sidtrace.cpp` | CLI that plays a tune and writes the oracle (`.bin`) + metadata (`.json`). |
| `preframr_playroutine/` | numpy package that loads the oracle and recovers timing + register frames. |
| `Dockerfile` | Multi-stage build: instrumented libsidplayfp + `sidtrace`, then a python test image. |

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
