# Roadmap

The goal is a **universal tracker playroutine**: losslessly decompile any SID
song into a common primitive IR (CONST / SEQ / BACC / TABLE_WALK / COMPOSITE +
per-song tuning) that a single replayer can play back. `round_trip == 1.0` is the
correctness gate; a register that does not yet round-trip is a *not-yet-modelled*
primitive (a fixable gap `round_trip` localises), never an unrecoverable one.

## Near-term

- **Drive the XSTATE / round-trip gap to zero.** Every fixture asserts perfect
  round-trip + zero XSTATE; currently-imperfect tunes are `xfail(strict=True)` so
  each fix flips a tune to XPASS and forces removing its marker. Biggest current
  causes: FutureComposer FREQ (tick-driven pitch-table walk), defMON, and the
  filter-cutoff / CTRL stragglers.
- **Global tuning (`recover_tuning`).** Per-song reference A4 (Hz),
  cents-from-A440, and temperament, fit from the recovered note→frequency table
  (cleaner than live freq). Makes the IR's note numbers absolute pitch so NOTES
  are comparable across tunes. Recovered offline; **zero runtime cost** (the
  note→freq table *is* the tuning). Add fixtures: Commando (Hubbard, own player),
  Grid Runner (Jammer, GoatTracker V2), Cauldron II Remix (Linus, GoatTracker V2
  — uses a **detune** effect: two voices on the same note with a small constant
  frequency-space offset; must recover as same-note + per-voice detune, not as a
  note difference, and stay round-trip-exact).

## Reference C64 player (future)

Prove the IR is practical on the original platform, not just by estimate:

- Implement a small **6502 reference replayer** for the IR (assembled with the
  `xa65` already in the Docker build), interpreting the common primitives over a
  per-tune IR data blob.
- Run it under the libsidplayfp emulator (the same one the tracer uses) and
  **round-trip its SID writes against the oracle** — on-platform losslessness.
- The primitives are integer table-lookups + bounded accumulators + latched
  writes (no runtime multiply/float; tuning baked into the note→freq table), so a
  single-speed update fits comfortably in a PAL frame; for the tightest
  multispeed, add an IR→6502 compile step (the form the original players already
  take).

## Known not-yet-modelled gaps (fixable; no missing instrumentation assumed)

- **FutureComposer FREQ vibrato / portamento** (keeps Tune_06 at ~0.90). The
  PITCHWALK base is recovered; the residual is the SMC vibrato (self-modified
  depth, delayed triangle) and 16-bit portamento glide, whose instantaneous freq
  is computed in CPU registers and not held in a normal RAM cell. Before adding
  any new observable, try recovering it from data we **already** log: the
  self-modifying-code immediate writes (depth/glide bytes patched into the player
  code) are captured by the RAM-write hook, so the modulation should be
  recoverable as a register-space BACC from those code-region writes. Only if
  that genuinely fails, log the emitted accumulator at each `$D400/$D401` `STA`
  (a small tracer hook).

## Scope: generative tunes

For algorithmically-generated melodies (e.g. *A Mind Is Born*) we model the
melodic **output as a note stream**, never the generating algorithm. The note
stream is captured data; the generator/modulation layer is recovered primitives.
A self-contained on-platform player stores such a note stream as data (cheap in
cycles), or — only if desired — a generative rule could be added as a new
primitive (e.g. an LFSR/PRNG node).
