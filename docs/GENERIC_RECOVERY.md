# Generic recovery: implementation plan

A phased plan to convert `preframr_playroutine/recover.py` from an ordered
per-primitive cascade into a generic, efficient hypothesis-search: an explicit
expression IR, a proposer/arbiter architecture with MDL scoring, dataflow-narrowed
candidate sets, and (last) an emit-slice lifter that generalizes the `CUTOFF`
approach. Each phase is independently landable, independently testable, and
protected by the guardrails in section 2.

Background: the analysis that produced this plan audited every primitive in
`recover.py` against the nine reverse-engineered players in
`/scratch/anarkiwi/cbm/re-trackers/*/\*-generators.md` (DMC, GoatTracker2,
JCH_NewPlayer, MusicAssembler, Soundmonitor, FutureComposer, defMON, WEMUSIC,
AMindIsBorn). Every documented generator reduces to one small grammar
(section 3); the current code implements scattered points of that grammar as
special cases, each with its own search, its own fidelity threshold, and its own
"never displace a better fit" guard.

## 1. Problems being fixed

1. **Ordered cascade, not evidence-based arbitration.** `classify_register`
   (recover.py:1193) hard-codes the hypothesis order (CONST → CUTOFF → SEQ
   short-circuits → FREQ path → BACC → walk/composite → XOR → AND → FEEDER → OR).
   Priors live in the ordering; every new primitive needs defensive
   "strictly better / never steals" guards and its own magic threshold
   (0.5, 0.6, 0.8, 0.82, 0.85, 0.95, 0.999 all appear).
2. **Copy-paste primitives.** `_xor_pair`/`_and_pair`/`_or_pair`
   (recover.py:1945/2016/2164) are one binop search ×3; override recovery exists
   ×3 (`_recover_walk_overrides` 1485, `_recover_pair_overrides` 2083,
   `_best_composite_override` 1898); the held-seed prelude ×2
   (`_attach_seed_prelude` 1303, `_attach_or_prelude` 2135); the
   segment/strip/fit skeleton ×3 (`segmented_bacc` 539, `segmented_pingpong` 621,
   `segmented_tickband` 683).
3. **Entangled axes.** A bounded accumulator has a *step model* and a *boundary
   behavior*; the current modes are four hand-picked (step, boundary) points.
   The RE corpus needs the full product (section 3.2).
4. **Unconstrained hypothesis space.** Feeder/pair/override searches scan every
   changing RAM cell (thousands); the 0.999 gates exist to suppress the
   resulting spurious matches. The read log already names the handful of cells
   the emitting code actually read.
5. **No decompilation metric.** `FEEDER`/cell-fed descriptors replay captured
   per-frame RAM streams; `round_trip == 1.0` cannot distinguish a recovered
   generator from a replayed value stream. Nothing scores captured-state
   consumption.
6. **Silent regressions on imperfect tunes.** The xfail ratchet
   (tests/test_hvsc.py `_PERFECT`) only protects perfect tunes; a 0.96 tune can
   drop to 0.90 invisibly.
7. **Per-register independence.** JCH and FC docs both state the recovery rule
   explicitly: recover the per-voice wavetable tick/cursor ONCE, then CTRL and
   FREQ collapse to `table[tick]`. The code re-derives (or misses) it per
   register.

## 2. Guardrails (apply to every phase)

- **Never regress the perfect set.** All 20 `_PERFECT` tunes in
  tests/test_hvsc.py must stay `overall == 1.0`, zero XSTATE, after every phase.
- **Fidelity snapshot ratchet** (built in Phase 0) must not decrease for any
  (tune, register) pair. Improvements update the snapshot in the same PR.
- Python: black, pylint clean (no unused imports/vars), pytest `-n auto`
  (xdist), coverage > 85%, numpy-first (per-frame stateful recurrences may loop;
  everything else vectorized). No test/script > 60s CPU.
- CI runs in Docker (existing multistage `Dockerfile`); whole-song HVSC fixtures
  keep their current skip-if-unfetchable behavior.
- **License wall:** this repo is public (Apache-2.0 python).
  `/scratch/anarkiwi/cbm/re-trackers` is a private repo of copyrighted-derivative
  disassemblies. Never copy its prose, asm, or table *contents* into this repo.
  Cross-checks against it (Phase 8) must read from the sibling path at test time
  and `pytest.skip` when absent. Cell addresses used as fixture assertions are
  facts and fine; quoted disassembly is not.
- Keep the public API: `analyze`, `round_trip`, `classify_register`,
  `reconstruct_register`, `state_sequence`, `fit_bacc`, `detect_table_walk`,
  `recover_tuning`, `voice_detune` keep their signatures. Descriptor dicts may
  gain keys but existing keys keep meaning (render.py and tests read them).

## 3. Target expression IR

### 3.1 Grammar

Descriptors become trees. A node is a dict with `"op"` plus children/params:

```
reg     := {"op": "post", "expr": expr, "mask": int, "scale": int,
            "overrides": [override, ...], "prelude": prelude|None,
            "byte_role": "full"|"lo"|"hi"}          # one outer wrapper per register
expr    := {"op": "cell",  "addr": a, "sample": "write"|"eof"|"operand"}
         | {"op": "chip",  "src": "osc3"|"env3"|"cia_ta"|"cia_tb"}
                             # live chip-state read at the emit instant — the
                             # player's own LDA $D41B idiom; semantics in §3.5
         | {"op": "const", "value": v}
         | {"op": "table", "base": b, "data": bytes, "index": expr,
            "index_mask": int|None, "offset": int}   # ram_image[b + idx]
         | {"op": "binop", "fn": "or"|"and"|"xor"|"add"|"sub", "a": expr, "b": expr}
         | {"op": "recur", "seeds": [...], "resets": [...], "step": step,
            "boundary": boundary, "width": 8|16, "directions": [...]}
         | {"op": "select", "arms": [(predicate, expr), ...], "default": expr}
         | {"op": "program", "table": {...}, "cursor_seed": ...}   # Phase 6
step     := {"kind": "const", "value": k}
         | {"kind": "updown", "up": k1, "down": k2}
         | {"kind": "table", "tables": [...], "seg_tables": [...],
            "index": "tick"}                        # hold-last past end
         | any of the above + {"divide": n}          # apply every n-th frame
boundary := {"kind": "wrap", "modulus": m} | {"kind": "saw"}
         | {"kind": "reflect", "lo": l, "hi": h}
         | {"kind": "clampflip", "lo": l, "hi": h, "clamp_lo": cl, "clamp_hi": ch}
         | {"kind": "countflip", "up_n": a, "down_n": b}          # Phase 6
         | {"kind": "target", "target_expr": expr}                # Phase 6
predicate := [(cell, mask, value) | (cell, "in", (v0, ...)), ...]  # conjunction,
                                                    # as _find_override today
override := {"predicate": predicate, "force": v}
prelude  := {"end": f, "frames": [...], "values": [...]}
```

Every current descriptor is a point in this grammar:

| current type | tree |
|---|---|
| CONST | `post(const)` |
| SEQ | keep as-is (latch list; it is already minimal) |
| FEEDER / XSTATE+cell | `post(cell@write)` |
| TABLE_WALK (+gate) | `post(binop(and, table[cell@eof], cell@eof or const-mask))` |
| XOR / AND / OR | `post(binop(fn, cell@write, cell@write or const))` |
| BACC (all modes) | `post(recur(...))`, cell-fed variant = `post(cell@write)` + prelude |
| COMPOSITE | `post(binop(add, base, mod))` with overrides |
| PITCHWALK | `post(table16[add(cell, cell, ...)])` (n-ary add = nested binop) |
| CUTOFF | `post(scale · clamp-select(add(cell@write, const, carry-expr)))` — Phase 7 lifter output; until then keep the hand-ported node |

**Song-level IR.** Above the per-register trees sits the song container:
tuning descriptor, per-voice note-event streams, and a **definition pool** —
`{"defs": {id: node}, ...}` with a `{"op": "ref", "id": id, "args": {...}}`
node so subtrees (tables, recurrence configs, instrument-like parameter sets)
are defined once and referenced with per-use parameters (seed, transpose,
phase). Proposers emit inline trees; the optimize pass (Phase 9) is what
factors them into the pool. `evaluate` resolves `ref` transparently.

**Closure requirement.** The grammar must be closed under composition: any
`expr` may appear wherever an `expr` is typed (a `table` index may be a
`binop`, a `select` arm may be a `recur`, a `recur` seed may come from a
`table`, a `binop` operand may be a masked nibble of another node). Proposers
propose shapes; the *evaluator and scorer must accept arbitrary well-typed
trees*. Do not special-case the compositions the current fixtures happen to
need (e.g. Soundmonitor's `$D418` is `or(mode_cell, recur_low_nibble)` — a
composition no current primitive can express; the grammar gets it for free
only if composition is unrestricted).

### 3.2 Grammar nodes the RE corpus requires (currently missing)

Ranked by leverage (engines / registers / known-imperfect fixtures covered):

1. **`select` (mux).** WEMUSIC FREQ/PW/CTRL (A/B sets by square-LFO toggle),
   DMC FREQ (slide vs vibrato), FC FREQ (six emit blocks — the dominant xfail
   cause), MusicAssembler PW (MODE-byte select), Soundmonitor stacked enables.
2. **`program` (table-program interpreter).** One archetype: cursor over
   (duration, step-or-SET) records with `$FF`-style loop rows. GT2 pulse+filter,
   JCH PW+filter, DMC filter, FC filter. Covers the two known "genuinely
   imperfect" registers: Guns_n_Ghosts `$D416` (0.99999 FEEDER today) and the FC
   `$D416` reflect idiom.
3. **Boundary axes `countflip` and `target`.** WEMUSIC/Soundmonitor dwell-counted
   flips; GT2 toneporta / Soundmonitor glide / DMC slide clamp-to-target-latch.
   Plus the `divide` step modifier (MA half-rate slide, Soundmonitor volume
   ramp, FC parity bias).
4. **Index sources beyond the note-on tick.** Global frame counter (Commando —
   Hubbard's reflected triangle), masked captured cell as table cursor
   (`$F7[LFSR & 7]`, A Mind Is Born), multi-bit mask-equality override predicate
   (`(ctr & $30) == 0`, AMIB rhythm gate — extend `_find_override`, which today
   builds only single-bit and full-byte-equality terms).
5. **LFSR — explicitly out of scope.** A Mind Is Born's `$B8` Galois LFSR is
   *generative melody code*, and the project deliberately does not model melody
   generators (README "Scope of generative tunes"): the melodic output is
   captured note-stream data. The LFSR state cell (`$14`) is therefore replayed
   as captured data, never fit as a recurrence. Do NOT add an LFSR node. What
   IS in scope from AMIB is the generic machinery around it: the masked-cursor
   table lookup (`$F7[cell & 7]` — a *mapping* from captured data to a
   register, item 4) and multi-bit mask predicates (`(ctr & $30) == 0`, where
   the counter is an ordinary BACC).

### 3.3 Design against the capability envelope, not the examples

The fixture tunes exercise only a slice of each engine's code. The RE corpus
documents capabilities that are present in the players but idle in the
captured tunes — recovery must handle these when an unseen tune exercises
them, so the grammar and fitters are specified from the *code's* envelope:

| documented-but-idle capability (source) | grammar coverage required |
|---|---|
| GT2 filter program "not triggered in the capture"; Soundmonitor filter sweep "mechanism fully present, unseeded"; MA FREQ ping-pong dormant (pre-dwell > note length) | `program` node + step×boundary product must be fit from *any* tune's series, never keyed to fixture-specific values |
| GT2 build flags: the fixture player was compiled with `NOFUNKTEMPO / NOCALCULATEDSPEED / NOWAVEDELAY / NOTRANS / NOREPEAT` — full builds in HVSC include per-channel tempo, funktempo, wave delay, transpose, repeat | none of these change the per-frame generator *shapes*; they change pacing and latch timing. Recovery must not assume fixed rows-per-note, fixed note-on cadence, or gate-transition-only reseeds (segmentation must use discontinuities as well as gates — legato/tie notes reseed without a gate edge) |
| DMC pitch-bend variant at half frame rate (`$1035` toggle); MA half-rate slide; FC parity bias | `divide`/parity step modifier on every step kind, not on the one kind an example used |
| Soundmonitor reprograms CIA Timer-A per section — the play cadence itself is state; defMON is 8× multispeed | frame binning must stay driven by observed interrupt ticks (`tick_cycles`), never an assumed constant; nothing downstream may bake in a cadence |
| WEMUSIC portamento SMC save/restore; JCH `$90` mid-note AD override; per-note instrument re-blits | SMC cells are ordinary `cell@*` inputs (write log covers code addresses); SEQ must tolerate mid-note latches, not only note-on latches |
| defMON instrument bytecode can flip any CTRL bit combination via eor masks; observed tunes only used a few values | bit-level nodes (`binop`, predicates) operate on full 8-bit domains; never enumerate "known" waveform bytes |

Three mechanisms enforce generalization; all three are requirements, not
suggestions:

1. **Orthogonal parameterization.** Every fitter searches the full axis
   product it claims (all step kinds × all boundaries × widths × divide), with
   pruning for *cost* only — never "this combination hasn't appeared yet".
   New idioms should land as new axis values, not new node types, wherever
   possible.
2. **The lifter (Phase 7) is the unseen-code path.** Grammar proposers are the
   efficient fast path for known shapes; code nobody has RE'd gets recovered by
   slicing its actual emit code. Phase ordering must keep the lifter
   engine-agnostic: no opcode-signature allowlists, no per-engine tables — its
   only gate is verified reconstruction fidelity.
3. **Synthetic capability suite + ablation tests** (section 5) — test coverage
   is generated from the documented capability envelope, not from HVSC tunes.

### 3.4 Canonical text serialization (the LLM training surface)

The recovered IR doubles as an LLM training corpus. The nested-dict form stays
the in-memory/programmatic representation; the *dataset surface* is a
canonical, line-oriented text form emitted by Phase 10. Normative rules —
these bind the emitter, and several bind the IR design itself:

1. **Line-oriented, fixed field order, no unions.** One op per line,
   `op field=value ...` with a fixed field order per op and omit-if-default
   (defaults defined once, in this doc). No field may hold more than one type:
   split the predicate term `(cell, mask|"in", value)` into distinct ops
   `eq`, `bit`, `mask`, `in` with fixed arity (this changes the in-memory
   predicate format too — do it in Phase 1).
2. **Bounded surface depth via named defs.** Inline nesting at most ~2 levels;
   any deeper or reused subtree becomes a pooled definition referenced by id
   (SSA-style). Closure is unrestricted in memory; the *surface* is a shallow
   DAG with names.
3. **Low-entropy identifiers.** Pool ids are short, kind-prefixed, sequential
   in **first-use order**: `t0..` tables, `i0..` instruments, `g0..`
   generators, `p0..` patterns. Content hashing is internal dedup machinery
   only and must never leak into emitted ids or ordering.
4. **Note-space over frequency-space.** Pitch as MIDI note numbers plus the
   per-song tuning line; a pitch table byte-identical to the 12-TET ladder
   under the recovered tuning is emitted as one intrinsic
   (`pitchtable 12tet cents=-35`), verified lossless, with literal bytes (or
   intrinsic + sparse corrections) as fallback. Generators use note-relative
   forms where Phase 9c lifted them (`vibrato`, `arp`, `transpose`,
   `detune` are surface ops). This buys transposition invariance (and cheap
   augmentation) plus large compaction.
5. **Numeric discipline.** Hex for addresses/masks/raw bytes, decimal for
   counts/steps/notes/frames — one rule, no exceptions. Signed values as
   signed decimals, never two's-complement bytes. No floats: tuning as
   integer cents, fidelity as parts-per-million integer. Event streams use
   delta times. Literal tables are RLE/delta encoded with an explicit length.
6. **Fenced data vs structure.** Captured streams (note events, replayed
   cells, any residual XSTATE) are emitted inside explicit `data`/`stream`
   blocks, never interleaved with generator ops — the model must see a hard
   structure/data boundary. Per-register fidelity (ppm) is recorded in the
   `meta` trailer (rule 9), so dataset builds can filter imperfect recoveries
   and prompt builds can strip analytics entirely.
7. **Prefix-streamable layout (the prompt requirement).** The IR is also the
   *prompt format*: the typical use is "first 30s of a song as prompt, LLM
   extends it" — so the document is a short global prelude (format version,
   chip, cadence/speed, tuning — facts knowable up front) followed by a
   **time-ordered body**. In the body, definitions (tables, instruments,
   generator shapes, patterns) are emitted **at first use**, immediately
   before the time block that first references them — never gathered up
   front. Truncating the text at any block boundary must yield a
   syntactically valid, fully-resolvable, evaluable IR of the song-so-far;
   appending well-formed blocks (including new defs minting the next
   sequential ids — rule 3 makes id allocation append-friendly) is the
   continuation operation. Voice symmetry is still expressed by refs.
8. **Stable small vocabulary.** Lower-case keywords, one keyword per op/field,
   no synonyms, versioned by the header line; grammar additions append, never
   rename.
9. **Causality — no whole-song lookahead in the body.** A body line may only
   depend on the past: per-note generator parameters (seed, rate, direction)
   attach to the note events / instrument refs that introduce them, never to
   whole-song arrays on the register binding (this makes the Phase 9b
   "one instrument + per-note seed" factoring *normative* for emission, not
   just an optimization). Captured `data`/`stream` blocks are chunked per
   time block. Whole-song analytics (per-register fidelity, note_range,
   corpus stats) live in strippable `meta` trailer lines, not in the body, so
   prompt datasets can drop them without breaking parseability.

Illustrative fragment (shapes, not final syntax):

```
preframr-ir 1
chip 8580 pal cadence irq cpf=19656 speed=1
tuning 12tet cents=-35 a4=note69
; --- time-ordered body: defs at first use, then the block using them ---
table t0 len=8 hex 41 11 21 81 09 91 93 9c
inst i0 ad=08 sr=88 wave=t0 pwseed=$0141 pwmod=saw16 step=64
bind v0 ctrl = and (walk t0 idx=tick) gate
pattern p0
  0 v0 on n=47 i=i0
  6 v0 off
play p0
play p0 transpose=+7
; a later block may mint new defs mid-stream (the continuation case):
inst i1 ad=0a sr=99 wave=t0 vib depth=2 rate=6 delay=12
pattern p1
  0 v0 on n=52 i=i1
play p1
meta fid v0.ctrl=1000000 v0.pw=1000000
```

### 3.5 Recovery totality

There is no "bespoke player" category anywhere in this design: every player
is machine code implementing a tracker-like program, and recovery is the same
problem for all of them — how many HVSC tunes happen to share an engine is a
statistic about reuse, not a property of the code. Accordingly, recovery is
**total by construction**, on this argument:

1. **All inter-call state lives in memory.** On the 6502, no CPU register
   survives between play calls as usable player state: the play entry arrives
   through interrupt dispatch (or the driver's JSR) with A/X/Y/P dead.
   Every value a player carries from one frame to the next therefore resides
   in RAM (including SMC'd code bytes) or in hardware-readable registers
   (SID osc3/env3, CIA timers/ports).
2. **Memory state is fully logged.** Initial image + the cycle-stamped RAM
   write log reconstruct RAM at any cycle; the read log resolves all dynamic
   indexing with both effective address and value.
3. **Within one call, every store is a finite computation over logged
   inputs.** A backward slice from `STA $D4xx` terminates at logged reads,
   image bytes, and SMC immediates — always. So every register write value is
   a deterministic function of observables *within a single play call*; the
   only open question is how compactly we can express that function, never
   whether we can.

Two instrumentation gaps currently break premise 2 and must be closed
(both hooks in `patches/instrument.patch` exclude `(addr & 0xff00) == 0x0100`
and `(addr & 0xf000) == 0xd000`):

- **I/O reads are not logged.** Players that read SID osc3/env3 (`$D41B/1C`
  — the classic noise-driven vibrato/drum idiom) or CIA timers as
  entropy/time sources have inputs recovery cannot see. Extend the tracer to
  log `$D000–$DFFF` reads (same record shape, `.iord.bin` or a kind bit; tiny
  volume) and update `docs/INSTRUMENTATION.md`. Such a read is decompiled to
  the **`chip` source node** (§3.1) — a *live* chip-state read at the emit
  instant, exactly the original code's `LDA $D41B`. Semantics are split by
  context: during **recovery/scoring**, the logged values drive
  reconstruction through the sampler like any input cell (no circularity —
  they are data the trace observed); at **replay**, the playroutine is itself
  driving a SID/CIA, so the node reads the live chip, and determinism (fixed
  LFSR seed, fixed power-on delay, cadence carried by the IR) makes the round
  trip exact under the emulator — while on real hardware it degrades
  gracefully to the same phase-dependence the original player had. The read
  values are deliberately NOT emitted as captured data: the chip coupling
  *is* the generator's semantics, and MDL would misprice it as replay
  otherwise.
- **Stack-page accesses are excluded.** A player that passes a computed
  operand through PHA/…/PLA within a call is invisible to a dynamic slice.
  Handle statically (the lifter tracks PHA/PLA pairing symbolically — no log
  needed), or include `$01xx` accesses behind the reads flag.

Consequence for the taxonomy: **XSTATE is retired as a terminal category.**
The generic pipeline is three tiers, all emitting the same IR:

- **Tier 1 — grammar proposers** (fast, semantic; Phases 2–6).
- **Tier 2 — emit-slice lifter** (static backward slice → grammar tree;
  Phase 7).
- **Tier 3 — dynamic input witness** (totality backstop; Phase 7): for a
  store the first two tiers leave imperfect, take the ordered logged inputs
  of the emitting code's dataflow cone each frame and recover
  `value = f(inputs)` by bounded exact expression synthesis over the op
  grammar (bitops/add/sub/shift/clamp/select, verified frame-exact); if no
  small `f` is found, the observed input→output mapping itself is the
  descriptor (exact, replayable — the inputs are sampler-replayable cells).
  Tier 3 always succeeds given premises 1–3; MDL charges it heavily, so
  Tiers 1–2 win wherever they fit.

`round_trip == 1.0` for every tune is thereby reachable by construction; the
remaining engineering axis is **description quality** (captured-state cost,
which Phases 2–6 drive down), measured by MDL — not reachability. The ratchet
goal "zero XSTATE" becomes "zero Tier-3 descriptors above a captured-state
budget".

Corpus correction feeding this section: **Blackout is a JCH_NewPlayer tune**
(`tests/fixtures/tunes.json`) — its gap is an
unexercised JCH capability (the §3.3 envelope: PW/filter program lanes,
vibrato bias ramp, absolute-path FREQ), expected to fall to Phase 6b/the
envelope work, with the Phase 8 anchors identifying the exact lane.
**Commando** (Hubbard) is grammar-covered (6d global-counter phase + 6a mux)
— and if its player reads osc3 noise, the I/O-read extension above is what
makes it observable. No fixture remains outside the tiers.

## 4. Phases

Each phase = one PR. Land in order; 0–2 are prerequisites for the rest.
Use subagents per phase; keep diffs reviewable (split Phase 1 if > ~1500 lines).

**Tracer slice (land first):** all tracer-side work (I/O read/write logs,
`--stack`, loader accessors, instrumentation contract v3) is specified as a
standalone mechanical brief in [`SIDTRACE_PR.md`](SIDTRACE_PR.md) — one PR,
after which every phase below is python-only.

**Foundation slice:** Phase 0 plus the *reconstruction side* of Phase 1
(ir.py, evaluator, to_ir adapter, typed predicates, golden parity fixtures)
is specified as a standalone, mechanically-executable brief in
[`IR_FOUNDATION_PR.md`](IR_FOUNDATION_PR.md) — hand that doc alone to a
subagent for the first PR. Phase 1's remaining items (unifying the
override/prelude *recovery* passes, the `state_sequence` group-by) follow in
the next PR against the goldens and snapshot that the foundation lands.

### Phase 0 — fidelity snapshot ratchet (small, do first)

**Goal:** make regressions on *imperfect* tunes visible before any refactor.

- Add `tests/fixtures/fidelity_snapshot.json`: `{tune_key: {register_hex:
  fidelity, "overall": x}}` for every catalog fixture, generated by a one-shot
  script (guard: whole-song render per tune already happens in CI; reuse
  `test_hvsc` plumbing, do not add a second render).
- In the existing whole-song test path, after `round_trip`, assert
  `fidelity[addr] >= snapshot[addr] - 1e-9` for every register, and
  `overall >= snapshot.overall`. On improvement, the test does NOT fail;
  a separate check (or the PR author) refreshes the file.
- Add `tests/update_snapshot.py` (not collected by pytest, like
  `straggler_report.py`) to regenerate it.

**Accept:** CI red if any (tune, register) fidelity drops; snapshot committed.

### Phase 1 — expression IR + single evaluator (behavior-preserving refactor)

**Goal:** one recursive evaluator; overrides/prelude/masking become generic
post-passes; kill the `_recon_*` family and the ×3/×2 duplication.

- New module `preframr_playroutine/ir.py`:
  - `evaluate(node, n_frames, sampler) -> np.ndarray` — recursive, vectorized
    per node; `recur` nodes loop per segment as today (numpy-first rule allows
    stateful loops).
  - `complexity(node) -> float` — MDL cost (section 4 Phase 2).
  - Node constructors/validators (plain dicts, JSON-serializable; numpy arrays
    allowed for tables, matching current descriptors).
- Translate the ten `_recon_*` builders (recover.py:2492–2819) into `evaluate`
  cases. Keep `reconstruct_register` as a thin adapter: legacy descriptor →
  tree (a `to_ir(descriptor)` shim) → `evaluate`, so render.py and stored
  descriptors keep working. Keep `CUTOFF` as one hand-ported node evaluated by the
  existing `_recon_cutoff` until Phase 7.
- Unify: one `attach_prelude(node, series, ctx, cells)` (replaces both
  `_attach_seed_prelude`/`_attach_or_prelude`; the OR variant's
  `max(first_live_frame)` over multiple cells is the general form) and one
  `recover_overrides(node, series, ctx, max_overrides)` greedy strictly-improving
  pass (replaces all three copies; parameterize the pre-first-write zero
  masking, cf. `_and_recon_masked`).
- `_default_until_first_write` stays the single outermost step.
- Normalize the predicate representation while touching it: replace the
  polymorphic `(cell, mask|"in", value)` tuples with typed terms
  (`eq`/`bit`/`mask`/`in`, fixed arity — 3.4 rule 1); keep a shim for stored
  legacy descriptors.

**Accept:** pure refactor — Phase 0 snapshot byte-identical (every fidelity
equal, classifications equal on the anchor tests in
`test_real_tune_anchors`). Unit tests: `to_ir` + `evaluate` reproduces
`reconstruct_register` exactly for a saved descriptor of every current type
(build these fixtures from the minisid tests + one HVSC tune per type).

### Phase 2 — proposers + global arbiter with MDL scoring

**Goal:** delete the cascade ordering and the per-primitive adoption guards.

- Each existing search becomes a proposer: `propose_*(series, sid_addr, ctx)
  -> [tree, ...]` (may return several candidates). Proposers keep their cheap
  prefilters but drop their *adoption* logic and their per-proposer thresholds.
- Arbiter, per register:
  ```
  score(tree) = fidelity(tree)  -  LAMBDA * complexity(tree)
  fidelity    = mean(evaluate(tree) == series)   # after default-until-first-write
  ```
  `complexity` counts: nodes (weight ~1 each), override terms (~1 each), and
  **captured-state cost**: each `cell@*` reference costs
  `w * changed_frames(cell) / n_frames` — a feeder replaying a per-frame stream
  is expensive; a recurrence with a seed list is cheap; a SEQ latch list costs
  per latch. Start `LAMBDA ≈ 1e-3` and `w ≈ 0.5`, then tune so that on the 20
  perfect tunes the winning tree is unchanged (write a calibration test that
  asserts exactly this).
  Scope note: the captured-state cost targets *per-frame modulation* smuggled
  through cell replay. Event-latched song data (SEQ latches, note streams —
  including generative-melody output like AMIB's LFSR cell) is captured **by
  design** (README "Scope of generative tunes") and must stay cheap: charge
  per *latch/change*, not per frame held, so the scorer never pressures toward
  "decompiling" melody data.
  Tie-break equal scores by lower complexity, then by proposer priority order
  (documented, single place).
- Routing priors (XOR only-CTRL, PITCHWALK only-FREQ, CUTOFF only `$D415–18`)
  become *proposer hints* (which proposers run for which register class), not
  adoption rules; the CUTOFF opcode-signature gate stays inside its proposer
  (it is a legitimate precondition, not an arbitration rule).
- Keep two structural short-circuits as proposers with high priors, not
  hard returns: AD/SR → SEQ; sparse/note-gated → SEQ (`_seq_correlation`).
- Report: descriptor gains `"score"`, `"complexity"`, `"captured_frames"` keys;
  `round_trip` output unchanged; `analyze` summary unchanged.

**Accept:** perfect set intact; snapshot non-decreasing; the calibration test
pins winners on the perfect set. Expect (and update snapshot for) improvements
where the cascade previously picked a worse-ordered branch. Delete
`_maybe_xor_ctrl`/`_maybe_and_ctrl`/`_maybe_or_reg`/`_maybe_feeder_upgrade`.

### Phase 3 — unify binop search; step×boundary product

**Goal:** collapse duplicates and cover the axes the corpus needs.

- One `propose_binop(series, sid_addr, ctx)` for `{or, and, xor, add, sub}` ×
  (cell, cell) and (cell, const): generalize `_or_pair`'s structure (it already
  does cell|const with per-bit-optimal constant; do the analogue per op —
  for `and` per-bit, for `add/sub` search small constant set from residual
  histogram). Same subsample-then-verify pattern; one entropy prefilter
  (≤ 24 distinct values, the union of today's three). Delete
  `_xor_pair`/`_and_pair`/`_or_pair`. `add` at width 16 over lo/hi pairs
  subsumes most of `_composite16`'s base+mod (keep `_composite` as a proposer
  until Phase 5, then re-evaluate).
- One `propose_recurrence(series, resets, ctx)`: segment once (note-ons +
  `_discontinuities`), fit each segment against the step×boundary product
  {const, updown, table[tick]} × {wrap, saw, reflect, clampflip}, share the
  vote/consistency logic, emit `recur` trees. Delete
  `segmented_bacc`/`segmented_pingpong`/`segmented_tickband` (keep `fit_bacc`
  public API as a thin wrapper). Preserve the existing anti-theft heuristics as
  *scoring inputs* (e.g. tickband's "few shared tables" check becomes part of
  complexity: many distinct rate tables = high cost).

**Accept:** perfect set intact; snapshot non-decreasing. Expected wins:
Vacuole-class defMON PW expressed as clampflip×table-step if fidelity says so;
JCH `$D418`-style folds unchanged. Unit-test the product fitter on synthetic
series for all 12 cells of the product.

### Phase 4 — read-log dataflow narrowing

**Goal:** shrink every cell search from "all changing cells" to "cells the
emitting code read", removing the spurious-match pressure and most of the cost.

- In `_build_context`: from `trace.ram_reads()` (when present) build
  `reads_near_store[sid_addr] = {cells read within W cycles before any store to
  sid_addr}` (W ≈ one play-call; use store cycles from `sid_writes`). Vectorize:
  sort reads by cycle, `searchsorted` the store cycles, take a bounded window.
- All proposers take candidate cells from
  `ctx.candidates(sid_addr) = reads_near_store ∪ {cells written near store}`
  when the read log exists, else fall back to today's global `changing cells`.
  Keep the fallback path tested (CI fixtures render without `--reads` — that
  stays the CI default, so add one `--reads` fixture to exercise narrowing, and
  benchmark both).
- Consider enabling `--reads` for CI fixture renders if trace size is
  acceptable; measure first (INSTRUMENTATION.md documents the format/cost).

**Accept:** identical or better snapshot; `analyze` wall-time on the longest
JCH fixture improves measurably (record before/after in the PR); with narrowing
active, relax nothing yet — thresholds/λ unchanged.

### Phase 5 — shared per-voice latent state

**Goal:** recover tick/cursor/note once per voice; let all proposers reference
them (the JCH/FC docs' explicit recommendation).

- New context field `ctx.latents[voice] = {"tick": series, "cursors": [(addr,
  series), ...], "note_cell": addr|None}`:
  - tick: synthesized from note-ons (as `segmented_tickband` does today) AND
    matched against captured cells that reset-to-0 at note-ons and step +1
    (prefer the captured cell when one matches — it survives retrigger nuances).
  - cursors: `_cursor_columns` output grouped by voice via reset correlation
    (`correlate_event_reset` exists).
  - global counter: any cell (or synthesized frame index) advancing +k every
    frame without note resets — for Commando-class phase sources.
- `table` and `recur` index/step models accept `"index": "tick"|"frame"|cell`.
- PITCHWALK's `_pitch_index_cells` becomes a general
  `propose_index_sum(idx_series, ctx)` usable by any table node.

**Accept:** snapshot non-decreasing; FC fixtures (Hawkeye stays perfect,
Tune_06/Manchester fidelity should rise via CTRL/filter lanes indexing the
shared tick). Add a synthetic minisid test with two registers driven by one
cursor and assert both recover referencing the same latent.

### Phase 6 — new grammar nodes: select, program, boundaries, index sources

Land as separate PRs in this order (biggest first):

**6a. `select` (mux).**
- Proposer: given the best two (or three) non-overlapping partial hypotheses
  for a register (arbiter keeps runners-up), find a predicate over
  `ctx.candidates` cells that separates their correct-frame sets; reuse
  `_find_override`'s candidate-predicate machinery (equality / bit / membership,
  extended with multi-bit mask-equality terms — enumerate masks from the cell's
  distinct-value XOR structure, don't brute-force 256 masks).
- Guard against overfitting (this node is powerful): a `select` arm must each
  explain ≥ some floor of frames (say ≥ 5%), the predicate complexity is
  charged by MDL, and adoption still requires beating the best single-arm tree
  on score, not just fidelity.
- Targets: FC FREQ (six blocks — start with 2–3-arm mux of
  pitchwalk/vibrato-BACC/porta-BACC), MusicAssembler PW, DMC FREQ.

**6b. `program` (table-program interpreter).**
- Fitter: segment the register/accumulator series into constant-slope runs
  `(len_i, slope_i)`; search RAM-image table pairs (read-log tables first) whose
  (duration, step) columns reproduce the run sequence, honoring SET rows
  (slope-∞ jumps) and loop markers. Emit
  `{"op": "program", "time_base": t, "spd_base": s, "cursor_seed": ...}` with an
  interpreter in `evaluate` matching the GT2/JCH/DMC/FC record semantics
  (parameterize: SET-if-value≥$80 vs value-column variants; keep the variants
  data-driven, not per-engine code paths).
- Targets: Guns_n_Ghosts `$D416` (→ perfect; remove from xfail),
  FC `$D416` (Manchester/Tune_06), and replacing FEEDER with closed forms on
  GT2/JCH PW/filter lanes (MDL will prefer the program: far less captured
  state).

**6c. Boundary/step extensions.** `countflip`, `target`, `divide` in the
Phase 3 product fitter. Targets: Soundmonitor PW/filter (already perfect via
other paths — assert no snapshot change, classification may improve),
GT2 toneporta, future WEMUSIC fixtures.

**6d. Index sources + predicate masks.** Global-frame-counter phase for `recur`
(Commando), `index_mask` on `table` (AMIB bass — the *lookup* is a generator;
the LFSR cell it indexes with is captured melody data, per the scope rule),
multi-bit mask predicates (AMIB gate, an ordinary counter tap). Targets:
Commando fidelity ↑ from 0.78, AMIB → toward 1.0 with its melody stream
remaining captured data throughout.

**Accept (each):** snapshot non-decreasing everywhere, targeted fixtures
improve, xfail markers removed only when a tune reaches perfect (the strict
xfail forces this). Every new node gets synthetic unit tests plus at least one
real-tune assertion.

### Phase 7 — emit-slice lifter + dynamic witness (generalizes CUTOFF; totality)

**Goal:** for a register the value-stream proposers leave imperfect, lift the
emitting code into the grammar automatically (Tier 2), with the dynamic input
witness (Tier 3) as the always-succeeds backstop — see §3.5. After this phase
XSTATE no longer exists as an output type.

**Prerequisite (tracer extension):** the §3.5 observability gaps (I/O
reads/writes, opt-in stack logging, `Trace.io_reads()`/`io_writes()`) are
carved out as a standalone mechanical brief —
[`SIDTRACE_PR.md`](SIDTRACE_PR.md) — intended to land as its own PR **up
front**, so all phases here are python-only. The slicer still handles
stack-page dataflow statically (PHA/PLA pairing) by default; the `--stack`
log exists as ground truth if a player defeats static tracking.

- New module `preframr_playroutine/lift.py`:
  1. For each store PC of the register (already in `store_pcs`): disassemble
     backward/forward within the executed-PC coverage bitmap
     (`trace` exposes the `.cov` bitmap; use a small tabular 6502 decoder —
     no new heavy dependency; ~50 opcodes suffice: LDA/LDX/LDY, ADC/SBC,
     AND/ORA/EOR, ASL/LSR/ROL/ROR, CLC/SEC, CMP/CPX/CPY, branches, STA, TAX/TXA
     etc., INC/DEC).
  2. Backward slice on the A register's def-use chain from the `STA $D4xx`,
     bounded (≤ ~24 instructions, no calls/loops crossed; branches allowed —
     they become `select` arms or clamp patterns).
  3. Ground operands: absolute → `cell@write`; immediate operand bytes are RAM
     addresses inside code → SMC cells, `cell@write` (this is how CUTOFF reads
     `imm`); indexed (`abs,X`) → resolve via the read log's effective addresses
     per frame (a per-frame varying address becomes `table[index-latent]` via
     Phase 5 latents); read-modify-write patterns → `cell@operand`
     (`_CellSampler.operand` exists). Carry chains: model `adc/sbc` pairs as
     16-bit `add` where the slice shows lo→hi carry, as `_recon_cutoff` does by
     hand today.
  4. Emit a tree; verify with `evaluate` vs the oracle; hand to the arbiter
     like any proposer (fidelity gate does the rest — no opcode-signature
     allowlists needed, though keep the lift bounded and fail-closed).
- **Tier 3 — dynamic input witness (totality backstop, §3.5).** When the
  static slice fails (within-call loops, unusual dataflow): per frame, take
  the ordered `(addr, value)` reads of the emitting window's dataflow cone as
  the input tuple and synthesize `value = f(inputs)` by bounded exact
  enumeration over the op grammar (verified frame-exact on all frames); if no
  small `f` exists, emit the witness descriptor itself — the input cells with
  sample modes plus the observed mapping — which is exact and replayable via
  the sampler machinery. Heavily MDL-charged, so anything Tiers 1–2 can
  express displaces it; a surviving Tier-3 descriptor is also the actionable
  *name* of the code that deserves a new grammar node or axis.
- Port `CUTOFF`: the defMON signature test becomes the lifter's first
  regression (lifting Automatas `$D416` must produce a tree ≥ 0.999 and the
  legacy `_recon_cutoff` result). Then delete `_CUTOFF_SIG`/`_defmon_cutoff`/
  `_recon_cutoff` once the lifted tree matches on Vacuole/Stargazer/Automatas.
- This proposer runs LAST and only for registers below 1.0 after all others —
  it is the most expensive and the most general.

**Accept:** defMON trio still perfect via lifted trees; at least one previously
imperfect register recovered by lift alone (candidate: MusicAssembler-class SMC
accumulators on any new fixture; FC vibrato depth via SMC immediate `$12F7`).
Lifter unit tests on hand-assembled minisid tunes (the oracle tests already
assemble tiny PSIDs with xa65 — add slices covering: plain feeder store,
add-fold, SMC immediate, clamp branch, indexed table read).

### Phase 8 — re-trackers cross-checks (fixtures, not code)

**Goal:** use the RE ground truth as a regression suite, respecting the
license wall (section 2).

- `tests/test_retrackers.py`, skipped unless
  `/scratch/anarkiwi/cbm/re-trackers` exists (env var override for path).
- For each engine with a catalog fixture, assert recovered structure against
  *facts* (cell addresses, node shapes) — e.g. DMC Doctagop: PW recur references
  bounds cells `$1756/$1759`-seeded values, CTRL is `binop(and, table[...],
  gate)` with gate cell `$100F`; GT2 Raindrops CTRL cells `$93B9`/`$93D0`;
  JCH `$D418` = `or($1793, $1009)`. Extend the existing `_ANCHORS` mechanism
  (tests/test_hvsc.py:76) rather than duplicating render plumbing.
- Do NOT build an engine-template recovery path (template matching per engine
  is whack-a-mole at engine granularity; the lifter is the general mechanism).
  Templates exist here only as assertions.

**Accept:** anchors pass locally with re-trackers present; CI (without the
private repo) skips them.

### Phase 9 — generic IR optimize pass (canonicalize, dedup, abstraction-lift)

**Goal:** the recovered song IR is *generically ordered and minimized*. Tracker
tunes routinely use their own abstractions suboptimally — the same instrument
defined multiple times, per-note absolute modulation where a relative
abstraction exists, transposed copies of patterns — and hand-coded players may
not use tracker abstractions at all. A generic optimizer recognizes these and
rewrites to the minimal shared form, **losslessly**: after `optimize`,
`round_trip` must still be exactly what it was, verified by re-evaluation.

Dependencies: Phases 1–2 only (IR + MDL cost). Can land in parallel with 3–8;
gains value as the grammar grows. New module
`preframr_playroutine/optimize.py`, entry `optimize(song_ir) -> song_ir`,
called at the end of `analyze` (flag-gated `optimize=True`).

Structure: a fixpoint loop of rewrite rules. Every rule application is
**(a) verified lossless** — `evaluate(new) == evaluate(old)` frame-exact for
every affected register — and **(b) MDL-decreasing** under the Phase 2 cost
function extended so pooled definitions are charged once and `ref`s cheaply
(that extension is what makes dedup pay). Iterate to fixpoint with an
iteration cap; the pass must be deterministic and idempotent
(`optimize(optimize(x)) == optimize(x)`).

**9a. Canonicalization (local, always-safe rewrites).**
- Sort commutative binop operands by canonical key; constant-fold
  (`or(x, 0) → x`, `and(x, $FF) → x`, nested masks collapse); degenerate
  recurrences to `const`/`SEQ` (step 0, lo==hi, single segment of length 1);
  merge overrides with identical predicates; strip empty preludes; normalize
  equivalent boundary parameterizations to one spelling.
- Deterministic ordering everywhere: descriptor keys, definition pool in
  **first-use order** with sequential kind-prefixed ids (content hashing is
  internal dedup machinery only — hash-derived ids/ordering must never appear
  in output; see 3.4 rule 3), segments/latches by frame — so IR output is
  stable and diffable across runs (extends the existing determinism guarantee
  from the trace to the IR).

**9b. Global dedup / structure sharing (the multiply-defined-instrument case).**
- Hash-cons canonicalized subtrees across all registers/voices/segments;
  byte-identical subtrees (tables, rate tables, recurrence configs, envelope
  parameter sets) become one pool definition + `ref`s.
- Parameterized dedup: subtrees identical up to a per-use scalar become one
  definition with a `ref` arg — recurrences differing only by seed → one
  "instrument" seeded per note; per-note (AD, SR, wave, PW-seed, sweep) tuples
  recurring across notes/voices → one instrument record referenced by note
  events. This is where "same instrument multiply defined" collapses even when
  the tune's own instrument table never shared them.
- Tables equal under a constant index shift or value offset → one table +
  offset arg (also catches lo/hi sub-table overlap).

**9c. Abstraction lifting (lossless recognition of tracker idioms the source
may not have used).** Each lift is a pattern → higher-level rewrite, applied
only when verification passes; the target abstractions are the ones the
6502 reference replayer will implement natively:
- **Arpeggio:** a FREQ/note lane cycling through a small pitch set with fixed
  period (SEQ latches or table-walk output) → `arp(base_note, offsets[],
  rate)`, offsets in semitones via the recovered tuning (`recover_tuning`
  note numbers). Frequencies must map exactly onto the note grid for the lift
  to apply; otherwise leave as-is.
- **Transpose:** note streams / pattern segments / pitch tables identical
  modulo a constant semitone shift → shared definition + `transpose(k)`.
  Applies across voices and across repeats within a voice.
- **Vibrato:** an absolute-frequency `recur` (reflect/triangle) whose center
  tracks the note pitch and whose depth/rate repeat across notes → relative
  `vibrato(depth, rate, delay)` attached to the note layer — one definition
  replacing N per-note absolute recurrences. Depth may be per-note-scaled
  (e.g. DMC's `NoteFreqHi[note]/2` scale) — allow a scale arg.
- **Detune:** fold `voice_detune`'s recovered constant offsets into the IR as
  a per-voice `detune(cents_or_freq_delta)` rather than divergent note data.
- Lift order: canonicalize → lift → dedup again (lifting creates sharing).

**Accept:**
- `round_trip` unchanged (exact per-register fidelity equality, not ≥) on
  every catalog fixture with `optimize=True` — the lossless gate, asserted in
  CI via the Phase 0 snapshot.
- Idempotence + determinism tests; total MDL cost non-increasing and strictly
  decreasing on at least the synthetic dedup fixtures.
- Synthetic micro-tunes (section 5a additions): (i) one instrument defined
  twice byte-identically → pool of one; (ii) two voices playing the same
  pattern transposed +7 → shared pattern + transpose; (iii) per-note absolute
  triangle FREQ modulation with shared depth/rate → single vibrato definition;
  (iv) three-pitch cycle at fixed rate → arp. Each asserts the *shape* of the
  optimized IR, not just fidelity.
- At least one real-tune win recorded in the PR (e.g. a GT2/JCH fixture whose
  per-voice tables/instrument tuples dedup measurably — report pool size and
  MDL before/after).

### Phase 10 — canonical text emitter + parser (dataset surface)

**Goal:** implement §3.4. New module `preframr_playroutine/text_ir.py`:
`emit(song_ir) -> str` and `parse(str) -> song_ir`, run after Phase 9's
`optimize` (the emitter consumes optimized IR; emitting unoptimized IR is
allowed for debugging but not for datasets).

- The pair must be bijective on canonical IR: `parse(emit(x)) == x` and
  `emit(parse(s)) == s` for canonical `s`.
- Lossless end-to-end: `evaluate(parse(emit(x)))` reproduces every register
  frame-exactly (assert via `round_trip` on all catalog fixtures).
- Deterministic: same trace → byte-identical text (extend `test_determinism`).
- Pitch-table intrinsics (3.4 rule 4) verified byte-exact against the RAM
  table before emission; fallback to literal on any mismatch.
- Corpus stats in CI logs: emitted bytes per tune, structure/data byte split,
  pool sizes — the compactness metrics for the training corpus.
- Unit tests: golden files for one fixture per engine family plus the Phase 5a
  micro-tunes; a fuzz test that round-trips randomized (deterministic-seed)
  canonical IRs through emit/parse.
- **Prompt-prefix tests (§3.4 rules 7/9).** For every catalog fixture:
  (i) truncate the emitted text at *each* block boundary — every prefix must
  parse, resolve all refs, and evaluate frame-exactly against the oracle for
  the frames it covers (a time-cut prefix IS a valid IR of the song-so-far);
  (ii) streaming associativity — `emit(song[0..30s])` must be byte-identical
  to the corresponding prefix of `emit(song)`, so (prompt, continuation)
  training pairs are exact prefix/suffix splits of one emission;
  (iii) continuation append — parse a 30s prefix, append the remainder's
  blocks (which mint new ids mid-stream), re-evaluate, and match the full
  song; (iv) a grep-level assertion that no def line appears before the first
  block needing it and no body line carries whole-song analytics (`meta` only
  in the trailer).

**Accept:** bijectivity, losslessness, determinism, and the prompt-prefix
suite green; golden files committed; no float appears anywhere in emitted
text.

## 5. Synthetic capability suite + ablation tests (cross-cutting)

Grows with every phase from Phase 3 onward; this is the anti-overfitting
mechanism. HVSC fixtures prove real-world recovery; these prove *envelope*
recovery.

**5a. Assembled micro-tunes.** The oracle tests already assemble tiny original
PSIDs with the in-build `xa65` and trace them. Extend `tests/_minisid.py` with
a small builder that emits a player exercising exactly one capability with
parameterized constants, then assert byte-exact recovery of the intended tree
shape. Required micro-tunes (one each, cheap — a few seconds of trace):

- each cell of the step×boundary product not exercised by any HVSC fixture
  (notably `countflip`, `target`-latch glide, `divide` on table-step);
- a table-program lane with SET rows + loop marker (GT2/JCH record semantics);
- a 2-arm and a 3-arm `select` (toggle-cell mux; mode-byte mux);
- an SMC-immediate accumulator (CUTOFF-style, but *different* opcode layout
  than defMON — this is the lifter's anti-overfit test);
- a mid-note SEQ latch (tie/legato: reseed without a gate edge);
- a 2× and an 8× multispeed variant of one of the above (binning envelope);
- a mid-tune CIA-latch change (cadence-as-state).

**5b. Parameter randomization.** Each micro-tune builder takes its constants
(steps, bounds, dwell counts, table contents, seed values) as arguments; tests
run a small fixed set of diverse parameterizations (deterministic seeds — CI
must stay reproducible), so fitters are exercised across their parameter
space, not at one point. A fitter that only recovers the magic values from an
HVSC example fails here.

**5c. Ablation tests.** Run `analyze` with engine-specific proposer *hints*
disabled (the Phase 2 hint table gets a test-only off switch) on one perfect
tune per engine and assert fidelity still reaches ≥ 0.999 via generic paths
alone (grammar proposers + lifter). This is the direct "generalizes to code
that hasn't been seen" check: if recovery of a known engine collapses without
its hints, the hints have become load-bearing knowledge and the generic path
has a gap. Hints may buy *speed*, never *correctness*.

**Accept:** suite green in CI (Docker, no HVSC needed — micro-tunes are
original and committed as source, satisfying the no-copyrighted-material
rule); every new grammar node or axis value lands with its micro-tune in the
same PR.

## 6. Deletions checklist (end state)

Gone: `_maybe_xor_ctrl`, `_maybe_and_ctrl`, `_maybe_or_reg`,
`_maybe_feeder_upgrade`, `_xor_pair`, `_and_pair`, `_or_pair`,
`segmented_pingpong`, `segmented_tickband` (folded into the product fitter),
`_recover_pair_overrides`, `_best_composite_override`, `_attach_or_prelude`,
`_recon_xor/_recon_and/_recon_or/_recon_feeder/_recon_composite/_recon_pitchwalk/
_recon_table/_recon_bacc*` (folded into `evaluate`), `_CUTOFF_SIG`/
`_defmon_cutoff`/`_recon_cutoff` (Phase 7), the threshold constants scattered
through adoption logic (arbiter λ + per-proposer prefilters remain, in one
documented place).

Stays: `_CellSampler` (unchanged — it is the grounding layer),
`state_sequence`, `_find_override` (extended), `_seq_correlation`,
`recover_tuning`/`voice_detune` (untouched), `trace.py`, `render.py`
(adapter-shimmed), the SEQ latch representation.

## 7. Efficiency items (fold into the phases they touch)

- `state_sequence` per-address boolean scans (recover.py:113–118): single
  argsort by `addr` + slice boundaries (Phase 1, while touching the file;
  keep the frozen-reference parity style of test used for `_table_walk_scan`).
- `_find_override` rebuilds its full candidate list per greedy iteration:
  build once per call, filter incrementally (Phase 2).
- Feeder/pair searches: Phase 4 narrowing is the real fix; do not
  micro-optimize them before it.
- Budget: `analyze` on the longest catalog fixture must stay < 60s CPU
  (current CI bound); add a timing assertion or at least log it in CI.

## 8. Expected outcomes (how to know it worked)

- Perfect set ≥ 20 throughout; expected to grow: Guns_n_Ghosts (6b),
  FC Tune_06/Manchester (5+6a+6b), Blackout (JCH — §3.3 envelope/6b, exact
  lane identified by the Phase 8 anchors), Commando fidelity 0.78 → ≥0.9
  (6d, plus the §3.5 I/O-read extension if its player reads osc3), AMIB
  0.991 → ≥0.999 (6d).
- After Phase 7, every register of every tune has an exact executable
  descriptor (§3.5 totality) — XSTATE is retired; the tracked axis becomes
  captured-state cost per register, driven down by Phases 2–6, with
  "zero Tier-3 descriptors above budget" as the new ratchet.
- `recover.py` net LOC shrinks despite new capability (the ×3 duplication and
  the guard zoo outweigh the new nodes).
- Adding the *next* idiom means: add a grammar node + proposer + synthetic
  test — no adoption guards, no threshold, no cascade edit. That is the
  definition of "no more whack-a-mole".
- Captured-state cost (`captured_frames`) becomes visible per register in
  `analyze` output — decompilation quality is now measured, not just
  round-trip.
- Optimized IR (Phase 9) is deterministic, canonically ordered, and minimal:
  multiply-defined instruments collapse to one pooled definition; transposed
  patterns, arpeggios, and per-note vibrato are expressed through the shared
  abstractions even where the original player never used them — with
  `round_trip` bit-identical before/after (the lossless gate). IR size (MDL
  total, pool size) becomes a tracked metric alongside fidelity, and is the
  input the 6502 reference replayer consumes.
- The emitted text IR (Phase 10) is a viable LLM training surface: canonical
  and deterministic, shallow (named defs, no deep nesting), low-entropy ids,
  note-space pitch with intrinsic 12-TET tables, integer-only numerics,
  captured data fenced from structure, per-register fidelity annotations for
  dataset filtering — while `parse∘emit` losslessness guarantees the corpus
  still regenerates every tune byte-exactly.
- The text IR works directly as a *prompt*: any 30s prefix is a valid,
  evaluable document; defs appear at first use so continuations naturally
  define new instruments/tables/patterns mid-stream with the next sequential
  ids; (prompt, continuation) pairs are exact prefix/suffix splits of a
  single canonical emission.
