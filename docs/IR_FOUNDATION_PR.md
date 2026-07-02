# Foundation PR: expression IR + parity harness + fidelity snapshot

Standalone, mechanically-executable instructions for ONE pull request. No
research or design decisions are required: every schema, mapping, and test is
specified here. Parent design (context only, do NOT implement its later
phases): `docs/GENERIC_RECOVERY.md`. This PR implements its Phase 0 and the
reconstruction side of Phase 1.

## Mission

Introduce `preframr_playroutine/ir.py` — a typed expression IR with a single
recursive evaluator — and switch `reconstruct_register` to run through it,
**byte-for-byte behavior-preserving**, proven by golden fixtures captured from
the unmodified code before the refactor. Also land the fidelity-snapshot
ratchet mechanism. Open a PR, watch CI, fix failures, merge on green.

## Scope

IN: golden parity fixtures; `ir.py` (`to_ir`, `evaluate`); typed override
predicates with a legacy shim; deletion of the superseded `_recon_*` bodies;
snapshot mechanism (`tests/update_snapshot.py`, snapshot assertions,
committed snapshot file — empty `{}` if it cannot be generated in this
environment).

OUT (do not do): any change to classification/search logic (`classify_register`
and everything it calls stays semantically identical), the proposer/arbiter,
new grammar nodes, the optimizer, the text serialization, performance work,
render.py changes beyond none. If a change outside scope seems needed, stop
and report instead.

## Hard invariants

- `pytest -q` green locally (`pip install -e ".[test]"`; config in
  pyproject.toml runs xdist + coverage `--cov-fail-under=85`).
- `black --check preframr_playroutine tests` and `pylint preframr_playroutine`
  clean (CI runs exactly these).
- Public API signatures unchanged: `reconstruct_register(descriptor, ticks,
  trace=None, sampler=None)`, `analyze`, `round_trip`, `classify_register`,
  `state_sequence`, `fit_bacc`, `detect_table_walk`.
- Golden parity: for every fixture descriptor, the new path reproduces the
  old output exactly (including `None` results).
- Do not reformat or restructure code you are not changing.

## Step-by-step (commit after each step; PR after step 1)

### Step 1 — golden fixtures from UNMODIFIED code (first commit)

Create `tests/make_ir_golden.py` (executable script, NOT named `test_*` so
pytest does not collect it) and `tests/test_ir.py`. Run the script on the
unmodified tree to produce `tests/fixtures/ir_golden.npz`; commit script,
fixture, and test together, and verify `pytest tests/test_ir.py` passes
BEFORE any refactor (the test asserts old-code self-consistency at this
point).

**Synthetic trace.** Both files share one deterministic fake trace (no
sidtrace binary, no HVSC, no randomness). Implement in `tests/test_ir.py`
(imported by the script):

```python
class FakeTrace:
    """Minimal duck-type for _CellSampler + _default_until_first_write."""
    def __init__(self, ram_writes, sid_writes):
        self._rw, self._sw = ram_writes, sid_writes
    def ram_writes(self, kind=None):
        del kind
        return self._rw
    def sid_writes(self, chip=None):
        del chip
        return self._sw
```

Build with `preframr_playroutine.EVENT_DTYPE` and
`preframr_playroutine.trace.RAMACCESS_DTYPE`:

- `ticks = np.arange(16, dtype=np.uint64) * 1000` (16 frames).
- SID writes: for each frame `f` and each SID addr in
  `{0xD402, 0xD404, 0xD416, 0xD418, 0xD400, 0xD401}`, one EVENT record with
  `cycle = ticks[f] + 10`, `etype=0`, `addr=<sid addr>`, `value=0`,
  `aux=0x1200` — EXCEPT: `0xD418` gets its first write only at frame 3
  (frames 0–2 have none), to exercise `written_mask` /
  `_default_until_first_write`.
- RAM writes (RAMACCESS records, `kind=0`, `pc=0x1200`), per frame `f`:
  - `0x10` value `f & 0xFF` at `cycle = ticks[f] + 5` (written BEFORE the SID
    write: `at_write` sees frame `f`'s value).
  - `0x11` value `(f * 3) & 0xFF` at `cycle = ticks[f] + 50` (written AFTER
    the SID write: `at_write` sees the PREVIOUS frame's value; `eof` sees
    frame `f`'s).
  - `0x12` TWO writes per frame: value `(f * 5) & 0xFF` at `+4` and value
    `(f * 5 + 1) & 0xFF` at `+6` (exercises `operand`, i.e. `back=2`).
  - `0x13` value `0xFF if f < 8 else 0xFE` at `+5` (a gate cell).
  - `0x14` value `f // 4` at `+5` (a slow cursor).
  - `0x15` value `0x40 if f % 2 == 0 else 0x41` at `+5`.
  - `0x16` first written at frame 6 (value `0x30 + f` at `+5`, frames ≥ 6
    only) — exercises `first_live_frame` and preludes.

Concatenate, do NOT sort (the code sorts internally); wrap in `FakeTrace`.

**RAM image** (for table descriptors): `ram = np.zeros(65536, np.uint8)` with
`ram[0x2000:0x2010] = [0x41,0x11,0x21,0x81,0x09,0x41,0x11,0x21,
0x81,0x09,0x41,0x11,0x21,0x81,0x09,0x41]`, and
`ram[0x3000:0x3010] = np.arange(16)`, `ram[0x3100:0x3110] =
np.arange(16) + 1` (pitch lo/hi tables).

**Descriptor set.** Reconstruct each of the following with
`reconstruct_register(desc, ticks, sampler=_CellSampler(fake_trace, ticks))`
and save `(key, output-or-sentinel)` into the `.npz` (store `None` results as
an array `[-1]` under key `<name>__none`). Every descriptor includes
`"addr"` (use the SID addr it targets) so `_default_until_first_write` runs.
Exact set (name: fields — copy verbatim, values chosen to exercise the
branches noted):

 1. `const_v`: `{"type":"CONST","value":7,"addr":0xD402}`
 2. `const_none`: `{"type":"CONST","value":None,"addr":0xD402}`
 3. `seq`: `{"type":"SEQ","latch_frames":[0,4,9],"latch_values":[3,8,2],
    "addr":0xD402}`
 4. `bacc_saw8`: `{"type":"BACC","mode":"saw","step":3,"lo":0,"hi":12,
    "phase":0,"modulus":15,"width":8,"byte_role":"full","resets":[0],
    "seeds":[0],"addr":0xD402}`
 5. `bacc_reflect16_lo` / `bacc_reflect16_hi`: `{"type":"BACC",
    "mode":"reflect","step":40,"lo":256,"hi":900,"width":16,
    "byte_role":"lo"|"hi","resets":[0,6],"seeds":[300,700],
    "segmented":True,"addr":0xD402}`
 6. `bacc_pingpong`: same shape plus `"mode":"pingpong","down_step":25,
    "clamp_lo":256,"clamp_hi":900,"steps":[40,30],"down_steps":[25,20],
    "directions":[1,-1]`
 7. `bacc_tickband`: `{"type":"BACC","mode":"tickband","step":2,"lo":100,
    "hi":400,"segmented":True,"resets":[0,8],"seeds":[120,150],
    "directions":[1,1],
    "rate_tables":[np.array([2,2,4,4,8],dtype=np.int64)],
    "seg_tables":[0,0],"n_segments":2,"addr":0xD402}`
 8. `bacc_cellfed`: `{"type":"BACC","mode":"saw","step":1,"lo":0,"hi":255,
    "cell":0x16,"sid":0xD402,"prelude_end":6,"prelude_frames":[0,2],
    "prelude_values":[9,11],"resets":[0],"seeds":[0],"byte_role":"full",
    "width":8,"addr":0xD402}`
 9. `walk_mask`: `{"type":"TABLE_WALK","base":0x2000,"stride":1,"length":16,
    "loop":0,"table":ram[0x2000:0x2010].copy(),"mask":0xFE,
    "cursor_addr":0x14,"cursor_offset":1,"addr":0xD404}`
10. `walk_gate_override`: same as 9 plus `"gate_addr":0x13,"mask":0xFF,
    "overrides":[{"predicate":[(0x15,0xFF,0x41)],"force":0x08}]`
11. `composite16`: `{"type":"COMPOSITE","byte_role":"lo","width_mask":0xFFFF,
    "base":{"lo":(0x10,0xD400),"hi":(0x11,0xD401)},
    "mod":{"lo":(0x12,0xD400),"hi":(0x14,0xD401)},
    "overrides":[{"predicate":[(0x13,0x01,0x00)],"force":0xFF}],
    "addr":0xD400}`
12. `composite8`: `{"type":"COMPOSITE","byte_role":"full","width_mask":0xFF,
    "base":{"cell":0x10,"sid":0xD402},"mod":None,
    "overrides":[{"predicate":[(0x15,"in",(0x40,))],"force":0x77}],
    "addr":0xD402}`
13. `pitchwalk_lo` / `pitchwalk_hi`: `{"type":"PITCHWALK",
    "byte_role":"lo"|"hi","lo_base":0x3000,"hi_base":0x3100,
    "lo_table":ram[0x3000:0x3010].copy(),
    "hi_table":ram[0x3100:0x3110].copy(),"index_cells":[0x14,0x10],
    "overrides":[],"addr":0xD400}`
14. `feeder`: `{"type":"FEEDER","cell":0x10,"sid":0xD416,"addr":0xD416}`
15. `xor`: `{"type":"XOR","cell_a":0x10,"cell_b":0x15,"sid":0xD404,
    "addr":0xD404}`
16. `and_ov`: `{"type":"AND","cell_a":0x10,"cell_b":0x13,"sid":0xD404,
    "overrides":[{"predicate":[(0x14,0xFF,0x02)],"force":0x81}],
    "addr":0xD404}`
17. `or_cells`: `{"type":"OR","cell_a":0x10,"cell_b":0x15,"sid":0xD418,
    "addr":0xD418}` (0xD418's late first write exercises the default-0 mask)
18. `or_const_prelude`: `{"type":"OR","cell_a":0x16,"const":0x0F,
    "sid":0xD418,"prelude_end":6,"prelude_frames":[0,3],
    "prelude_values":[0x1F,0x2F],"addr":0xD418}`
19. `xstate_cell`: `{"type":"XSTATE","cell":0x11,"sid":0xD402,"addr":0xD402}`
20. `xstate_bare`: `{"type":"XSTATE","addr":0xD402}` → result is `None`
21. `cutoff`: build the CUTOFF descriptor mechanically: place the opcode/
    operand bytes required by `_CUTOFF_OFFS` into the RAM-write log as cells
    (addresses `0x1200+off` for each named offset, one write per frame at
    `+5`, values: `op_lo`/`op_hi` cells constant `0x69`, `slo`=2, `shi`=0,
    `lo`=frame`*2`, `hi`=frame, `imm`=5, `base`=2, `scale` cell `0x0A`), then
    `{"type":"CUTOFF","addr":0xD416,"sid":0xD416,"cells":{name: 0x1200+off
    for name, off in _CUTOFF_OFFS.items()},"base":2,"imm":5,"scale":2}`.
    (Import `_CUTOFF_OFFS` from `preframr_playroutine.recover`.)

**`tests/test_ir.py`** loads the `.npz` and asserts, for every key, that
`reconstruct_register(desc, ticks, sampler=fresh_sampler())` equals the golden
array (or is `None` for the sentinel). Descriptors are rebuilt in the test by
calling the same `build_descriptors()` helper the script uses (put
`build_descriptors`/`build_fake_trace` in `tests/_ir_fixture.py` shared by
both). Also assert one property directly: sampler mode coverage —
`at_write(0x11, 0xD402)[f] == ((f-1)*3)&0xFF for f>=1` (write-after-store
lag) and `operand(0x12, 0xD402)[f] == (f*5)&0xFF` (back=2).

Run `python tests/make_ir_golden.py`, then `pytest -q`; commit
(`ir: golden parity fixtures for reconstruction refactor`). Push the branch
and open the PR now (`gh pr create --fill`); subsequent steps push to it.

### Step 2 — `preframr_playroutine/ir.py`

New module. Node schemas (plain dicts; numpy arrays allowed as values):

```
{"op":"post", "expr":node, "byte_role":"full"|"lo"|"hi",
 "width_mask":int|None, "overrides":[...], "prelude":dict|None,
 "addr":int, "sid":int|None}
{"op":"const", "value":int}                      # evaluate -> full(value)
{"op":"seq", "frames":[...], "values":[...]}     # port _recon_seq
{"op":"cell", "addr":int, "sample":"write"|"eof"|"operand", "sid":int}
{"op":"lohi", "lo":node, "hi":node}              # lo | (hi<<8)
{"op":"table", "data":uint8[], "index":node, "stride":int, "offset":int}
                                                  # data[clip(index*stride+offset, 0, len-1)]
{"op":"binop", "fn":"or"|"and"|"xor"|"add", "a":node, "b":node}
{"op":"recur", ...}      # every BACC field verbatim; evaluate ports
                          # _recon_bacc_full (incl. tickband) unchanged
{"op":"cutoff", ...}      # every CUTOFF field verbatim; evaluate ports
                          # _recon_cutoff unchanged
```

`evaluate(node, n, sampler) -> np.ndarray | None`: recursive; all arithmetic
in int64 exactly as the current `_recon_*` bodies do (port their code — do
not rewrite logic). `post` applies, in this fixed order:

1. `v = evaluate(expr)`; if expr is None (bare XSTATE) return None.
2. `v &= width_mask` if width_mask is not None.
3. byte extract by `byte_role` (`lo`: `&0xFF`; `hi`: `>>8 &0xFF`; `full`:
   unchanged — note `full` recur output is NOT masked, matching
   `_recon_bacc`).
4. `_apply_overrides(v, overrides, sampler)`.
5. prelude overlay: frames `< prelude["end"]` replaced by the SEQ-latched
   prelude values `& 0xFF` (port from `_recon_bacc`/`_recon_or`).
6. `_default_until_first_write` using `addr` (move that function, unchanged,
   into ir.py; keep a re-export in recover.py).

`to_ir(descriptor) -> node | None` — exact mapping (fields not listed are
copied where named):

| legacy | tree |
|---|---|
| CONST | `post(const(value or 0))` |
| SEQ | `post(seq(latch_frames, latch_values))` |
| FEEDER | `post(cell(cell,"write",sid), width_mask=0xFF)` |
| XSTATE with cell | same as FEEDER |
| XSTATE without cell | `None` |
| BACC with `cell` | `post(cell(cell,"write",sid), width_mask=0xFF, prelude=…)` |
| BACC without `cell` | `post(recur(<all fields>), byte_role=byte_role or "full")` |
| TABLE_WALK | `idx = cell(cursor_addr,"eof",addr)`; `t = table(table, idx, stride, cursor_offset)`; expr = `binop(and, t, cell(gate_addr,"eof",addr))` if gate_addr else `binop(and, t, const(mask))`; `post(expr, overrides=overrides)` |
| TABLE_WALK cursor_addr None | `post(const(0))` (matches `_recon_table` zeros) — only when no `"cursor"` series either; a literal `"cursor"` series is not produced by classification, ignore it |
| COMPOSITE | parts: pair `{"lo":(c,s),"hi":(c,s)}` → `lohi(cell(c,"write",s), cell(c,"write",s))`; `{"cell","sid"}` → `cell(...,"write",...)`; `{"series"}` → `const`-like literal node `{"op":"literal","data":series}` (add it: evaluate returns the array); expr = `binop(add, base, mod)` (mod omitted → base alone); `post(expr, width_mask=width_mask, byte_role=byte_role, overrides=overrides)` |
| PITCHWALK | `idx = binop(add, cell(index_cells[0],"eof",addr), …)` folded left over index_cells (single cell → just the cell); expr = `lohi(table(lo_table, idx, 1, 0), table(hi_table, idx, 1, 0))`; `post(expr, byte_role=byte_role, overrides=overrides)`. CAUTION: `_recon_pitchwalk` clips the SUMMED index once against the table length — so evaluate `table` with a SHARED pre-clipped index: implement by giving both tables the same `index` node and clipping inside `table` (lengths are equal; verify in `to_ir`, assert equal lengths). |
| XOR / AND / OR | `binop(fn, cell(cell_a,"write",sid), second)` where second = `cell(cell_b,"write",sid)` or `const(const)`; `post(..., width_mask=0xFF, overrides=overrides (AND), prelude=… (OR))` |
| CUTOFF | `post(cutoff(<fields>), addr=addr)` (cutoff evaluates the full pipeline internally as today; post applies only step 6) |

Sanity note on override/prelude placement: only AND carries overrides among
the pair ops today, and only OR/cell-fed-BACC carry preludes; `to_ir` copies
whatever keys exist — the fixed `post` order reproduces all current
combinations (verified by the goldens).

### Step 3 — switch `reconstruct_register`; typed predicates

- `reconstruct_register` body becomes: build sampler as today →
  `node = to_ir(descriptor)` → `return ir.evaluate(node, len(ticks),
  sampler)` (None propagates). Keep the docstring.
- Typed predicates: in `_find_override` (recover.py), emit dict terms
  `{"kind":"eq","cell":a,"value":v}`, `{"kind":"bit","cell":a,"mask":m,
  "value":v}`, `{"kind":"in","cell":a,"values":(…)}` instead of tuples. Add
  `{"kind":"mask", "cell":a,"mask":m,"value":v}` as the general form ("eq" =
  mask 0xFF, "bit" = single-bit mask; implement all three via one code path).
  `_apply_overrides` (moved to ir.py) accepts BOTH forms: legacy tuples
  (`(cell, int_mask, value)` and `(cell, "in", values)`) via a
  `_predicate_terms()` normalizer, and the new dicts. Golden fixtures use
  legacy tuples — they prove the shim.
- Update any test in `tests/test_recover.py` that asserts tuple-form
  predicates to the dict form (mechanical substitution; list the changed
  assertions in the PR description).

Run `pytest -q` — the golden test now proves the new path equals the old
outputs. Commit (`ir: single evaluator + to_ir adapter, typed predicates`).

### Step 4 — delete superseded code

Delete from recover.py: `_recon_seq`, `_run_recurrence`, `_recon_tickband`,
`_recon_bacc_full`, `_recon_bacc`, `_recon_table`, `_recon_composite`,
`_recon_pitchwalk`, `_recon_feeder`, `_recon_cutoff`, `_recon_xor`,
`_recon_and`, `_recon_or`, `_comp_part`, and the builders dispatch — their
ports now live in ir.py (module-private there; internal recover.py callers
like `_seg_fidelity`, `_recon_cutoff` usage in `_defmon_cutoff`,
`_and_recon_masked`, `_recover_walk_overrides` switch to
`ir.evaluate(to_ir(...))` or to small ir.py helpers — keep those call sites'
behavior identical; the goldens plus the full unit suite are the check).
`_apply_overrides` and `_default_until_first_write` live in ir.py with
re-exports if any test imports them. Run black, pylint (no unused
imports/vars), `pytest -q`. Commit (`ir: remove superseded _recon_* bodies`).

### Step 5 — fidelity snapshot mechanism

- `tests/fixtures/fidelity_snapshot.json`: commit `{}` initially.
- `tests/update_snapshot.py` (not pytest-collected; mirror the structure of
  `tests/straggler_report.py`, which already renders every catalog fixture):
  for each renderable fixture, run `round_trip`, write
  `{"<family>:<basename>:<subtune>": {"overall": x, "regs":
  {"0xd402": f, ...}}}` sorted, floats rounded to 6 places.
- In `tests/test_hvsc.py`, inside the existing whole-song perfect/xfail test
  after `round_trip`: load the snapshot once (module level); if the tune's
  key is present, assert every recorded register's fidelity
  `>= snapshot - 1e-9` and `overall >= snapshot["overall"] - 1e-9`. Absent
  key → no assertion (keeps CI green with `{}`).
- Then TRY to populate it: if `docker` is available, run
  `docker build -t preframr-playroutine:snap .` and run the image with
  `-e HVSC_BASE_URL=https://hvsc.c64.org/download/C64Music` and entrypoint
  `python tests/update_snapshot.py` (mount or `docker cp` the json out). If
  docker or network is unavailable, keep `{}` and say so in the PR
  description — the mechanism still lands.

Commit (`tests: per-register fidelity snapshot ratchet`).

### Step 6 — finish the PR

- `black preframr_playroutine tests`, `pylint preframr_playroutine`,
  `pytest -q` — all clean locally.
- Push; watch `gh pr checks <num> --watch`. Both CI jobs must pass
  (`lint-and-unit`, and `docker-oracle` which runs the full suite including
  the 26 whole-song HVSC tunes — this is the real behavior-preservation
  gate: every `_PERFECT` tune must still assert `overall == 1.0`).
- Fix any failure and re-push. Merge on green (squash). Do not merge with
  any red check; do not disable or xfail a failing test to get green.

## Pitfalls (read before coding)

- **Do not "improve" logic while porting.** `_recon_*` bodies move verbatim;
  int64 dtypes, `np.clip` bounds, `& 0xFF` placements, and the unmasked
  `full`-role recur output are all load-bearing. The goldens will catch you.
- Cycle arithmetic is uint64 (`sample = write_cycles + np.uint64(2)`); keep
  numpy unsigned types intact when building the fake trace.
- `_CellSampler` caches by int keys; always `int()` addresses.
- `reconstruct_register` must return `None` (not zeros) for bare XSTATE and
  unknown types — `round_trip` depends on it.
- `test_analysis.py` / `test_recover.py` may exercise internals you moved;
  prefer re-exports in recover.py over editing those tests, except the
  predicate-tuple assertions (Step 3).
- Coverage must stay ≥ 85% overall; ir.py is fully covered by the golden test
  if the descriptor set above is implemented completely — do not skip
  descriptors.
- pyproject pytest addopts already include `-n auto --cov`; just run
  `pytest -q`.

## Definition of done

PR merged to main with: golden fixtures committed and passing against the new
evaluator; `ir.py` the only reconstruction path; typed predicates emitted (+
legacy shim proven by goldens); superseded `_recon_*` deleted; snapshot
mechanism live (populated if the environment allowed); both CI jobs green;
no fidelity change on any HVSC fixture.
