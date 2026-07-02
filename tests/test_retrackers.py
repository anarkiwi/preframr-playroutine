"""Deeper re-trackers cross-check (license-walled; skipped in CI).

Phase 8 of docs/GENERIC_RECOVERY.md: use the reverse-engineering ground truth in
the PRIVATE ``re-trackers`` repo as a regression suite, respecting the license
wall (section 2). This module is TEST-ONLY and touches NO recovery code.

License wall (binding): ``re-trackers`` is a private repo of copyrighted-derived
disassemblies. This test READS it at runtime to confirm the FACTS (cell
addresses) that the public ``tests/test_hvsc.py`` anchors hardcode -- it never
copies prose, asm, or table contents into this repo. It ``pytest.skip``s cleanly
when the repo is absent (the CI case) or the render toolchain is unavailable, so
CI (which has neither the private repo nor, by default, these HVSC renders in
this module) never runs it.

The check has two halves per anchored engine:
  1. content cross-check -- grep the engine's generator doc for the documented
     cell-address tokens, so drift between the public hardcoded facts and the
     private ground truth is caught; and
  2. structure check -- render the catalog fixture (reusing test_hvsc's render
     plumbing via ``_render``) and assert the recovered descriptors reference
     those exact cells / shapes.
"""

import os

import pytest

from preframr_playroutine import analyze, round_trip

from _hvsc import fetchable, load_catalog
from _render import HAVE_SIDTRACE, render_tune

RETRACKERS_PATH = os.environ.get("RETRACKERS_PATH", "/scratch/anarkiwi/cbm/re-trackers")

CATALOG = load_catalog()

if not os.path.isdir(RETRACKERS_PATH):
    pytest.skip(
        f"re-trackers not present at {RETRACKERS_PATH} (set RETRACKERS_PATH)",
        allow_module_level=True,
    )
if not HAVE_SIDTRACE:
    pytest.skip("sidtrace binary not available", allow_module_level=True)
if not CATALOG:
    pytest.skip("tests/fixtures/tunes.json missing", allow_module_level=True)

_GATE_MASKS = (0xFE, 0xFF)


def _doc_tokens(relpath, tokens):
    """Assert every ``token`` string appears in the re-trackers doc ``relpath``.

    Reads the private file at test time (never vendored); returns the text so a
    check can grep further. Missing file -> skip (repo layout changed upstream).
    """
    path = os.path.join(RETRACKERS_PATH, relpath)
    if not os.path.exists(path):
        pytest.skip(f"re-trackers doc missing: {relpath}")
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    missing = [t for t in tokens if t not in text]
    assert not missing, (relpath, missing)
    return text


def _walk_ctrls(result, addrs):
    return {a: result[a] for a in addrs if result.get(a, {}).get("type") == "TABLE_WALK"}


_CTRL = (0xD404, 0xD40B, 0xD412)


def _check_dmc(result, trace):
    # dmc-generators.md documents: CTRL = waveval($1780) AND gatemask($100F), a
    # TABLE-WALK $18AD[wavecur $177x]; PW pw_min/pw_max bound cells $1756/$1759.
    _doc_tokens("DMC/dmc-generators.md", ("$177A", "$100F", "$1756", "$1759"))
    walks = _walk_ctrls(result, _CTRL)
    assert walks, {hex(a): result.get(a, {}).get("type") for a in _CTRL}
    for addr, desc in walks.items():
        cur = int(desc["cursor_addr"])
        assert 0x1770 <= cur <= 0x178F, (hex(addr), hex(cur))
        assert int(desc["mask"]) in _GATE_MASKS, (hex(addr), hex(int(desc["mask"])))
    # PW voice 1 ($D40A hi byte) reflects between pw_min ($1756) and pw_max
    # ($1759): the recovered reflect bounds stay within those image-seeded values.
    img = trace.ram_image()
    b0, b1 = int(img[0x1756]), int(img[0x1759])
    lo_b, hi_b = min(b0, b1), max(b0, b1)
    pw = result.get(0xD40A, {})
    if pw.get("type") == "BACC" and "lo" in pw and "hi" in pw:
        assert lo_b <= min(int(pw["lo"]), int(pw["hi"])), (pw.get("lo"), pw.get("hi"), lo_b)
        assert max(int(pw["lo"]), int(pw["hi"])) <= hi_b, (pw.get("lo"), pw.get("hi"), hi_b)


def _check_gt2(result, _trace):
    # goattracker2-generators.md: CTRL = chnwave ($93B9) AND chngate ($93D0),
    # chnwave = wavetbl[chnwaveptr $93B8].
    _doc_tokens("GoatTracker2/goattracker2-generators.md", ("$93B9", "$93B8", "$93D0"))
    for addr, desc in _walk_ctrls(result, _CTRL).items():
        cur = int(desc["cursor_addr"])
        assert 0x93B0 <= cur <= 0x93BF, (hex(addr), hex(cur))
        assert int(desc["mask"]) in _GATE_MASKS, (hex(addr), hex(int(desc["mask"])))


def _check_jch(result, _trace):
    # jch-generators.md: MODE/VOL $D418 = $1793 | $1009.
    _doc_tokens("JCH_NewPlayer/jch-generators.md", ("$1793", "$1009", "$D418"))
    d418 = result.get(0xD418, {})
    assert d418.get("type") in ("OR", "SEQ", "CONST"), d418
    if d418.get("type") == "OR":
        cells = {int(d418[k]) for k in ("cell_a", "cell_b") if k in d418}
        assert {0x1793, 0x1009} <= cells, d418


_CHECKS = {
    ("DMC", "Doctagop.sid"): _check_dmc,
    ("GoatTracker2", "Raindrops.sid"): _check_gt2,
    ("JCH_NewPlayer", "24th_Amaranth_Grand_Prix_3.sid"): _check_jch,
}


def _key(entry):
    return (entry.get("family"), os.path.basename(entry["path"]))


def _ids(entry):
    return f"{entry.get('family', '?')}:{os.path.basename(entry['path'])}:{entry.get('subtune', 1)}"


_ENTRIES = [e for e in CATALOG if _key(e) in _CHECKS]


@pytest.mark.parametrize("entry", _ENTRIES, ids=[_ids(e) for e in _ENTRIES])
def test_retrackers_facts(entry, tmp_path_factory):
    """Cross-check the recovered structure against re-trackers cell-address facts."""
    if not fetchable(entry):
        pytest.skip(f"tune not fetchable: {entry['path']}")
    work = tmp_path_factory.mktemp("retrackers")
    _sid, _prefix, trace = render_tune(entry, work)
    result = analyze(trace)
    fid = round_trip(trace)
    assert fid["overall"] >= 0.95, fid["overall"]
    _CHECKS[_key(entry)](result, trace)
