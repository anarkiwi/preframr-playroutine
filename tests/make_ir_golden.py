"""Capture golden reconstruction outputs for the parity test.

Runs ``reconstruct_register`` over the shared descriptor set against the shared
fake trace and writes ``tests/fixtures/ir_golden.npz``. NOT named ``test_*`` so
pytest does not collect it. Run once on the unmodified tree before the IR
refactor; ``tests/test_ir.py`` then asserts the evaluator still reproduces it.

    python tests/make_ir_golden.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preframr_playroutine.recover import _CellSampler, reconstruct_register  # noqa: E402
from _ir_fixture import build_descriptors, build_fake_trace, build_ticks  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "ir_golden.npz")


def main():
    ticks = build_ticks()
    trace = build_fake_trace()
    descs = build_descriptors()
    saved = {}
    for name, desc in descs.items():
        recon = reconstruct_register(desc, ticks, sampler=_CellSampler(trace, ticks))
        if recon is None:
            saved[f"{name}__none"] = np.array([-1], dtype=np.int64)
        else:
            saved[name] = np.asarray(recon, dtype=np.int64)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, **saved)
    print(f"wrote {OUT} ({len(saved)} keys)")


if __name__ == "__main__":
    main()
