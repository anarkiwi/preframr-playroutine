"""Render a recovered tune to WAV by replaying the IR-regenerated register stream.

The SID register stream is regenerated from the *recovered* tracker descriptors
(``analyze`` + ``reconstruct_register``), not the oracle, then driven through
reSIDfp (pyresidfp) to synthesise audio for ear-verification of a recovery.

``pyresidfp`` is an optional dependency (the ``audio`` extra). The pure-numpy
matrix builder needs only numpy; the audio path imports pyresidfp lazily.
"""

from __future__ import annotations

import argparse
import datetime
import wave

import numpy as np

from .recover import analyze, reconstruct_register, round_trip
from .recover import _CellSampler  # internal shared sampler
from .trace import Trace

# SID register file: $D400..$D418 (25 registers).
N_REGS = 25
SID_BASE = 0xD400

_MODELS = {"6581": "MOS6581", "8580": "MOS8580"}


def recovered_register_matrix(trace: Trace, kind: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    """Regenerate the ``$D400..$D418`` register stream from recovered descriptors.

    Pure numpy (no pyresidfp). Returns ``(ticks, matrix)`` where ``matrix`` has
    shape ``(n, 25)`` of uint8 register values per play tick, built from the
    recovered per-register descriptors via one shared ``_CellSampler``.
    """
    result = analyze(trace, kind)
    ticks = trace.tick_cycles(kind)
    n = len(ticks)
    sampler = _CellSampler(trace, ticks)
    matrix = np.zeros((n, N_REGS), dtype=np.uint8)
    for off in range(N_REGS):
        desc = result.get(SID_BASE + off)
        if desc is None:
            continue
        recon = reconstruct_register(desc, ticks, sampler=sampler)
        if recon is not None:
            matrix[:, off] = np.asarray(recon, dtype=np.int64) & 0xFF
    return ticks, matrix


def render_wav(
    trace: Trace,
    out_path: str,
    model: str = "6581",
    sample_rate: int = 44100,
    kind: str = "auto",
) -> dict:
    """Render the recovered register stream to a 16-bit mono WAV via reSIDfp.

    ``model`` is ``"6581"`` or ``"8580"``. The CPU clock comes from
    ``trace.cpu_hz``. Returns a stats dict including the recovery ``overall``
    round-trip fidelity and the count of register mismatches versus the oracle
    (``trace.register_frames()``) as a correctness signal.
    """
    try:
        from pyresidfp import SoundInterfaceDevice  # pylint: disable=import-outside-toplevel
        from pyresidfp._pyresidfp import (  # pylint: disable=import-outside-toplevel
            ChipModel,
            SamplingMethod,
        )
        from pyresidfp.registers import (  # pylint: disable=import-outside-toplevel
            WritableRegister,
        )
    except ImportError as exc:
        raise ImportError(
            "pyresidfp is required for WAV rendering: pip install 'preframr_playroutine[audio]'"
        ) from exc

    if model not in _MODELS:
        raise ValueError(f"model must be one of {sorted(_MODELS)}, got {model!r}")

    ticks, matrix = recovered_register_matrix(trace, kind)
    n = len(ticks)
    clock_hz = trace.cpu_hz

    chip_model = getattr(ChipModel, _MODELS[model])
    sid = SoundInterfaceDevice(
        model=chip_model,
        sampling_method=SamplingMethod.RESAMPLE,
        clock_frequency=clock_hz,
        sampling_frequency=float(sample_rate),
    )
    regs = [WritableRegister(off) for off in range(N_REGS)]

    deltas = np.diff(ticks.astype(np.int64))
    deltas = np.append(deltas, int(np.median(deltas)) if len(deltas) else 0)

    samples: list = []
    prev = np.full(N_REGS, -1, dtype=np.int64)
    for i in range(n):
        row = matrix[i]
        changed = np.nonzero(row != prev)[0]
        for off in changed:
            sid.write_register(regs[off], int(row[off]))
        prev = row.astype(np.int64)
        samples.extend(sid.clock(datetime.timedelta(seconds=float(deltas[i]) / clock_hz)))

    audio = np.asarray(samples, dtype=np.int16)
    _write_mono16(out_path, audio, int(sample_rate))

    oracle = trace.register_frames(kind=kind)[1][:, :N_REGS]
    mismatches = int(np.count_nonzero(oracle.astype(np.int64) != matrix.astype(np.int64)))
    return {
        "overall": round_trip(trace, kind)["overall"],
        "register_mismatches": mismatches,
        "samples": int(len(audio)),
        "seconds": len(audio) / float(sample_rate),
        "model": model,
    }


def _write_mono16(out_path: str, audio: np.ndarray, sample_rate: int) -> None:
    """Write ``audio`` (int16) as a 16-bit mono WAV at ``sample_rate``."""
    # pylint: disable=no-member
    with wave.open(out_path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio.tobytes())


def sid_model_from_file(sid_path: str) -> str:
    """Return ``"6581"`` or ``"8580"`` from a ``.sid`` header's model flags.

    Reads the big-endian u16 ``flags`` field at offset 0x76; bits 4-5 encode the
    model (1=6581, 2=8580). Returns ``"6581"`` when unknown or both are set.
    """
    with open(sid_path, "rb") as handle:
        head = handle.read(0x78)
    if len(head) < 0x78:
        return "6581"
    flags = (head[0x76] << 8) | head[0x77]
    model_bits = (flags >> 4) & 0x3
    return "8580" if model_bits == 2 else "6581"


def main(argv: list | None = None) -> None:
    """CLI: render a recovered trace to a WAV file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prefix", help="trace prefix, .bin or .json (Trace.load accepts any)")
    parser.add_argument("out", help="output WAV path")
    parser.add_argument("--model", choices=sorted(_MODELS), default="6581")
    parser.add_argument("--sid", help="auto-detect model from a .sid header (overrides --model)")
    parser.add_argument("--rate", type=int, default=44100, help="sample rate (Hz)")
    parser.add_argument("--kind", default="auto", help="tick kind: auto/irq/nmi/both")
    args = parser.parse_args(argv)

    model = sid_model_from_file(args.sid) if args.sid else args.model
    trace = Trace.load(args.prefix)
    stats = render_wav(trace, args.out, model=model, sample_rate=args.rate, kind=args.kind)
    print(stats)


if __name__ == "__main__":
    main()
