"""Load and analyse sidtrace oracle output.

``sidtrace`` writes a flat ``<prefix>.bin`` of 16-byte event records plus a
``<prefix>.json`` metadata sidecar. This module loads them into numpy and
recovers the periodic update structure (PAL/NTSC raster vs CIA timer driven)
and the per-frame SID register tables used by table/sweep/BACC players.
"""

from __future__ import annotations

import json
import os

import numpy as np

# Must match the Record layout in app/sidtrace.cpp (fields little-endian).
EVENT_DTYPE = np.dtype(
    [
        ("cycle", "<u8"),
        ("etype", "u1"),
        ("chip", "u1"),
        ("reg", "u1"),
        ("value", "u1"),
        ("addr", "<u2"),
        ("aux", "<u2"),
    ]
)

# Event types.
SID_WRITE = 0
CIA_IRQ = 1
VIC_IRQ = 2
CPU_VECTOR = 3

# CPU vector kinds (low byte of the 6502 interrupt vector address).
VEC_IRQ = 0xFE
VEC_NMI = 0xFA
VEC_RST = 0xFC

# CIA interrupt sources (chip field of CIA_IRQ events).
SRC_CIA1 = 1  # drives IRQ
SRC_CIA2 = 2  # drives NMI


def load_events(path: str) -> np.ndarray:
    """Load a ``.bin`` event file as a structured array."""
    return np.fromfile(path, dtype=EVENT_DTYPE)


def _resolve(prefix: str) -> tuple[str, str]:
    """Return (bin_path, json_path) from a prefix, .bin or .json path."""
    if prefix.endswith(".bin"):
        base = prefix[:-4]
    elif prefix.endswith(".json"):
        base = prefix[:-5]
    else:
        base = prefix
    return base + ".bin", base + ".json"


class Trace:
    """A loaded oracle trace: structured events plus metadata."""

    def __init__(self, events: np.ndarray, meta: dict):
        self.events = events
        self.meta = meta

    # -- loading -----------------------------------------------------------

    @classmethod
    def load(cls, prefix: str) -> "Trace":
        """Load a trace from a prefix, ``.bin`` or ``.json`` path."""
        bin_path, json_path = _resolve(prefix)
        events = load_events(bin_path)
        meta = {}
        if os.path.exists(json_path):
            with open(json_path, encoding="utf-8") as handle:
                meta = json.load(handle)
        return cls(events, meta)

    @classmethod
    def from_events(cls, events: np.ndarray, meta: dict | None = None) -> "Trace":
        """Build a trace directly from an event array (e.g. for tests)."""
        return cls(events, meta or {})

    # -- selectors ---------------------------------------------------------

    def sid_writes(self, chip: int | None = None) -> np.ndarray:
        """SID write events, optionally filtered to one chip index."""
        evs = self.events[self.events["etype"] == SID_WRITE]
        if chip is not None:
            evs = evs[evs["chip"] == chip]
        return evs

    def interrupts(self) -> np.ndarray:
        """All interrupt-line assertion events (CIA and VIC)."""
        et = self.events["etype"]
        return self.events[(et == CIA_IRQ) | (et == VIC_IRQ)]

    def cpu_vectors(self, kind: int | None = None) -> np.ndarray:
        """CPU interrupt vector-through events, optionally filtered by kind."""
        evs = self.events[self.events["etype"] == CPU_VECTOR]
        if kind is not None:
            evs = evs[evs["value"] == kind]
        return evs

    @property
    def cpu_hz(self) -> float:
        """Effective CPU clock in Hz (from metadata, default PAL)."""
        return float(self.meta.get("cpu_hz", 985248.444))

    @property
    def refresh_hz(self) -> float:
        """Nominal video refresh (frame) rate for the effective model."""
        model = self.meta.get("effective_model", "PAL")
        return 60.0 if model.startswith("NTSC") else 50.0

    @property
    def frame_cycles(self) -> float:
        """Cycles in one video frame (single-speed VBI period)."""
        return self.cpu_hz / self.refresh_hz

    # -- timing analysis ---------------------------------------------------

    def tick_cycles(self, kind: str = "auto") -> np.ndarray:
        """Cycles at which the play routine ran (CPU interrupt entries).

        ``kind`` is ``'irq'``, ``'nmi'``, ``'both'`` or ``'auto'`` (pick the
        vector kind with the most regular cadence / highest count).
        """
        vec = self.cpu_vectors()
        if len(vec) == 0:
            return np.empty(0, dtype=np.uint64)
        irq = np.unique(vec[vec["value"] == VEC_IRQ]["cycle"])
        nmi = np.unique(vec[vec["value"] == VEC_NMI]["cycle"])
        if kind == "irq":
            return irq
        if kind == "nmi":
            return nmi
        if kind == "both":
            return np.unique(vec[(vec["value"] == VEC_IRQ) | (vec["value"] == VEC_NMI)]["cycle"])
        # auto: prefer the more frequent source (the actual play cadence).
        return irq if len(irq) >= len(nmi) else nmi

    def interval_stats(self, kind: str = "auto") -> dict:
        """Period statistics of the play cadence, in CPU cycles."""
        ticks = self.tick_cycles(kind)
        if len(ticks) < 2:
            return {"count": int(len(ticks)), "period": None}
        diffs = np.diff(ticks.astype(np.int64))
        period = float(np.median(diffs))
        return {
            "count": int(len(ticks)),
            "period": period,
            "period_min": int(diffs.min()),
            "period_max": int(diffs.max()),
            "period_std": float(diffs.std()),
            "calls_per_frame": self.frame_cycles / period if period else None,
        }

    def interrupt_source_counts(self) -> dict:
        """How many interrupt-line assertions came from each source."""
        ints = self.interrupts()
        cia = ints[ints["etype"] == CIA_IRQ]
        vic = ints[ints["etype"] == VIC_IRQ]
        return {
            "vic_raster": int(len(vic)),
            "cia1_irq": int(np.count_nonzero(cia["chip"] == SRC_CIA1)),
            "cia2_nmi": int(np.count_nonzero(cia["chip"] == SRC_CIA2)),
        }

    def classify(self) -> dict:
        """Best-effort classification of the periodic update structure."""
        counts = self.interrupt_source_counts()
        stats = self.interval_stats()
        cia = self.interrupts()
        cia = cia[cia["etype"] == CIA_IRQ]
        vic = self.interrupts()
        vic = vic[vic["etype"] == VIC_IRQ]

        # Dominant driver by interrupt-line activity.
        if counts["cia1_irq"] > counts["vic_raster"]:
            driver = "CIA"
        elif counts["vic_raster"] > 0:
            driver = "RASTER"
        else:
            driver = "UNKNOWN"

        period = stats.get("period")
        calls = stats.get("calls_per_frame")
        speed = int(round(calls)) if calls else None

        result = {
            "driver": driver,
            "speed_string": self.meta.get("speed_string", ""),
            "effective_model": self.meta.get("effective_model", ""),
            "period_cycles": period,
            "frame_cycles": self.frame_cycles,
            "calls_per_frame": calls,
            "speed": speed,
            "multispeed": bool(speed and speed > 1),
            "interrupt_sources": counts,
        }
        if len(cia):
            latches = cia[cia["chip"] == SRC_CIA1]["addr"]
            if len(latches):
                vals, freq = np.unique(latches, return_counts=True)
                result["cia_timer_latch"] = int(vals[freq.argmax()])
        if len(vic):
            lines = np.unique(vic["addr"])
            result["raster_lines"] = lines.tolist()
            result["raster_split"] = bool(len(lines) > 1)
        return result

    # -- register reconstruction ------------------------------------------

    def register_frames(self, chip: int = 0, kind: str = "auto") -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct per-frame SID register snapshots.

        Returns ``(tick_cycles, frames)`` where ``frames`` has shape
        ``(n_ticks, 32)``: the cumulative register state (last value written
        to each register, carried forward) at the end of each play tick.
        This is the table that table/sweep/arpeggio players step through.
        """
        ticks = self.tick_cycles(kind)
        frames = np.zeros((len(ticks), 32), dtype=np.uint8)
        if len(ticks) == 0:
            return ticks, frames

        writes = self.sid_writes(chip)
        # Frame i covers writes with cycle < ticks[i+1]; last frame is open.
        bound = np.append(ticks[1:], np.uint64(np.iinfo(np.uint64).max))
        for reg in range(32):
            wr = writes[writes["reg"] == reg]
            if len(wr) == 0:
                continue
            wc = wr["cycle"]
            wv = wr["value"]
            pos = np.searchsorted(wc, bound, side="left")
            taken = pos > 0
            idx = np.clip(pos - 1, 0, len(wv) - 1)
            frames[taken, reg] = wv[idx][taken]
        return ticks, frames


def decode_voices(frames: np.ndarray) -> dict:
    """Decode per-voice SID parameters from a ``(n, 32)`` frame table.

    Returns a dict with arrays per voice (0..2): ``freq``, ``pulse``,
    ``ctrl``, ``waveform``, ``gate``, ``attack``, ``decay``, ``sustain``,
    ``release`` plus global ``volume``, ``filter_cutoff``, ``filter_res``,
    ``filter_mode``.
    """
    frames = np.asarray(frames, dtype=np.uint16)
    out: dict = {"voices": []}
    for v in range(3):
        b = v * 7
        freq = frames[:, b] | (frames[:, b + 1] << 8)
        pulse = (frames[:, b + 2] | (frames[:, b + 3] << 8)) & 0x0FFF
        ctrl = frames[:, b + 4]
        ad = frames[:, b + 5]
        sr = frames[:, b + 6]
        out["voices"].append(
            {
                "freq": freq.astype(np.uint16),
                "pulse": pulse.astype(np.uint16),
                "ctrl": ctrl.astype(np.uint8),
                "gate": (ctrl & 0x01).astype(np.uint8),
                "sync": ((ctrl >> 1) & 0x01).astype(np.uint8),
                "ring": ((ctrl >> 2) & 0x01).astype(np.uint8),
                "test": ((ctrl >> 3) & 0x01).astype(np.uint8),
                "waveform": ((ctrl >> 4) & 0x0F).astype(np.uint8),
                "attack": ((ad >> 4) & 0x0F).astype(np.uint8),
                "decay": (ad & 0x0F).astype(np.uint8),
                "sustain": ((sr >> 4) & 0x0F).astype(np.uint8),
                "release": (sr & 0x0F).astype(np.uint8),
            }
        )
    cutoff = ((frames[:, 21] & 0x07) | (frames[:, 22] << 3)).astype(np.uint16)
    out["filter_cutoff"] = cutoff
    out["filter_res"] = ((frames[:, 23] >> 4) & 0x0F).astype(np.uint8)
    out["filter_route"] = (frames[:, 23] & 0x07).astype(np.uint8)
    out["volume"] = (frames[:, 24] & 0x0F).astype(np.uint8)
    out["filter_mode"] = ((frames[:, 24] >> 4) & 0x07).astype(np.uint8)
    return out
