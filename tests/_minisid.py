"""Build tiny, original (non-copyrighted) PSID tunes for end-to-end tests.

The C64 player is a few bytes of hand-assembled 6502 that, each call,
increments a zero-page counter and writes it to voice-1 frequency low plus a
fixed master volume. With the PSID speed flag clear the libsidplayfp driver
calls it from a VIC raster IRQ (VBI); set, from a CIA timer IRQ.
"""

import struct

LOAD = 0x1000
INIT = 0x1000
PLAY = 0x1003

# 6502 player, assembled at $1000:
#   1000 60           init: RTS
#   1001 EA EA        pad
#   1003 E6 FB        play: INC $FB        ; frame counter
#   1005 A5 FB              LDA $FB
#   1007 8D 00 D4           STA $D400      ; voice1 freq lo = counter
#   100A A9 0F              LDA #$0F
#   100C 8D 18 D4           STA $D418      ; master volume
#   100F 60                 RTS
CODE = bytes(
    [
        0x60,
        0xEA,
        0xEA,
        0xE6,
        0xFB,
        0xA5,
        0xFB,
        0x8D,
        0x00,
        0xD4,
        0xA9,
        0x0F,
        0x8D,
        0x18,
        0xD4,
        0x60,
    ]
)


# I/O-probe player, assembled at $1000: init sets voice 3 to max-frequency
# noise, play copies the osc3 readback ($D41B) and CIA1 timer A lo ($DC04)
# into voice-1 pulse width and INCs the VIC border colour (I/O read + write).
#   1000 A9 FF     init: LDA #$FF
#   1002 8D 0E D4        STA $D40E
#   1005 8D 0F D4        STA $D40F
#   1008 A9 80           LDA #$80
#   100A 8D 12 D4        STA $D412
#   100D 60              RTS
#   100E AD 1B D4  play: LDA $D41B
#   1011 8D 02 D4        STA $D402
#   1014 AD 04 DC        LDA $DC04
#   1017 8D 03 D4        STA $D403
#   101A A9 0F           LDA #$0F
#   101C 8D 18 D4        STA $D418
#   101F EE 20 D0        INC $D020
#   1022 60              RTS
IOPROBE_INIT = 0x1000
IOPROBE_PLAY = 0x100E
IOPROBE_CODE = bytes(
    [
        0xA9,
        0xFF,
        0x8D,
        0x0E,
        0xD4,
        0x8D,
        0x0F,
        0xD4,
        0xA9,
        0x80,
        0x8D,
        0x12,
        0xD4,
        0x60,
        0xAD,
        0x1B,
        0xD4,
        0x8D,
        0x02,
        0xD4,
        0xAD,
        0x04,
        0xDC,
        0x8D,
        0x03,
        0xD4,
        0xA9,
        0x0F,
        0x8D,
        0x18,
        0xD4,
        0xEE,
        0x20,
        0xD0,
        0x60,
    ]
)


def _build_psid(speed: int, init: int, play: int, code: bytes) -> bytes:
    """Assemble PSID v2 bytes for a player loaded at LOAD ($1000)."""
    magic = b"PSID"
    version = 2
    data_offset = 0x7C
    load_address = 0  # embedded in C64 data below
    songs = 1
    start_song = 1
    name = b"preframr test"
    author = b"preframr-playroutine"
    released = b"2026 test"
    # flags: clock PAL (bit2), sid model 6581 (bit4)
    flags = (1 << 2) | (1 << 4)

    header = magic
    header += struct.pack(">H", version)
    header += struct.pack(">H", data_offset)
    header += struct.pack(">H", load_address)
    header += struct.pack(">H", init)
    header += struct.pack(">H", play)
    header += struct.pack(">H", songs)
    header += struct.pack(">H", start_song)
    header += struct.pack(">I", speed)
    header += name.ljust(32, b"\x00")[:32]
    header += author.ljust(32, b"\x00")[:32]
    header += released.ljust(32, b"\x00")[:32]
    header += struct.pack(">H", flags)
    header += struct.pack(">B", 0)  # startPage
    header += struct.pack(">B", 0)  # pageLength
    header += struct.pack(">B", 0)  # secondSIDAddress
    header += struct.pack(">B", 0)  # thirdSIDAddress
    assert len(header) == data_offset, len(header)

    c64data = struct.pack("<H", LOAD) + code
    return header + c64data


def build_psid(speed: int = 0) -> bytes:
    """Return PSID v2 bytes. ``speed`` bit0: 0=VBI (raster), 1=CIA timer."""
    return _build_psid(speed, INIT, PLAY, CODE)


def build_ioprobe_psid() -> bytes:
    """Return the I/O-probe PSID (VBI-driven; exercises I/O reads + writes)."""
    return _build_psid(0, IOPROBE_INIT, IOPROBE_PLAY, IOPROBE_CODE)
