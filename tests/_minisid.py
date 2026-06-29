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


def build_psid(speed: int = 0) -> bytes:
    """Return PSID v2 bytes. ``speed`` bit0: 0=VBI (raster), 1=CIA timer."""
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
    header += struct.pack(">H", INIT)
    header += struct.pack(">H", PLAY)
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

    c64data = struct.pack("<H", LOAD) + CODE
    return header + c64data
