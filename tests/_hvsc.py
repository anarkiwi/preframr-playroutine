"""HVSC tune cache/download fixture helper.

Resolves a tune entry from ``tests/fixtures/tunes.json`` to a local ``.sid``
path, with md5 verification. Resolution order (per the v2 contract):

1. cache dir ``${PREFRAMR_HVSC_CACHE:-<repo>/tests/.hvsc_cache}`` (if md5 ok);
2. local mirror ``${HVSC_ROOT:-/scratch/hvsc/C64Music}/<path>``;
3. download ``${HVSC_BASE_URL:-https://hvsc.c64.org/download/C64Music}/<path>``.

After fetching, the file md5 is verified against ``entry['md5']`` and cached.
Writes are atomic (temp file + rename) so concurrent xdist workers are safe.
No third-party deps: stdlib ``urllib``/``hashlib`` only.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_FIXTURES = os.path.join(_HERE, "fixtures", "tunes.json")

DEFAULT_BASE_URL = "https://hvsc.c64.org/download/C64Music"
DEFAULT_MIRROR = "/scratch/hvsc/C64Music"


def catalog_path() -> str:
    """Path to the committed tune catalog."""
    return _FIXTURES


def load_catalog() -> list:
    """Return the list of tune entries (empty list if the file is missing)."""
    if not os.path.exists(_FIXTURES):
        return []
    with open(_FIXTURES, encoding="utf-8") as handle:
        return json.load(handle)


def cache_dir() -> str:
    """Resolved cache directory."""
    return os.environ.get("PREFRAMR_HVSC_CACHE", os.path.join(_HERE, ".hvsc_cache"))


def _md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(dst: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dst), suffix=".part")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def ensure_tune(entry: dict) -> str:
    """Return a local path to the cached ``.sid`` for ``entry`` (fetch if needed).

    Raises ``FileNotFoundError`` if the tune cannot be obtained from cache,
    mirror, or download, and ``ValueError`` on md5 mismatch.
    """
    rel = entry["path"]
    want = entry["md5"].lower()
    dst = os.path.join(cache_dir(), rel)

    if os.path.exists(dst) and _md5(dst) == want:
        return dst

    mirror_root = os.environ.get("HVSC_ROOT", DEFAULT_MIRROR)
    mirror = os.path.join(mirror_root, rel)
    if os.path.exists(mirror):
        with open(mirror, "rb") as handle:
            data = handle.read()
    else:
        base = os.environ.get("HVSC_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        url = base + "/" + rel.lstrip("/")
        with urllib.request.urlopen(url, timeout=60) as resp:  # nosec - fixed HVSC host
            data = resp.read()

    got = hashlib.md5(data).hexdigest()
    if got != want:
        raise ValueError(f"md5 mismatch for {rel}: got {got}, want {want}")
    _atomic_write(dst, data)
    return dst


def fetchable(entry: dict) -> bool:
    """Whether ``ensure_tune`` is likely to succeed (cache/mirror/network)."""
    dst = os.path.join(cache_dir(), entry["path"])
    if os.path.exists(dst):
        return True
    if os.path.exists(os.path.join(os.environ.get("HVSC_ROOT", DEFAULT_MIRROR), entry["path"])):
        return True
    try:
        ensure_tune(entry)
        return True
    except (OSError, ValueError):
        return False
