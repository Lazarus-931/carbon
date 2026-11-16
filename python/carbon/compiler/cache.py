"""Persistent compiled-artifact cache under ~/.cache/carbon.

CARBON_CACHE_DIR moves the cache; CARBON_CACHE=0 disables it.
"""

import hashlib
import json
import os
import threading
from functools import cache
from pathlib import Path


def directory(kind):
    """The cache directory for one kind of entry, or None when caching is disabled."""
    if os.environ.get("CARBON_CACHE") == "0":
        return None
    root = (
        Path(os.environ.get("CARBON_CACHE_DIR") or Path.home() / ".cache" / "carbon")
        / kind
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


@cache
def binary_identity(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(kind, key):
    """The JSON value stored under `key`, or None when it is absent, unreadable or
    caching is disabled."""
    root = directory(kind)
    if root is None:
        return None
    try:
        return json.loads((root / f"{key}.json").read_text())
    except (OSError, ValueError):
        return None


def store(kind, key, value):
    """Store a JSON value under `key`, replacing the entry atomically so concurrent
    readers see whole files."""
    root = directory(kind)
    if root is None:
        return
    pending = root / f"{key}.{os.getpid()}.{threading.get_ident()}.json"
    pending.write_text(json.dumps(value))
    pending.replace(root / f"{key}.json")
