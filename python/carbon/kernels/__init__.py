_SHARED = {}


def shared_workspace(device, size, use="scratch"):
    """Reuse a buffer of at least `size` bytes, keyed by device and `use`.
    Ordered dispatches may share a use when their workspace lifetimes do not overlap;
    simultaneous lifetimes require distinct names. The buffer grows as needed.
    """
    held = _SHARED.get((id(device), use))
    if held is None or held[1].size < size:
        held = (device, device.buffer(size))
        _SHARED[(id(device), use)] = held
    return held[1]
