import struct
from dataclasses import dataclass

from .view import BoundView


@dataclass(frozen=True)
class WriteConflict:
    """Two threads wrote `binding` (parameter name) in one dispatch; `element` is the
    lowest such element."""

    binding: str
    element: int


def check_writes(
    kernel,
    device,
    buffers,
    *,
    constants,
    threads,
    grid,
    scalars=None,
    views=None,
    max_scratch=None,
):
    """Dispatch `kernel` once on `buffers` with write recording and return its
    conflicts, empty when none.

    The run writes its outputs as a normal dispatch would. Atomics and reads are not
    recorded; matrix-unit products are stored element by element under write checking,
    so their stores are.
    """
    artifact = kernel.compile(
        constants=constants,
        threads=threads,
        device=device,
        views=views,
        max_scratch=max_scratch,
        check_writes=True,
    )
    if not artifact.checked_bindings:
        device.prepare(
            [artifact.dispatch(device, buffers, grid=grid, scalars=scalars)],
        ).submit().wait()
        return []
    kinds = [kind for _, kind in artifact.parameters if hasattr(kind, "element")]
    shadows = []
    for binding in artifact.checked_bindings:
        storage = (
            buffers[binding].buffer
            if isinstance(buffers[binding], BoundView)
            else buffers[binding]
        )
        shadow = device.buffer(
            max(storage.size // kinds[binding].element.itemsize * 4, 4),
        )
        shadow.write(bytes(shadow.size))
        shadows.append(shadow)
    # Word 0 counts conflicts; word 1 + i keeps the lowest element of checked binding i
    # that two threads wrote.
    record = device.buffer(4 * (len(shadows) + 1))
    record.write(
        struct.pack(f"<{len(shadows) + 1}I", 0, *([0xFFFFFFFF] * len(shadows))),
    )
    dispatch = artifact.dispatch(
        device,
        buffers,
        grid=grid,
        scalars=scalars,
        checks=[*shadows, record],
    )
    device.prepare([dispatch]).submit().wait()
    count, *first = struct.unpack(f"<{len(shadows) + 1}I", record.read())
    if not count:
        return []
    names = [name for name, kind in artifact.parameters if hasattr(kind, "element")]
    return [
        WriteConflict(names[binding], element)
        for binding, element in zip(artifact.checked_bindings, first)
        if element != 0xFFFFFFFF
    ]
