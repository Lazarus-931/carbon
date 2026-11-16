from dataclasses import dataclass
from typing import Callable

from carbon import language as cl


@dataclass(frozen=True)
class Hook:
    """Call `function(*values, *operands)` with one buffer per operand type, bound last.
    Equal hooks share compiled kernels, so `function` must be hashable."""

    function: Callable
    operands: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "operands", tuple(self.operands))

    def __call__(self, *arguments):
        *values, operands = arguments
        return self.function(*values, *operands)


def operand_buffers(hook):
    return () if hook is None else tuple(cl.buffer(dtype) for dtype in hook.operands)


def applied(hook, value, *arguments):
    return value if hook is None else hook(value, *arguments)


def round_to(value, dtype):
    return cl.cast(cl.cast(value, dtype), cl.f32)


def place(layout):
    e = cl.arange(layout.size, layout=layout)
    return e // 64, e // 8 % 8, e % 8
