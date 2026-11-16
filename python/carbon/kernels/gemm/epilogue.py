"""Transform f32 output tiles before the store rounds them to the output type.

Hooks take (value, row, column, *operands) and return an f32 tile. Row and column are
u32 tiles in the value's layout; rows flatten batches as b * M + r. Padded elements
can name the next batch, so hooks may read memory but must not write it. Loads are
bounds-checked. Operand buffers bind after the output, in order.
"""

from dataclasses import dataclass
from math import pi, sqrt
from typing import Callable

from carbon import language as cl
from carbon.kernels.common import Hook, round_to

Epilogue = Hook


@dataclass(frozen=True)
class Glu:
    """Apply hashable `function(gate, up)` elementwise to f32 paired products.
    Equal pairings share compiled kernels."""

    function: Callable

    def __call__(self, gate, up):
        return self.function(gate, up)


def paired(glu, value):
    *outer, columns = value.layout.shape
    pairs = cl.reshape(value, (*outer, columns // 2, 2))
    axis = len(outer) + 1
    lines = pairs.layout.reduced(axis)
    up = cl.arange(pairs.layout.size, layout=pairs.layout) % 2 == 1
    gate = cl.sum(cl.select(up, 0.0, pairs), axis, layout=lines)
    return glu(gate, cl.sum(cl.select(up, pairs, 0.0), axis, layout=lines))


def _relu(value, row, column):
    return cl.maximum(value, 0.0)


def _silu(value, row, column):
    return value * cl.sigmoid(value)


def _gelu(value, row, column):
    return 0.5 * value * (1.0 + cl.erf(value * sqrt(0.5)))


def _gelu_tanh(value, row, column):
    return (
        0.5
        * value
        * (1.0 + cl.tanh(sqrt(2 / pi) * (value + 0.044715 * value * value * value)))
    )


relu = Epilogue(_relu)
silu = Epilogue(_silu)
gelu = Epilogue(_gelu)
"""x * Phi(x), through the error function."""
gelu_tanh = Epilogue(_gelu_tanh)


def _swiglu(gate, up):
    return gate * cl.sigmoid(gate) * up


swiglu = Glu(_swiglu)


@dataclass(frozen=True)
class _Rounded:
    dtype: object

    def __call__(self, value, row, column):
        return round_to(value, self.dtype)


def rounded(dtype):
    """Round through `dtype` and back to f32 before later epilogues."""
    return Epilogue(_Rounded(dtype))


@dataclass(frozen=True)
class _Softcap:
    cap: float

    def __call__(self, value, row, column):
        return cl.tanh(value / self.cap) * self.cap


def softcap(cap):
    """cap * tanh(x / cap); apply `rounded` first to match stored logits."""
    return Epilogue(_Softcap(float(cap)))


def _bias(value, row, column, vector):
    return value + cl.cast(cl.load(vector, column, True, other=0.0), cl.f32)


@dataclass(frozen=True)
class _Periodic:
    period: int

    def __call__(self, value, row, column, vector):
        return value + cl.cast(
            cl.load(vector, column % self.period, True, other=0.0),
            cl.f32,
        )


def bias(dtype, period=None):
    """x + bias[column], from a vector of `dtype` with one element per output column."""
    return Epilogue(_bias if period is None else _Periodic(period), (dtype,))


@dataclass(frozen=True)
class _Matrix:
    columns: int
    product: bool
    stride: int

    def __call__(self, value, row, column, matrix):
        other = cl.cast(
            cl.load(
                matrix,
                row * self.stride + column,
                column < self.columns,
                other=0.0,
            ),
            cl.f32,
        )
        return value * other if self.product else value + other


def residual(dtype, columns, stride=None):
    """Add a row-major residual of `dtype`; stride defaults to `columns`."""
    return Epilogue(
        _Matrix(columns, False, columns if stride is None else stride),
        (dtype,),
    )


def multiply(dtype, columns, stride=None):
    """Multiply by a row-major matrix of `dtype`; stride defaults to `columns`."""
    return Epilogue(
        _Matrix(columns, True, columns if stride is None else stride),
        (dtype,),
    )


@dataclass(frozen=True)
class _Chain:
    parts: tuple

    def __call__(self, value, row, column, *operands):
        for part in self.parts:
            count = len(part.operands)
            value, operands = (
                part(value, row, column, operands[:count]),
                operands[count:],
            )
        return value


def chain(*epilogues):
    """`epilogues` applied in order; their operands follow one another."""
    return Epilogue(
        _Chain(epilogues),
        tuple(dtype for part in epilogues for dtype in part.operands),
    )
