"""NumPy references for the kernels in carbon.kernels, in float64."""

from math import erf, sqrt

import numpy as np

from carbon import language as cl
from testbed import NUMPY, storage, widen

FP4 = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    np.float32,
)


def rounded(values, dtype=cl.bf16):
    """`values` as `dtype` stores them, widened to float64."""
    return widen(storage(values, dtype), dtype).astype(np.float64)


def wide(bits, dtype=cl.bf16):
    """Stored `dtype` values (bf16 as raw bits) widened to float64."""
    return widen(bits, dtype).astype(np.float64)


def read(buffer, dtype, *shape):
    """A buffer of `dtype` values as float64 in `shape`."""
    raw = np.frombuffer(buffer.read(), NUMPY[dtype.name])[: int(np.prod(shape))]
    return widen(raw, dtype).astype(np.float64).reshape(shape)


def silu(x):
    return x / (1 + np.exp(-x))


def gelu(x):
    return 0.5 * x * (1 + np.vectorize(erf)(x / sqrt(2)))


def attention(q, keys, values, positions, scale, *, window=0, causal=True, bias=None):
    """softmax(scale · q kᵀ + bias) v per row and head; q (rows, heads, depth), keys
    and values (kv_heads, positions, depth), row r at position positions[r]. `bias`
    maps (row, head, key positions) to what each score gains."""
    rows, heads, depth = q.shape
    group = heads // keys.shape[0]
    positions = list(positions)
    last_position = keys.shape[1] - 1
    output = np.zeros((rows, heads, depth))
    for row, position in enumerate(positions):
        if causal:
            first = max(0, position - window + 1) if window else 0
            last = position
        else:
            first = max(0, position - window) if window else 0
            last = min(last_position, position + window) if window else last_position
        for head in range(heads):
            k = keys[head // group, first : last + 1]
            v = values[head // group, first : last + 1]
            scores = k @ q[row, head] * scale
            if bias is not None:
                scores = scores + bias(row, head, np.arange(first, last + 1))
            weights = np.exp(scores - scores.max())
            output[row, head] = weights @ v / weights.sum()
    return output


def pack(codes, bits):
    """Codes packed little-endian into u32 words, `bits` each."""
    stream = np.zeros(codes.size * bits, np.uint8)
    for bit in range(bits):
        stream[bit::bits] = (codes.ravel() >> bit) & 1
    return np.packbits(stream.reshape(-1, 8)[:, ::-1]).view(np.uint32)


def affine_weight(rng, n, k, bits, group):
    """Packed codes, scales, biases and the exact (n, k) weight of an Affine format."""
    codes = rng.integers(0, 1 << bits, (n, k))
    scales = (rng.integers(1, 5, (n, k // group)) / 8).astype(np.float32)
    biases = (rng.integers(-4, 5, (n, k // group)) / 8).astype(np.float32)
    weight = codes * np.repeat(scales, group, 1) + np.repeat(biases, group, 1)
    return pack(codes, bits), scales, biases, weight


def mxfp4_weight(rng, n, k):
    """Packed FP4 codes, E8M0 scale bytes and the exact (n, k) weight of MXFP4."""
    codes = rng.integers(0, 16, (n, k))
    exponents = rng.integers(124, 130, (n, k // 32)).astype(np.uint8)
    weight = FP4[codes] * np.repeat(2.0 ** (exponents.astype(np.float64) - 127), 32, 1)
    return pack(codes, 4), exponents.view(np.uint32), None, weight
