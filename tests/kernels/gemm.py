from dataclasses import dataclass
from itertools import product

import numpy as np
import pytest

import reference
from carbon import language as cl
from carbon.kernels.gemm import (
    MXFP4,
    Affine,
    QuantizedRegisters,
    QuantizedRows,
    dense_matmul,
    epilogue,
    interleave_halves,
    quantized_matmul,
)
from testbed import device, storage, upload


@dataclass(frozen=True)
class Dense:
    rows: int
    n: int
    k: int
    after: str = ""

    def __str__(self):
        return f"dense-{self.rows}x{self.n}x{self.k}-{self.after or 'plain'}"


@dataclass(frozen=True)
class Quantized:
    rows: int
    n: int
    k: int
    fmt: object
    glu: bool = False
    plan: object = None
    dtype: object = cl.f32
    after: str = ""

    def __str__(self):
        name = "mxfp4" if self.fmt == MXFP4 else f"q{self.fmt.bits}g{self.fmt.group}"
        glu = "-glu" if self.glu else ""
        plan = ""
        if isinstance(self.plan, QuantizedRows):
            plan = f"-rows{self.plan.features}x{self.plan.simds}"
        elif self.plan:
            plan = f"-registers{self.plan.split}x{self.plan.extra}"
        dtype = "" if self.dtype == cl.f32 else f"-{self.dtype.name}"
        after = f"-{self.after.replace(' ', '-')}" if self.after else ""
        return f"{name}-{self.rows}x{self.n}x{self.k}{glu}{plan}{dtype}{after}"


AFTER = {
    "": (lambda dtype: {}, lambda total, bias: total),
    "bias": (
        lambda dtype: {"epilogue": epilogue.bias(dtype)},
        lambda total, bias: total + bias,
    ),
    "bias gelu": (
        lambda dtype: {"epilogue": epilogue.chain(epilogue.bias(dtype), epilogue.gelu)},
        lambda total, bias: reference.gelu(total + bias),
    ),
    "bias relu": (
        lambda dtype: {"epilogue": epilogue.chain(epilogue.bias(dtype), epilogue.relu)},
        lambda total, bias: np.maximum(total + bias, 0.0),
    ),
}

DENSE = [
    Dense(1, 1000, 384),
    Dense(37, 520, 1000),
    Dense(37, 1100, 1024, "bias"),
    Dense(37, 2000, 256, "bias relu"),
    Dense(4, 800, 3000, "bias gelu"),
]
DENSE_SWEEP = [
    Dense(rows, n, k, after)
    for rows, (n, k), after in product(
        (1, 2, 4, 8, 37, 200),
        ((512, 512), (1000, 384), (520, 1000)),
        AFTER,
    )
]
QUANTIZED = [
    Quantized(1, 100, 512, Affine(4, 64)),
    Quantized(3, 100, 512, Affine(8, 64)),
    Quantized(12, 100, 512, Affine(3, 64)),
    Quantized(40, 37, 96, Affine(4, 32)),
    Quantized(1, 64, 256, MXFP4),
    Quantized(40, 64, 256, MXFP4),
    Quantized(3, 352, 512, Affine(4, 64), glu=True),
    Quantized(2, 100, 512, Affine(4, 64)),
    Quantized(4, 96, 1024, Affine(2, 64)),
    Quantized(4, 64, 256, MXFP4),
    Quantized(4, 100, 512, Affine(4, 64), glu=True),
    Quantized(5, 100, 512, Affine(4, 64), plan=QuantizedRows(2, 4)),
    Quantized(8, 70, 512, Affine(8, 64), plan=QuantizedRows(4, 2)),
    Quantized(16, 100, 512, Affine(4, 64), plan=QuantizedRows(2, 4)),
    Quantized(16, 64, 256, MXFP4, glu=True, plan=QuantizedRows(4, 1)),
    Quantized(2, 100, 512, Affine(4, 64), dtype=cl.bf16),
    Quantized(3, 100, 512, Affine(8, 64), dtype=cl.bf16),
    Quantized(4, 352, 512, Affine(4, 64), glu=True, dtype=cl.bf16),
    Quantized(8, 100, 512, Affine(4, 64), dtype=cl.bf16),
    Quantized(12, 100, 512, Affine(3, 64), dtype=cl.bf16),
    Quantized(2, 100, 512, Affine(4, 64), after="bias"),
    Quantized(4, 64, 256, MXFP4, after="bias gelu"),
    Quantized(3, 100, 512, Affine(8, 64), dtype=cl.bf16, after="bias relu"),
    Quantized(4, 352, 512, Affine(4, 64), glu=True, after="bias"),
    Quantized(9, 100, 512, Affine(4, 64), dtype=cl.bf16),
    Quantized(11, 352, 512, Affine(4, 64), glu=True, dtype=cl.bf16),
    Quantized(10, 100, 512, Affine(4, 64), after="bias gelu"),
    Quantized(13, 100, 512, Affine(4, 64), plan=QuantizedRegisters(4, 7)),
    Quantized(20, 64, 256, MXFP4),
]
QUANTIZED_SWEEP = [
    Quantized(rows, n, k, Affine(bits, 64))
    for rows, bits, (n, k) in product(
        (1, 2, 3, 4, 8, 12, 40),
        (2, 3, 4, 5, 6, 8),
        ((100, 512), (256, 1024)),
    )
]


def check_dense(case):
    metal, dtype = device(), cl.bf16
    rng = np.random.default_rng(case.n + case.k + case.rows)
    x = storage(rng.standard_normal((case.rows, case.k)) * 0.5, dtype)
    weight = storage(rng.standard_normal((case.n, case.k)) * case.k**-0.5, dtype)
    bias = storage(rng.standard_normal(case.n) * 0.1, dtype)
    options, expected = AFTER[case.after]
    options = options(dtype)
    if "epilogue" in options and case.after:
        options["operands"] = (upload(metal, bias),)
    total = reference.wide(x) @ reference.wide(weight).T
    expected = expected(total, reference.wide(bias))
    output = metal.buffer(case.rows * case.n * 2)
    metal.prepare(
        dense_matmul(
            metal,
            upload(metal, x),
            upload(metal, weight),
            output,
            rows=case.rows,
            n=case.n,
            k=case.k,
            dtype=dtype,
            **options,
        ),
    ).submit().wait()
    actual = reference.read(output, dtype, case.rows, case.n)
    np.testing.assert_allclose(
        actual,
        expected,
        atol=2e-2 * np.abs(expected).max(),
        err_msg=str(case),
    )


def check_quantized(case):
    metal = device()
    rng = np.random.default_rng(case.n + case.k + case.rows)
    if case.fmt == MXFP4:
        packed, scales, biases, weight = reference.mxfp4_weight(rng, case.n, case.k)
    else:
        packed, scales, biases, weight = reference.affine_weight(
            rng,
            case.n,
            case.k,
            case.fmt.bits,
            case.fmt.group,
        )
    x = (rng.integers(-4, 5, (case.rows, case.k)) / 16).astype(np.float32)
    width = case.n // 2 if case.glu else case.n
    dtype = case.dtype
    bias = storage(rng.standard_normal(width) * 0.1, dtype)
    options, after = AFTER[case.after]
    options = options(dtype)
    if case.after:
        options["operands"] = (upload(metal, bias),)
    if biases is not None:
        scales, biases = storage(scales, dtype), storage(biases, dtype)
    output = metal.buffer(case.rows * width * dtype.itemsize)
    metal.prepare(
        quantized_matmul(
            metal,
            upload(metal, storage(x, dtype)),
            upload(metal, packed),
            upload(metal, scales),
            upload(metal, scales if biases is None else biases),
            output,
            rows=case.rows,
            n=case.n,
            k=case.k,
            dtype=dtype,
            fmt=case.fmt,
            glu=epilogue.swiglu if case.glu else None,
            plan=case.plan,
            **options,
        ),
    ).submit().wait()
    actual = reference.read(output, dtype, case.rows, width)
    expected = x.astype(np.float64) @ weight.T
    if case.glu:
        expected = reference.silu(expected[:, :width]) * expected[:, width:]
    expected = after(expected, reference.wide(bias, dtype))
    rtol, atol = (1e-5, 1e-4) if dtype == cl.f32 else (2**-8, 2**-8)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=rtol,
        atol=atol * max(1.0, np.abs(expected).max()),
        err_msg=str(case),
    )


@pytest.mark.parametrize("case", DENSE, ids=str)
def test_dense(case):
    check_dense(case)


@pytest.mark.sweep
@pytest.mark.parametrize("case", DENSE_SWEEP, ids=str)
def test_dense_sweep(case):
    check_dense(case)


@pytest.mark.parametrize("case", QUANTIZED, ids=str)
def test_quantized(case):
    check_quantized(case)


@pytest.mark.sweep
@pytest.mark.parametrize("case", QUANTIZED_SWEEP, ids=str)
def test_quantized_sweep(case):
    check_quantized(case)


def test_dense_glu_pairs_interleaved_halves():
    metal, dtype, rows, n, k = device(), cl.bf16, 5, 512, 256
    rng = np.random.default_rng(1)
    x = storage(rng.standard_normal((rows, k)) * 0.5, dtype)
    weight = storage(rng.standard_normal((n, k)) * k**-0.5, dtype)
    output = metal.buffer(rows * n // 2 * 2)
    metal.prepare(
        dense_matmul(
            metal,
            upload(metal, x),
            upload(metal, interleave_halves(weight)),
            output,
            rows=rows,
            n=n,
            k=k,
            dtype=dtype,
            glu=epilogue.swiglu,
        ),
    ).submit().wait()
    total = reference.wide(x) @ reference.wide(weight).T
    expected = reference.silu(total[:, : n // 2]) * total[:, n // 2 :]
    actual = reference.read(output, dtype, rows, n // 2)
    np.testing.assert_allclose(actual, expected, atol=3e-2 * np.abs(expected).max())
