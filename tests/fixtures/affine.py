import hashlib
import struct

from carbon import Device, language as cl
from carbon.bench import Case, Prepared


@cl.kernel
def affine(
    x: cl.buffer(cl.f32),
    y: cl.buffer(cl.f32),
    output: cl.buffer(cl.f32),
    count: cl.u32,
    scale: cl.f32,
    BLOCK: cl.constexpr,
):
    indices = cl.program_id() * BLOCK + cl.arange(BLOCK)
    active = indices < count
    a = cl.load(x, indices, active, other=0.0)
    b = cl.load(y, indices, active, other=0.0)
    cl.store(output, indices, a * scale + b, active)


def handwritten(block, threads):
    return f"""
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
struct Constants {{ uint count; float scale; }};
kernel void carbon_kernel(device const float* x [[buffer(0)]],
                          device const float* y [[buffer(1)]],
                          device float* output [[buffer(2)]],
                          constant Constants& params [[buffer(3)]],
                          uint lane [[thread_position_in_threadgroup]],
                          uint group [[threadgroup_position_in_grid]]) {{
    for (uint element = lane; element < {block}u; element += {threads}u) {{
        uint index = group * {block}u + element;
        if (index < params.count) output[index] = x[index] * params.scale + y[index];
    }}
}}
"""


def prepare(variant="generated", count=1031, block=128, threads=64):
    device = Device()
    x = [(i % 31 - 15) / 8 for i in range(count)]
    y = [(i % 17 - 8) / 4 for i in range(count)]
    payloads = [struct.pack(f"<{count}f", *values) for values in (x, y)]
    inputs = [device.buffer(max(4, count * 4)) for _ in payloads]
    for buffer, payload in zip(inputs, payloads):
        buffer.write(payload)
    output = device.buffer(max(4, count * 4) + 32)
    view = output.view(16, max(4, count * 4))
    bindings = [*inputs, view]
    grid = (max(1, (count + block - 1) // block), 1, 1)
    artifact = (
        affine.compile(constants={"BLOCK": block}, threads=threads)
        if variant == "generated"
        else None
    )
    source = artifact.source if artifact else handwritten(block, threads)
    pipeline = device.compile(source, "carbon_kernel", fast_math=False)
    if artifact:
        dispatch = artifact.dispatch(
            device,
            bindings,
            grid=grid,
            scalars={"count": count, "scale": 2.0},
        )
    else:
        dispatch = device.dispatch(
            pipeline,
            bindings,
            struct.pack("<If", count, 2.0),
            grid=grid,
            threads=(threads, 1, 1),
        )
    executable = device.prepare([dispatch])
    initial = bytes([0xA5]) * output.size
    expected = (
        initial[:16]
        + struct.pack(f"<{count}f", *(a * 2 + b for a, b in zip(x, y)))
        + initial[16 + count * 4 :]
    )

    def reset():
        output.write(initial)

    def verify(repetitions):
        if output.read() != expected:
            raise AssertionError("affine result or guard bytes differ")
        for buffer, payload in zip(inputs, payloads):
            if buffer.read()[: len(payload)] != payload:
                raise AssertionError("input was modified")

    return Prepared(
        executable.submit,
        reset,
        verify,
        {
            "device": device.name,
            "variant": variant,
            "count": count,
            "block": block,
            "threads": threads,
            "grid": grid,
            "dtype": "f32",
            "math_mode": "precise; contraction off",
            "input_sha256": hashlib.sha256(b"".join(payloads)).hexdigest(),
            "expected_sha256": hashlib.sha256(expected).hexdigest(),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "ir_sha256": hashlib.sha256(artifact.ir.encode()).hexdigest()
            if artifact
            else None,
            "compiler_ms": artifact.compile_ms if artifact else None,
            "llvm_version": artifact.llvm_version if artifact else None,
            "metal_compile_ms": pipeline.compile_ms,
            "purpose": "compiler equivalence diagnostic, not a tuned affine benchmark",
        },
    )


def cases():
    return [
        Case("handwritten", lambda: prepare("handwritten")),
        Case("generated", prepare),
    ]
