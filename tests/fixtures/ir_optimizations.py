"""Compare matching case names across compiler revisions with `carbon.bench`."""

import hashlib
from functools import cache, partial

import numpy as np

from carbon import Device, language as cl
from carbon.bench import Case, Prepared
from carbon.compiler import Layout, Mode


@cache
def copy_storage(element):
    @cl.kernel
    def copy(source: cl.buffer(element), output: cl.buffer(element), A: cl.constexpr):
        index = cl.program_id() * A.size + cl.arange(A.size, layout=A)
        cl.store(output, index, cl.load(source, index, True, other=0), True)

    return copy


@cl.kernel
def transform(
    source: cl.buffer(cl.f32),
    output: cl.buffer(cl.f32),
    A: cl.constexpr,
    B: cl.constexpr,
    KIND: cl.constexpr,
    REPEAT: cl.constexpr,
):
    index = cl.program_id() * A.size + cl.arange(A.size, layout=A)
    if KIND == "loads" or KIND == "loads-sync" or KIND == "scratch-loop":
        if KIND == "loads" or KIND == "loads-sync":
            state = cl.full(A.shape, 0.0, cl.f32, layout=A)
        else:
            state = cl.load(source, index, True, other=0.0)
            scratch = cl.scratch(A.shape, cl.f32)
        if KIND == "loads-sync":
            scratch = cl.scratch(A.shape, cl.f32)
            local = cl.arange(A.size, layout=A)
        for step in range(REPEAT):
            if KIND == "loads" or KIND == "loads-sync":
                value = cl.load(source, index, True, other=0.0)
                state = cl.fma(state, 0.99, value)
                if KIND == "loads-sync":
                    cl.store(scratch, local, state, True)
                    cl.barrier()
                    state = cl.load(scratch, A.size - 1 - local, True, other=0.0)
                    cl.barrier()
            else:
                value = cl.convert_layout(state, B, scratch=scratch)
                state = cl.convert_layout(value * value + 0.25, A, scratch=scratch)
        cl.store(output, index, state, True)
    else:
        value = cl.load(source, index, True, other=0.0)
        if KIND == "layouts" or KIND == "scratch-layouts":
            if KIND == "scratch-layouts":
                scratch = cl.scratch(A.shape, cl.f32)
                value = cl.convert_layout(value, B, scratch=scratch)
            else:
                value = cl.convert_layout(value, B)
            value = value * value + 0.25
            if KIND == "scratch-layouts":
                value = cl.convert_layout(value, A, scratch=scratch)
            else:
                value = cl.convert_layout(value, A)
            cl.store(output, index, value, True)
        elif KIND == "norm":
            scratch = cl.scratch(A.shape, cl.f32)
            squared = cl.item(
                cl.sum(value * value, 0, scratch=scratch),
                scratch=scratch,
            )
            scale = cl.rsqrt(squared / A.size + 1e-5)
            cl.store(output, index, value * scale, True)
        elif KIND == "norm-rows":
            scratch = cl.scratch(A.shape, cl.f32)
            squared = cl.sum(value * value, 1, scratch=scratch, layout=A.reduced(1))
            scale = cl.broadcast_to(
                cl.rsqrt(squared / A.shape[1] + 1e-5),
                A.shape,
                axes=(0,),
                layout=A,
            )
            cl.store(output, index, value * scale, True)


def prepare(
    kind,
    block,
    *,
    repeat=16,
    rows=1024,
    element=cl.f32,
    blocked=False,
):
    metal = Device()
    threads = {
        "layouts": 32,
        "loads": 64,
        "loads-sync": 64,
        "scratch-loop": 64,
        "scratch-layouts": 64,
        "norm": 128,
        "norm-rows": 128,
        "copy": 64,
    }[kind]
    rng = np.random.default_rng(471)
    if kind == "copy":
        dtype = np.uint16 if element.itemsize == 2 else np.uint8
        data = rng.integers(0, 256, (rows, block), dtype=dtype)
    else:
        dtype = np.float32
        data = rng.uniform(-0.5, 0.5, (rows, block)).astype(dtype)
    if kind == "copy":
        expected = data
    elif kind in ("layouts", "scratch-layouts"):
        expected = data * data + np.float32(0.25)
    elif kind == "scratch-loop":
        expected = data.copy()
        for _ in range(repeat):
            expected = expected * expected + np.float32(0.25)
    elif kind == "loads":
        expected = data * sum(float(np.float32(0.99)) ** i for i in range(repeat))
    elif kind == "loads-sync":
        expected = np.zeros_like(data, dtype=np.float64)
        for _ in range(repeat):
            expected = (expected * float(np.float32(0.99)) + data)[:, ::-1]
    elif kind in ("norm", "norm-rows"):
        wide = data.astype(np.float64).reshape(
            -1,
            block // 4 if kind == "norm-rows" else block,
        )
        expected = wide / np.sqrt((wide * wide).mean(1, keepdims=True) + 1e-5)
    payload = data.tobytes()
    source = metal.buffer(len(payload))
    source.write(payload)
    output_bytes = expected.size * np.dtype(dtype).itemsize
    output = metal.buffer(output_bytes + 32)
    initial = bytes([0xA5]) * output.size
    constants = (
        {"A": Layout.blocked((block,), threads=threads)}
        if kind == "copy"
        else {
            "A": (Layout.blocked if blocked else Layout.striped)(
                (block,),
                threads=threads,
            ),
            "B": Layout.blocked((block,), threads=threads),
            "KIND": kind,
            "REPEAT": repeat,
        }
    )
    if kind == "norm-rows":
        constants["A"] = Layout(
            (4, block // 4),
            Mode(4, block // 4),
            Mode(32, 1),
            Mode(block // 128, 32),
        )
    kernel = copy_storage(element) if kind == "copy" else transform
    artifact = kernel.compile(
        constants=constants,
        threads=threads,
        device=metal,
        max_grid=rows,
    )
    executable = metal.prepare(
        [
            artifact.dispatch(
                metal,
                [source, output.view(16, output_bytes)],
                grid=(rows, 1, 1),
            ),
        ],
    )

    def verify(repetitions):
        raw = output.read()
        assert raw[:16] == raw[-16:] == initial[:16]
        assert source.read() == payload
        actual = np.frombuffer(raw[16:-16], dtype).reshape(expected.shape)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-4)

    return Prepared(
        executable.submit,
        lambda: output.write(initial),
        verify,
        {
            "device": metal.name,
            "kind": kind,
            "shape": [rows, block],
            "threads": threads,
            "repeat": repeat,
            "element": element.name,
            "blocked": blocked,
            "scratch_bytes": artifact.scratch_bytes,
            "barriers": artifact.source.count("threadgroup_barrier("),
            "shuffles": artifact.source.count("simd_shuffle"),
            "source_bytes": len(artifact.source),
            "source_sha256": hashlib.sha256(artifact.source.encode()).hexdigest(),
            "compiler_ms": artifact.compile_ms,
        },
    )


def cases():
    return [
        Case(
            "copy-u8-streaming",
            partial(prepare, "copy", 1024, element=cl.u8, rows=32768),
        ),
        *[
            Case(
                f"copy-{element.name}-{block}",
                partial(prepare, "copy", block, element=element),
            )
            for element in (cl.u8, cl.u16)
            for block in (512, 1024, 2048)
        ],
        *[
            Case(f"layouts-{block}", partial(prepare, "layouts", block))
            for block in (128, 257, 512, 2048)
        ],
        *[
            Case(f"{kind}-{block}", partial(prepare, kind, block))
            for kind in ("scratch-layouts", "scratch-loop")
            for block in (128, 257, 512, 2048)
        ],
        *[
            Case(f"loads-{repeat}", partial(prepare, "loads", 256, repeat=repeat))
            for repeat in (16, 128)
        ],
        *[
            Case(
                f"{kind}-blocked-{block}-{repeat}",
                partial(prepare, kind, block, repeat=repeat, blocked=True),
            )
            for kind in ("loads", "loads-sync")
            for block in (128, 257, 512, 2048)
            for repeat in (1, 16)
        ],
        *[
            Case(f"{kind}-{block}", partial(prepare, kind, block))
            for kind in ("norm", "norm-rows")
            for block in (512, 2048, 4096, 8192)
        ],
    ]
