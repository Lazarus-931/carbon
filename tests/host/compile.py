import json
import operator
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from carbon import Device, language as cl
from testbed import run, run_compiled


def doubling_kernel():
    @cl.kernel
    def double(
        x: cl.buffer(cl.f32),
        out: cl.buffer(cl.f32),
        count: cl.u32,
        BLOCK: cl.constexpr,
    ):
        index = cl.program_id() * BLOCK + cl.arange(BLOCK)
        active = index < count
        cl.store(out, index, cl.load(x, index, active, other=0.0) * 2.0, active)

    return double


def test_stage_shows_the_carbon_ir():
    staged = doubling_kernel().stage(constants={"BLOCK": 64}, threads=32)
    assert '"carbon.' in staged.ir


def test_language_callables_follow_local_bindings():
    language = cl

    def make_kernel(functions):
        @cl.kernel
        def apply(output: cl.buffer(cl.f32)):
            cl = functions
            index = language.arange(32)
            operation = cl.maximum
            value = operation(language.cast(index, language.f32), 2.0)
            language.store(output, index, value, True)

        return apply

    values = np.arange(32, dtype=np.float32)
    for functions, expected in (
        (cl, np.maximum(values, 2.0)),
        (SimpleNamespace(maximum=operator.add), values + 2.0),
    ):
        (actual,) = run(
            make_kernel(functions),
            outputs=[(32, np.float32)],
        )
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="callable"):
        make_kernel(SimpleNamespace(maximum=42)).stage(constants={}, threads=32)


def test_compile_caches_by_arguments(monkeypatch):
    from carbon.compiler import kernel as kernel_module

    calls = []
    original = kernel_module.stage
    monkeypatch.setattr(
        kernel_module,
        "stage",
        lambda *a, **k: calls.append(k["constants"]) or original(*a, **k),
    )
    kernel = doubling_kernel()
    first = kernel.compile(constants={"BLOCK": 64}, threads=32)
    assert (
        kernel.compile(constants={"BLOCK": 64}, threads=32) is first and len(calls) == 1
    )
    kernel.compile(constants={"BLOCK": 128}, threads=32)
    kernel.compile(constants={"BLOCK": 64}, threads=64)
    assert len(calls) == 3


def test_disk_cache_skips_the_compiler(tmp_path, monkeypatch):
    native = pytest.importorskip("carbon._compiler")
    from carbon.compiler import kernel as kernel_module

    monkeypatch.delenv("CARBON_CACHE", raising=False)
    monkeypatch.setenv("CARBON_CACHE_DIR", str(tmp_path))
    first = doubling_kernel().compile(constants={"BLOCK": 64}, threads=32)
    assert len(list((tmp_path / "artifacts").glob("*.json"))) == 1

    class Refusing:
        __file__ = native.__file__

        def __getattr__(self, name):
            return getattr(native, name)

        def compile(self, ir, target="metal"):
            raise AssertionError("a cached artifact must not be recompiled")

    monkeypatch.setattr(kernel_module, "load_compiler", Refusing)
    second = doubling_kernel().compile(constants={"BLOCK": 64}, threads=32)
    assert (second.ir, second.source) == (first.ir, first.source)
    x = np.arange(100, dtype=np.float32)
    (out,) = run_compiled(
        second,
        inputs=[x],
        outputs=[(100, np.float32)],
        scalars={"count": 100},
        grid=(2, 1, 1),
    )
    assert np.array_equal(out, 2 * x)


def test_buffers_views_and_overlap():
    metal = Device()
    buffer = metal.buffer(16)
    buffer.write(bytes(range(16)))
    assert buffer.read() == bytes(range(16))
    assert buffer.view(4, 8).read() == bytes(range(4, 12))
    assert buffer.overlaps(buffer.view(4, 8))
    assert not buffer.view(0, 4).overlaps(buffer.view(4, 8))
    assert not buffer.overlaps(metal.buffer(16))
    for invalid in (lambda: buffer.view(12, 8), lambda: buffer.write(bytes(17))):
        with pytest.raises((ValueError, IndexError)):
            invalid()


def test_one_submission_runs_many_dispatches():
    metal = Device()
    artifact = doubling_kernel().compile(
        constants={"BLOCK": 64},
        threads=64,
        device=metal,
    )
    values = np.arange(100, dtype=np.float32)
    buffers = [metal.buffer(400) for _ in range(4)]
    buffers[0].write(values.tobytes())
    dispatches = [
        artifact.dispatch(
            metal,
            [source, target],
            grid=(2, 1, 1),
            scalars={"count": 100},
        )
        for source, target in zip(buffers, buffers[1:])
    ]
    completion = metal.prepare(dispatches).submit()
    completion.wait()
    assert completion.done
    assert np.array_equal(np.frombuffer(buffers[3].read(), np.float32), values * 8)


def test_bench_command_line_checks_every_case():
    fixture = Path(__file__).parents[1] / "fixtures" / "affine.py"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "carbon.bench",
            str(fixture),
            "--blocks",
            "1",
            "--samples",
            "1",
            "--warmup",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(result.stdout)["correct"]
