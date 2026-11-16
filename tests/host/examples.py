import runpy
from pathlib import Path

import numpy as np
import pytest

from carbon.compiler import Layout, View
from testbed import device, execute, guarded, run_compiled, unguarded

EXAMPLES = Path(__file__).parents[2] / "examples"


def example(name):
    return runpy.run_path(str(EXAMPLES / f"{name}.py"))


def test_dot_example():
    dot_rows = example("dot")["dot_rows"]
    schedule = Layout.striped((93,), threads=64)
    staged = dot_rows.stage(constants={"SCHEDULE": schedule}, threads=64)
    assert '"carbon.reduce_sum"' in staged.ir and 'order = "fast"' in staged.ir
    artifact = dot_rows.compile(
        constants={"SCHEDULE": schedule},
        threads=64,
        device=device(),
    )
    for columns in (0, 1, 97, 257):
        left = ((np.arange(3 * columns) % 11 - 5) / 8).astype(np.float16)
        right = ((np.arange(3 * columns) % 7 - 3) / 8).astype(np.float16)
        (out,) = run_compiled(
            artifact,
            inputs=[left, right],
            outputs=[(3, np.float32)],
            scalars={"columns": columns},
            grid=(3, 1, 1),
        )
        products = left.astype(np.float32) * right.astype(np.float32)
        assert np.array_equal(out, products.reshape(3, columns).sum(1)), columns


def test_elementwise_example():
    multiply = example("elementwise")["multiply"]
    staged = multiply.stage(constants={"BM": 3, "BN": 31}, threads=32)
    assert '!carbon.view<f32, [-1, -1], [], -1, -1, "device">' in staged.ir
    assert staged.ir.count('"carbon.partition"') == 3
    metal = device()
    artifact = multiply.compile(constants={"BM": 3, "BN": 31}, threads=32, device=metal)
    for rows, columns in ((0, 7), (1, 1), (4, 33), (7, 97)):
        stride = columns + 5
        size = max(1, rows * stride)
        left = ((np.arange(size) % 19 - 9) / 8).astype(np.float32)
        right = ((np.arange(size) % 13 - 6) / 16).astype(np.float32)
        inputs = [metal.buffer(size * 4) for _ in range(2)]
        for buffer, values in zip(inputs, (left, right)):
            buffer.write(values.tobytes())
        storage, output = guarded(metal, size * 4)
        normal = View("f32", (rows, columns), (stride, 1), size * 4)
        execute(
            artifact,
            [
                normal.bind(inputs[0]),
                normal[::-1, :].bind(inputs[1]),
                normal.bind(output),
            ],
            grid=(max(1, -(-rows // 3) * -(-columns // 31)) + 2, 1, 1),
            scalars={"scale": 1.5},
        )
        expected = np.frombuffer(bytes([0xA5]) * size * 4, np.float32).copy()
        for row in range(rows):
            mirrored = (rows - 1 - row) * stride
            expected[row * stride : row * stride + columns] = (
                left[row * stride : row * stride + columns]
                * right[mirrored : mirrored + columns]
                * np.float32(1.5)
            )
        assert unguarded(storage) == expected.tobytes(), (rows, columns)
    buffers = [metal.buffer(128) for _ in range(3)]
    dense = View("f32", (4, 8), (8, 1), 128)
    with pytest.raises(TypeError, match="View.bind"):
        artifact.dispatch(metal, buffers, grid=(1, 1, 1), scalars={"scale": 1.0})
    with pytest.raises(ValueError, match="valid storage"):
        artifact.dispatch(
            metal,
            [
                View("f32", (4, 8), (8, 1), 256).bind(buffers[0]),
                dense.bind(buffers[1]),
                dense.bind(buffers[2]),
            ],
            grid=(1, 1, 1),
            scalars={"scale": 1.0},
        )


def test_persistent_mlp_example():
    mlp = example("persistent_mlp")
    metal = device()
    for width, rows, groups in ((128, 16, 4), (256, 8, 8), (512, 8, 16)):
        rng = np.random.default_rng(width)
        values = rng.normal(size=(32, width)).astype(np.float16)
        weights = (rng.normal(size=(6, width, width)) / np.sqrt(width)).astype(
            np.float16,
        )
        options = {"constants": {"BLOCKS": 2}, "threads": 32 * groups, "device": metal}
        forward = mlp["make_mlp"](width, rows=rows, groups=groups).compile(**options)
        layer = mlp["make_mlp_layer"](width, rows=rows, groups=groups).compile(
            **options,
        )
        assert (
            forward.scratch_bytes == layer.scratch_bytes == rows * width * 2
            and len(forward.scratch_layout) == 3
        )
        grid = (32 // rows, 1, 1)
        source, matrices = metal.buffer(values.nbytes), metal.buffer(weights.nbytes)
        source.write(values.tobytes())
        matrices.write(weights.tobytes())
        for blocks in (0, 1, 2):
            expected = values.copy()
            for matrix in weights[: 3 * blocks]:
                expected = np.maximum(
                    expected.astype(np.float32) @ matrix.astype(np.float32),
                    0,
                ).astype(np.float16)
            (fused, fused_view), (split, split_view), (spare, spare_view) = (
                guarded(metal, values.nbytes) for _ in range(3)
            )
            dispatches = [
                forward.dispatch(
                    metal,
                    [source, matrices, fused_view],
                    grid=grid,
                    scalars={"blocks": blocks},
                ),
            ]
            current = source
            for number in range(3 * blocks):
                output = split_view if (3 * blocks - number) % 2 else spare_view
                dispatches.append(
                    layer.dispatch(
                        metal,
                        [current, matrices, output],
                        grid=grid,
                        scalars={"number": number},
                    ),
                )
                current = output
            metal.prepare(dispatches).submit().wait()
            result = np.frombuffer(unguarded(fused), np.float16).reshape(values.shape)
            np.testing.assert_allclose(result, expected, rtol=4e-3, atol=3e-3)
            if blocks:
                assert unguarded(fused) == unguarded(split), (width, blocks)
            unguarded(spare)
        assert (
            source.read() == values.tobytes() and matrices.read() == weights.tobytes()
        )
