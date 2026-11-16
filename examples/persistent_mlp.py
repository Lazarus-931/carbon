from carbon import language as cl
from carbon.compiler import Atom, TiledAtom, View


def _kernels(width, rows, groups):
    if (
        any(type(value) is not int for value in (width, rows, groups))
        or not 1 <= groups <= 32
        or width < 8 * groups
        or width % (8 * groups)
        or rows < 8
        or rows % 8
    ):
        raise ValueError(
            "width must tile across 1 to 32 SIMD groups and rows must be a positive "
            "multiple of 8",
        )
    tiled = TiledAtom(
        Atom(cl.f16),
        groups=(1, groups),
        fragments=(rows // 8, width // (8 * groups)),
    )
    layout = tiled.layout()

    @cl.inline
    def layer(value, weights, number, BLOCKS):
        shared = cl.scratch((rows, width), cl.f16)
        index = cl.arange(rows * width, layout=layout)
        cl.store(shared, index, value, True)
        cl.barrier()
        storage = cl.view(
            weights,
            View(
                "f16",
                (3 * BLOCKS * width, width),
                (width, 1),
                3 * BLOCKS * width * width * 2,
            ),
        )
        matrix = cl.subview(storage, (number * width, 0), (width, width))
        total = tiled.zeros()
        for depth in range(width // 8):
            left = tiled.load_a(shared, 0, depth)
            right = tiled.load_b(matrix, cl.simd_id() * tiled.fragments[1], depth)
            total = tiled.mma(left, right, total)
        cl.barrier()
        return cl.cast(cl.maximum(tiled.unpack(total), 0.0), cl.f16)

    @cl.kernel
    def forward(
        x: cl.buffer(cl.f16),
        weights: cl.buffer(cl.f16),
        output: cl.buffer(cl.f16),
        blocks: cl.u32,
        BLOCKS: cl.constexpr,
    ):
        index = cl.program_id() * rows * width + cl.arange(rows * width, layout=layout)
        value = cl.load(x, index, True, other=0.0)
        for block in range(blocks):
            value = layer(value, weights, 3 * block, BLOCKS)
            value = layer(value, weights, 3 * block + 1, BLOCKS)
            value = layer(value, weights, 3 * block + 2, BLOCKS)
        cl.store(output, index, value, True)

    @cl.kernel
    def single_layer(
        x: cl.buffer(cl.f16),
        weights: cl.buffer(cl.f16),
        output: cl.buffer(cl.f16),
        number: cl.u32,
        BLOCKS: cl.constexpr,
    ):
        index = cl.program_id() * rows * width + cl.arange(rows * width, layout=layout)
        value = cl.load(x, index, True, other=0.0)
        value = layer(value, weights, number, BLOCKS)
        cl.store(output, index, value, True)

    return forward, single_layer


def make_mlp(width, *, rows=16, groups=4):
    """Build an f16 MLP with f32 accumulation and ReLU after each layer.

    Weights have shape (3 * BLOCKS, width, width), inputs have a multiple of `rows`
    examples, and runtime `blocks` must be at most BLOCKS. Each threadgroup owns all
    hidden features of its examples, so layers need no cross-threadgroup barrier.
    """
    return _kernels(width, rows, groups)[0]


def make_mlp_layer(width, *, rows=16, groups=4):
    """Build one layer of the same computation and tile schedule for split-dispatch
    comparisons.

    Weights have shape (3 * BLOCKS, width, width); runtime `number` selects a layer
    in [0, 3 * BLOCKS). Each call reads and writes one complete activation tensor.
    """
    return _kernels(width, rows, groups)[1]
