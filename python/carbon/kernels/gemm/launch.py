from dataclasses import dataclass, replace
from math import ceil, prod

from carbon import language as cl
from carbon.compiler import Atom, Layout, View
from carbon.kernels import search
from carbon.kernels.gemm.gemv import make_cooperative_gemv
from carbon.kernels.gemm.quantized import (
    make_dequantize,
    make_qmm,
    make_qmm_registers,
    make_qmv,
    make_qmv_rows,
    packing,
    per_word,
    vector_words,
)
from carbon.kernels.gemm.tensor import make_tensor_matmul
from carbon.kernels.gemm.tiled import (
    dimensions,
    make_sum_partials,
    make_tiled_matmul,
    operand_staging,
    tile_shape,
)

DTYPES = {"f16": cl.f16, "bf16": cl.bf16, "f32": cl.f32}
DENSE_ROWS, FEW_ROWS, CANDIDATES = 512, 8, 6
ROW_DEFAULT, ROW_PLAN = 4, 16
WORKSPACE_LIMIT, TARGET_GROUPS, TILE_GROUPS = 64 << 20, 64, 96
TILES = (
    ((4, 1), (4, 4)),
    ((2, 2), (4, 4)),
    ((1, 1), (4, 4)),
    ((1, 1), (4, 2)),
    ((1, 1), (2, 4)),
    ((1, 1), (2, 2)),
    ((1, 4), (1, 2)),
    ((1, 1), (1, 4)),
    ((1, 4), (1, 1)),
)
SMALL_TILES = (((1, 1), (1, 4)), ((1, 4), (1, 2)), ((1, 4), (1, 1)), ((1, 1), (2, 2)))
WIDE_SMALL_TILES = (
    ((1, 4), (1, 2)),
    ((1, 1), (1, 4)),
    ((1, 4), (1, 1)),
    ((1, 1), (2, 2)),
)


@dataclass(frozen=True)
class Tiles:
    """Output tiles on the 8 x 8 matrix units: `groups` SIMD groups of `fragments` 8 x 8
    blocks each, `block_k`-deep steps, and `splits` parts of K that a second pass
    sums."""

    groups: tuple = (2, 2)
    fragments: tuple = (2, 4)
    block_k: int = 32
    splits: int = 1

    @property
    def threads(self):
        return 32 * self.groups[0] * self.groups[1]

    @property
    def layout(self):
        return Layout.striped(
            tile_shape(self.groups, self.fragments),
            threads=self.threads,
        )


@dataclass(frozen=True)
class Tensor:
    """Output tiles on the tensor ops, the Neural Accelerators on M5."""

    tile: tuple = (64, 64)


@dataclass(frozen=True)
class Vector:
    """One row of output: `reduction` threads over (K, N), and `splits` parts of K that
    a second pass sums."""

    reduction: tuple = (4, 32)
    splits: int = 1


@dataclass(frozen=True)
class QuantizedVector:
    """One or two rows over packed weights: a SIMD group per output feature, `words`
    words of K per lane."""

    outputs: int = 4
    words: int = 2


@dataclass(frozen=True)
class QuantizedRows:
    """A few rows over packed whole-word codes: `features` output features per SIMD
    group and `simds` SIMD groups, lanes over words of K. The rows share each read of
    the weights."""

    features: int = 2
    simds: int = 4


@dataclass(frozen=True)
class QuantizedRegisters:
    """Up to 32 rows on the 8 x 8 matrix units with the weights unpacked in registers,
    `split` SIMD groups along K. Up to `extra` rows past the last whole tile of eight
    take fused multiply-adds instead of one more tile."""

    split: int = 4
    extra: int = 3


@dataclass(frozen=True)
class QuantizedTiles:
    """Output `block`s from staged x and dequantized weight tiles, on the tensor ops
    with `tensor`."""

    block: tuple = (32, 32)
    tensor: bool = False


@dataclass(frozen=True)
class Dequantized:
    """The weight dequantized once into the caller's scratch, then a dense product."""


@dataclass(frozen=True, eq=False)
class Product:
    left: object
    right: object
    output: object
    LEFT: object
    RIGHT: object
    OUTPUT: object
    PRODUCT: object
    glu: object
    epilogue: object
    operands: tuple

    @property
    def dtype(self):
        return DTYPES[self.LEFT.element]


@dataclass(frozen=True, eq=False)
class QuantizedProduct:
    x: object
    packed: object
    scales: object
    biases: object
    output: object
    rows: int
    n: int
    k: int
    dtype: object
    fmt: object
    glu: object
    epilogue: object
    operands: tuple
    scratch: object
    stride: object

    @property
    def width(self):
        return self.n if self.glu is None else self.n // 2


def matmul(
    device,
    left,
    LEFT,
    right,
    RIGHT,
    output,
    OUTPUT,
    *,
    glu=None,
    epilogue=None,
    operands=(),
    plan=None,
):
    """Return the dispatches for output = left @ right.

    Args:
        left, right, output: the bindings.
        LEFT, RIGHT, OUTPUT: their Views: (M, K), (K, N) and (M, N), or batched with a
            leading axis.
        glu: a Glu that pairs product columns 2j and 2j + 1 into output column j, or
            None.
        epilogue: an Epilogue that maps the product before the store, or None.
        operands: the epilogue's buffers, one per operand type, in order.
        plan: a Tiles, Tensor or Vector in place of the runtime search's choice.

    Returns:
        The dispatches to prepare and submit, in order.
    """
    return launch(
        device,
        dense_product(
            left,
            LEFT,
            right,
            RIGHT,
            output,
            OUTPUT,
            glu,
            epilogue,
            operands,
        ),
        plan,
    )


def dense_matmul(
    device,
    x,
    weight,
    output,
    *,
    rows,
    n,
    k,
    dtype,
    glu=None,
    epilogue=None,
    operands=(),
    stride=None,
    plan=None,
):
    """Return the dispatches for output (rows, n) = x (rows, k) @ weight.T with a dense
    (n, k) weight.

    Args:
        x, weight, output: the bindings.
        rows, n, k: the product's shape, n the weight's rows.
        dtype: cl.f16, cl.bf16 or cl.f32, for x, the weight and the output.
        glu: a Glu for a weight whose rows `interleave_halves` arranged; the output is
            then (rows, n / 2). Or None.
        epilogue, operands: an Epilogue that maps the product before the store, and its
            buffers.
        stride: the elements between output rows, the output's width by default.
        plan: a Tiles, Tensor or Vector in place of the runtime search's choice.

    Returns:
        The dispatches to prepare and submit, in order.
    """
    if glu is not None and n % 2:
        raise ValueError("a paired product needs an even n")
    width = n // 2 if glu is not None else n
    between = width if stride is None else stride
    size = dtype.itemsize
    return matmul(
        device,
        x,
        View(dtype.name, (rows, k), (k, 1), rows * k * size),
        weight,
        View(dtype.name, (k, n), (1, k), n * k * size),
        output,
        View(
            dtype.name,
            (rows, width),
            (between, 1),
            ((rows - 1) * between + width) * size,
        ),
        glu=glu,
        epilogue=epilogue,
        operands=operands,
        plan=plan,
    )


def quantized_matmul(
    device,
    x,
    packed,
    scales,
    biases,
    output,
    *,
    rows,
    n,
    k,
    dtype,
    fmt,
    glu=None,
    epilogue=None,
    operands=(),
    scratch=None,
    stride=None,
    plan=None,
):
    """Return the dispatches for output (rows, n) = x (rows, k) @ weight.T with
    quantized weights.

    Args:
        x, packed, scales, biases, output: the bindings.
        rows, n, k: the product's shape.
        dtype: the activation type: cl.f16, cl.bf16 or cl.f32.
        fmt: the quantization format, such as Affine(4, 64).
        glu: a Glu for a [gate | up] weight; the output is then (rows, n / 2). Or None.
        epilogue, operands: an Epilogue that maps the product before the store, and its
            buffers.
        scratch: a buffer of n * k elements of `dtype` that a Dequantized plan unpacks
            the weight into, or None.
        stride: the elements between output rows, the output's width by default.
        plan: a QuantizedVector, QuantizedRegisters, QuantizedTiles or Dequantized in
            place of the runtime search's choice.

    Returns:
        The dispatches to prepare and submit, in order.
    """
    product = QuantizedProduct(
        x,
        packed,
        scales,
        biases,
        output,
        rows,
        n,
        k,
        dtype,
        fmt,
        glu,
        epilogue,
        tuple(operands),
        scratch,
        stride,
    )
    if len(product.operands) != len(() if epilogue is None else epilogue.operands):
        raise ValueError(
            "operands must give one buffer per operand type of the epilogue",
        )
    if stride is not None and (type(stride) is not int or stride < product.width):
        raise ValueError("an output stride is an int of at least the output's width")
    return launch(device, product, plan)


def interleave_halves(weight):
    """The rows of a [gate | up] (n, k) weight array reordered for `dense_matmul`'s glu:
    gate row j to row 2j and up row j to row 2j + 1."""
    n, k = weight.shape
    return weight.reshape(2, n // 2, k).transpose(1, 0, 2).reshape(n, k)


def launch(device, product, plan=None):
    if plan is None:
        plan = search.choose(
            device,
            key(product),
            candidates(device, product),
            lambda sample, candidate: dispatches(device, sample, candidate),
            lambda: trial(device, product),
        )
    return dispatches(device, product, plan)


def dense_product(
    left,
    LEFT,
    right,
    RIGHT,
    output,
    OUTPUT,
    glu=None,
    epilogue=None,
    operands=(),
):
    operands = tuple(operands)
    if len(operands) != len(() if epilogue is None else epilogue.operands):
        raise ValueError(
            "operands must give one buffer per operand type of the epilogue",
        )
    PRODUCT = OUTPUT
    if glu is not None:
        if (
            len(OUTPUT.shape) != 2
            or len(RIGHT.shape) != 2
            or 2 * OUTPUT.shape[1] != RIGHT.shape[1]
        ):
            raise ValueError(
                "a paired product is rank two, its output half as wide "
                "as the right operand",
            )
        PRODUCT = View(
            OUTPUT.element,
            (OUTPUT.shape[0], RIGHT.shape[1]),
            (RIGHT.shape[1], 1),
            2 * OUTPUT.extent_bytes,
        )
    dimensions(LEFT, RIGHT, PRODUCT)
    if (
        LEFT.element not in DTYPES
        or RIGHT.element != LEFT.element
        or OUTPUT.element != LEFT.element
    ):
        raise ValueError("contractions require matching f16, bf16, or f32 storage")
    LEFT, RIGHT, PRODUCT = canonical(LEFT, RIGHT, PRODUCT)
    return Product(
        left,
        right,
        output,
        LEFT,
        RIGHT,
        PRODUCT if glu is None else OUTPUT,
        PRODUCT,
        glu,
        epilogue,
        operands,
    )


def canonical(left, right, output):
    if len(left.shape) != 3 or right.strides[0] != 0 or left.shape[0] == 1:
        return left, right, output
    batches, rows, depth = left.shape
    try:
        return (
            left.reshape((batches * rows, depth)),
            right[0],
            output.reshape((batches * rows, output.shape[2])),
        )
    except ValueError:
        return left, right, output


def dispatches(device, product, plan):
    steps = DENSE_STEPS if isinstance(product, Product) else QUANTIZED_STEPS
    if type(plan) not in steps:
        raise ValueError(
            f"a {type(product).__name__} takes one of "
            f"{', '.join(kind.__name__ for kind in steps)}",
        )
    return steps[type(plan)](device, product, plan)


def tiled_steps(device, product, plan):
    LEFT, RIGHT, OUTPUT, PRODUCT = (
        product.LEFT,
        product.RIGHT,
        product.OUTPUT,
        product.PRODUCT,
    )
    if plan.splits > 1 and (product.glu is not None or len(LEFT.shape) != 2):
        raise ValueError("split K takes a rank-two product without a glu")
    dtype, (rows, columns) = product.dtype, PRODUCT.shape[-2:]
    batches = OUTPUT.shape[0] if len(OUTPUT.shape) == 3 else 1
    workspace = (
        device.buffer(plan.splits * rows * columns * 4) if plan.splits > 1 else None
    )
    target, fused, bound = (
        (product.output, product.epilogue, product.operands)
        if workspace is None
        else (workspace, None, ())
    )
    partials = matrix((plan.splits, rows, columns)) if plan.splits > 1 else OUTPUT
    bm, bn = plan.layout.shape
    grid = plan.splits * batches * -(-rows // bm) * -(-columns // bn)
    constants = {
        "LEFT": LEFT,
        "RIGHT": RIGHT,
        "OUTPUT": partials,
        "GROUPS": plan.groups,
        "FRAGMENTS": plan.fragments,
        "ATOM": Atom(dtype),
        "SCHEDULE": plan.layout,
    }
    tiles = (2**32 - 1) // max(LEFT.strides[0], OUTPUT.strides[0], 1) // bm
    row_major = all(
        len(view.shape) == 2 and view.strides[1] == 1 and not view.offset_bytes
        for view in (LEFT, OUTPUT)
    )
    left = product.left
    if plan.splits == 1 and row_major and 2 * bm < rows <= tiles * bm:
        views = {
            name: View(view.element, (None, view.shape[1]), view.strides, None)
            for name, view in (("left", LEFT), ("output", OUTPUT))
        }
        kernel = make_tiled_matmul(
            dtype,
            block_k=plan.block_k,
            glu=product.glu,
            epilogue=fused,
            runtime_rows=True,
        ).compile(
            constants={**constants, "LEFT": views["left"], "OUTPUT": views["output"]},
            threads=plan.threads,
            device=device,
            max_grid=tiles * -(-columns // bn),
            views=views,
        )
        left, target = LEFT.bind(left), OUTPUT.bind(target)
    else:
        kernel = make_tiled_matmul(
            dtype,
            block_k=plan.block_k,
            split_k=plan.splits,
            glu=product.glu,
            epilogue=fused,
        ).compile(
            constants=constants,
            threads=plan.threads,
            device=device,
            max_grid=grid,
        )
    steps = [
        kernel.dispatch(
            device,
            [left, product.right, target, *bound],
            grid=(grid, 1, 1),
        ),
    ]
    return steps + summed(device, product, workspace, partials, plan.splits)


def tensor_steps(device, product, plan):
    if (
        product.glu is not None
        or len(product.LEFT.shape) != 2
        or product.dtype not in (cl.f16, cl.bf16)
    ):
        raise ValueError(
            "tensor tiles take a rank-two f16 or bf16 product without a glu",
        )
    rows, columns = product.PRODUCT.shape
    grid = -(-rows // plan.tile[0]) * -(-columns // plan.tile[1])
    kernel = make_tensor_matmul(
        product.dtype,
        tile=plan.tile,
        epilogue=product.epilogue,
    ).compile(
        constants={
            "LEFT": product.LEFT,
            "RIGHT": product.RIGHT,
            "OUTPUT": product.OUTPUT,
        },
        threads=128,
        device=device,
        max_grid=grid,
    )
    return [
        kernel.dispatch(
            device,
            [product.left, product.right, product.output, *product.operands],
            grid=(grid, 1, 1),
        ),
    ]


def vector_steps(device, product, plan):
    LEFT, RIGHT, OUTPUT = product.LEFT, product.RIGHT, product.OUTPUT
    if product.glu is not None or len(LEFT.shape) != 2 or LEFT.shape[0] != 1:
        raise ValueError(
            "a vector plan takes one row of a rank-two product without a glu",
        )
    columns = OUTPUT.shape[1]
    workspace = device.buffer(plan.splits * columns * 4) if plan.splits > 1 else None
    target, fused, bound = (
        (product.output, product.epilogue, product.operands)
        if workspace is None
        else (workspace, None, ())
    )
    partials = matrix((plan.splits, columns)) if plan.splits > 1 else OUTPUT
    schedule = Layout.striped(plan.reduction, threads=128)
    grid = plan.splits * -(-columns // plan.reduction[1])
    kernel = make_cooperative_gemv(
        product.dtype,
        splits=plan.splits,
        epilogue=fused,
    ).compile(
        constants={
            "LEFT": LEFT,
            "RIGHT": RIGHT,
            "OUTPUT": partials,
            "SCHEDULE": schedule,
        },
        threads=128,
        device=device,
        max_grid=grid,
    )
    steps = [
        kernel.dispatch(
            device,
            [product.left, product.right, target, *bound],
            grid=(grid, 1, 1),
        ),
    ]
    return steps + summed(
        device,
        product,
        workspace,
        matrix((plan.splits, 1, columns)),
        plan.splits,
    )


def summed(device, product, workspace, partials, splits):
    if splits == 1:
        return []
    sums = -(-product.OUTPUT.size // 1024)
    kernel = make_sum_partials(product.dtype, epilogue=product.epilogue).compile(
        constants={
            "PARTIALS": partials,
            "OUTPUT": product.OUTPUT,
            "SCHEDULE": Layout.striped((1024,), threads=256),
        },
        threads=256,
        device=device,
        max_grid=sums,
    )
    return [
        kernel.dispatch(
            device,
            [workspace, product.output, *product.operands],
            grid=(sums, 1, 1),
        ),
    ]


def matrix(shape):
    return View(
        "f32",
        shape,
        tuple(prod(shape[i + 1 :]) for i in range(len(shape))),
        prod(shape) * 4,
    )


def quantized_vector_steps(device, product, plan):
    p = product
    if not (p.rows == 1 or p.rows == 2 and per_word(p.fmt)):
        raise ValueError(
            "a quantized vector plan takes one row, or two of whole-word codes",
        )
    grid = -(-p.width // plan.outputs)
    kernel = make_qmv(
        p.dtype,
        p.fmt,
        outputs=plan.outputs,
        words=plan.words,
        rows=p.rows,
        glu=p.glu,
        epilogue=p.epilogue,
        stride=p.stride,
    ).compile(
        constants={"N": p.n, "K": p.k},
        threads=32 * plan.outputs,
        device=device,
        max_grid=grid,
    )
    return [
        kernel.dispatch(
            device,
            [p.x, p.packed, p.scales, p.biases, p.output, *p.operands],
            grid=(grid, 1, 1),
        ),
    ]


def quantized_rows_steps(device, product, plan):
    p = product
    if not (2 <= p.rows <= ROW_PLAN and per_word(p.fmt)):
        raise ValueError(
            f"a quantized rows plan takes 2 to {ROW_PLAN} rows of whole-word codes",
        )
    grid = -(-p.width // (plan.features * plan.simds))
    kernel = make_qmv_rows(
        p.dtype,
        p.fmt,
        rows=p.rows,
        features=plan.features,
        simds=plan.simds,
        glu=p.glu,
        epilogue=p.epilogue,
        stride=p.stride,
    ).compile(
        constants={"N": p.n, "K": p.k},
        threads=32 * plan.simds,
        device=device,
        max_grid=grid,
    )
    return [
        kernel.dispatch(
            device,
            [p.x, p.packed, p.scales, p.biases, p.output, *p.operands],
            grid=(grid, 1, 1),
        ),
    ]


def quantized_registers_steps(device, product, plan):
    p = product
    codes = packing(p.fmt)[1]
    if p.rows > 32 or p.fmt.group % codes or p.k % (8 * codes * plan.split):
        raise ValueError(
            "register tiles take up to 32 rows, with groups and K split "
            "into whole words",
        )
    tiles, extra = divmod(p.rows, 8)
    if not tiles or extra > plan.extra:
        tiles, extra = -(-p.rows // 8), 0
    grid = -(-p.width // (16 if p.glu is None else 8))
    kernel = make_qmm_registers(
        p.dtype,
        p.fmt,
        tiles=tiles,
        extra=extra,
        split=plan.split,
        glu=p.glu,
        epilogue=p.epilogue,
        stride=p.stride,
    ).compile(
        constants={"N": p.n, "K": p.k},
        threads=32 * plan.split,
        device=device,
        max_grid=grid,
        views={"x": View(p.dtype.name, (None, p.k), (p.k, 1), None)},
    )
    x = View(
        p.dtype.name,
        (p.rows, p.k),
        (p.k, 1),
        p.rows * p.k * p.dtype.itemsize,
    ).bind(p.x)
    return [
        kernel.dispatch(
            device,
            [x, p.packed, p.scales, p.biases, p.output, *p.operands],
            grid=(grid, 1, 1),
        ),
    ]


def quantized_tiles_steps(device, product, plan):
    p = product
    limit = (2**32 - 1) // max(p.n, p.k)
    if p.rows > limit:
        raise ValueError(
            f"quantized_matmul takes at most {limit} rows of these "
            "shapes, for u32 indexing",
        )
    block, tensor = plan.block, plan.tensor
    columns = -(-p.width // (block[1] if p.glu is None else block[1] // 2))
    grid = -(-p.rows // block[0]) * columns
    views = (
        {
            "x": View(
                p.dtype.name,
                (p.rows, p.k),
                (p.k, 1),
                p.rows * p.k * p.dtype.itemsize,
            ),
        }
        if tensor
        else None
    )
    kernel = make_qmm(
        p.dtype,
        p.fmt,
        block=block,
        tensor=tensor,
        glu=p.glu,
        epilogue=p.epilogue,
        stride=p.stride,
    ).compile(
        constants={"N": p.n, "K": p.k},
        threads=128,
        device=device,
        max_grid=grid if tensor else -(-limit // block[0]) * columns,
        views=views,
    )
    x = views["x"].bind(p.x) if tensor else p.x
    return [
        kernel.dispatch(
            device,
            [x, p.packed, p.scales, p.biases, p.output, *p.operands],
            grid=(grid, 1, 1),
            scalars={"rows": p.rows},
        ),
    ]


def dequantized_steps(device, product, plan):
    p = product
    if p.scratch is None:
        raise ValueError("a dequantized plan needs scratch for the weight")
    count = per_word(p.fmt)
    grid = -(-p.n // 32) * -(-p.k // (8 * (count or 8)))
    kernel = make_dequantize(p.dtype, p.fmt, interleave=p.glu is not None).compile(
        constants={"N": p.n, "K": p.k},
        threads=256,
        device=device,
        max_grid=grid,
    )
    size, between = p.dtype.itemsize, p.width if p.stride is None else p.stride
    return [
        kernel.dispatch(
            device,
            [p.packed, p.scales, p.biases, p.scratch],
            grid=(grid, 1, 1),
        ),
    ] + matmul(
        device,
        p.x,
        View(p.dtype.name, (p.rows, p.k), (p.k, 1), p.rows * p.k * size),
        p.scratch,
        View(p.dtype.name, (p.k, p.n), (p.n, 1), p.n * p.k * size),
        p.output,
        View(
            p.dtype.name,
            (p.rows, p.width),
            (between, 1),
            ((p.rows - 1) * between + p.width) * size,
        ),
        glu=p.glu,
        epilogue=p.epilogue,
        operands=p.operands,
    )


DENSE_STEPS = {Tiles: tiled_steps, Tensor: tensor_steps, Vector: vector_steps}
QUANTIZED_STEPS = {
    QuantizedVector: quantized_vector_steps,
    QuantizedRows: quantized_rows_steps,
    QuantizedRegisters: quantized_registers_steps,
    QuantizedTiles: quantized_tiles_steps,
    Dequantized: dequantized_steps,
}


def candidates(device, product):
    plans = (
        dense_plans(device, product)
        if isinstance(product, Product)
        else quantized_plans(device, product)
    )
    return list(dict.fromkeys(plans))[:CANDIDATES]


def dense_plans(device, product):
    left, right, output = product.LEFT, product.RIGHT, product.PRODUCT
    batches, m, n, k = dimensions(left, right, output)
    rank_two = len(left.shape) == 2 and m * n > 0
    single = product.glu is None
    tiles = tile_plan(product)
    plans = [default_plan(device, product)]
    if single and rank_two and m == 1:
        vector = gemv_plan(m, n, k)
        plans += [vector, replace(vector, splits=max(1, vector.splits // 2))]
    tensor = left.element in ("f16", "bf16") and getattr(device, "tensor_ops", False)
    if single and rank_two and m > FEW_ROWS and tensor:
        plans += [Tensor((128, 64) if m > 64 else (64, 64)), Tensor((64, 64))]
    plans.append(tiles)
    table = TILES if m > FEW_ROWS else WIDE_SMALL_TILES if n >= 256 else SMALL_TILES
    shapes = list(table)
    if (tiles.groups, tiles.fragments) in shapes:
        place = shapes.index((tiles.groups, tiles.fragments))
        plans += [
            replace(tiles, groups=groups, fragments=fragments)
            for groups, fragments in shapes[max(0, place - 1) : place + 2]
        ]
    if single and rank_two:
        plans += [
            replace(tiles, splits=count)
            for count in (tiles.splits * 2, tiles.splits // 2)
            if 1 <= count <= 64 and fits(m, n, count)
        ]
    return [
        fitted for plan in plans if (fitted := fit(device, product, plan)) is not None
    ]


def default_plan(device, product):
    left, right, output = product.LEFT, product.RIGHT, product.PRODUCT
    batches, m, n, k = dimensions(left, right, output)
    rank_two = len(left.shape) == 2 and m * n > 0
    halves = rank_two and left.element in ("f16", "bf16") and product.glu is None
    if (
        halves
        and getattr(device, "apple_family", 0) >= 10
        and m > 8
        and (m > 32 or n > 64)
    ):
        return Tensor((64, 64) if m <= 64 else (128, 64))
    wide = m >= 512 and n >= 1024 or m >= 4096 and n >= 512
    if halves and getattr(device, "tensor_ops", False) and wide and k < 6144:
        return Tensor((64, 64))
    if product.glu is None and rank_two and m == 1 and n <= 64:
        return gemv_plan(m, n, k)
    if (
        product.glu is None
        and len(left.shape) == 2
        and m == 1
        and left.element == "f32"
    ):
        if k <= 512:
            return Vector((8, 16))
        if k >= 8192 and n <= 1024:
            return split_vector(Vector((4, 32)), m, n, k, rank_two)
    plan = fit(device, product, tile_plan(product))
    if plan is None:
        raise ValueError("the operand tiles exceed the device's threadgroup memory")
    return plan


def tile_plan(product):
    left, right, output = product.LEFT, product.RIGHT, product.PRODUCT
    batches, m, n, k = dimensions(left, right, output)
    rank_two = len(left.shape) == 2 and m * n > 0
    single = product.glu is None
    if m <= FEW_ROWS:
        limit = max_partitions(k, 8, 256) if rank_two and single else 1
        while limit > 1 and not fits(m, n, limit):
            limit //= 2
        groups, fragments, splits = select_small_tile(batches, m, n, limit)
    else:
        splittable = rank_two and single and k >= 4096 and fits(m, n, 8)
        groups, fragments, splits = select_tile(batches, m, n, 8 if splittable else 1)
    return Tiles(groups, fragments, 32, splits)


def gemv_plan(m, n, k):
    columns = min(32, 1 << max(0, (n - 1).bit_length()))
    rows = 128 // columns
    splits = 1
    while splits < max_partitions(k, rows, 64) and ceil(n / columns) * splits < 32:
        splits *= 2
    return split_vector(
        Vector((rows, columns), splits if fits(m, n, splits) else 1),
        m,
        n,
        k,
        True,
    )


def split_vector(plan, m, n, k, rank_two):
    if (
        plan.splits == 1
        and rank_two
        and k >= 8192
        and fits(m, n, 8)
        and ceil(n / plan.reduction[1]) < TARGET_GROUPS
    ):
        return replace(plan, splits=8)
    return plan


def fit(device, product, plan):
    if not isinstance(plan, Tiles):
        return plan
    bm, bn = tile_shape(plan.groups, plan.fragments)
    stage_left, stage_right, _ = operand_staging(product.LEFT, product.RIGHT, bn)
    step = (bm * stage_left + bn * stage_right) * product.LEFT.itemsize
    if step * plan.block_k <= device.max_threadgroup_memory:
        return plan
    if step * 8 <= device.max_threadgroup_memory:
        return replace(plan, block_k=8)
    return None


def fits(m, n, splits):
    workspace = splits * m * n * 4
    return workspace <= WORKSPACE_LIMIT and workspace // 4 < 2**32


def select_tile(batches, m, n, splits):
    for groups, fragments in TILES:
        bm, bn = tile_shape(groups, fragments)
        if 4 * m < 3 * bm and not (m < 8 and bm == 8):
            continue
        for split in sorted({1, splits}):
            if split * batches * ceil(m / bm) * ceil(n / bn) >= TILE_GROUPS:
                return groups, fragments, split
    return (1, 4), (1, 2), 1


def select_small_tile(batches, m, n, max_splits):
    split = 1
    while True:
        for groups, fragments in WIDE_SMALL_TILES if n >= 256 else SMALL_TILES:
            bm, bn = tile_shape(groups, fragments)
            if (4 * m >= 3 * bm or bm == 8) and split * batches * ceil(m / bm) * ceil(
                n / bn,
            ) >= TILE_GROUPS:
                return groups, fragments, split
        if split >= max_splits:
            return (1, 4), (1, 2), max_splits
        split *= 2


def max_partitions(k, rows, limit):
    splits = 1
    while splits < limit and k // (rows * splits * 2) >= 8:
        splits *= 2
    return splits


def quantized_plans(device, product):
    p = product
    count, codes = per_word(p.fmt), packing(p.fmt)[1]
    plans = [default_quantized(device, p)]
    if p.rows == 1 or p.rows == 2 and count:
        plans += [QuantizedVector(words=words) for words in (2, 1)]
    if 2 <= p.rows <= FEW_ROWS and count:
        plans += [
            plan for plan in (QuantizedRows(), QuantizedRows(4, 2)) if plan not in plans
        ]
    if p.rows <= 32 and p.fmt.group % codes == 0 and p.k % (8 * codes) == 0:
        plans += [
            QuantizedRegisters(split)
            for split in (4, 2, 1)
            if p.k % (8 * codes * split) == 0
        ][:2]
    if p.rows > 2:
        plans += [QuantizedTiles((32, 32)), QuantizedTiles((64, 32))]
        if getattr(device, "tensor_ops", False) and p.dtype in (cl.f16, cl.bf16):
            plans.append(QuantizedTiles((64, 64), True))
        if p.scratch is not None and p.rows > 32:
            plans.append(Dequantized())
    return plans


def default_quantized(device, p):
    count, codes = per_word(p.fmt), packing(p.fmt)[1]
    halves = p.dtype in (cl.f16, cl.bf16)
    tensor = (
        getattr(device, "apple_family", 0) >= 10
        and halves
        and (p.rows > 32 or p.n > 1024)
    )
    if p.rows == 1:
        return QuantizedVector(words=vector_words(p.fmt, p.k))
    if p.rows <= ROW_DEFAULT and count:
        return QuantizedRows()
    if (
        (p.rows <= 8 or p.rows <= 32 and not tensor)
        and p.fmt.group % codes == 0
        and p.k % (8 * codes) == 0
    ):
        return QuantizedRegisters(
            next(split for split in (4, 2, 1) if p.k % (8 * codes * split) == 0),
        )
    if p.scratch is not None and not tensor and p.rows >= DENSE_ROWS:
        return Dequantized()
    if tensor:
        return QuantizedTiles((64, 64), True)
    return QuantizedTiles(
        (64, 32) if -(-p.rows // 64) * 1.8 < -(-p.rows // 32) else (32, 32),
    )


def key(product):
    if isinstance(product, Product):
        batches, m, n, k = dimensions(product.LEFT, product.RIGHT, product.PRODUCT)
        views = (product.LEFT, product.RIGHT, product.OUTPUT)
        layouts = tuple((view.strides, view.offset_bytes % 16) for view in views)
        return (
            "matmul",
            product.LEFT.element,
            batches,
            bucket(m),
            n,
            k,
            layouts,
            search.stable(product.glu),
            search.stable(product.epilogue),
        )
    return (
        "quantized",
        product.dtype.name,
        search.stable(product.fmt),
        bucket(product.rows),
        product.n,
        product.k,
        product.stride,
        product.scratch is not None,
        search.stable(product.glu),
        search.stable(product.epilogue),
    )


def bucket(rows):
    return rows if rows <= FEW_ROWS else 1 << (rows - 1).bit_length()


def trial(device, product):
    if isinstance(product, Product):
        return replace(product, output=device.buffer(product.output.size))
    scratch = None if product.scratch is None else device.buffer(product.scratch.size)
    return replace(product, output=device.buffer(product.output.size), scratch=scratch)
