import struct
from dataclasses import dataclass
from functools import cache
from math import prod

from ..language import STORAGE, Scalar, _builder, bf16, f16, f32
from .layout import Layout, Mode

_KINDS = {"f16": "half", "bf16": "bfloat", "f32": "float"}
_TENSOR_HEADER = (
    "#include <metal_stdlib>\n#include <metal_tensor>\n"
    "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
    "using namespace metal;\nusing namespace mpp::tensor_ops;\n"
)


@cache
def _device():
    from .._metal import Device

    return Device()


@cache
def _measured(source, shape, groups, count=1):
    """The layouts a probe kernel reports for `count` tensors: each thread writes,
    tensor by tensor and slot by slot, the row-major index of the element it holds, or
    -1 when the device does not hold `shape` in whole registers."""
    slots = prod(shape) // (32 * groups)
    size = groups * 32 * slots
    device = _device()
    pipeline = device.compile(source, "probe", fast_math=False)
    if pipeline.simd_width != 32:
        raise RuntimeError("matrix units need 32-lane SIMD groups")
    output = device.buffer(count * size * 4)
    device.prepare(
        [device.dispatch(pipeline, [output], threads=(groups * 32, 1, 1))],
    ).submit().wait()
    elements = struct.unpack(f"<{count * size}i", output.read())
    if min(elements) < 0:
        raise ValueError(
            f"this device does not hold a {shape[0]}x{shape[1]} block in {slots} "
            "registers per thread",
        )
    return [
        Layout.measured(
            shape,
            groups,
            slots,
            list(elements[at * size : (at + 1) * size]),
        )
        for at in range(count)
    ]


@cache
def _simdgroup(dtype):
    kind = _KINDS[dtype.name]
    return _measured(
        f"""#include <metal_stdlib>
using namespace metal;
kernel void probe(device int* output [[buffer(0)]],
                  ushort lane [[thread_index_in_simdgroup]]) {{
    threadgroup {kind} values[64];
    values[lane] = {kind}(lane);
    values[lane + 32] = {kind}(lane + 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<{kind}, 8, 8> matrix;
    simdgroup_load(matrix, values, 8);
    output[lane * 2] = int(float(matrix.thread_elements()[0]));
    output[lane * 2 + 1] = int(float(matrix.thread_elements()[1]));
}}
""",
        (8, 8),
        1,
    )[0]


def _cooperative(blocks, tensor, shape, groups):
    """The registers of the cooperative tensor `tensor` that each of `blocks` declares,
    in `groups` SIMD groups. Multidimensional indices list the column first."""
    rows, columns = shape
    slots = rows * columns // (32 * groups)
    size = groups * 32 * slots
    bodies = "".join(
        f"""    {{
    {declarations}
    bool whole = {tensor}.get_capacity() == {slots};
    for (ushort slot = 0; slot < {slots}; ++slot) {{
        auto at = {tensor}.get_multidimensional_index(whole ? slot : 0);
        output[{index * size} + (group * 32 + lane) * {slots} + slot] =
            whole && {tensor}.is_valid_element(slot)
                ? int(at[1]) * {columns} + int(at[0]) : -1;
    }}
    }}
"""
        for index, declarations in enumerate(blocks)
    )
    return _measured(
        f"""{_TENSOR_HEADER}kernel void probe(device int* output [[buffer(0)]],
                  ushort lane [[thread_index_in_simdgroup]],
                  ushort group [[simdgroup_index_in_threadgroup]]) {{
{bodies}}}
""",
        shape,
        groups,
        len(blocks),
    )


def _descriptor(m, n, k, relaxed, transposes=(False, False)):
    left, right = (str(flag).lower() for flag in transposes)
    return (
        f"constexpr auto descriptor = matmul2d_descriptor({m}, {n}, {k}, {left}, "
        f"{right}, {str(relaxed).lower()}, "
        "matmul2d_descriptor::mode::multiply_accumulate);\n    "
    )


@cache
def _tensor_unit(shape, dtype, relaxed, role):
    """The registers of one SIMD group's `role` cooperative tensor for an m x n x k
    matmul2d. The probe runs the product too, so a shape the device cannot multiply
    fails here."""
    m, n, k = shape
    kind = _KINDS[dtype.name]
    tensor, block = {
        "a": ("left", (m, k)),
        "b": ("right", (k, n)),
        "accumulator": ("destination", (m, n)),
    }[role]
    return _cooperative(
        [
            f"{_descriptor(m, n, k, relaxed)}matmul2d<descriptor, "
            "execution_simdgroup> operation;\n    auto left = "
            f"operation.get_left_input_cooperative_tensor<{kind}, {kind}, "
            "float>();\n    auto right = "
            f"operation.get_right_input_cooperative_tensor<{kind}, {kind}, "
            "float>();\n    auto destination = "
            "operation.get_destination_cooperative_tensor<decltype(left), "
            "decltype(right), float>();\n    operation.run(left, right, "
            "destination);",
        ],
        tensor,
        block,
        1,
    )[0]


@cache
def _destination(shape, groups, dtype, relaxed):
    """The registers of the product of a matmul2d that `groups` SIMD groups run on
    tensors of `dtype`. Apple leaves them implementation defined, and on M5 relaxed f32
    products use another order, so the probe asks for every descriptor and operand that
    code generation may emit with this precision (either operand transposed, device or
    threadgroup memory) and requires one answer."""
    dynamic = "static_cast<int>(dynamic_extent)"
    blocks = [
        f"{_descriptor(*shape, dynamic, relaxed, transposes)}matmul2d<descriptor, "
        f"execution_simdgroups<{groups}>> operation;\n    auto destination = "
        f"operation.get_destination_cooperative_tensor<{operand}, {operand}, "
        "float>();"
        for transposes in ((False, False), (False, True), (True, False), (True, True))
        for operand in (
            f"tensor<{space} {_KINDS[dtype.name]}, dextents<int32_t, 2>, tensor_inline>"
            for space in ("device", "threadgroup")
        )
    ]
    layouts = _cooperative(blocks, "destination", shape, groups)
    if any(layout != layouts[0] for layout in layouts):
        raise ValueError(
            "this device lays out a matrix product's registers differently for "
            "transposed or threadgroup operands; Carbon needs one layout per shape "
            "and precision",
        )
    return layouts[0]


def _groups(threads):
    if type(threads) is not int or not 32 <= threads <= 1024 or threads % 32:
        raise ValueError(
            "matrix layouts require full 32-lane SIMD groups, up to 1024 threads",
        )
    return threads // 32


def _blocks(view, index, layout):
    """Coordinates of each SIMD group's block of a rank-two view at `index`, a block
    number in row-major order or a (row, column) block pair, and whether that block
    starts inside the view; accesses check each coordinate."""
    from .. import language as cl

    builder = _builder()
    if not getattr(view, "view", False) or builder.view_rank(view) != 2:
        raise ValueError("matrix atom memory needs a matching rank-two view")
    groups = layout.groups.size
    rows, columns = layout.shape[0] // groups, layout.shape[1]
    if layout.size != groups * rows * columns or any(
        layout.groups.offset(g) != g * rows * columns for g in range(groups)
    ):
        raise TypeError("matrix memory moves tiles that stack one block per SIMD group")
    height, width = (
        view.descriptor.shape
        if view.descriptor is not None
        else (cl.dim(view, 0), cl.dim(view, 1))
    )
    if isinstance(index, tuple):
        if len(index) != 2:
            raise ValueError(
                "matrix memory takes a block number or a (row, column) block pair",
            )
        row, column = index
    else:
        across = (width + columns - 1) // columns
        across = max(across, 1) if isinstance(across, int) else across
        row, column = index // across, index % across
    element = cl.arange(layout.size, layout=layout) % (rows * columns)
    inside = (row < (height + rows - 1) // rows) & (
        column < (width + columns - 1) // columns
    )
    return (
        row * rows + element // columns,
        column * columns + element % columns,
    ), inside


class _Unit:
    def _block(self, role):
        m, n, k = self.shape
        if role not in ("a", "b", "accumulator"):
            raise ValueError("a matrix role is 'a', 'b' or 'accumulator'")
        return {"a": (m, k), "b": (k, n), "accumulator": (m, n)}[role]

    def _element(self, role):
        self._block(role)
        return f32 if role == "accumulator" else self.operand

    def layout(self, role="accumulator", *, threads=32):
        """Each SIMD group's `role` block, stacked by rows, in the registers the device
        holds it in."""
        groups = _groups(threads)
        self._block(role)
        block = self._registers(role)
        rows, columns = block.shape
        return Layout(
            (groups * rows, columns),
            Mode(groups, rows * columns),
            block.lanes,
            block.slots,
        )

    def zeros(self):
        """An f32 accumulator of zeros: one block per SIMD group of the kernel."""
        layout = self.layout(threads=_builder().threads)
        return _builder().full(layout.shape, 0.0, f32, layout=layout)

    def load(self, view, index, *, role):
        """Each SIMD group's `role` block of a rank-two view at block `index`; elements
        past the edge are zero, and f16/bf16 views widen to f32 operands."""
        from .. import language as cl

        element = self._element(role)
        layout = self.layout(role, threads=_builder().threads)
        if getattr(view, "element", None) not in (
            element.name,
            *(("f16", "bf16") if element == f32 else ()),
        ):
            raise ValueError("matrix atom memory needs a matching rank-two view")
        coordinates, inside = _blocks(view, index, layout)
        value = cl.load(view, coordinates, inside, other=0.0)
        return cl.cast(value, element) if value.element != element.name else value

    def store(self, view, index, value):
        """Store each SIMD group's block of `value` at block `index` of a rank-two view;
        elements past the edge are dropped."""
        from .. import language as cl

        layout = getattr(value, "layout", None)
        if layout is None or getattr(view, "element", None) != value.element:
            raise ValueError("matrix atom memory needs a matching rank-two view")
        coordinates, inside = _blocks(view, index, layout)
        cl.store(view, coordinates, value, inside)

    def _mma(self, unit, relaxed, left, right, accumulator):
        builder = _builder()
        for value, role in ((left, "a"), (right, "b"), (accumulator, "accumulator")):
            element = self._element(role).name
            if (
                getattr(value, "layout", None)
                != self.layout(role, threads=builder.threads)
                or value.element != element
            ):
                raise TypeError(
                    f"mma needs its {role} operand as {element} in this unit's {role} "
                    "layout",
                )
        return builder.mma(unit, self.shape, relaxed, left, right, accumulator)


@dataclass(frozen=True)
class Atom(_Unit):
    """An 8x8x8 Metal SIMD-group multiply with f32 accumulation."""

    operand: Scalar = f16
    accumulator: Scalar = f32

    def __post_init__(self):
        if self.operand not in (f16, bf16, f32) or self.accumulator != f32:
            raise ValueError(
                "matrix atoms require f16/bf16/f32 operands and f32 accumulation",
            )

    @property
    def shape(self):
        """(m, n, k) of one multiply: 8 x 8 x 8."""
        return (8, 8, 8)

    def _registers(self, role):
        return _simdgroup(self._element(role))

    def mma(self, left, right, accumulator):
        """accumulator + left x right, one block per SIMD group, as
        simdgroup_multiply_accumulate."""
        return self._mma("simdgroup", False, left, right, accumulator)


@dataclass(frozen=True)
class TensorAtom(_Unit):
    """An m x n x k multiply on each SIMD group's matrix unit, operands and accumulator
    in registers: Metal 4 matmul2d on cooperative tensors, run on the Neural
    Accelerators of M5. f16 or bf16 operands, f32 accumulation."""

    shape: tuple = (16, 32, 16)
    operand: Scalar = f16
    relaxed: bool = False

    def __post_init__(self):
        if (
            not isinstance(self.shape, tuple)
            or len(self.shape) != 3
            or any(type(n) is not int or n < 8 or n % 8 for n in self.shape)
        ):
            raise ValueError("a tensor atom shape is three positive multiples of eight")
        if self.operand not in (f16, bf16):
            raise ValueError("tensor atoms multiply f16 or bf16 operands")
        if type(self.relaxed) is not bool:
            raise TypeError("relaxed must be a boolean")

    def _registers(self, role):
        return _tensor_unit(self.shape, self.operand, self.relaxed, role)

    def mma(self, left, right, accumulator):
        """accumulator + left x right, one block per SIMD group, as one matmul2d per
        SIMD group."""
        return self._mma("tensor", self.relaxed, left, right, accumulator)


@dataclass(frozen=True)
class CopyAtom:
    """A layout for moving `width` neighbouring elements per thread with one memory
    instruction.

    `layout()` gives each thread runs of `width` consecutive elements, interleaved so
    one SIMD group covers 32 * width contiguous elements per access. The compiler turns
    each run it proves contiguous and aligned into one vector access; runs whose mask is
    not all true fall back to per-element accesses, so tails stay exact.
    """

    dtype: Scalar = f32
    width: int = 4

    def __post_init__(self):
        if self.dtype not in STORAGE:
            raise ValueError(
                "copy atoms move f16, bf16, f32, u32, u8, i8, u16 or i16 elements",
            )
        if type(self.width) is not int or self.width not in (1, 2, 4):
            raise ValueError("copy atom width must be 1, 2 or 4")

    def layout(self, shape, *, threads=32):
        """Runs of `width` per thread, consecutive threads on consecutive runs."""
        if type(threads) is not int or not 32 <= threads <= 1024 or threads % 32:
            raise ValueError(
                "copy layouts require full 32-lane SIMD groups, up to 1024 threads",
            )
        if not isinstance(shape, tuple) or any(
            type(n) is not int or n < 1 for n in shape
        ):
            raise ValueError("tile shape must contain positive integer extents")
        span = threads * self.width
        runs = -(-prod(shape) // span)
        return Layout(
            shape,
            Mode(threads // 32, 32 * self.width),
            Mode(32, self.width),
            Mode((runs, self.width), (span, 1)),
        )


def _pair(value, name):
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(type(n) is not int or n < 1 for n in value)
    ):
        raise ValueError(f"{name} must be two positive integers")
    return value


@dataclass(frozen=True)
class TiledAtom:
    """A grid of SIMD groups, each holding a grid of one unit's blocks, acting as one
    tile-sized operation.

    `groups` arranges SIMD groups over the tile and `fragments` arranges blocks within
    each group, so for an m x n x k unit the tile is m * groups * fragments by n *
    groups * fragments. Fragment grids are nested tuples indexed [row][column].
    """

    atom: Atom | TensorAtom
    groups: tuple = (1, 1)
    fragments: tuple = (1, 1)

    def __post_init__(self):
        if not isinstance(self.atom, (Atom, TensorAtom)):
            raise TypeError("a tiled atom needs an Atom or a TensorAtom")
        _pair(self.groups, "groups")
        _pair(self.fragments, "fragments")
        if self.threads > 1024:
            raise ValueError("a tiled atom can use at most 32 SIMD groups")

    @property
    def shape(self):
        """(rows, columns) of the whole tile."""
        m, n, _ = self.atom.shape
        return (
            m * self.groups[0] * self.fragments[0],
            n * self.groups[1] * self.fragments[1],
        )

    @property
    def threads(self):
        """The threads the tile spans: 32 per SIMD group."""
        return 32 * self.groups[0] * self.groups[1]

    def zeros(self):
        """Zero accumulators: a tuple of rows of this group's blocks, as mma takes and
        returns them."""
        return tuple(
            tuple(self.atom.zeros() for _ in range(self.fragments[1]))
            for _ in range(self.fragments[0])
        )

    def load_a(self, view, row, step):
        """This group's column of left blocks: block rows row.., at reduction step
        `step`."""
        return tuple(
            self.atom.load(view, (row + i, step), role="a")
            for i in range(self.fragments[0])
        )

    def load_b(self, view, column, step):
        """This group's row of right blocks: block columns column.., at reduction step
        `step`."""
        return tuple(
            self.atom.load(view, (step, column + j), role="b")
            for j in range(self.fragments[1])
        )

    def mma(self, left, right, accumulators):
        """Accumulator block (i, j) plus left block i times right block j, for every
        block of this group."""
        return tuple(
            tuple(self.atom.mma(a, b, value) for b, value in zip(right, row))
            for a, row in zip(left, accumulators)
        )

    def store(self, view, group, accumulators):
        """Store one group's blocks contiguously at block index `group` * blocks per
        group."""
        rows, columns = self.fragments
        for i, row in enumerate(accumulators):
            for j, value in enumerate(row):
                self.atom.store(view, group * rows * columns + i * columns + j, value)

    def layout(self):
        """Register ownership of the whole tile, built from the unit's accumulator
        layout."""
        shape, axes = self._grid()
        layout = self.atom.layout(threads=self.threads)
        layout = (
            layout.stack(self.fragments[1])
            .stack(self.fragments[0])
            .reshape(shape)
            .transpose(axes)
        )
        return layout.reshape(self.shape)

    def unpack(self, accumulators):
        """The whole tile as one value in `layout()`."""
        from .. import language as cl

        shape, axes = self._grid()
        tiles = cl.stack(tuple(cl.stack(tuple(row)) for row in accumulators))
        return cl.reshape(cl.transpose(cl.reshape(tiles, shape), axes), self.shape)

    def _grid(self):
        """Fragment and group row axes come before fragment and group column axes."""
        m, n, _ = self.atom.shape
        return (*self.fragments, *self.groups, m, n), (2, 0, 4, 3, 1, 5)


@dataclass(frozen=True)
class TensorMatmul:
    """accumulator + left @ right on the GPU's matrix units, every SIMD group of the
    threadgroup cooperating.

    Metal 4 tensor ops; on M5 they run on the Neural Accelerators. `left` and `right`
    are rank-two views of `operand` that the hardware reads itself: their strides choose
    row- or column-major reads, and their clipped extents bound every access, so partial
    tiles need no padding. The accumulator is an f32 tile in `layout()`, the registers
    the device gives the product. `relaxed` lets the hardware trade precision for speed.
    """

    operand: Scalar = f16
    relaxed: bool = False

    def __post_init__(self):
        if self.operand not in (f16, bf16, f32):
            raise ValueError("tensor matmul operands are f16, bf16 or f32")
        if type(self.relaxed) is not bool:
            raise TypeError("relaxed must be a boolean")

    def layout(self, shape, *, threads):
        """Which elements of an (m, n) product each thread holds."""
        if (
            not isinstance(shape, tuple)
            or len(shape) != 2
            or any(type(n) is not int or n < 1 for n in shape)
        ):
            raise ValueError("a tensor matmul product is two positive extents")
        return _destination(shape, _groups(threads), self.operand, self.relaxed)

    def zeros(self, shape):
        """An f32 accumulator of zeros for an (m, n) product, in `layout`."""
        layout = self.layout(shape, threads=_builder().threads)
        return _builder().full(shape, 0.0, f32, layout=layout)

    def run(self, left, right, accumulator):
        """accumulator + left @ right, for views left (m, k) and right (k, n)."""
        builder = _builder()
        layout = getattr(accumulator, "layout", None)
        if (
            layout is None
            or accumulator.element != "f32"
            or len(layout.shape) != 2
            or layout != self.layout(layout.shape, threads=builder.threads)
        ):
            raise TypeError(
                "the tensor matmul accumulator must be an f32 tile in this "
                "operation's layout()",
            )
        for view in (left, right):
            if getattr(view, "element", None) != self.operand.name:
                raise ValueError(
                    f"tensor matmul operands must be {self.operand.name} views",
                )
        return builder.tensor_matmul(self, left, right, accumulator)
