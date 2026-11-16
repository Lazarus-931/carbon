from functools import cache

from carbon import language as cl
from carbon.compiler import CopyAtom, Layout, TiledAtom, View
from carbon.kernels.common import operand_buffers
from carbon.kernels.gemm.epilogue import paired


def dimensions(left, right, output):
    """(batches, rows, columns, depth) of a product of rank-two or batched rank-three
    Views that agree."""
    views = (left, right, output)
    if not all(isinstance(view, View) for view in views):
        raise TypeError("matmul requires View descriptors")
    rank = len(left.shape)
    if rank not in (2, 3) or any(len(view.shape) != rank for view in views):
        raise ValueError("matmul views must all have rank two or all have rank three")
    a, b, c = [view.shape if rank == 3 else (1, *view.shape) for view in views]
    batches, rows, depth = a
    columns = b[2]
    if b != (batches, depth, columns) or c != (batches, rows, columns):
        raise ValueError("incompatible matmul dimensions")
    if any(n is not None and n >= 2**32 for view in views for n in view.shape) or any(
        not view.runtime and view.size >= 2**32 for view in views
    ):
        raise ValueError("matrix element counts must fit u32")
    return batches, rows, columns, depth


def tile_shape(groups, fragments):
    if any(
        len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape)
        for shape in (groups, fragments)
    ):
        raise ValueError("group and fragment shapes need two positive integers")
    return tuple(8 * group * fragment for group, fragment in zip(groups, fragments))


def configuration(left, right, output, schedule, groups, fragments):
    batches, rows, columns, depth = dimensions(left, right, output)
    shape = tile_shape(groups, fragments)
    if schedule.shape != shape or schedule.threads != 32 * groups[0] * groups[1]:
        raise ValueError("the output layout must match the group and fragment shapes")
    return batches, rows, columns, depth, *shape


def matrix_view(view):
    if len(view.shape) == 2:
        return view
    batches, rows, columns = view.shape
    try:
        return view.reshape((batches * rows, columns))
    except ValueError:
        return view


def matrix_tile(view, shape):
    return shape if len(view.shape) == 2 else (1, *shape)


def direct_matrix_loads(*views):
    return all(
        all(n is None or n % 8 == 0 for n in view.shape)
        and (
            (not view.runtime and not view.size)
            or (view.strides[1] == 1 and view.strides[0] >= 8)
            or (view.strides[0] == 1 and view.strides[1] >= 8)
        )
        for view in views
    )


def operand_staging(left, right, columns):
    """Which operands go through scratch, as (left, right, either); direct matrix loads
    read the others."""

    def direct(view):
        flat = matrix_view(view)
        return (
            len(flat.shape) == 2
            and all(n is None or n % 8 == 0 for n in view.shape[-2:])
            and direct_matrix_loads(flat)
        )

    stage_left = not direct(left)
    stage_right = not (direct(right) and right.shape[-1] % columns == 0)
    return stage_left, stage_right, stage_left or stage_right


def copy_width(view, shape):
    """The widest copy whose runs stay contiguous and aligned in every partition tile of
    `view`."""
    axes = list(zip(view.shape, matrix_tile(view, shape), view.strides))
    *outer, (_, inner_size, inner_stride) = [axis for axis in axes if axis[1] > 1]
    for width in (4, 2):
        if (
            view.offset_bytes // view.itemsize % width == 0
            and inner_stride == 1
            and inner_size % width == 0
            and all(stride % width == 0 for _, _, stride in outer)
            and all(
                size * stride % width == 0
                for extent, size, stride in axes
                if extent is None or extent > size
            )
        ):
            return width
    return 1


def staging_layout(shape, schedule, width, dtype):
    if width == 1:
        return Layout.striped(shape, threads=schedule.threads)
    return CopyAtom(dtype, width).layout(shape, threads=schedule.threads)


@cl.inline
def reduction_steps(block, width, total):
    """The 8-deep steps in reduction block `block`: `width`, or what remains for the
    last block."""
    count = width
    if total % width:
        if block == total // width:
            count = total % width
    return count


@cl.inline
def split_range(partition, count, partitions):
    """The [first, last) range of `count` items for `partition` of `partitions`; earlier
    ones take the remainder."""
    first = partition * (count // partitions)
    size = count // partitions
    if count % partitions:
        if partition < count % partitions:
            first = first + partition
            size = size + 1
        else:
            first = first + count % partitions
    return first, first + size


@cache
def make_tiled_matmul(
    dtype=cl.f32,
    *,
    block_k=8,
    split_k=1,
    glu=None,
    epilogue=None,
    runtime_rows=False,
):
    """Output tiles on the 8 x 8 matrix units in a caller-selected two-dimensional
    schedule. Operands whose layout allows it load straight into fragments; the others
    stage through scratch. `epilogue` maps each output tile before the store, and its
    operands bind after `output`; a split-K product's epilogue goes to
    make_sum_partials."""
    if dtype not in (cl.f16, cl.bf16, cl.f32):
        raise ValueError("matmul storage must be f16, bf16, or f32")
    if type(block_k) is not int or block_k <= 0 or block_k % 8:
        raise ValueError("reduction block size must be a positive multiple of 8")
    if type(split_k) is not int or not 1 <= split_k < 2**32:
        raise ValueError("split_k must be a positive u32")
    if split_k > 1 and epilogue is not None:
        raise ValueError("a split-K product's epilogue belongs to make_sum_partials")
    if split_k > 1 and runtime_rows:
        raise ValueError("a split-K product fixes its rows")
    if split_k > 1 and glu is not None:
        raise ValueError("a split-K product's partials do not pair")
    rank = 2 if runtime_rows else None
    output_type = cl.f32 if split_k > 1 else dtype
    epilogue_buffers = operand_buffers(epilogue)

    def configure(left, right, output, schedule, groups, fragments):
        if runtime_rows and len(left.shape) != 2:
            raise ValueError("runtime rows need rank-two matrices")
        if glu is not None:
            if len(left.shape) != 2 or 2 * output.shape[1] != right.shape[1]:
                raise ValueError(
                    "a paired product is rank two, its output half as "
                    "wide as the right operand",
                )
            output = View(
                output.element,
                (output.shape[0], right.shape[1]),
                (right.shape[1], 1),
                None if output.runtime else 2 * output.extent_bytes,
            )
        if split_k > 1:
            if len(left.shape) != 2:
                raise ValueError("split-K currently requires rank-two matrices")
            if (
                len(output.shape) != 3
                or output.shape[0] != split_k
                or output.element != "f32"
            ):
                raise ValueError(
                    "split-K output must be an f32 View with shape (split_k, M, N)",
                )
            if output.size >= 2**32:
                raise ValueError("partial element count must fit u32")
            output = output[0]
        return configuration(left, right, output, schedule, groups, fragments)

    @cl.kernel
    def matmul(
        left: cl.buffer(dtype, rank=rank),
        right: cl.buffer(dtype),
        output: cl.buffer(output_type, rank=rank),
        operands: epilogue_buffers,
        LEFT: cl.constexpr,
        RIGHT: cl.constexpr,
        OUTPUT: cl.constexpr,
        SCHEDULE: cl.constexpr,
        GROUPS: cl.constexpr,
        FRAGMENTS: cl.constexpr,
        ATOM: cl.constexpr,
    ):
        BATCHES, M, N, K, BLOCK_M, BLOCK_N = configure(
            LEFT,
            RIGHT,
            OUTPUT,
            SCHEDULE,
            GROUPS,
            FRAGMENTS,
        )
        a_view = left if runtime_rows else cl.view(left, LEFT)
        b_view = cl.view(right, RIGHT)
        c_view = output if runtime_rows else cl.view(output, OUTPUT)
        if runtime_rows:
            M = cl.dim(a_view, 0)
        if runtime_rows or BATCHES * M * N > 0:
            GROUP_M, GROUP_N = GROUPS
            FRAGMENT_M, FRAGMENT_N = FRAGMENTS
            BLOCK_K, DEPTH_TILES, SPLITS = block_k, block_k // 8, split_k
            N_BLOCKS = (N + BLOCK_N - 1) // BLOCK_N
            M_BLOCKS = (M + BLOCK_M - 1) // BLOCK_M
            K_BLOCKS = (K + BLOCK_K - 1) // BLOCK_K
            K_TILES = (K + 7) // 8
            if SPLITS > 1:
                OUTPUT_BLOCKS = M_BLOCKS * N_BLOCKS
                partition = cl.program_id() // OUTPUT_BLOCKS
                block_row = (cl.program_id() % OUTPUT_BLOCKS) // N_BLOCKS
                batch = 0
                first, last = split_range(partition, K_BLOCKS, SPLITS)
            else:
                if BATCHES > 1:
                    batch = cl.program_id() // (M_BLOCKS * N_BLOCKS)
                    block_row = (cl.program_id() // N_BLOCKS) % M_BLOCKS
                else:
                    batch = 0
                    block_row = cl.program_id() // N_BLOCKS
                first, last = 0, K_BLOCKS
            block_column = cl.program_id() % N_BLOCKS
            group_row = cl.simd_id() // GROUP_N
            group_column = cl.simd_id() % GROUP_N
            tiled = TiledAtom(ATOM, GROUPS, FRAGMENTS)
            accumulators = tiled.zeros()
            STAGE_A, STAGE_B, SYNCHRONIZE = operand_staging(LEFT, RIGHT, BLOCK_N)
            if STAGE_A:
                a_storage = cl.scratch((BLOCK_M, BLOCK_K), ATOM.operand)
                A_WIDTH = copy_width(LEFT, (BLOCK_M, BLOCK_K))
                a_index = cl.arange(
                    BLOCK_M * BLOCK_K,
                    layout=staging_layout(
                        (BLOCK_M, BLOCK_K),
                        SCHEDULE,
                        A_WIDTH,
                        ATOM.operand,
                    ),
                )
            else:
                a_matrix = a_view if runtime_rows else cl.view(left, matrix_view(LEFT))
                row = batch * (M // 8) + (block_row * GROUP_M + group_row) * FRAGMENT_M
            if STAGE_B:
                b_storage = cl.scratch((BLOCK_K, BLOCK_N), ATOM.operand)
                B_WIDTH = copy_width(RIGHT, (BLOCK_K, BLOCK_N))
                b_index = cl.arange(
                    BLOCK_K * BLOCK_N,
                    layout=staging_layout(
                        (BLOCK_K, BLOCK_N),
                        SCHEDULE,
                        B_WIDTH,
                        ATOM.operand,
                    ),
                )
            else:
                b_matrix = cl.view(right, matrix_view(RIGHT))
                column = (block_column * GROUP_N + group_column) * FRAGMENT_N
            for block in range(first, last):
                if STAGE_A:
                    a_tile = cl.partition(
                        a_view,
                        matrix_tile(LEFT, (BLOCK_M, BLOCK_K)),
                        (batch * M_BLOCKS + block_row) * K_BLOCKS + block,
                    )
                    cl.store(
                        a_storage,
                        a_index,
                        cl.load(a_tile, a_index, True, other=0.0),
                        True,
                    )
                if STAGE_B:
                    b_tile = cl.partition(
                        b_view,
                        matrix_tile(RIGHT, (BLOCK_K, BLOCK_N)),
                        (batch * K_BLOCKS + block) * N_BLOCKS + block_column,
                    )
                    cl.store(
                        b_storage,
                        b_index,
                        cl.load(b_tile, b_index, True, other=0.0),
                        True,
                    )
                if SYNCHRONIZE:
                    cl.barrier()
                steps = reduction_steps(block, DEPTH_TILES, K_TILES)
                for inner in range(steps):
                    step = block * DEPTH_TILES + inner
                    if STAGE_A:
                        a = tiled.load_a(a_storage, group_row * FRAGMENT_M, inner)
                    else:
                        a = tiled.load_a(a_matrix, row, step)
                    if STAGE_B:
                        b = tiled.load_b(b_storage, group_column * FRAGMENT_N, inner)
                    else:
                        b = tiled.load_b(b_matrix, column, batch * K_TILES + step)
                    accumulators = tiled.mma(a, b, accumulators)
                if SYNCHRONIZE:
                    cl.barrier()
            result = tiled.unpack(accumulators)
            index = cl.arange(SCHEDULE.size, layout=tiled.layout())
            WIDTH = BLOCK_N if glu is None else BLOCK_N // 2
            if glu is not None:
                result = paired(glu, cl.reshape(result, SCHEDULE.shape))
                index = cl.arange(BLOCK_M * WIDTH, layout=result.layout)
            if SPLITS > 1:
                destination = cl.partition(
                    c_view,
                    (1, BLOCK_M, BLOCK_N),
                    cl.program_id(),
                )
            else:
                destination = cl.partition(
                    c_view,
                    matrix_tile(OUTPUT, (BLOCK_M, WIDTH)),
                    cl.program_id(),
                )
            if epilogue is not None:
                out_row = batch * M + block_row * BLOCK_M + index // WIDTH
                out_column = block_column * WIDTH + index % WIDTH
                result = epilogue(result, out_row, out_column, operands)
            cl.store(destination, index, cl.cast(result, output_type), True)

    return matmul


@cache
def make_sum_partials(dtype=cl.f32, *, epilogue=None):
    """Sum f32 partial matrices in partition order and round once to the output type.
    `epilogue` maps the sums before the store, and its operands bind after `output`."""
    if dtype not in (cl.f16, cl.bf16, cl.f32):
        raise ValueError("output storage must be f16, bf16, or f32")
    epilogue_buffers = operand_buffers(epilogue)

    def shape(partials, output):
        if (
            len(partials.shape) != 3
            or partials.element != "f32"
            or partials.shape[1:] != output.shape
            or output.element != dtype.name
        ):
            raise ValueError(
                "partials must have shape (partitions, M, N) matching the output",
            )
        if partials.size >= 2**32 or any(n >= 2**32 for n in partials.shape):
            raise ValueError("partial dimensions and element count must fit u32")
        return partials.shape[0], output.size, max(output.shape[-1], 1)

    @cl.kernel
    def sum_partials(
        partials: cl.buffer(cl.f32),
        output: cl.buffer(dtype),
        operands: epilogue_buffers,
        PARTIALS: cl.constexpr,
        OUTPUT: cl.constexpr,
        SCHEDULE: cl.constexpr,
    ):
        PARTITIONS, SIZE, COLUMNS = shape(PARTIALS, OUTPUT)
        source = cl.view(partials, PARTIALS)
        destination = cl.view(output, OUTPUT)
        index = cl.program_id() * SCHEDULE.size + cl.arange(
            SCHEDULE.size,
            layout=SCHEDULE,
        )
        active = index < SIZE
        total = cl.full(SCHEDULE.shape, 0.0, cl.f32, layout=SCHEDULE)
        for partition in range(PARTITIONS):
            total = total + cl.load(source, partition * SIZE + index, active, other=0.0)
        if epilogue is not None:
            total = epilogue(total, index // COLUMNS, index % COLUMNS, operands)
        cl.store(destination, index, cl.cast(total, dtype), active)

    return sum_partials
