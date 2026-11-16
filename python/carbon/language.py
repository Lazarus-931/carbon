"""Kernel types and explicit tile operations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Scalar:
    """An element type: its name in the IR, where u32 is i32, and its size in bytes."""

    name: str
    itemsize: int


@dataclass(frozen=True)
class Buffer:
    """A buffer parameter's type; a ranked buffer arrives as a runtime view with `rank`
    axes."""

    element: Scalar
    rank: int | None = None

    def __post_init__(self):
        if self.rank is not None and (type(self.rank) is not int or self.rank < 0):
            raise ValueError("buffer rank must be a nonnegative integer")


f16 = Scalar("f16", 2)
"""IEEE half precision, for storage."""
bf16 = Scalar("bf16", 2)
"""bfloat16, f32's exponent with 8 bits of precision, for storage."""
f32 = Scalar("f32", 4)
"""IEEE single precision, the type of floating-point arithmetic."""
u32 = Scalar("i32", 4)
"""Unsigned 32-bit integers; the IR names them i32."""
u8 = Scalar("ui8", 1)
"""Unsigned 8-bit integers, for storage."""
i8 = Scalar("si8", 1)
"""Signed 8-bit integers, for storage."""
u16 = Scalar("ui16", 2)
"""Unsigned 16-bit integers, for storage."""
i16 = Scalar("si16", 2)
"""Signed 16-bit integers, for storage."""
STORAGE = (f16, bf16, f32, u32, u8, i8, u16, i16)
"""The element types of buffers, views and scratch."""
inf = float("inf")
"""Positive infinity, for f32 constants."""
constexpr = object()
"""Annotates a kernel parameter whose value `compile(constants=...)` supplies."""


def buffer(element, *, rank=None):
    """A buffer parameter's annotation. Every access is bounds-checked; which thread
    writes which element is the kernel's choice, and `carbon.compiler.check_writes`
    finds two threads writing one element.

    Args:
        element: f16, bf16, f32, u32, or the storage integers u8, i8, u16 and i16, which
            kernels cast to u32 or f32 to compute with.
        rank: None for a flat binding; otherwise the axes of a runtime view, whose
            shape, strides and offset arrive with each dispatch.

    Returns:
        The Buffer annotation.
    """
    if not isinstance(element, Scalar) or element not in STORAGE:
        raise TypeError(
            "buffer elements must be f16, bf16, f32, u32, u8, i8, u16 or i16",
        )
    return Buffer(element, rank)


def _builder():
    from .compiler.builder import current_builder

    return current_builder()


def program_id():
    """This threadgroup's index in the one-dimensional grid, a u32 scalar."""
    return _builder().program_id()


def simd_id():
    """This thread's SIMD-group index within the threadgroup, a u32 scalar."""
    return _builder().simd_id()


def arange(width, *, layout=None):
    """The u32 tile 0, 1, ..., width - 1, striped across the kernel's threads unless
    `layout` places it."""
    return _builder().arange(width, layout=layout)


def full(shape, value, dtype=f32, *, layout=None):
    """A tile of `shape` filled with `value`, striped across the kernel's threads unless
    `layout` places it."""
    return _builder().full(shape, value, dtype, layout=layout)


def load(view, indices, mask, *, other):
    """Load a tile, or a uniform scalar from read-only device storage.

    Neighbouring elements one thread holds become one vector access wherever the
    compiler proves them contiguous and aligned; carbon.compiler.CopyAtom gives layouts
    that make them so.

    Args:
        view: a buffer parameter, a view of one, or scratch.
        indices: a u32 tile of flat indices, or a tuple with one coordinate tile per
            view axis; scalars broadcast to the tiles' layout. A scalar index loads one
            scalar, the same in every thread, from storage the kernel never writes.
        mask: a bool tile or scalar; an element whose mask is false, or whose index is
            outside the view, is not read.
        other: the value of the elements not read.

    Returns:
        A tile of the view's element type in the indices' layout, or a scalar.
    """
    return _builder().load(view, indices, mask, other=other)


def store(view, indices, values, mask):
    """Store a tile where `mask` holds. Where a layout repeats an element, only its
    first holder stores it.

    Args:
        view: a buffer parameter, a view of one, or scratch.
        indices: as for cl.load; elements outside the view are not written.
        values: a tile in the indices' layout of the view's element type, or a scalar to
            broadcast.
        mask: a bool tile or scalar.
    """
    return _builder().store(view, indices, values, mask)


def atomic_add(view, indices, values, mask=True):
    """Relaxed device fetch-add on u32 or f32 elements.

    Args:
        view: a device buffer parameter or a view of one.
        indices: a u32 tile of flat indices.
        values: the addends, a tile in the indices' layout or a scalar.
        mask: a bool tile or scalar; nothing is added where it is false.

    Returns:
        The previous values, zero where `mask` is false.
    """
    return _builder().atomic(view, indices, values, mask, "add")


def atomic_min(view, indices, values, mask=True):
    """Relaxed device fetch-min on u32 elements, with arguments as for atomic_add;
    returns the previous values."""
    return _builder().atomic(view, indices, values, mask, "min")


def atomic_max(view, indices, values, mask=True):
    """Relaxed device fetch-max on u32 elements, with arguments as for atomic_add;
    returns the previous values."""
    return _builder().atomic(view, indices, values, mask, "max")


def atomic_store(view, indices, values, mask=True):
    """Relaxed device store of u32 or f32 elements, with arguments as for atomic_add;
    masked memory is unchanged."""
    return _builder().atomic(view, indices, values, mask, "store")


def view(buffer, descriptor):
    """`buffer` seen through a static View.

    Args:
        buffer: a flat buffer parameter, or scratch.
        descriptor: a carbon.compiler.View: element type, shape, strides in elements,
            and an offset and extent in bytes from the start of the allocation.

    Returns:
        The view.
    """
    return _builder().make_view(buffer, descriptor)


def dim(view, axis):
    """The extent of `view` along `axis`: a constant for a static view, a runtime u32
    otherwise. For a partition it is the tile's extent, even where the tile is clipped
    at an edge; for a subview it is the clipped extent."""
    return _builder().dim(view, axis)


def partition(view, shape, index):
    """Tile `index` of `view` cut into tiles of `shape`, numbered in row-major order and
    clipped to the view.

    Args:
        view: a device view.
        shape: the tile's extents, one per axis.
        index: the tile's number, a u32 the same in every thread of the threadgroup.

    Returns:
        A runtime view of the tile; accesses past the view's edge are masked.
    """
    return _builder().partition(view, shape, index)


def subview(view, offsets, shape):
    """A region of `view`, clipped to the view's bounds, with the view's strides.

    Args:
        view: a flat device buffer or a device view.
        offsets: the region's first coordinate, one u32 per axis, the same in every
            thread of the threadgroup.
        shape: the region's extents, one u32 per axis, the same in every thread of the
            threadgroup.

    Returns:
        A runtime view whose extents are clipped where the region crosses the view's
        edge.
    """
    return _builder().subview(view, offsets, shape)


def scratch(shape, dtype=f32, *, strides=None):
    """Allocate kernel-lifetime threadgroup storage. Allocations share bytes only when a
    barrier separates their lifetimes.

    Args:
        shape: positive compile-time extents.
        dtype: f16, bf16, f32 or u32.
        strides: element strides, one per axis; row-major by default.

    Returns:
        A threadgroup view of the allocation.
    """
    return _builder().scratch(shape, dtype, strides=strides)


def barrier():
    """Wait for every thread of the threadgroup and make their scratch writes visible;
    device memory is not ordered."""
    return _builder().barrier()


def double_buffer(count, *, buffers, state, produce, consume, args=()):
    """Consume `count` tiles using two distinct sets of scratch buffers.

    `produce(index, buffers, args)` fills a stage; `consume(index, buffers, state,
    args)` returns the next scalar/tile state, which may be a tuple. The next tile is
    produced before the current tile is consumed. Threadgroup barriers publish writes
    and complete readers before storage is reused. Callbacks execute exactly once per
    tile, including odd counts; a zero count returns the initial state without calling
    them. The count and surrounding control flow must be threadgroup-uniform. The
    schedule is synchronous and does not imply asynchronous Metal copies. Callbacks with
    runtime control flow must use `@inline`.
    """
    from .compiler.pipeline import double_buffer as schedule

    return schedule(
        count,
        buffers=buffers,
        state=state,
        produce=produce,
        consume=consume,
        args=args,
    )


def convert_layout(value, layout, *, scratch=None, free=False):
    """Move a tile to `layout`.

    Args:
        value: a tile.
        layout: the target Layout, of the tile's shape.
        scratch: a threadgroup view of the tile's shape and element type. The conversion
            then stores the tile, waits at a barrier, loads it in `layout` and waits
            again; without scratch, values move between lanes by shuffles, which cannot
            cross SIMD groups.
        free: require every element to stay in the thread that holds it, so the
            conversion only renames registers; staging fails otherwise.

    Returns:
        The tile in `layout`.
    """
    return _builder().convert_layout(value, layout, scratch=scratch, free=free)


def sum(value, axis, *, layout=None, scratch=None, order="fast"):
    """Sum along `axis`. `order="fast"` lets f32 sums use the hardware's order, so their
    last bits can depend on the layout and target; `order="pairwise"` adds logical
    neighbours in a fixed tree, the same bits everywhere.

    Args:
        value: an f32 or u32 tile.
        axis: the axis to sum, a compile-time index.
        layout: the result's layout; striped over the input's threads by default.
        scratch: a workspace of the input's shape and type, needed when values cross
            SIMD groups.
        order: "fast" or "pairwise".

    Returns:
        A tile without `axis`.
    """
    return _builder().reduce(
        value,
        axis,
        "sum",
        layout=layout,
        scratch=scratch,
        order=order,
    )


def max(value, axis, *, layout=None, scratch=None):
    """Maximum along `axis`, in a fixed tree; floating NaNs propagate and +0 wins over
    -0. `value`, `axis`, `layout` and `scratch` are as for cl.sum, as is the result's
    shape."""
    return _builder().reduce(value, axis, "max", layout=layout, scratch=scratch)


def cumsum(value, axis, *, scratch=None, order="fast"):
    """Inclusive prefix sums along `axis`. `order="fast"` lets each thread add its own
    run of the axis serially and SIMD groups use the hardware prefix, so f32 bits can
    depend on the layout and target; `order="pairwise"` adds, at each distance 2^k, the
    element 2^k before, the same bits for every layout.

    Args:
        value: an f32 or u32 tile.
        axis: the axis to scan, a compile-time index.
        scratch: a workspace of the input's shape and type, needed when partners sit in
            other SIMD groups.
        order: "fast" or "pairwise".

    Returns:
        The running sums, in the input's layout.
    """
    return _builder().scan(value, axis, "sum", scratch=scratch, order=order)


def cummax(value, axis, *, scratch=None, order="fast"):
    """Inclusive running maximum along `axis`, with cl.maximum's NaN and zero rules;
    arguments as for cl.cumsum."""
    return _builder().scan(value, axis, "max", scratch=scratch, order=order)


def maximum(left, right):
    """Elementwise maximum; for f32, NaN if either side is NaN, and +0 above -0."""
    return _builder().extremum("maximum", left, right)


def minimum(left, right):
    """Elementwise minimum; for f32, NaN if either side is NaN, and -0 below +0."""
    return _builder().extremum("minimum", left, right)


def exp(value):
    """e to the power of each f32 element."""
    return _builder().math("exp", value)


def exp2(value):
    """2 to the power of each f32 element."""
    return _builder().math("exp2", value)


def log(value):
    """Natural logarithm of f32 values, subject to target subnormal behavior."""
    return _builder().math("log", value)


def log2(value):
    """Base-2 logarithm of f32 values, subject to target subnormal behavior."""
    return _builder().math("log2", value)


def reciprocal(value):
    """1 / x for f32 values."""
    return _builder().math("reciprocal", value)


def rsqrt(value):
    """1 / sqrt(x) for f32 values."""
    return _builder().math("rsqrt", value)


def sqrt(value):
    """Square root of f32 values."""
    return _builder().math("sqrt", value)


def sin(value):
    """Sine of f32 values in radians."""
    return _builder().math("sin", value)


def cos(value):
    """Cosine of f32 values in radians."""
    return _builder().math("cos", value)


def tan(value):
    """Tangent of f32 values in radians."""
    return _builder().math("tan", value)


def tanh(value):
    """Hyperbolic tangent of f32 values."""
    return _builder().math("tanh", value)


def abs(value):
    """Absolute value of f32 values."""
    return _builder().math("abs", value)


def floor(value):
    """f32 values rounded down to integers."""
    return _builder().math("floor", value)


def ceil(value):
    """f32 values rounded up to integers."""
    return _builder().math("ceil", value)


def fma(a, b, c):
    """a * b + c in f32 with one rounding."""
    return _builder().fma(a, b, c)


def erf(value):
    """The error function: Norbert Juffa's minimax polynomials in fma, both evaluated
    and chosen at |x| = 0.927734375; above it erf(x) = copysign(1 - exp(r), x)."""
    t = abs(value)
    s = value * value
    r = fma(
        fma(-1.72853470e-5, t, 3.83197126e-4),
        s,
        fma(-3.88396438e-3, t, 2.42546219e-2),
    )
    r = fma(
        fma(fma(fma(r, t, -1.06777877e-1), t, -6.34846687e-1), t, -1.28717512e-1),
        t,
        -t,
    )
    large = bitcast(
        bitcast(1.0 - exp(r), u32) | (bitcast(value, u32) & 0x80000000),
        f32,
    )
    q = fma(
        fma(
            fma(fma(-5.96761703e-4, s, 4.99119423e-3), s, -2.67681349e-2),
            s,
            1.12819925e-1,
        ),
        s,
        -3.76125336e-1,
    )
    small = fma(fma(q, s, 1.28379166e-1), value, value)
    return select(t > 0.927734375, large, small)


def sigmoid(value):
    """1 / (1 + exp(-x)); exp overflows to infinity for large -x, which gives exactly
    0."""
    return reciprocal(1.0 + exp(-value))


def select(condition, left, right):
    """Select values, not control flow; both operands are evaluated.

    Args:
        condition: a bool tile or scalar.
        left, right: values of one type, taken where `condition` is true and false;
            scalars broadcast to the other arguments' layout.

    Returns:
        The selected values.
    """
    return _builder().select(condition, left, right)


def broadcast_to(value, shape, *, axes=None, layout=None, scratch=None):
    """Map the axes of `value` into a tile of `shape`; extents of 1 repeat.

    Args:
        value: a scalar or tile.
        shape: the output shape.
        axes: the output axis of each input axis, all distinct; the trailing axes by
            default.
        layout: the output layout; striped by default.
        scratch: a workspace of the input's shape, needed when elements move between
            SIMD groups; predicates use u32 storage.

    Returns:
        A tile of `shape`.
    """
    return _builder().broadcast_to(
        value,
        shape,
        axes=axes,
        layout=layout,
        scratch=scratch,
    )


def item(value, *, scratch=None):
    """A one-element tile as a scalar every thread holds, without host readback.

    Args:
        value: a one-element tile.
        scratch: a nonempty workspace of the element's storage type, needed when the
            tile spans SIMD groups.

    Returns:
        The element, the same in every thread of the threadgroup.
    """
    return _builder().item(value, scratch=scratch)


def cast(value, dtype):
    """Convert elements between f16, bf16 and f32, from u32 or an 8- or 16-bit integer
    to f32, or between u32 and an 8- or 16-bit integer, which extends by its sign or
    truncates; each value stays in its thread."""
    return _builder().cast(value, dtype)


def bitcast(value, dtype):
    """Reinterpret f32/u32 bits without changing shape, ownership, or payloads."""
    return _builder().bitcast(value, dtype)


def reshape(value, shape):
    """The same elements in row-major order under a new `shape`; each value stays in its
    thread."""
    return _builder().reshape(value, shape)


def stack(values, axis=0):
    """Stack equal tiles along a new axis without communicating between threads."""
    return _builder().stack(values, axis)


def transpose(value, axes=None):
    """Permute tile axes while retaining each value in its current thread."""
    return _builder().transpose(value, axes)


def inline(function):
    """Make `function` an inline helper: each call stages its body in place, runtime
    control flow included.

    Returns:
        The helper, callable from kernels and other helpers.
    """
    from .compiler.frontend import Inline

    return Inline(function)


def kernel(function):
    """Make `function` a kernel.

    Annotate its parameters with cl.buffer(...) for bindings, a tuple of cl.buffer(...)
    for a group of bindings that the body sees as a tuple, cl.u32 or cl.f32 for runtime
    scalars after every binding, and cl.constexpr for values given at compile time. The
    body is staged from its source: compile-time values fold in Python, and `for ... in
    range(...)` and `if` on runtime values become structured IR.

    Returns:
        A Kernel; `compile(constants=..., threads=...)` gives the Artifact to dispatch.
    """
    from .compiler import Kernel

    return Kernel(function)
