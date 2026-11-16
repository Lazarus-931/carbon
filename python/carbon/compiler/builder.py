import math
import operator
import struct
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import cached_property

from ..language import STORAGE, Buffer, Scalar, bf16, f16, f32, i8, i16, u8, u16, u32
from .layout import Layout, Tile
from .native import THREADGROUP_MEMORY, load_compiler
from .shape import reshape_shape, transpose_axes
from .view import View

_active = ContextVar("carbon_builder", default=None)
_comparisons = ("lt", "le", "gt", "ge", "eq", "ne")
native = load_compiler()
INTEGERS = {dtype.name: dtype for dtype in (u8, i8, u16, i16)}


def value_type(
    element,
    width=0,
    *,
    view=False,
    layout=None,
    descriptor=None,
    view_shape=None,
):
    if descriptor is not None:
        return native.view_type(
            descriptor.element,
            descriptor.shape,
            descriptor.strides,
            descriptor.offset_bytes,
            descriptor.extent_bytes,
            descriptor.address_space,
        )
    if view_shape is not None:
        return native.view_type(element, view_shape, (), -1, -1, "device")
    if view:
        return native.buffer_type(element)
    if width:
        return native.tile_type(element, layout)
    return native.scalar_type(element)


def current_builder():
    builder = _active.get()
    if builder is None:
        raise RuntimeError("language operations require an active kernel compilation")
    return builder


@dataclass(frozen=True, eq=False)
class Value:
    handle: object
    owner: object = field(default=None, repr=False)

    def __bool__(self):
        raise TypeError("a runtime value cannot control Python execution")

    def __index__(self):
        raise TypeError(
            "a runtime value cannot be a Python integer; only for statements loop "
            "over runtime counts",
        )

    def __add__(self, other):
        return current_builder().binary("add", self, other)

    def __radd__(self, other):
        return current_builder().binary("add", other, self)

    def __sub__(self, other):
        return current_builder().binary("sub", self, other)

    def __rsub__(self, other):
        return current_builder().binary("sub", other, self)

    def __mul__(self, other):
        return current_builder().binary("mul", self, other)

    def __rmul__(self, other):
        return current_builder().binary("mul", other, self)

    def __floordiv__(self, other):
        return current_builder().binary("floordiv", self, other)

    def __rfloordiv__(self, other):
        return current_builder().binary("floordiv", other, self)

    def __mod__(self, other):
        return current_builder().binary("mod", self, other)

    def __rmod__(self, other):
        return current_builder().binary("mod", other, self)

    def __truediv__(self, other):
        return current_builder().binary("truediv", self, other)

    def __rtruediv__(self, other):
        return current_builder().binary("truediv", other, self)

    def __lt__(self, other):
        return current_builder().binary("lt", self, other)

    def __le__(self, other):
        return current_builder().binary("le", self, other)

    def __gt__(self, other):
        return current_builder().binary("gt", self, other)

    def __ge__(self, other):
        return current_builder().binary("ge", self, other)

    def __eq__(self, other):
        return current_builder().binary("eq", self, other)

    def __ne__(self, other):
        return current_builder().binary("ne", self, other)

    def __neg__(self):
        return current_builder().negate(self)

    def __and__(self, other):
        return current_builder().bitwise("and", self, other)

    def __rand__(self, other):
        return current_builder().bitwise("and", other, self)

    def __or__(self, other):
        return current_builder().bitwise("or", self, other)

    def __ror__(self, other):
        return current_builder().bitwise("or", other, self)

    def __xor__(self, other):
        return current_builder().bitwise("xor", self, other)

    def __rxor__(self, other):
        return current_builder().bitwise("xor", other, self)

    def __lshift__(self, other):
        return current_builder().bitwise("shl", self, other)

    def __rlshift__(self, other):
        return current_builder().bitwise("shl", other, self)

    def __rshift__(self, other):
        return current_builder().bitwise("shr", self, other)

    def __rrshift__(self, other):
        return current_builder().bitwise("shr", other, self)

    def __invert__(self):
        return current_builder().invert(self)

    @cached_property
    def _described(self):
        return self.handle.type.describe()

    @property
    def type(self):
        return self.handle.type

    @property
    def element(self):
        """The element type's IR name."""
        return self._described[1]

    @property
    def layout(self):
        """A tile's Layout; None for scalars and memory."""
        return self._described[2]

    @property
    def width(self):
        """A tile's number of elements; 0 for scalars and memory."""
        return 0 if self.layout is None else self.layout.size

    @property
    def view(self):
        return self._described[0] in ("buffer", "view")

    @cached_property
    def descriptor(self):
        """A static view's View; None for anything else."""
        kind, element, _, shape, strides, offset, extent, space = self._described
        if kind != "view" or extent < 0:
            return None
        return View(element, tuple(shape), tuple(strides), extent, offset, space)

    @property
    def view_shape(self):
        """A runtime view's shape, -1 where the dispatch supplies the extent; None for
        anything else."""
        kind, _, _, shape, _, _, extent, _ = self._described
        return tuple(shape) if kind == "view" and extent < 0 else None

    @property
    def tile(self):
        return Tile(self.element, self.layout) if self.width else None


@dataclass(frozen=True)
class StagedKernel:
    fingerprint: str
    source: str
    parameters: tuple[tuple[str, Scalar | Buffer], ...]
    threads: int
    tiles: tuple[tuple[str, Tile], ...] = ()
    views: tuple[tuple[str, View], ...] = ()
    staging: object = field(default=None, repr=False, compare=False)
    scalar_ranges: tuple[tuple[str, int, int], ...] = ()

    @property
    def ir(self):
        return self.staging.ir

    def verify(self, threadgroup_memory=THREADGROUP_MEMORY):
        return native.canonical(self.staging, threadgroup_memory)


class Builder:
    def __init__(self, threads, max_grid=None, indirect=False):
        if type(threads) is not int or not 32 <= threads <= 1024 or threads % 32:
            raise ValueError(
                "kernels require full 32-lane SIMD groups, up to 1024 threads",
            )
        self.threads = threads
        if max_grid is not None and (
            type(max_grid) is not int or not 1 <= max_grid < 2**32
        ):
            raise ValueError("max_grid must be a positive u32 threadgroup count")
        self.max_grid = max_grid
        self.native = native.Staging(threads, max_grid, bool(indirect))
        self.parameters = []
        self.specialized = []
        self.scalar_ranges = []
        self.tile_values = []
        self.location = None
        self.owner = object()

    @contextmanager
    def activate(self):
        token = _active.set(self)
        try:
            yield self
        finally:
            _active.reset(token)

    def check(self, value):
        if isinstance(value, Value) and value.owner is not self.owner:
            raise ValueError("values cannot cross kernel compilations")

    def bind(self, name, kind):
        element = kind.element.name if isinstance(kind, Buffer) else kind.name
        shape = (
            (-1,) * kind.rank
            if isinstance(kind, Buffer) and kind.rank is not None
            else None
        )
        view = isinstance(kind, Buffer)
        handle = self.native.argument(value_type(element, view=view, view_shape=shape))
        self.parameters.append((name, kind))
        return self.wrap(handle)

    def specialize(self, name, kind, view):
        """A ranked buffer parameter fixed to `view`: a flat binding seen through that
        static view, or a runtime view with constant strides when some extent of `view`
        stays a runtime value."""
        if not isinstance(kind, Buffer) or kind.rank is None:
            raise TypeError(
                f"only ranked buffer parameters can be specialized; {name} is not one",
            )
        if (
            not isinstance(view, View)
            or view.element != kind.element.name
            or len(view.shape) != kind.rank
        ):
            raise ValueError(
                f"the view for {name} must be a View of matching element type and rank",
            )
        self.specialized.append((name, view))
        if view.runtime:
            shape = tuple(-1 if extent is None else extent for extent in view.shape)
            handle = self.native.argument(
                native.view_type(
                    view.element,
                    shape,
                    view.strides,
                    view.offset_bytes,
                    -1,
                    "device",
                ),
            )
            self.parameters.append((name, kind))
            return self.wrap(handle)
        return self.make_view(self.bind(name, Buffer(kind.element)), view)

    def constrain(self, name, kind, value, bounds):
        """Give u32 parameter `name` the inclusive range `bounds`, checked at
        dispatch."""
        if kind != u32:
            raise TypeError(f"scalar ranges require u32 parameters; {name} is not one")
        if (
            type(bounds) is not tuple
            or len(bounds) != 2
            or any(type(bound) is not int for bound in bounds)
        ):
            raise ValueError(
                f"the range for {name} must be an inclusive (minimum, maximum) u32 "
                "pair",
            )
        self.native.set_range(value.handle, *bounds)
        self.scalar_ranges.append((name, *bounds))

    def require_bytes(self, name, kind, value, minimum):
        """Give buffer parameter `name` a minimum size of `minimum` bytes, checked at
        dispatch."""
        if not isinstance(kind, Buffer):
            raise TypeError(
                f"minimum bytes require buffer parameters; {name} is not one",
            )
        if type(minimum) is not int:
            raise ValueError(
                f"minimum bytes for {name} must describe whole u32-indexed elements",
            )
        self.native.set_min_bytes(value.handle, minimum)

    def wrap(self, handle):
        value = Value(handle, self.owner)
        if value.width:
            if value.layout.threads != self.threads or value.layout.lanes.size != 32:
                raise ValueError("tile ownership must match the kernel launch geometry")
            self.tile_values.append(value)
        return value

    def emit(self, name, operands, attributes=None, *, result=None, at_entry=False):
        """Omit `result` to infer the type natively; supply it only when the operation
        chooses its output type."""
        (handle,) = self.native.create(
            name,
            [value.handle for value in operands],
            None if result is None else [result],
            attributes or {},
            self.location,
            at_entry,
        )
        return self.wrap(handle)

    def effect(self, name, operands, attributes=None):
        self.native.create(
            name,
            [value.handle for value in operands],
            [],
            attributes or {},
            self.location,
        )

    def literal(self, value, element=None):
        """A constant: a bool is i1, a float f32 and an int u32 unless `element` says
        otherwise."""
        if type(value) is bool and element in (None, "i1"):
            return self.wrap(
                self.native.constant(value, native.scalar_type("i1"), self.location),
            )
        if type(value) is bool or not isinstance(value, (int, float)):
            raise ValueError("only numeric constants are supported")
        element = element or ("f32" if isinstance(value, float) else "i32")
        if element in ("f16", "bf16"):
            return self.cast(
                self.literal(value, "f32"),
                f16 if element == "f16" else bf16,
            )
        if element in INTEGERS:
            bits = 8 * INTEGERS[element].itemsize
            low = -(1 << (bits - 1)) if element.startswith("si") else 0
            if type(value) is not int or not low <= value < low + (1 << bits):
                raise ValueError(
                    f"integer constants must lie in {low} to {low + (1 << bits) - 1}",
                )
            return self.cast(self.literal(value % 2**32, "i32"), INTEGERS[element])
        if element == "i32":
            if type(value) is not int or not 0 <= value < 2**32:
                raise ValueError("integer constants must fit u32")
            handle = self.native.constant(
                value,
                native.scalar_type("i32"),
                self.location,
            )
        elif element == "f32":
            handle = (
                self.native.constant(
                    float(value),
                    native.scalar_type("f32"),
                    self.location,
                )
                if math.isfinite(value)
                else self.native.bits(
                    struct.unpack("<I", struct.pack("<f", value))[0],
                    self.location,
                )
            )
        else:
            raise ValueError("unsupported literal type")
        return self.wrap(handle)

    def coerce(self, value, element, width, *, layout=None):
        """`value` as an `element` scalar, or a `width`-element tile in `layout`:
        constants become literals and scalars broadcast; element types and layouts are
        never converted implicitly."""
        self.check(value)
        if width and layout is None:
            layout = Layout.striped((width,), threads=self.threads)
        if not isinstance(value, Value):
            value = self.literal(value, element)
        if value.view or value.element != element:
            raise ValueError(
                "operand types differ; implicit runtime casts are not supported",
            )
        if value.width == width:
            if value.layout != layout:
                raise ValueError(
                    "tile layouts differ; an explicit layout conversion is required",
                )
            return value
        if value.width or not width:
            raise ValueError("tile shapes differ")
        return self.emit(
            "carbon.broadcast",
            [value],
            result=native.tile_type(element, layout),
        )

    def operands(self, left, right):
        """Both operands of a binary operation in one type: Python constants take the
        type of the staged side."""
        values = [item for item in (left, right) if isinstance(item, Value)]
        element = (
            values[0].element
            if values
            else (
                "i1"
                if type(left) is bool and type(right) is bool
                else "f32"
                if isinstance(left, float) or isinstance(right, float)
                else "i32"
            )
        )
        width = max((item.width for item in values), default=0)
        layout = next((item.layout for item in values if item.width), None)
        return (
            self.coerce(left, element, width, layout=layout),
            self.coerce(right, element, width, layout=layout),
        )

    def binary(self, opcode, left, right):
        if not isinstance(left, Value) and not isinstance(right, Value):
            return getattr(operator, opcode)(left, right)
        division = opcode in ("floordiv", "mod")
        runtime_division = division and isinstance(right, Value)
        if (
            division
            and not runtime_division
            and (type(right) is not int or not 0 < right < 2**32)
        ):
            raise ValueError(
                "integer division requires a positive compile-time u32 divisor",
            )
        left, right = self.operands(left, right)
        if division and left.element != "i32":
            raise ValueError("integer division requires u32 operands")
        if opcode == "truediv" and left.element != "f32":
            raise ValueError("true division requires f32 operands")
        compare = opcode in _comparisons
        if left.element not in ("i32", "f32"):
            raise ValueError("arithmetic and comparisons require f32 or u32")
        if runtime_division:
            return self.emit("carbon.integer_div", [left, right], {"opcode": opcode})
        if left.width:
            return self.map(opcode, left, right)
        if compare:
            predicate = "u" + opcode if opcode not in ("eq", "ne") else opcode
            if left.element == "f32":
                predicate = "une" if opcode == "ne" else "o" + opcode
            return self.wrap(
                self.native.compare(
                    predicate,
                    left.handle,
                    right.handle,
                    self.location,
                ),
            )
        name = {"floordiv": "divui", "mod": "remui", "truediv": "divf"}.get(opcode)
        if name is None:
            name = opcode + ("f" if left.element == "f32" else "i")
        return self.emit(f"arith.{name}", [left, right])

    def negate(self, value):
        if not isinstance(value, Value):
            return -value
        self.check(value)
        if value.element != "f32":
            raise ValueError("negation requires f32")
        if value.width:
            return self.map("neg", value)
        return self.emit("arith.negf", [value])

    def bitwise(self, opcode, left, right):
        if not isinstance(left, Value) and not isinstance(right, Value):
            operation = {
                "and": operator.and_,
                "or": operator.or_,
                "xor": operator.xor,
                "shl": operator.lshift,
                "shr": operator.rshift,
            }[opcode]
            return operation(left, right)
        return self.emit(
            "carbon.bitwise",
            list(self.operands(left, right)),
            {"opcode": opcode},
        )

    def invert(self, value):
        if not isinstance(value, Value):
            return ~value
        return self.bitwise("xor", value, True if value.element == "i1" else 0xFFFFFFFF)

    def bitcast(self, value, dtype):
        self.check(value)
        if not isinstance(value, Value) or dtype not in (u32, f32):
            raise TypeError("bitcast requires f32 or u32 scalars or tiles")
        if value.element == dtype.name:
            return value
        return self.emit(
            "carbon.bitcast",
            [value],
            result=value_type(dtype.name, value.width, layout=value.layout),
        )

    def math(self, opcode, value):
        self.check(value)
        if not isinstance(value, Value):
            value = self.literal(value, "f32")
        return self.emit("carbon.math", [value], {"opcode": opcode})

    def extremum(self, kind, left, right):
        return self.emit(f"carbon.{kind}", list(self.operands(left, right)))

    def fma(self, a, b, c):
        for item in (a, b, c):
            self.check(item)
        values = [item for item in (a, b, c) if isinstance(item, Value)]
        if any(item.view or item.element != "f32" for item in values):
            raise TypeError(
                "fma requires f32 scalars or tiles; cast storage explicitly",
            )
        width = max((item.width for item in values), default=0)
        layout = next((item.layout for item in values if item.width), None)
        return self.emit(
            "carbon.fma",
            [self.coerce(item, "f32", width, layout=layout) for item in (a, b, c)],
        )

    def select(self, condition, left, right):
        self.check(condition)
        left, right = self.operands(left, right)
        layout = left.layout
        if isinstance(condition, Value) and condition.width and layout is None:
            layout = condition.layout
            left = self.coerce(left, left.element, condition.width, layout=layout)
            right = self.coerce(right, right.element, condition.width, layout=layout)
        condition = self.coerce(condition, "i1", left.width, layout=layout)
        return self.emit("carbon.select", [condition, left, right])

    def broadcast_to(self, value, shape, *, axes=None, layout=None, scratch=None):
        self.check(value)
        layout = (
            Layout.striped(shape, threads=self.threads) if layout is None else layout
        )
        if not isinstance(layout, Layout) or layout.shape != shape:
            raise ValueError("broadcast layout must match the output shape")
        if not isinstance(value, Value):
            value = self.literal(value)
        if value.view:
            raise TypeError("broadcast requires a scalar or tile")
        if not value.width:
            if axes not in (None, ()) or scratch is not None:
                raise ValueError("scalar broadcast has no input axes or scratch")
            return self.coerce(value, value.element, layout.size, layout=layout)
        rank, source = len(shape), value.layout.shape
        axes = tuple(range(rank - len(source), rank)) if axes is None else axes
        if (
            not isinstance(axes, tuple)
            or len(axes) != len(source)
            or any(type(axis) is not int or not 0 <= axis < rank for axis in axes)
            or len(set(axes)) != len(axes)
        ):
            raise ValueError(
                "broadcast axes must map each input axis to a distinct output axis",
            )
        operands = [value]
        if scratch is not None:
            operands.append(self.collective_scratch(scratch))
        return self.emit(
            "carbon.broadcast_in_dim",
            operands,
            {"axes": tuple(axes)},
            result=native.tile_type(value.element, layout),
        )

    def full(self, shape, value, dtype=f32, *, layout=None):
        if not isinstance(dtype, Scalar):
            raise TypeError("full requires a scalar element type")
        if layout is None:
            layout = Layout.striped(shape, threads=self.threads)
        if not isinstance(layout, Layout) or layout.shape != shape:
            raise ValueError("full shape must match its layout")
        return self.coerce(value, dtype.name, layout.size, layout=layout)

    def program_id(self):
        return self.emit("carbon.program_id", [])

    def simd_id(self):
        return self.emit("carbon.simd_id", [])

    def mma(self, unit, shape, relaxed, left, right, accumulator):
        for value in (left, right, accumulator):
            self.check(value)
        return self.emit(
            "carbon.mma",
            [left, right, accumulator],
            {"unit": unit, "shape": tuple(shape), "relaxed": relaxed},
        )

    def tensor_matmul(self, operation, left, right, accumulator):
        for value in (left, right, accumulator):
            self.check(value)
        return self.emit(
            "carbon.tensor_matmul",
            [left, right, accumulator],
            {"relaxed": operation.relaxed},
        )

    def match(self, value, reference):
        return self.coerce(
            value,
            reference.element,
            reference.width,
            layout=reference.layout,
        )

    def arange(self, width, *, layout=None):
        if type(width) is not int or not 0 < width < 2**32:
            raise ValueError("tile width must be a positive compile-time u32")
        if layout is None:
            layout = Layout.striped((width,), threads=self.threads)
        if not isinstance(layout, Layout) or layout.size != width:
            raise ValueError("arange width must match the layout's logical size")
        return self.emit("carbon.iota", [], result=native.tile_type("i32", layout))

    def make_view(self, buffer, descriptor):
        self.check(buffer)
        if not isinstance(descriptor, View):
            raise TypeError("view requires a compile-time View descriptor")
        if (
            not isinstance(buffer, Value)
            or not buffer.view
            or buffer.view_shape is not None
        ):
            raise ValueError(
                "view construction requires a raw device binding or static scratch",
            )
        return self.emit(
            "carbon.view",
            [buffer],
            result=value_type(buffer.element, descriptor=descriptor),
        )

    def view_rank(self, value):
        self.check(value)
        if not isinstance(value, Value) or not value.view:
            raise TypeError("expected a buffer or view")
        if value.descriptor is not None:
            return len(value.descriptor.shape)
        return len(value.view_shape) if value.view_shape is not None else 1

    def dim(self, view, axis):
        rank = self.view_rank(view)
        if type(axis) is not int or not 0 <= axis < rank:
            raise ValueError(
                "dimension axis must be a compile-time index within the view rank",
            )
        return self.emit("carbon.dim", [view], {"axis": axis})

    def partition(self, view, shape, index):
        rank = self.view_rank(view)
        if (
            not isinstance(shape, tuple)
            or len(shape) != rank
            or any(type(n) is not int or n <= 0 for n in shape)
            or math.prod(shape) >= 2**32
        ):
            raise ValueError(
                "partition shape must match the rank and have a positive u32 size",
            )
        return self.emit(
            "carbon.partition",
            [view, self.coerce(index, "i32", 0)],
            {"shape": shape},
        )

    def subview(self, view, offsets, shape):
        rank = self.view_rank(view)
        if any(
            not isinstance(values, tuple) or len(values) != rank
            for values in (offsets, shape)
        ):
            raise ValueError(
                "subview offsets and shape must be tuples matching the view rank",
            )
        values = [view, *(self.coerce(value, "i32", 0) for value in (*offsets, *shape))]
        return self.emit(
            "carbon.subview",
            values,
            {"operandSegmentSizes": [1, rank, rank]},
        )

    def scratch(self, shape, dtype, *, strides=None):
        if not isinstance(dtype, Scalar) or dtype not in STORAGE:
            raise TypeError(
                "scratch storage must be f16, bf16, f32, u32, u8, i8, u16 or i16",
            )
        if not isinstance(shape, tuple) or any(
            type(n) is not int or n <= 0 for n in shape
        ):
            raise ValueError("scratch shape must contain positive compile-time extents")
        size = math.prod(shape)
        if strides is not None:
            if (
                not isinstance(strides, tuple)
                or len(strides) != len(shape)
                or any(type(stride) is not int for stride in strides)
            ):
                raise ValueError(
                    "scratch strides must be an integer tuple matching its rank",
                )
            spans = [(extent - 1) * stride for extent, stride in zip(shape, strides)]
            lower, upper = (
                sum(min(0, span) for span in spans),
                sum(max(0, span) for span in spans),
            )
            storage = self.scratch((upper - lower + 1,), dtype)
            descriptor = View(
                dtype.name,
                shape,
                strides,
                storage.descriptor.extent_bytes,
                -lower * dtype.itemsize,
                "threadgroup",
            )
            return self.make_view(storage, descriptor)
        strides, stride = [], 1
        for extent in reversed(shape):
            strides.insert(0, stride)
            stride *= extent
        descriptor = View(
            dtype.name,
            shape,
            tuple(strides),
            size * dtype.itemsize,
            address_space="threadgroup",
        )
        return self.emit(
            "carbon.scratch",
            [],
            result=value_type(dtype.name, descriptor=descriptor),
            at_entry=True,
        )

    def barrier(self):
        self.effect("carbon.barrier", [])

    def convert_layout(self, value, layout, *, scratch=None, free=False):
        self.check(value)
        if not isinstance(value, Value) or not value.width:
            raise ValueError(
                "layout conversion requires a tile value in layout-aware IR",
            )
        if free and scratch is not None:
            raise ValueError(
                "a free conversion keeps every element in its thread and takes no "
                "scratch",
            )
        if scratch is not None:
            scratch = self.collective_scratch(scratch)
        return self.emit(
            "carbon.convert_layout",
            [value] if scratch is None else [value, scratch],
            {"free": True} if free else None,
            result=native.tile_type(value.element, layout),
        )

    def cast(self, value, dtype):
        self.check(value)
        if not isinstance(dtype, Scalar) or dtype not in STORAGE:
            raise TypeError(
                "cast requires an f16, bf16, f32, u32, u8, i8, u16 or i16 destination",
            )
        floating = ("f16", "bf16", "f32")
        if not isinstance(value, Value):
            value = self.literal(value, "f32" if dtype.name in floating else "i32")
        if dtype.name not in floating and value.element in floating:
            raise TypeError("cast to an integer destination requires an integer value")
        if value.element == dtype.name:
            return value
        return self.emit(
            "carbon.cast",
            [value],
            result=value_type(dtype.name, value.width, layout=value.layout),
        )

    def collective_scratch(self, scratch):
        self.check(scratch)
        if not isinstance(scratch, Value) or not scratch.view:
            raise TypeError("collective scratch must be a scratch view")
        return scratch

    def reduce(self, value, axis, kind, *, layout=None, scratch=None, order="fast"):
        self.check(value)
        if order not in ("fast", "pairwise"):
            raise ValueError('reduction order must be "fast" or "pairwise"')
        if not isinstance(value, Value) or not value.width:
            raise TypeError(
                f"{kind} requires an f32 or u32 tile; cast storage values explicitly",
            )
        rank = len(value.layout.shape)
        if type(axis) is not int or not -rank <= axis < rank:
            raise ValueError(
                "reduction axis must be a compile-time index within the tile rank",
            )
        axis %= rank
        if layout is None:
            shape = value.layout.shape[:axis] + value.layout.shape[axis + 1 :]
            layout = Layout.striped(shape, threads=value.layout.threads)
        operands = [value]
        if scratch is not None:
            operands.append(self.collective_scratch(scratch))
        return self.emit(
            f"carbon.reduce_{kind}",
            operands,
            {"axis": axis, "order": order},
            result=native.tile_type(value.element, layout),
        )

    def scan(self, value, axis, kind, *, scratch=None, order="fast"):
        self.check(value)
        if order not in ("fast", "pairwise"):
            raise ValueError('scan order must be "fast" or "pairwise"')
        if not isinstance(value, Value) or not value.width:
            raise TypeError(
                f"cum{kind} requires an f32 or u32 tile; cast storage values "
                "explicitly",
            )
        rank = len(value.layout.shape)
        if type(axis) is not int or not -rank <= axis < rank:
            raise ValueError(
                "scan axis must be a compile-time index within the tile rank",
            )
        operands = [value]
        if scratch is not None:
            operands.append(self.collective_scratch(scratch))
        return self.emit(
            f"carbon.scan_{kind}",
            operands,
            {"axis": axis % rank, "order": order},
        )

    def item(self, value, *, scratch=None):
        for operand in (value, scratch):
            self.check(operand)
        return self.emit(
            "carbon.item",
            [value] if scratch is None else [value, scratch],
        )

    def reshape(self, value, shape):
        self.check(value)
        if not isinstance(value, Value) or not value.width:
            raise ValueError("reshape requires a tile value")
        shape = reshape_shape(shape, value.width)
        if shape == value.layout.shape:
            return value
        return self.emit("carbon.reshape", [value], {"shape": shape})

    def stack(self, values, axis):
        if not isinstance(values, (tuple, list)) or not values:
            raise ValueError("stack needs a nonempty tuple or list of equal tiles")
        for value in values:
            self.check(value)
            if not isinstance(value, Value) or not value.width:
                raise TypeError("stack inputs must have the same tile type and layout")
        rank = len(values[0].layout.shape)
        if type(axis) is not int or not -rank - 1 <= axis <= rank:
            raise ValueError("stack axis must select a position within the new rank")
        return self.emit("carbon.stack", list(values), {"axis": axis % (rank + 1)})

    def transpose(self, value, axes):
        self.check(value)
        if not isinstance(value, Value) or not value.width:
            raise TypeError("transpose requires a tile value")
        axes = transpose_axes(axes, len(value.layout.shape))
        if axes == tuple(range(len(axes))):
            return value
        return self.emit("carbon.transpose", [value], {"axes": axes})

    def map(self, opcode, *values):
        return self.emit("carbon.map", list(values), {"opcode": opcode})

    def memory_operands(self, view, indices, mask):
        self.check(view)
        coordinates = indices if isinstance(indices, tuple) else (indices,)
        tile = next(
            (
                index
                for index in coordinates
                if isinstance(index, Value) and index.width
            ),
            None,
        )
        if not isinstance(view, Value) or not view.view or tile is None:
            raise ValueError("memory operations require a buffer and tile indices")
        if len(coordinates) > 1 and len(coordinates) != self.view_rank(view):
            raise ValueError(
                "memory operations take flat indices or one coordinate per view axis",
            )
        indices = [
            self.coerce(index, "i32", tile.width, layout=tile.layout)
            for index in coordinates
        ]
        return indices, self.coerce(mask, "i1", tile.width, layout=tile.layout)

    def load(self, view, indices, mask, *, other):
        if not isinstance(indices, tuple) and (
            not isinstance(indices, Value) or not indices.width
        ):
            self.view_rank(view)
            indices = self.coerce(indices, "i32", 0)
            mask = self.coerce(mask, "i1", 0)
            fallback = self.coerce(other, view.element, 0)
            return self.emit("carbon.load_scalar", [view, indices, mask, fallback])
        indices, mask = self.memory_operands(view, indices, mask)
        fallback = self.coerce(other, view.element, mask.width, layout=mask.layout)
        return self.emit("carbon.load", [view, mask, fallback, *indices])

    def store(self, view, indices, values, mask):
        indices, mask = self.memory_operands(view, indices, mask)
        values = self.coerce(values, view.element, mask.width, layout=mask.layout)
        self.effect("carbon.store", [view, values, mask, *indices])

    def atomic(self, view, indices, values, mask, opcode):
        if isinstance(indices, tuple):
            raise ValueError("atomics take flat indices")
        (indices,), mask = self.memory_operands(view, indices, mask)
        values = self.coerce(values, view.element, indices.width, layout=indices.layout)
        if opcode == "store":
            self.effect("carbon.atomic_store", [view, indices, values, mask])
            return
        return self.emit(
            "carbon.atomic",
            [view, indices, values, mask],
            {"opcode": opcode},
        )

    @contextmanager
    def region(self, block=None):
        block = self.native.enter(block)
        try:
            yield block
        finally:
            self.native.leave()

    def condition(self, value):
        self.check(value)
        if value.view or value.width or value.element not in ("i1", "i32", "f32"):
            raise TypeError(
                "runtime conditions require a scalar predicate, u32, or f32",
            )
        return value if value.element == "i1" else self.binary("ne", value, 0)

    def if_else(self, condition, branches):
        merged = [[], []]
        for pair in zip(*(values for _, values in branches)):
            reference = next(
                (value for value in pair if isinstance(value, Value)),
                None,
            )
            arms = []
            for value, (block, _) in zip(pair, branches):
                with self.region(block):
                    if not isinstance(value, Value):
                        if type(value) not in (int, float, bool):
                            raise TypeError(
                                "runtime branches cannot select compile-time metadata",
                            )
                        if reference is None:
                            value = self.literal(value)
                        else:
                            value = self.match(value, reference)
                    self.check(value)
                    if value.view:
                        raise TypeError(
                            "branch results must be scalars or tiles, not views",
                        )
                    arms.append(value)
            left, right = arms
            if left.type != right.type:
                raise TypeError(
                    "branch results must have matching types, shapes, and layouts",
                )
            merged[0].append(left)
            merged[1].append(right)
        (then, _), (otherwise, _) = branches
        handles = self.native.branch(
            condition.handle,
            [value.type for value in merged[0]],
            then,
            otherwise,
            [value.handle for value in merged[0]],
            [value.handle for value in merged[1]],
            self.location,
        )
        return [self.wrap(handle) for handle in handles]

    def for_loop(self, start, stop, step, initial, body):
        """A runtime loop from `start` to `stop` by a constant `step`: `body(index,
        carried)` stages one trip and returns the next carried values, which keep their
        initial types."""
        if type(step) is not int or not 0 < step < 2**32:
            raise ValueError("loop step must be a positive compile-time u32")
        bounds = [self.coerce(bound, "i32", 0) for bound in (start, stop)]
        lower, upper = [
            self.emit("arith.index_castui", [bound], result=native.scalar_type("index"))
            for bound in bounds
        ]
        increment = self.wrap(
            self.native.constant(step, native.scalar_type("index"), self.location),
        )
        initial = [
            value if isinstance(value, Value) else self.literal(value)
            for value in initial
        ]
        for value in initial:
            self.check(value)
            if value.view:
                raise TypeError(
                    "loop-carried values must be scalars or tiles, not views",
                )
        induction, *carried = self.native.loop(
            lower.handle,
            upper.handle,
            increment.handle,
            [value.handle for value in initial],
            self.location,
        )
        carried = [self.wrap(handle) for handle in carried]
        index = self.emit(
            "arith.index_castui",
            [self.wrap(induction)],
            result=native.scalar_type("i32"),
        )
        yielded = body(index, carried)
        if len(yielded) != len(initial):
            raise ValueError("loop yield arity differs from its initial values")
        yielded = [self.match(value, init) for value, init in zip(yielded, initial)]
        handles = self.native.close([value.handle for value in yielded], self.location)
        return [self.wrap(handle) for handle in handles]

    def finish(self, source):
        self.native.finish()
        tiles = tuple(
            (str(index), value.tile) for index, value in enumerate(self.tile_values)
        )
        return StagedKernel(
            self.native.fingerprint,
            source,
            tuple(self.parameters),
            self.threads,
            tiles,
            tuple(self.specialized),
            self.native,
            tuple(self.scalar_ranges),
        )
