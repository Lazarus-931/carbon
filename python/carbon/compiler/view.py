from dataclasses import dataclass, replace
from math import prod

from ..language import STORAGE
from .shape import reshape_shape, reshape_strides, transpose_axes

NAMES = {"u32": "i32", "u8": "ui8", "i8": "si8", "u16": "ui16", "i16": "si16"}
"""Element names a View accepts besides the IR's own, mapped to the IR's."""
ITEMSIZE = {dtype.name: dtype.itemsize for dtype in STORAGE}


@dataclass(frozen=True)
class View:
    """Strides count elements; offset and extent count bytes of the bound storage.

    As a `views=` specialization of a ranked buffer, an extent may be None: it stays a
    runtime value, the other extents, the strides and the zero offset become constants,
    `extent_bytes` is None, and dispatch takes any bound view that `matches` it.
    """

    element: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    extent_bytes: int
    offset_bytes: int = 0
    address_space: str = "device"

    def __post_init__(self):
        object.__setattr__(self, "element", NAMES.get(self.element, self.element))
        if self.element not in ITEMSIZE:
            raise TypeError(
                'views support "f16", "bf16", "f32", "u32", "u8", "i8", "u16" and '
                '"i16" storage',
            )
        if self.address_space not in ("device", "threadgroup"):
            raise ValueError("unsupported address space")
        if not isinstance(self.shape, tuple) or not isinstance(self.strides, tuple):
            raise TypeError("view shape and strides must be tuples")
        if len(self.shape) != len(self.strides):
            raise ValueError("view shape and stride ranks differ")
        if any(type(stride) is not int for stride in self.strides):
            raise TypeError("view strides must be integers")
        if self.runtime:
            if (
                any(n is not None and (type(n) is not int or n < 0) for n in self.shape)
                or self.extent_bytes is not None
                or self.offset_bytes
                or self.address_space != "device"
            ):
                raise ValueError(
                    "a view with runtime extents has no byte extent or offset and "
                    "lives in device storage",
                )
            return
        if any(type(n) is not int or n < 0 for n in self.shape):
            raise ValueError("view extents must be nonnegative integers")
        if type(self.extent_bytes) is not int or self.extent_bytes < 0:
            raise ValueError("storage extent must be a nonnegative byte count")
        if type(self.offset_bytes) is not int or self.offset_bytes % self.itemsize:
            raise ValueError("view offset must be element-aligned bytes")
        lower, upper = self.byte_bounds
        if lower < 0 or upper > self.extent_bytes:
            raise ValueError("view addresses exceed the bound storage extent")

    @property
    def itemsize(self):
        return ITEMSIZE[self.element]

    @property
    def runtime(self):
        """Whether some extent stays a runtime value."""
        return None in self.shape

    def matches(self, view):
        """Whether `view`, a View with every extent known, fits this one: same element,
        strides, offset and fixed extents."""
        return (
            not view.runtime
            and view.element == self.element
            and view.strides == self.strides
            and view.address_space == "device"
            and view.offset_bytes == self.offset_bytes
            and len(view.shape) == len(self.shape)
            and all(
                fixed is None or fixed == extent
                for fixed, extent in zip(self.shape, view.shape)
            )
        )

    @property
    def size(self):
        return prod(self.shape)

    @property
    def byte_bounds(self):
        """Occupied byte range in the bound storage, [start, end)."""
        if not self.size:
            return self.offset_bytes, self.offset_bytes
        spans = [
            (extent - 1) * stride * self.itemsize
            for extent, stride in zip(self.shape, self.strides)
        ]
        lower = self.offset_bytes + sum(min(0, span) for span in spans)
        upper = self.offset_bytes + sum(max(0, span) for span in spans) + self.itemsize
        return lower, upper

    def byte_offset(self, coordinate):
        """The byte offset in the bound storage of the element at `coordinate`."""
        if not isinstance(coordinate, tuple) or len(coordinate) != len(self.shape):
            raise ValueError("coordinate rank must match the view")
        if any(
            type(index) is not int or not 0 <= index < extent
            for index, extent in zip(coordinate, self.shape)
        ):
            raise ValueError("coordinate is outside the view")
        return self.offset_bytes + self.itemsize * sum(
            index * stride for index, stride in zip(coordinate, self.strides)
        )

    def __getitem__(self, key):
        """Apply basic Python indexing without copying or changing the binding."""
        key = key if isinstance(key, tuple) else (key,)
        if sum(item is Ellipsis for item in key) > 1:
            raise IndexError("a view index can contain only one ellipsis")
        missing = len(self.shape) - sum(
            item is not None and item is not Ellipsis for item in key
        )
        if missing < 0:
            raise IndexError("too many indices for the view")
        expanded = []
        for item in key:
            expanded.extend([slice(None)] * missing if item is Ellipsis else [item])
        if not any(item is Ellipsis for item in key):
            expanded.extend([slice(None)] * missing)
        shape, strides = [], []
        axis, offset = 0, self.offset_bytes
        for item in expanded:
            if item is None:
                shape.append(1)
                strides.append(0)
                continue
            extent, stride = self.shape[axis], self.strides[axis]
            axis += 1
            if type(item) is int:
                index = item + extent if item < 0 else item
                if not 0 <= index < extent:
                    raise IndexError("index is outside the view")
                offset += index * stride * self.itemsize
            elif isinstance(item, slice):
                if any(
                    value is not None and type(value) is not int
                    for value in (item.start, item.stop, item.step)
                ):
                    raise TypeError("slice bounds and steps must be integers")
                start, stop, step = item.indices(extent)
                shape.append(len(range(start, stop, step)))
                strides.append(stride * step)
                offset += start * stride * self.itemsize
            else:
                raise TypeError(
                    "view indices must be integers, slices, ellipsis, or None",
                )
        if not prod(shape):
            offset = self.offset_bytes
        return replace(
            self,
            shape=tuple(shape),
            strides=tuple(strides),
            offset_bytes=offset,
        )

    def transpose(self, axes=None):
        """Permute logical axes while preserving the underlying memory."""
        axes = transpose_axes(axes, len(self.shape), subject="view")
        return replace(
            self,
            shape=tuple(self.shape[i] for i in axes),
            strides=tuple(self.strides[i] for i in axes),
        )

    def reshape(self, shape):
        """Change shape without copying; reject non-affine address mappings."""
        shape = reshape_shape(shape, self.size)
        if shape == self.shape:
            return self
        strides = reshape_strides(self.shape, self.strides, shape)
        return replace(self, shape=shape, strides=strides)

    def require_unique_stores(self):
        """Reject overlapping or unproven store mappings; reads may broadcast."""
        if not self.size:
            return
        span = 1
        for stride, extent in sorted(
            (abs(stride), extent)
            for extent, stride in zip(self.shape, self.strides)
            if extent > 1
        ):
            if stride < span:
                raise ValueError("cannot prove unique addresses for view stores")
            span += (extent - 1) * stride

    def bind(self, buffer):
        """Attach this descriptor to storage for a runtime-shaped kernel argument."""
        return BoundView(buffer, self)


@dataclass(frozen=True)
class BoundView:
    """A View attached to the buffer that holds its elements, as a dispatch binds it."""

    buffer: object
    descriptor: View
