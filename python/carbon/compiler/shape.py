from math import prod


def transpose_axes(axes, rank, *, subject="tile"):
    """`axes` as a permutation of `rank` axes with negative axes resolved; reversed axes
    when None."""
    axes = tuple(reversed(range(rank))) if axes is None else axes
    if not isinstance(axes, tuple) or any(
        type(axis) is not int or not -rank <= axis < rank for axis in axes
    ):
        raise ValueError(f"transpose axes must be a permutation of the {subject} axes")
    axes = tuple(axis % rank for axis in axes)
    if sorted(axes) != list(range(rank)):
        raise ValueError(f"transpose axes must be a permutation of the {subject} axes")
    return axes


def reshape_shape(shape, size):
    """`shape` with its one -1 extent inferred, checked to hold `size` elements."""
    if not isinstance(shape, tuple) or any(type(n) is not int or n < -1 for n in shape):
        raise ValueError(
            "reshape needs a tuple of nonnegative extents and at most one -1",
        )
    if shape.count(-1) > 1:
        raise ValueError("reshape can infer only one extent")
    if -1 in shape:
        known = prod(n for n in shape if n != -1)
        if not known or size % known:
            raise ValueError("cannot infer reshape extent")
        shape = tuple(size // known if n == -1 else n for n in shape)
    if prod(shape) != size:
        raise ValueError("reshape must preserve the element count")
    return shape


def reshape_strides(shape, strides, target):
    """Regroup contiguous runs without changing the row-major element order."""
    chunks = []
    for extent, stride in reversed(tuple(zip(shape, strides))):
        if extent == 1:
            continue
        if chunks and stride == chunks[-1][0] * chunks[-1][1]:
            size, step = chunks[-1]
            chunks[-1] = (size * extent, step)
        else:
            chunks.append((extent, stride))
    if not prod(shape) or not chunks:
        chunks = [(prod(target), 1)]
    result = [0] * len(target)
    axis = len(target) - 1
    for size, step in chunks:
        consumed = 1
        while axis >= 0 and (consumed < size or target[axis] == 1 or not size):
            result[axis] = step * consumed
            consumed *= target[axis]
            axis -= 1
        if consumed != size:
            raise ValueError("reshape would require a copy")
    if axis >= 0:
        raise ValueError("reshape would require a copy")
    return tuple(result)
