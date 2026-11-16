import itertools

import pytest

from carbon.compiler import Layout, Owner

SHAPES = [((93,), 32), ((3, 31), 64), ((4, 8), 32), ((257,), 128), ((2, 3, 16), 32)]


@pytest.mark.parametrize(
    "make",
    [Layout.striped, Layout.blocked],
    ids=["striped", "blocked"],
)
@pytest.mark.parametrize("shape, threads", SHAPES, ids=[str(s) for s, _ in SHAPES])
def test_every_element_has_one_owner(make, shape, threads):
    layout = make(shape, threads=threads)
    assert layout.shape == shape and layout.threads == threads
    owners = {}
    for coordinate in itertools.product(*(range(extent) for extent in shape)):
        owner = layout.owner(coordinate)
        assert owner not in owners, (coordinate, owners.get(owner))
        owners[owner] = coordinate
        assert layout.coordinate(*owner) == coordinate
    assert len(owners) == layout.size <= layout.capacity


def test_reshape_stack_and_transpose_keep_owners():
    reshaped = Layout.striped((93,)).reshape((3, -1))
    assert reshaped.shape == (3, 31) and reshaped.owner((2, 30)) == Owner(0, 28, 2)
    stacked = Layout.striped((4, 8)).stack(2).transpose((1, 0, 2))
    assert stacked.shape == (4, 2, 8)
    assert all(
        stacked.coordinate(0, lane, 1) == (lane // 8, 1, lane % 8) for lane in range(32)
    )


@pytest.mark.parametrize(
    "make, message",
    [
        (lambda: Layout.striped((4, 8)).transpose((0, 0)), "permutation"),
        (lambda: Layout.striped((4, 8)).stack(0), "positive"),
        (lambda: Layout.striped((3, 31), threads=64).reduced(2), "axis"),
    ],
)
def test_invalid_layouts_raise(make, message):
    with pytest.raises(ValueError, match=message):
        make()
