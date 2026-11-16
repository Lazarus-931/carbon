from dataclasses import dataclass

from ..language import STORAGE
from .native import load_compiler

_ELEMENTS = {dtype.name for dtype in STORAGE} | {"i1"}

_native = load_compiler()
Mode, Layout, Owner = _native.Mode, _native.Layout, _native.Owner


@dataclass(frozen=True)
class Tile:
    """Distributed value type; neither an allocation nor physical registers."""

    element: str
    layout: Layout

    def __post_init__(self):
        if self.element not in _ELEMENTS or not isinstance(self.layout, Layout):
            raise TypeError("tile requires a supported element type and a Layout")

    @property
    def shape(self):
        return self.layout.shape

    @property
    def size(self):
        return self.layout.size
