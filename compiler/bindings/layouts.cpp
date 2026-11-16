#include "tile/dialect.h"
#include "tile/module.h"
#include "mlir/IR/Diagnostics.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <numeric>
#include <sstream>

namespace nb = nanobind;
using namespace mlir;

namespace carbon {
namespace {

struct Owner {
    int64_t group, lane, slot;
};

// Python integers only: a bool is not an index.
int64_t whole(nb::handle value, const char* what) {
    if (!nb::isinstance<nb::int_>(value) || nb::isinstance<nb::bool_>(value))
        throw nb::value_error((std::string(what) + " must be an integer").c_str());
    return nb::cast<int64_t>(value);
}

int64_t index(nb::handle value, int64_t extent) {
    if (!nb::isinstance<nb::int_>(value) || nb::isinstance<nb::bool_>(value) || nb::cast<int64_t>(value) < 0 ||
        nb::cast<int64_t>(value) >= extent)
        throw nb::value_error(("index must be an integer in [0, " + std::to_string(extent) + ")").c_str());
    return nb::cast<int64_t>(value);
}

SmallVector<int64_t> extents(nb::handle shape) {
    if (!nb::isinstance<nb::tuple>(shape))
        throw nb::type_error("tile shape must be a tuple");
    SmallVector<int64_t> result;
    for (auto extent : nb::cast<nb::tuple>(shape)) {
        auto value = whole(extent, "tile extents");
        if (value < 1)
            throw nb::value_error("tile extents must be positive integers");
        result.push_back(value);
    }
    return result;
}

nb::tuple tuple(ArrayRef<int64_t> values) {
    nb::list items;
    for (auto value : values)
        items.append(value);
    return nb::steal<nb::tuple>(PyList_AsTuple(items.ptr()));
}

nb::object tree(Attribute attribute) {
    if (auto items = dyn_cast<ArrayAttr>(attribute)) {
        nb::list result;
        for (auto item : items)
            result.append(tree(item));
        return nb::steal(PyList_AsTuple(result.ptr()));
    }
    return nb::int_(cast<IntegerAttr>(attribute).getInt());
}

Attribute attribute(nb::handle value, bool extent) {
    auto& context = ir::context();
    if (nb::isinstance<nb::tuple>(value)) {
        SmallVector<Attribute> items;
        for (auto item : nb::cast<nb::tuple>(value))
            items.push_back(attribute(item, extent));
        if (items.empty())
            throw nb::type_error("hierarchical modes require nonempty shape and stride tuples");
        return ArrayAttr::get(&context, items);
    }
    if (!nb::isinstance<nb::int_>(value) || nb::isinstance<nb::bool_>(value) ||
        (extent && nb::cast<int64_t>(value) < 1))
        throw nb::value_error("mode extents must be positive integers with integer strides");
    return IntegerAttr::get(IntegerType::get(&context, 64), nb::cast<int64_t>(value));
}

bool sameTree(Attribute a, Attribute b) {
    auto x = dyn_cast<ArrayAttr>(a), y = dyn_cast<ArrayAttr>(b);
    if (!x || !y)
        return !x && !y;
    if (x.size() != y.size())
        return false;
    for (auto [left, right] : llvm::zip(x, y))
        if (!sameTree(left, right))
            return false;
    return true;
}

template <class Build>
auto checked(Build build) {
    std::string message;
    ScopedDiagnosticHandler handler(&ir::context(), [&](Diagnostic& diagnostic) {
        message = diagnostic.str();
        return success();
    });
    auto result = build([] { return emitError(UnknownLoc::get(&ir::context())); });
    if (!result)
        throw nb::value_error(message.c_str());
    return result;
}

ir::LayoutAttr make(ArrayRef<int64_t> shape, ir::ModeAttr groups, ir::ModeAttr lanes, ir::ModeAttr slots) {
    auto& context = ir::context();
    return checked([&](auto error) {
        return ir::LayoutAttr::getChecked(error, &context, DenseI64ArrayAttr::get(&context, shape), groups, lanes,
                                          slots);
    });
}

ir::LayoutAttr derived(ir::LayoutAttr layout, const char* failure) {
    if (!layout)
        throw nb::value_error(failure);
    return make(layout.getShape().asArrayRef(), layout.getGroups(), layout.getLanes(), layout.getSlots());
}

std::string text(nb::handle value) { return nb::cast<std::string>(nb::repr(value)); }

std::string text(Attribute attribute) { return text(tree(attribute)); }

nb::tuple fields(const Owner& owner) { return nb::make_tuple(owner.group, owner.lane, owner.slot); }

std::string repr(ir::ModeAttr mode) {
    return "Mode(shape=" + text(mode.getShape()) + ", stride=" + text(mode.getStride()) + ")";
}

int64_t geometry(int64_t threads) {
    if (threads < 32 || threads % 32)
        throw nb::value_error("layout requires a whole number of SIMD groups");
    return threads;
}

// One digit per bit of a power-of-two count, moving `element(bit) - element(0)`, with neighbouring bits that continue
// each other's stride merged into one digit.
ir::ModeAttr fitted(int64_t count, llvm::function_ref<int64_t(int64_t)> element) {
    if (count & (count - 1))
        throw nb::value_error("measured SIMD groups and slots must be powers of two");
    SmallVector<std::pair<int64_t, int64_t>> digits;
    for (int64_t bit = 1; bit < count; bit <<= 1) {
        auto stride = element(bit) - element(0);
        if (!digits.empty() && stride == digits.back().first * digits.back().second)
            digits.back().first *= 2;
        else
            digits.push_back({2, stride});
    }
    auto& context = ir::context();
    if (digits.size() <= 1)
        return digits.empty() ? ir::ModeAttr::of(&context, 1, 0)
                              : ir::ModeAttr::of(&context, digits[0].first, digits[0].second);
    SmallVector<Attribute> sizes, strides;
    for (auto [extent, stride] : llvm::reverse(digits)) {
        sizes.push_back(IntegerAttr::get(IntegerType::get(&context, 64), extent));
        strides.push_back(IntegerAttr::get(IntegerType::get(&context, 64), stride));
    }
    return ir::ModeAttr::get(&context, ArrayAttr::get(&context, sizes), ArrayAttr::get(&context, strides));
}

int64_t axis(nb::handle value, int64_t rank, const char* what) {
    auto position = whole(value, what);
    if (position < -rank || position >= rank)
        throw nb::value_error(what);
    return position < 0 ? position + rank : position;
}

}

void bindLayouts(nb::module_& module) {
    nb::class_<Owner>(module, "Owner",
                      "Where an element lives: SIMD group, lane and register slot. Unpacks and compares like the "
                      "tuple (group, lane, slot).")
        .def(nb::init<int64_t, int64_t, int64_t>(), nb::arg("group"), nb::arg("lane"), nb::arg("slot"))
        .def_ro("group", &Owner::group, "The SIMD group within the threadgroup.")
        .def_ro("lane", &Owner::lane, "The lane within the SIMD group.")
        .def_ro("slot", &Owner::slot, "The register slot within the lane.")
        .def("__len__", [](const Owner&) { return 3; })
        .def("__getitem__", [](const Owner& owner, nb::handle key) {
            auto item = nb::steal(PyObject_GetItem(fields(owner).ptr(), key.ptr()));
            if (!item.is_valid())
                throw nb::python_error();
            return item;
        })
        .def("__iter__", [](const Owner& owner) { return nb::iter(fields(owner)); })
        .def("__eq__", [](const Owner& owner, nb::handle other) {
            if (nb::isinstance<Owner>(other))
                return fields(owner).equal(fields(nb::cast<Owner>(other)));
            return nb::isinstance<nb::tuple>(other) && fields(owner).equal(other);
        })
        .def("__hash__", [](const Owner& owner) { return PyObject_Hash(fields(owner).ptr()); })
        .def("__repr__", [](const Owner& owner) {
            return "Owner(group=" + std::to_string(owner.group) + ", lane=" + std::to_string(owner.lane) +
                   ", slot=" + std::to_string(owner.slot) + ")";
        });

    nb::class_<ir::ModeAttr>(module, "Mode",
                             "A hierarchical shape:stride map from an index to an element, with the last leaf varying "
                             "fastest.\n\n"
                             "Each digit of the index, most significant first, moves the element by its stride:\n\n"
                             "    Mode((4, 8), (8, 1)).offset(11) == 1 * 8 + 3 * 1 == 11\n\n"
                             "A stride-0 digit repeats elements: Mode((2, 16), (0, 1)) gives indices 1 and 17 the "
                             "same element.")
        .def("__init__", [](ir::ModeAttr* self, nb::handle shape, nb::handle stride) {
            auto extents = attribute(shape, true), strides = attribute(stride, false);
            if (!sameTree(extents, strides))
                throw nb::value_error("mode shape and stride trees differ");
            new (self) ir::ModeAttr(checked([&](auto error) {
                return ir::ModeAttr::getChecked(error, &ir::context(), extents, strides);
            }));
        }, nb::arg("shape"), nb::arg("stride"))
        .def_prop_ro("shape", [](ir::ModeAttr mode) { return tree(mode.getShape()); },
                     "The extents: an int, or nested tuples of ints.")
        .def_prop_ro("stride", [](ir::ModeAttr mode) { return tree(mode.getStride()); },
                     "The strides, in the same tree as the shape.")
        .def_prop_ro("size", &ir::ModeAttr::size, "The number of indices: the product of the extents.")
        .def_prop_ro("repeats", &ir::ModeAttr::repeats, "How many times each element appears across stride-0 digits.")
        .def("offset", [](ir::ModeAttr mode, nb::handle at) { return mode.offset(index(at, mode.size())); },
             nb::arg("index"), "The element `index` maps to: each of its digits times that digit's stride.")
        .def("__eq__", [](ir::ModeAttr mode, nb::handle other) {
            return nb::isinstance<ir::ModeAttr>(other) && nb::cast<ir::ModeAttr>(other) == mode;
        })
        .def("__hash__", [](ir::ModeAttr mode) { return size_t(mode.getAsOpaquePointer()); })
        .def("__repr__", &repr);

    nb::class_<ir::LayoutAttr>(module, "Layout",
                               "A dense mixed-radix ownership map, optionally padded past tile size.\n\n"
                               "Element groups.offset(g) + lanes.offset(l) + slots.offset(s) of the row-major shape "
                               "sits in slot s of lane l in SIMD group g; places at size or beyond are padding:\n\n"
                               "    Layout.striped((64,))  lane 5 holds elements 5 and 37\n"
                               "    Layout.blocked((64,))  lane 5 holds elements 10 and 11\n\n"
                               "A stride-0 lane or SIMD-group digit repeats elements: every thread along it holds the "
                               "same values. Loads and math run in every holder; stores and atomics come from the "
                               "first holder only.")
        .def("__init__", [](ir::LayoutAttr* self, nb::handle shape, nb::handle groups, nb::handle lanes,
                            nb::handle slots) {
            for (auto mode : {groups, lanes, slots})
                if (!nb::isinstance<ir::ModeAttr>(mode))
                    throw nb::type_error("ownership axes must be Mode objects");
            new (self) ir::LayoutAttr(make(extents(shape), nb::cast<ir::ModeAttr>(groups),
                                           nb::cast<ir::ModeAttr>(lanes), nb::cast<ir::ModeAttr>(slots)));
        }, nb::arg("shape"), nb::arg("groups"), nb::arg("lanes"), nb::arg("slots"))
        .def_static("striped", [](nb::handle shape, int64_t threads) {
            return derived(ir::LayoutAttr::striped(&ir::context(), extents(shape), geometry(threads)), "");
        }, nb::arg("shape"), nb::kw_only(), nb::arg("threads") = 32,
           "Consecutive threads on consecutive elements, then the next slot.")
        .def_static("blocked", [](nb::handle shape, int64_t threads) {
            return derived(ir::LayoutAttr::blocked(&ir::context(), extents(shape), geometry(threads)), "");
        }, nb::arg("shape"), nb::kw_only(), nb::arg("threads") = 32, "Each thread on consecutive elements.")
        .def_static("measured", [](nb::handle shape, int64_t groups, int64_t slots, std::vector<int64_t> elements) {
            if (groups < 1 || slots < 1 || int64_t(elements.size()) != groups * 32 * slots)
                throw nb::value_error("a measured layout needs one element for every SIMD group, lane and slot");
            if (llvm::any_of(elements, [](int64_t element) { return element < 0; }))
                throw nb::value_error("every measured register must hold an element of the tile");
            auto at = [&](int64_t group, int64_t lane, int64_t slot) {
                return elements[(group * 32 + lane) * slots + slot];
            };
            auto layout = make(extents(shape), fitted(groups, [&](int64_t group) { return at(group, 0, 0); }),
                               fitted(32, [&](int64_t lane) { return at(0, lane, 0); }),
                               fitted(slots, [&](int64_t slot) { return at(0, 0, slot); }));
            for (int64_t group = 0; group < groups; ++group)
                for (int64_t lane = 0; lane < 32; ++lane)
                    for (int64_t slot = 0; slot < slots; ++slot)
                        if (layout.index(group, lane, slot) != at(group, lane, slot))
                            throw nb::value_error("the measured registers are not a layout: each register's element "
                                                  "must be a sum over the bits of its SIMD group, lane and slot");
            return layout;
        }, nb::arg("shape"), nb::arg("groups"), nb::arg("slots"), nb::arg("elements"),
           "The layout in which SIMD group g, lane l and slot s hold `elements[(g * 32 + l) * slots + s]`, as a "
           "device reports it.")
        .def_prop_ro("shape", [](ir::LayoutAttr layout) { return tuple(layout.getShape().asArrayRef()); },
                     "The tile's logical shape.")
        .def_prop_ro("groups", &ir::LayoutAttr::getGroups, "The Mode from SIMD-group number to element.")
        .def_prop_ro("lanes", &ir::LayoutAttr::getLanes, "The Mode from lane to element.")
        .def_prop_ro("slots", &ir::LayoutAttr::getSlots, "The Mode from register slot to element.")
        .def_prop_ro("modes", [](ir::LayoutAttr layout) {
            return nb::make_tuple(layout.getGroups(), layout.getLanes(), layout.getSlots());
        }, "(groups, lanes, slots).")
        .def_prop_ro("size", &ir::LayoutAttr::size, "The number of logical elements.")
        .def_prop_ro("capacity", &ir::LayoutAttr::capacity,
                     "The number of (group, lane, slot) places, padding and repeats included.")
        .def_prop_ro("threads", &ir::LayoutAttr::threads, "The threads the layout spans: SIMD groups times 32.")
        .def_prop_ro("replicated", &ir::LayoutAttr::replicated, "Whether some element is held by more than one thread.")
        .def("flat_index", [](ir::LayoutAttr layout, nb::handle group, nb::handle lane, nb::handle slot) {
            return layout.index(index(group, layout.getGroups().size()), index(lane, layout.getLanes().size()),
                                index(slot, layout.getSlots().size()));
        }, nb::arg("group"), nb::arg("lane"), nb::arg("slot"),
           "The row-major index of the element in that place; size or more for padding.")
        .def("coordinate", [](ir::LayoutAttr layout, nb::handle group, nb::handle lane, nb::handle slot) -> nb::object {
            auto at = layout.index(index(group, layout.getGroups().size()), index(lane, layout.getLanes().size()),
                                   index(slot, layout.getSlots().size()));
            if (at >= layout.size())
                return nb::none();
            auto shape = layout.getShape().asArrayRef();
            SmallVector<int64_t> coordinate(shape.size());
            for (size_t axis = shape.size(); axis-- > 0;) {
                coordinate[axis] = at % shape[axis];
                at /= shape[axis];
            }
            return tuple(coordinate);
        }, nb::arg("group"), nb::arg("lane"), nb::arg("slot"),
           "The coordinate of the element in that place, or None for padding.")
        .def("owner_of", [](ir::LayoutAttr layout, nb::handle at) {
            auto element = index(at, layout.size());
            return Owner{layout.getGroups().inverse(element), layout.getLanes().inverse(element),
                         layout.getSlots().inverse(element)};
        }, nb::arg("index"), "The Owner of row-major element `index`; of its holders, the first.")
        .def("owner", [](ir::LayoutAttr layout, nb::handle coordinate) {
            auto shape = layout.getShape().asArrayRef();
            if (!nb::isinstance<nb::tuple>(coordinate) || nb::len(coordinate) != shape.size())
                throw nb::value_error("coordinate rank must match the tile");
            auto values = nb::cast<nb::tuple>(coordinate);
            int64_t element = 0;
            for (size_t axis = 0; axis < shape.size(); ++axis)
                element = element * shape[axis] + index(values[axis], shape[axis]);
            return Owner{layout.getGroups().inverse(element), layout.getLanes().inverse(element),
                         layout.getSlots().inverse(element)};
        }, nb::arg("coordinate"), "The Owner of the element at `coordinate`; of its holders, the first.")
        .def("reshape", [](ir::LayoutAttr layout, nb::handle shape) {
            if (!nb::isinstance<nb::tuple>(shape))
                throw nb::value_error("reshape needs a tuple of positive extents and at most one -1");
            auto extents = nb::cast<nb::tuple>(shape);
            SmallVector<int64_t> target;
            int64_t known = 1, inferred = -1;
            for (size_t axis = 0; axis < extents.size(); ++axis) {
                auto value = whole(extents[axis], "reshape extents");
                if (value == -1 && inferred < 0)
                    inferred = axis;
                else if (value < 1)
                    throw nb::value_error("reshape needs a tuple of positive extents and at most one -1");
                else
                    known *= value;
                target.push_back(value);
            }
            if (inferred >= 0) {
                if (layout.size() % known)
                    throw nb::value_error("cannot infer reshape extent");
                target[inferred] = layout.size() / known;
            }
            return derived(layout.reshape(target), "reshape must preserve the element count");
        }, nb::arg("shape"), "Reshape logical coordinates without moving values between owners.")
        .def("stack", [](ir::LayoutAttr layout, nb::handle count, nb::handle at) {
            auto copies = whole(count, "stack count");
            if (copies < 1)
                throw nb::value_error("stack count must be a positive integer");
            auto rank = int64_t(layout.getShape().size());
            auto position = whole(at, "stack axis");
            if (position < -rank - 1 || position > rank)
                throw nb::value_error("stack axis must select a position within the new rank");
            position = position < 0 ? position + rank + 1 : position;
            return derived(layout.stack(copies, position),
                           "tile axes split an ownership digit; convert the layout explicitly first");
        }, nb::arg("count"), nb::arg("axis") = 0, "Insert an axis of equal tiles by extending each thread's slot mode.")
        .def("transpose", [](ir::LayoutAttr layout, nb::handle axes) {
            auto rank = int64_t(layout.getShape().size());
            SmallVector<int64_t> order;
            if (axes.is_none())
                for (int64_t axis = rank; axis-- > 0;)
                    order.push_back(axis);
            else if (!nb::isinstance<nb::tuple>(axes) || nb::len(axes) != size_t(rank))
                throw nb::value_error("transpose axes must be a permutation of the tile axes");
            else
                for (auto item : nb::cast<nb::tuple>(axes))
                    order.push_back(axis(item, rank, "transpose axes must be a permutation of the tile axes"));
            SmallVector<int64_t> sorted(order);
            llvm::sort(sorted);
            for (int64_t at = 0; at < rank; ++at)
                if (sorted[at] != at)
                    throw nb::value_error("transpose axes must be a permutation of the tile axes");
            return derived(layout.transpose(order),
                           "tile axes split an ownership digit; convert the layout explicitly first");
        }, nb::arg("axes") = nb::none(), "Permute tile coordinates without changing their lane or register slot.")
        .def("reduced", [](ir::LayoutAttr layout, nb::handle at) {
            auto rank = int64_t(layout.getShape().size());
            return derived(layout.reduced(axis(at, rank, "reduction axis must select a tile dimension")),
                           "tile axes split an ownership digit; convert the layout explicitly first");
        }, nb::arg("axis"), "Keep reduction roots on their lanes; reject unrepresentable axis splits.")
        .def("across_groups", [](ir::LayoutAttr layout, nb::handle groups) {
            auto count = whole(groups, "group count");
            if (count < 1 || layout.getGroups().size() != 1)
                throw nb::value_error("across_groups repeats a one-SIMD-group layout over a positive group count");
            return derived(layout.acrossGroups(count), "across_groups needs a one-SIMD-group layout");
        }, nb::arg("groups"), "This one-SIMD-group layout, held whole by each of `groups` SIMD groups.")
        .def_prop_ro("mlir", [](ir::LayoutAttr layout) {
            std::string text;
            llvm::raw_string_ostream stream(text);
            Attribute(layout).print(stream);
            return text;
        }, "The layout as the compiler's IR spells it.")
        .def("__eq__", [](ir::LayoutAttr layout, nb::handle other) {
            return nb::isinstance<ir::LayoutAttr>(other) && nb::cast<ir::LayoutAttr>(other) == layout;
        })
        .def("__hash__", [](ir::LayoutAttr layout) { return size_t(layout.getAsOpaquePointer()); })
        .def("__repr__", [](ir::LayoutAttr layout) {
            return "Layout(shape=" + nb::cast<std::string>(nb::repr(tuple(layout.getShape().asArrayRef()))) +
                   ", groups=" + repr(layout.getGroups()) + ", lanes=" + repr(layout.getLanes()) +
                   ", slots=" + repr(layout.getSlots()) + ")";
        });
}

}
