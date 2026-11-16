import ast
import builtins
import inspect
import linecache
import math
import textwrap
from functools import cache, update_wrapper
from types import CodeType

from ..language import STORAGE, Buffer, Scalar, constexpr, f32, u32
from .atom import Atom, CopyAtom, TensorAtom, TensorMatmul, TiledAtom
from .builder import Builder, Value, current_builder
from .layout import Layout
from .view import View

ATOMICS = {"atomic_add", "atomic_min", "atomic_max"}
COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


class Inline:
    def __init__(self, function):
        self.function = function
        update_wrapper(self, function)

    def __call__(self, *args, **kwargs):
        return current_builder().inline(self.function, args, kwargs)


class Returned(Exception):
    def __init__(self, value):
        self.value = value


def flatten_values(value):
    leaves = []

    def visit(item):
        if type(item) in (tuple, list):
            return type(item), tuple(visit(child) for child in item)
        leaves.append(item)
        return None

    structure = visit(value)
    return leaves, structure


def restore_values(leaves, structure):
    values = iter(leaves)

    def visit(item):
        if item is None:
            return next(values)
        kind, children = item
        return kind(visit(child) for child in children)

    return visit(structure)


def same_value(left, right):
    if left is right:
        return True
    if (
        isinstance(left, Value)
        or isinstance(right, Value)
        or type(left) is not type(right)
    ):
        return False
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            same_value(a, b) for a, b in zip(left, right)
        )
    return left == right


@cache
def read_names(node):
    """Names read from the enclosing scope. In a comprehension, only the first
    iterable is outside its local scope. Cached ASTs must remain immutable."""
    if isinstance(node, COMPREHENSIONS):
        first, *rest = node.generators
        bound = frozenset(
            name.id
            for generator in node.generators
            for name in ast.walk(generator.target)
            if isinstance(name, ast.Name)
        )
        inside = [
            child
            for child in ast.iter_child_nodes(node)
            if not isinstance(child, ast.comprehension)
        ]
        inside += [
            *first.ifs,
            *(part for generator in rest for part in (generator.iter, *generator.ifs)),
        ]
        return (
            read_names(first.iter) | frozenset().union(*map(read_names, inside)) - bound
        )
    found = (
        {node.id}
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        else set()
    )
    for child in ast.iter_child_nodes(node):
        found |= read_names(child)
    return frozenset(found)


@cache
def stored_names(node):
    """Names a subtree assigns in its own scope; comprehension variables stay inside the
    comprehension."""
    if isinstance(node, COMPREHENSIONS):
        return frozenset()
    found = (
        {node.id}
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        else set()
    )
    for child in ast.iter_child_nodes(node):
        found |= stored_names(child)
    return frozenset(found)


@cache
def written_names(node):
    return frozenset(item.id for item in ast.walk(node) if isinstance(item, ast.Name))


@cache
def live_suffixes(statements, live_out):
    """Names needed from each statement onward."""
    needed, suffixes = set(live_out), [live_out]
    for statement in reversed(statements):
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                needed.difference_update(written_names(target))
            needed.update(read_names(statement.value))
        elif isinstance(statement, ast.AugAssign) and isinstance(
            statement.target,
            ast.Name,
        ):
            needed.add(statement.target.id)
            needed.update(read_names(statement.value))
        elif isinstance(statement, ast.If):
            needed = set().union(
                live_suffixes(tuple(statement.body), frozenset(needed))[0],
                live_suffixes(tuple(statement.orelse), frozenset(needed))[0],
                read_names(statement.test),
            )
        elif isinstance(statement, ast.For) and isinstance(statement.target, ast.Name):
            carried = set(needed)
            while True:
                live = live_suffixes(tuple(statement.body), frozenset(carried))[0]
                incoming = needed | (live - {statement.target.id})
                if incoming == carried:
                    break
                carried = incoming
            needed = carried | read_names(statement.iter)
        else:
            needed.update(read_names(statement))
        suffixes.append(frozenset(needed))
    return suffixes[::-1]


@cache
def parsed_source(code):
    lines = linecache.getlines(code.co_filename)
    if not lines:
        raise ValueError(
            "staged functions need inspectable source; save the kernel in a .py file "
            "and import it instead of defining it in a REPL or exec()",
        )
    first_line = code.co_firstlineno
    source = textwrap.dedent("".join(inspect.getblock(lines[first_line - 1 :])))
    tree = ast.parse(source, filename=code.co_filename)
    ast.increment_lineno(tree, first_line - 1)
    definition = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef)),
        None,
    )
    if definition is None:
        raise ValueError("staged functions need inspectable source")
    body = definition.body
    if ast.get_docstring(definition, clean=False) is not None:
        body = body[1:]
    return source, tuple(body)


@cache
def local_names(code):
    parameters = code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
    return frozenset(parameters).union(*map(stored_names, parsed_source(code)[1]))


def closure_environment(function):
    cells = zip(function.__code__.co_freevars, function.__closure__ or ())
    return {
        **function.__globals__,
        **{name: cell.cell_contents for name, cell in cells},
    }


def code_names(code):
    names = set(code.co_names) | set(code.co_freevars)
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            names |= code_names(constant)
    return names


def free_values(function):
    environment = closure_environment(function)
    return tuple(
        (name, type(environment[name]), environment[name])
        for name in sorted(code_names(function.__code__))
        if name in environment
    )


class ASTFrontend(Builder):
    def __init__(
        self,
        function,
        constants,
        threads,
        max_grid=None,
        views=None,
        ranges=None,
        min_bytes=None,
        indirect=False,
    ):
        super().__init__(threads, max_grid, indirect)
        self.function = function
        self.constants = constants
        self.views = dict(views or {})
        self.ranges = dict(ranges or {})
        self.min_bytes = dict(min_bytes or {})
        self.values = {}
        self.filename = inspect.getsourcefile(function) or function.__name__
        self.environment = closure_environment(function)
        self.locals = local_names(function.__code__)
        self.loop_depth = 0
        self.branch_depth = 0
        self.in_helper = False
        self.call_stack = [function]

    def error(self, node, message):
        raise ValueError(f"{self.filename}:{getattr(node, 'lineno', 0)}: {message}")

    def expression(self, node):
        if isinstance(node, ast.Name):
            if node.id in self.values:
                return self.values[node.id]
            if node.id in self.constants:
                return self.constants[node.id]
            if node.id not in self.locals:
                if node.id in self.environment:
                    return self.environment[node.id]
                if hasattr(builtins, node.id):
                    return getattr(builtins, node.id)
            self.error(node, "unknown value; pass constants explicitly")
        previous = self.location
        self.location = (self.filename, node.lineno, node.col_offset + 1)
        try:
            return self._expression(node)
        except (ValueError, TypeError, AttributeError, NotImplementedError) as failure:
            if str(failure).startswith(f"{self.filename}:"):
                raise
            self.error(node, str(failure))
        finally:
            self.location = previous

    def _expression(self, node):
        if isinstance(node, ast.Constant) and (
            node.value is None or type(node.value) in (int, float, bool, str)
        ):
            return node.value
        if isinstance(node, ast.Tuple):
            return tuple(self.expression(item) for item in node.elts)
        if isinstance(node, ast.Attribute):
            parent = self.expression(node.value)
            if isinstance(parent, Value):
                if node.attr == "layout" and parent.layout is not None:
                    return parent.layout
                self.error(node, "attribute access requires compile-time metadata")
            return getattr(parent, node.attr)
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                left, right = self.expression(node.left), self.expression(node.right)
                if isinstance(left, Value) or isinstance(right, Value):
                    self.error(node, "exponentiation requires compile-time operands")
                result = left**right
                if type(result) not in (int, float):
                    self.error(
                        node,
                        "compile-time exponentiation must produce a real number",
                    )
                return result
            bitwise = {
                ast.BitAnd: "and",
                ast.BitOr: "or",
                ast.BitXor: "xor",
                ast.LShift: "shl",
                ast.RShift: "shr",
            }
            if type(node.op) in bitwise:
                return self.bitwise(
                    bitwise[type(node.op)],
                    self.expression(node.left),
                    self.expression(node.right),
                )
            operations = {
                ast.Add: "add",
                ast.Sub: "sub",
                ast.Mult: "mul",
                ast.FloorDiv: "floordiv",
                ast.Mod: "mod",
                ast.Div: "truediv",
            }
            if type(node.op) not in operations:
                self.error(node, "unsupported arithmetic")
            return self.binary(
                operations[type(node.op)],
                self.expression(node.left),
                self.expression(node.right),
            )
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return self.negate(self.expression(node.operand))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
            return self.invert(self.expression(node.operand))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not self.compile_time(
                node.operand,
                "not needs a compile-time operand; invert masks with ~",
            )
        if isinstance(node, ast.BoolOp):
            for item in node.values:
                value = self.compile_time(
                    item,
                    "boolean operators need compile-time "
                    "operands; combine masks with & and |",
                )
                if bool(value) != isinstance(node.op, ast.And):
                    return value
            return value
        if isinstance(node, ast.IfExp):
            condition = self.compile_time(
                node.test,
                "conditional expressions need a "
                "compile-time condition; cl.select chooses per element",
            )
            return self.expression(node.body if condition else node.orelse)
        if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
            items = self.comprehension(node.elt, node.generators)
            return items if isinstance(node, ast.ListComp) else tuple(items)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            if isinstance(node.ops[0], (ast.Is, ast.IsNot)):
                left, right = (
                    self.expression(node.left),
                    self.expression(node.comparators[0]),
                )
                if (
                    left is not None
                    and right is not None
                    and any(isinstance(value, Value) for value in (left, right))
                ):
                    self.error(
                        node,
                        "identity checks on device values require a None operand",
                    )
                return (
                    (left is right)
                    if isinstance(node.ops[0], ast.Is)
                    else (left is not right)
                )
            predicates = {
                ast.Lt: "lt",
                ast.LtE: "le",
                ast.Gt: "gt",
                ast.GtE: "ge",
                ast.Eq: "eq",
                ast.NotEq: "ne",
            }
            if type(node.ops[0]) not in predicates:
                self.error(node, "unsupported comparison")
            return self.binary(
                predicates[type(node.ops[0])],
                self.expression(node.left),
                self.expression(node.comparators[0]),
            )
        if isinstance(node, ast.Call):
            return self.call(node)
        if isinstance(node, ast.Subscript):
            parent = self.expression(node.value)
            if isinstance(node.slice, ast.Slice):
                key = slice(
                    *(
                        None if part is None else self.expression(part)
                        for part in (
                            node.slice.lower,
                            node.slice.upper,
                            node.slice.step,
                        )
                    ),
                )
            else:
                key = self.expression(node.slice)
            if isinstance(key, slice):
                items = (key.start, key.stop, key.step)
            else:
                items = key if isinstance(key, tuple) else (key,)
            if isinstance(parent, Value) or any(
                isinstance(item, Value) for item in items
            ):
                self.error(
                    node,
                    "subscripts need compile-time values; index device "
                    "memory with cl.load",
                )
            return parent[key]
        self.error(node, "unsupported expression")

    def compile_time(self, node, message):
        value = self.expression(node)
        if isinstance(value, Value):
            self.error(node, message)
        return value

    def comprehension(self, element, generators):
        if not generators:
            return [self.expression(element)]
        first, *rest = generators
        if first.is_async:
            self.error(first.iter, "comprehensions cannot be asynchronous")
        iterable = self.compile_time(
            first.iter,
            "comprehensions need compile-time iterables",
        )
        saved, items = self.values, []
        try:
            for item in iterable:
                self.values = dict(saved)
                self.bind_names(first.target, item)
                if all(
                    self.compile_time(
                        test,
                        "comprehension conditions must be compile-time",
                    )
                    for test in first.ifs
                ):
                    items += self.comprehension(element, rest)
        finally:
            self.values = saved
        return items

    def call(self, node):
        if any(isinstance(item, ast.Starred) for item in node.args) or any(
            item.arg is None for item in node.keywords
        ):
            self.error(node, "call arguments must be explicit")
        arguments = [self.expression(item) for item in node.args]
        keywords = {item.arg: self.expression(item.value) for item in node.keywords}
        function = self.expression(node.func)
        if not callable(function):
            self.error(node, "call target must be a Python callable")
        return function(*arguments, **keywords)

    def assign(self, target, value):
        if value is None:
            self.error(target, "the expression has no value")
        self.bind_names(target, value)

    def bind_names(self, target, value):
        if isinstance(target, ast.Name):
            self.values[target.id] = value
        elif (
            isinstance(target, ast.Tuple)
            and isinstance(value, (tuple, list))
            and len(target.elts) == len(value)
        ):
            for item, result in zip(target.elts, value):
                self.bind_names(item, result)
        else:
            self.error(target, "assignment requires names or a matching tuple")

    def loop(self, node):
        if not isinstance(node.target, ast.Name) or node.orelse:
            self.error(node, "loops require one induction name and no else clause")
        if node.target.id in self.values or node.target.id in self.constants:
            self.error(node, "loop induction names must not shadow existing values")
        call = node.iter
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "range"
            and 1 <= len(call.args) <= 3
            and not call.keywords
        ):
            self.error(node, "runtime loops require range(start, stop, step)")
        if self.environment.get("range", range) is not range:
            self.error(node, "range must refer to Python's builtin")
        bounds = [self.expression(item) for item in call.args]
        if len(bounds) == 1:
            start, stop, step = 0, bounds[0], 1
        elif len(bounds) == 2:
            start, stop, step = *bounds, 1
        else:
            start, stop, step = bounds
        written = set().union(*map(stored_names, node.body))
        names = sorted(written.intersection(self.values))
        outer = self.values
        initial, structure = flatten_values(tuple(outer[name] for name in names))

        def body(index, carried):
            carried = restore_values(carried, structure)
            self.values = {**outer, **dict(zip(names, carried)), node.target.id: index}
            self.loop_depth += 1
            try:
                self.block(node.body, names)
                yielded, result_structure = flatten_values(
                    tuple(self.values[name] for name in names),
                )
                if result_structure != structure:
                    self.error(
                        node,
                        "loop-carried tuples must preserve their structure",
                    )
                return yielded
            finally:
                self.loop_depth -= 1
                self.values = outer

        results = self.for_loop(start, stop, step, initial, body)
        self.values.update(zip(names, restore_values(results, structure)))

    def branch(self, node, live_out):
        condition = self.expression(node.test)
        if not isinstance(condition, Value):
            self.block(node.body if condition else node.orelse, live_out)
            return
        condition = self.condition(condition)
        outer, branches = self.values, []
        for statements in (node.body, node.orelse):
            self.values = dict(outer)
            self.branch_depth += 1
            try:
                with self.region() as arm:
                    self.block(statements, live_out)
                branches.append((arm, self.values))
            finally:
                self.values = outer
                self.branch_depth -= 1
        left, right = (values for _, values in branches)
        names = []
        for name in sorted(left.keys() & right.keys() & live_out):
            a, b = left[name], right[name]
            if same_value(a, b):
                self.values[name] = a
            else:
                names.append(name)
        arms, structure = [], None
        for arm, values in branches:
            leaves, branch_structure = flatten_values(
                tuple(values[name] for name in names),
            )
            if structure is not None and branch_structure != structure:
                self.error(node, "branch-result tuples must have matching structures")
            structure = branch_structure
            arms.append((arm, leaves))
        results = restore_values(self.if_else(condition, arms), structure)
        self.values.update(zip(names, results))

    def block(self, statements, live_out=()):
        for index, statement in enumerate(statements):
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                self.assign(statement.targets[0], self.expression(statement.value))
            elif isinstance(statement, ast.AugAssign) and isinstance(
                statement.target,
                ast.Name,
            ):
                value = ast.BinOp(
                    left=statement.target,
                    op=statement.op,
                    right=statement.value,
                )
                ast.copy_location(value, statement)
                self.assign(statement.target, self.expression(value))
            elif isinstance(statement, ast.Expr) and isinstance(
                statement.value,
                ast.Call,
            ):
                function = statement.value.func
                name = (
                    function.attr
                    if isinstance(function, ast.Attribute)
                    else getattr(function, "id", None)
                )
                if self.expression(statement.value) is not None and name not in ATOMICS:
                    self.error(statement, "unused value expression")
            elif isinstance(statement, (ast.For, ast.If)):
                previous = self.location
                self.location = (
                    self.filename,
                    statement.lineno,
                    statement.col_offset + 1,
                )
                try:
                    if isinstance(statement, ast.For):
                        self.loop(statement)
                    else:
                        self.branch(
                            statement,
                            live_suffixes(tuple(statements), frozenset(live_out))[
                                index + 1
                            ],
                        )
                except (ValueError, TypeError) as failure:
                    if str(failure).startswith(f"{self.filename}:"):
                        raise
                    self.error(statement, str(failure))
                finally:
                    self.location = previous
            elif isinstance(statement, ast.Pass):
                continue
            elif isinstance(statement, ast.Return):
                if not self.in_helper or self.loop_depth or self.branch_depth:
                    self.error(
                        statement,
                        "return is supported only outside runtime control flow in an "
                        "inline helper",
                    )
                raise Returned(
                    self.expression(statement.value) if statement.value else None,
                )
            else:
                self.error(statement, "unsupported statement")

    def inline(self, function, arguments, keywords):
        if function in self.call_stack:
            raise ValueError("recursive inline helpers are not supported")
        bound = inspect.signature(function).bind(*arguments, **keywords)
        bound.apply_defaults()
        saved = (
            self.function,
            self.filename,
            self.environment,
            self.locals,
            self.values,
            self.constants,
            self.loop_depth,
            self.branch_depth,
            self.in_helper,
        )
        self.function = function
        self.filename = inspect.getsourcefile(function) or function.__name__
        self.environment = closure_environment(function)
        self.locals = local_names(function.__code__)
        self.values, self.constants = dict(bound.arguments), {}
        self.loop_depth, self.branch_depth, self.in_helper = 0, 0, True
        self.call_stack.append(function)
        try:
            _, body = parsed_source(function.__code__)
            try:
                self.block(body)
            except Returned as result:
                return result.value
        finally:
            self.call_stack.pop()
            (
                self.function,
                self.filename,
                self.environment,
                self.locals,
                self.values,
                self.constants,
                self.loop_depth,
                self.branch_depth,
                self.in_helper,
            ) = saved

    def parameter(self, name, kind):
        value = (
            self.specialize(name, kind, self.views.pop(name))
            if name in self.views
            else self.bind(name, kind)
        )
        if name in self.ranges:
            self.constrain(name, kind, value, self.ranges.pop(name))
        if name in self.min_bytes:
            self.require_bytes(name, kind, value, self.min_bytes.pop(name))
        return value

    def lower(self):
        signature = inspect.signature(self.function)
        annotations = inspect.get_annotations(self.function, eval_str=True)
        constant_names = {
            name for name, kind in annotations.items() if kind is constexpr
        }
        if set(self.constants) != constant_names:
            raise ValueError("compile-time constant names do not match the signature")
        seen_scalar = False
        for name, parameter in signature.parameters.items():
            if (
                parameter.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
                or parameter.default is not inspect.Parameter.empty
            ):
                raise ValueError(
                    "kernel parameters must be explicit positional arguments without "
                    "defaults",
                )
            kind = annotations.get(name)
            if kind is constexpr:
                continue
            group = isinstance(kind, tuple)
            members = (
                [(f"{name}[{index}]", part) for index, part in enumerate(kind)]
                if group
                else [(name, kind)]
            )
            for _, part in members:
                if not (
                    isinstance(part, Buffer)
                    and part.element in STORAGE
                    or not group
                    and isinstance(part, Scalar)
                    and part in (f32, u32)
                ):
                    raise TypeError(f"unsupported annotation for {name}")
                if isinstance(part, Buffer) and seen_scalar:
                    raise ValueError("buffers must precede runtime scalars")
                seen_scalar |= isinstance(part, Scalar)
            values = tuple(self.parameter(member, part) for member, part in members)
            self.values[name] = values if group else values[0]
        if self.views:
            raise ValueError(
                "views name no ranked buffer parameter: "
                f"{', '.join(sorted(self.views))}",
            )
        if self.ranges:
            raise ValueError(
                "ranges name no runtime scalar parameter: "
                f"{', '.join(sorted(self.ranges))}",
            )
        if self.min_bytes:
            raise ValueError(
                "min_bytes name no buffer parameter: "
                f"{', '.join(sorted(self.min_bytes))}",
            )
        source, body = parsed_source(self.function.__code__)
        self.block(body)
        return self.finish(source)


def stage(
    function,
    *,
    constants,
    threads,
    max_grid=None,
    views=None,
    ranges=None,
    min_bytes=None,
    indirect=False,
):
    """Construct typed tile IR without loading MLIR or executing the kernel.

    `views` fixes ranked buffer parameters to compile-time Views, so their shapes and
    strides become constants. `ranges` gives inclusive bounds for runtime u32 scalars,
    checked when an artifact is dispatched. `min_bytes` gives checked minimum buffer
    sizes without fixing their actual dimensions or strides.
    """
    if type(threads) is not int or not 1 <= threads <= 1024:
        raise ValueError("threads must be an integer between 1 and 1024")
    constants = dict(constants)
    if any(
        type(value) not in (int, float, bool, str)
        and not isinstance(
            value,
            (View, Layout, Scalar, Atom, TensorAtom, TiledAtom, CopyAtom, TensorMatmul),
        )
        and not (isinstance(value, tuple) and all(type(n) is int for n in value))
        for value in constants.values()
    ):
        raise TypeError(
            "compile-time parameters must be numeric, strings, shape tuples, Scalar, "
            "View, Layout, Atom, TensorAtom, TiledAtom, CopyAtom, or TensorMatmul",
        )
    if any(
        isinstance(value, float) and not math.isfinite(value)
        for value in constants.values()
    ):
        raise ValueError("compile-time parameters must be finite")
    frontend = ASTFrontend(
        function,
        constants,
        threads,
        max_grid,
        views,
        ranges,
        min_bytes,
        indirect,
    )
    with frontend.activate():
        return frontend.lower()
