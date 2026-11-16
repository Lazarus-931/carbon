
#include "thread/analysis.h"
#include "thread/dialect.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>

using namespace mlir;

namespace carbon::msl {
namespace {

unsigned widthOf(Type type) { return isa<IndexType>(type) ? 64 : type.getIntOrFloatBitWidth(); }

uint64_t mask(unsigned width) { return width >= 64 ? ~uint64_t(0) : (uint64_t(1) << width) - 1; }

uint64_t limit(unsigned width) { return width >= 64 ? uint64_t(1) << 63 : uint64_t(1) << width; }

// The largest power of two dividing a `width`-bit value; 2^width (2^63 for 64 bits) divides zero.
uint64_t lowest(uint64_t value, unsigned width) {
    value &= mask(width);
    return value ? std::min(value & -value, limit(width)) : limit(width);
}

uint64_t product(uint64_t a, uint64_t b, unsigned width) {
    return uint64_t(std::min<unsigned __int128>((unsigned __int128)a * b, limit(width)));
}

std::optional<uint64_t> constant(Value value) {
    APInt bits;
    if (matchPattern(value, m_ConstantInt(&bits)))
        return bits.getZExtValue();
    return std::nullopt;
}

// A power of two that divides an integer; 0 until a definition reaches it.
struct Divisor {
    uint64_t value = 0;
    static Divisor join(const Divisor& a, const Divisor& b) {
        return {!a.value ? b.value : !b.value ? a.value : std::min(a.value, b.value)};
    }
    bool operator==(const Divisor& other) const { return value == other.value; }
    void print(llvm::raw_ostream& stream) const { stream << value; }
};

class DivisorLattice : public dataflow::Lattice<Divisor> {
  public:
    using Lattice::Lattice;
};

// Optimistic: a value waits for all of its operands, and loop-carried values meet at the smallest divisor.
class DivisorAnalysis : public dataflow::SparseForwardDataFlowAnalysis<DivisorLattice> {
  public:
    using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

    void setToEntryState(DivisorLattice* lattice) override { propagateIfChanged(lattice, lattice->join(Divisor{1})); }

    LogicalResult visitOperation(Operation* operation, ArrayRef<const DivisorLattice*> operands,
                                 ArrayRef<DivisorLattice*> results) override {
        if (operation->getNumResults() != 1 || !operation->getResult(0).getType().isIntOrIndex()) {
            setAllToEntryStates(results);
            return success();
        }
        SmallVector<uint64_t> inputs;
        for (auto* operand : operands) {
            if (!operand->getValue().value)
                return success();
            inputs.push_back(operand->getValue().value);
        }
        auto width = widthOf(operation->getResult(0).getType());
        auto right = operation->getNumOperands() > 1 ? constant(operation->getOperand(1)) : std::nullopt;
        uint64_t result = 1;
        if (auto value = constant(operation->getResult(0)))
            result = lowest(*value, width);
        else if (isa<arith::AddIOp, arith::SubIOp, arith::OrIOp, arith::XOrIOp, arith::MinUIOp, arith::MaxUIOp>(
                     operation))
            result = std::min(inputs[0], inputs[1]);
        else if (isa<arith::SelectOp>(operation))
            result = std::min(inputs[1], inputs[2]);
        else if (isa<arith::MulIOp>(operation))
            result = product(inputs[0], inputs[1], width);
        else if (isa<arith::ShLIOp>(operation) && right && *right < width)
            result = product(inputs[0], uint64_t(1) << *right, width);
        else if (isa<arith::ShRUIOp>(operation) && right && *right < width)
            result = std::max<uint64_t>(inputs[0] >> *right, 1);
        else if (isa<arith::DivUIOp>(operation) && right && *right && !(*right & (*right - 1)))
            result = std::max<uint64_t>(inputs[0] / *right, 1);
        else if (isa<arith::RemUIOp>(operation))
            result = std::min(inputs[0], inputs[1]);
        else if (isa<arith::AndIOp>(operation))
            result = std::max(inputs[0], inputs[1]);
        else if (isa<arith::ExtUIOp, arith::TruncIOp, arith::IndexCastUIOp>(operation))
            result = std::min(inputs[0], limit(width));
        propagateIfChanged(results[0], results[0]->join(Divisor{result}));
        return success();
    }

    // A loop counter is divisible by what divides both its lower bound and its step.
    void visitNonControlFlowArguments(Operation* operation, const RegionSuccessor& successor,
                                      ArrayRef<DivisorLattice*> arguments, unsigned firstIndex) override {
        auto loop = dyn_cast<scf::ForOp>(operation);
        if (!loop)
            return SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(operation, successor, arguments,
                                                                               firstIndex);
        auto* point = getProgramPointBefore(loop.getBody());
        auto lower = getLatticeElementFor(point, loop.getLowerBound())->getValue().value;
        auto step = getLatticeElementFor(point, loop.getStep())->getValue().value;
        if (!lower || !step)
            return;
        auto* induction = getLatticeElement(loop.getInductionVar());
        propagateIfChanged(induction, induction->join(Divisor{std::min(lower, step)}));
    }
};

}

Facts::Facts(Operation* kernel) : solver_(std::make_unique<DataFlowSolver>()) {
    dataflow::loadBaselineAnalyses(*solver_);
    solver_->load<dataflow::IntegerRangeAnalysis>();
    solver_->load<DivisorAnalysis>();
    if (failed(solver_->initializeAndRun(kernel)))
        throw std::invalid_argument("internal compiler error: integer facts did not converge");
}

Facts::~Facts() = default;

std::pair<uint64_t, uint64_t> Facts::range(Value value) const {
    if (auto known = constant(value))
        return {*known, *known};
    auto* state = solver_->lookupState<dataflow::IntegerValueRangeLattice>(value);
    if (!state || state->getValue().isUninitialized())
        return {0, mask(widthOf(value.getType()))};
    const auto& bounds = state->getValue().getValue();
    return {bounds.umin().getZExtValue(), bounds.umax().getZExtValue()};
}

uint64_t Facts::divisor(Value value) const {
    auto [low, high] = range(value);
    if (low == high)
        return lowest(low, widthOf(value.getType()));
    auto* state = solver_->lookupState<DivisorLattice>(value);
    return state && state->getValue().value ? state->getValue().value : 1;
}

std::optional<int> Facts::form(Value value) {
    if (!value.getType().isIntOrIndex())
        return std::nullopt;
    if (auto found = values_.find(value); found != values_.end())
        return found->second;
    auto id = intern(build(value));
    values_[value] = id;
    return id;
}

std::optional<std::pair<int, uint64_t>> Facts::split(Value value) {
    auto id = form(value);
    if (!id)
        return std::nullopt;
    auto part = linear(*id);
    auto constant = part.constant;
    part.constant = 0;
    return std::make_pair(intern(part), constant);
}

int Facts::intern(const Linear& linear) {
    auto [found, inserted] = formIds_.try_emplace(linear, int(forms_.size()));
    if (inserted)
        forms_.push_back(linear);
    return found->second;
}

Facts::Linear Facts::atom(Atom atom, unsigned width) {
    auto key = std::make_tuple(int(atom.kind), atom.value.getAsOpaquePointer(), atom.left, atom.right, atom.divisor);
    auto [found, inserted] = atomIds_.try_emplace(key, int(atoms_.size()));
    if (inserted)
        atoms_.push_back(atom);
    return Linear{width, {{found->second, 1}}, 0};
}

Facts::Linear Facts::opaque(Value value, unsigned width) { return atom({Kind::Value, value}, width); }

// value = sum(k * other) + rest, with every coefficient nonnegative; each other's terms are at most its limit minus
// its constant, so value is at most the sum of those and the largest rest.
std::optional<uint64_t> Facts::bound(Value value, ArrayRef<std::pair<Value, uint64_t>> limits) {
    auto id = form(value);
    if (!id)
        return std::nullopt;
    auto rest = linear(*id);
    auto half = mask(rest.width) / 2;
    unsigned __int128 total = 0;
    for (auto [other, top] : limits) {
        auto key = form(other);
        if (!key)
            continue;
        auto assumed = linear(*key);
        auto reach = largest(assumed);
        if (assumed.width != rest.width || assumed.terms.empty() || !reach || top < assumed.constant)
            continue;
        uint64_t count = ~uint64_t(0);
        for (auto [atom, coefficient] : assumed.terms) {
            auto found = llvm::find_if(rest.terms, [&](auto term) { return term.first == atom; });
            if (found == rest.terms.end() || coefficient > half || found->second > half) {
                count = 0;
                break;
            }
            count = std::min(count, found->second / coefficient);
        }
        if (!count)
            continue;
        auto terms = assumed;
        terms.constant = 0;
        rest = combine(rest, terms, -count);
        total += (unsigned __int128)count * (std::min(top, *reach) - assumed.constant);
    }
    auto remainder = largest(rest);
    if (!remainder || total + *remainder > mask(rest.width))
        return std::nullopt;
    return uint64_t(total + *remainder);
}

std::optional<std::pair<int, std::array<uint64_t, 32>>> Facts::lanes(Value value, Value lane) {
    auto id = form(value);
    if (!id)
        return std::nullopt;
    auto source = linear(*id);
    Linear shared{source.width, {}, 0};
    std::array<uint64_t, 32> offsets;
    offsets.fill(source.constant);
    for (auto [atom, coefficient] : source.terms) {
        if (!mentions(atom, lane)) {
            shared.terms.emplace_back(atom, coefficient);
            continue;
        }
        for (uint64_t index = 0; index < offsets.size(); ++index) {
            auto part = atomAt(atom, lane, index);
            if (!part)
                return std::nullopt;
            offsets[index] = (offsets[index] + coefficient * *part) & mask(source.width);
        }
    }
    return std::make_pair(intern(shared), offsets);
}

void Facts::leaves(int id, SmallVectorImpl<Value>& values) const {
    for (auto [atom, coefficient] : linear(id).terms) {
        const auto& entry = atoms_[atom];
        if (entry.kind == Kind::Value)
            values.push_back(entry.value);
        for (auto inner : {entry.left, entry.right})
            if (inner >= 0)
                leaves(inner, values);
    }
}

Value Facts::materialize(int id, OpBuilder& builder, Location location) {
    const auto form = linear(id);
    auto type = builder.getIntegerType(form.width);
    auto constant = [&](uint64_t value) -> Value {
        return arith::ConstantOp::create(builder, location, builder.getIntegerAttr(type, value & mask(form.width)));
    };
    Value sum;
    for (auto [index, coefficient] : form.terms) {
        const auto atom = atoms_[index];
        Value term;
        if (atom.kind == Kind::Value) {
            term = atom.value;
        } else if (atom.kind == Kind::Extend) {
            if (auto inner = materialize(atom.left, builder, location))
                term = arith::ExtUIOp::create(builder, location, type, inner);
        } else {
            auto left = materialize(atom.left, builder, location);
            auto right = atom.right >= 0 ? materialize(atom.right, builder, location) : constant(atom.divisor);
            if (left && right && left.getType() == type && right.getType() == type) {
                if (atom.kind == Kind::Quotient)
                    term = arith::DivUIOp::create(builder, location, left, right);
                else if (atom.kind == Kind::Remainder)
                    term = arith::RemUIOp::create(builder, location, left, right);
                else
                    term = arith::MulIOp::create(builder, location, left, right);
            }
        }
        if (!term || term.getType() != type)
            return {};
        if (coefficient != 1)
            term = arith::MulIOp::create(builder, location, term, constant(coefficient));
        sum = sum ? arith::AddIOp::create(builder, location, sum, term).getResult() : term;
    }
    if (!sum || form.constant)
        sum = sum ? arith::AddIOp::create(builder, location, sum, constant(form.constant)).getResult()
                  : constant(form.constant);
    return sum;
}

bool Facts::mentions(int id, Value lane) const {
    const auto& atom = atoms_[id];
    if (atom.kind == Kind::Value)
        return atom.value == lane;
    return llvm::any_of(std::array<int, 2>{atom.left, atom.right}, [&](int inner) {
        return inner >= 0 && llvm::any_of(linear(inner).terms, [&](auto term) { return mentions(term.first, lane); });
    });
}

std::optional<uint64_t> Facts::at(const Linear& linear, Value lane, uint64_t index) const {
    uint64_t total = linear.constant;
    for (auto [atom, coefficient] : linear.terms) {
        auto part = atomAt(atom, lane, index);
        if (!part)
            return std::nullopt;
        total += coefficient * *part;
    }
    return total & mask(linear.width);
}

std::optional<uint64_t> Facts::atomAt(int id, Value lane, uint64_t index) const {
    const auto& atom = atoms_[id];
    if (atom.kind == Kind::Value)
        return atom.value == lane ? std::optional<uint64_t>(index) : std::nullopt;
    auto left = at(linear(atom.left), lane, index);
    auto right = atom.right >= 0 ? at(linear(atom.right), lane, index) : std::optional<uint64_t>(atom.divisor);
    if (!left || !right)
        return std::nullopt;
    switch (atom.kind) {
    case Kind::Quotient:
        return *right ? std::optional<uint64_t>(*left / *right) : std::nullopt;
    case Kind::Remainder:
        return *right ? std::optional<uint64_t>(*left % *right) : std::nullopt;
    case Kind::Product:
        return *left * *right;
    default:
        return left;
    }
}

Facts::Linear Facts::scaled(const Linear& linear, uint64_t factor) {
    Linear zero;
    zero.width = linear.width;
    return combine(zero, linear, factor);
}

Facts::Linear Facts::combine(Linear a, const Linear& b, uint64_t factor) {
    auto bits = mask(a.width);
    std::map<int, uint64_t> terms(a.terms.begin(), a.terms.end());
    for (auto [atom, coefficient] : b.terms)
        terms[atom] = (terms[atom] + coefficient * factor) & bits;
    a.terms.clear();
    for (auto [atom, coefficient] : terms)
        if (coefficient)
            a.terms.emplace_back(atom, coefficient);
    a.constant = (a.constant + b.constant * factor) & bits;
    return a;
}

Facts::Linear Facts::build(Value value) {
    auto width = widthOf(value.getType());
    if (auto [low, high] = range(value); low == high)
        return Linear{width, {}, low};
    auto* operation = value.getDefiningOp();
    if (!operation || operation->getNumOperands() == 0)
        return opaque(value, width);
    auto input = [&](unsigned index) { return linear(*form(operation->getOperand(index))); };
    auto known = [&](unsigned index) -> std::optional<uint64_t> {
        auto [low, high] = range(operation->getOperand(index));
        return low == high ? std::optional<uint64_t>(low) : constant(operation->getOperand(index));
    };
    auto right = operation->getNumOperands() == 2 ? known(1) : std::nullopt;
    if (isa<arith::AddIOp>(operation))
        return combine(input(0), input(1), 1);
    if (isa<arith::SubIOp>(operation))
        return combine(input(0), input(1), mask(width));
    if (isa<arith::MulIOp>(operation)) {
        if (right)
            return scaled(input(0), *right);
        if (auto left = known(0))
            return scaled(input(1), *left);
        auto a = *form(operation->getOperand(0)), b = *form(operation->getOperand(1));
        return atom({Kind::Product, {}, std::min(a, b), std::max(a, b)}, width);
    }
    if (isa<arith::ShLIOp>(operation) && right && *right < width)
        return scaled(input(0), uint64_t(1) << *right);
    if (isa<arith::DivUIOp>(operation) && right && *right)
        return quotient(value, operation->getOperand(0), *right);
    if (isa<arith::ShRUIOp>(operation) && right && *right < width)
        return quotient(value, operation->getOperand(0), uint64_t(1) << *right);
    if (isa<arith::RemUIOp>(operation) && right && *right)
        return remainder(value, operation->getOperand(0), *right);
    if (isa<arith::DivUIOp, arith::RemUIOp>(operation) && !right)
        return divided(value, operation->getOperand(0), operation->getOperand(1), isa<arith::RemUIOp>(operation));
    if (isa<arith::MaxUIOp, arith::MinUIOp>(operation)) {
        auto a = range(operation->getOperand(0)), b = range(operation->getOperand(1));
        if (isa<arith::MaxUIOp>(operation) ? a.first >= b.second : a.second <= b.first)
            return input(0);
        if (isa<arith::MaxUIOp>(operation) ? b.first >= a.second : b.second <= a.first)
            return input(1);
    }
    if (isa<arith::AndIOp>(operation) && right && !((*right + 1) & *right)) {
        if (*right == mask(width))
            return input(0);
        return remainder(value, operation->getOperand(0), *right + 1);
    }
    if (isa<arith::ExtUIOp, arith::IndexCastUIOp, arith::TruncIOp>(operation)) {
        auto source = widthOf(operation->getOperand(0).getType());
        if (source == width) {
            auto same = input(0);
            same.width = width;
            return same;
        }
        return source < width ? extend(value, operation->getOperand(0), width) : truncate(input(0), width);
    }
    return opaque(value, width);
}

// Without wrapping, numerator = divisor * (whole + offset / divisor) + rest + offset % divisor, so the quotient is
// whole + offset / divisor + (rest + offset % divisor) / divisor.
Facts::Linear Facts::quotient(Value value, Value numerator, uint64_t divisor) {
    auto source = linear(*form(numerator));
    auto width = source.width;
    auto [low, high] = range(numerator);
    if (high < divisor)
        return Linear{width, {}, 0};
    auto part = source;
    part.constant = 0;
    auto offset = source.constant;
    if (part.terms.empty())
        return Linear{width, {}, offset / divisor};
    if (auto top = largest(part, false); top && *top <= mask(width) - offset) {
        Linear whole{width, {}, 0}, rest{width, {}, 0};
        for (auto [atom, coefficient] : part.terms) {
            if (coefficient % divisor)
                rest.terms.emplace_back(atom, coefficient);
            else
                whole.terms.emplace_back(atom, coefficient / divisor);
        }
        whole.constant = offset / divisor;
        offset %= divisor;
        auto reach = largest(rest, false);
        if (rest.terms.empty() || (reach && *reach < divisor - offset))
            return whole;
        if (offset < std::gcd(divides(rest, false), divisor))
            return combine(whole, atom({Kind::Quotient, {}, intern(rest), -1, divisor}, width), 1);
        rest.constant = offset;
        return combine(whole, atom({Kind::Quotient, {}, intern(rest), -1, divisor}, width), 1);
    }
    if (low < offset || offset % divisor >= std::gcd(divides(part, false), divisor))
        return opaque(value, width);
    auto result = atom({Kind::Quotient, {}, intern(part), -1, divisor}, width);
    result.constant = offset / divisor;
    return result;
}

// Terms that are multiples of the divisor drop out when the sum does not wrap, or always for a power of two, which
// divides 2^width.
Facts::Linear Facts::remainder(Value value, Value numerator, uint64_t divisor) {
    auto source = linear(*form(numerator));
    auto width = source.width;
    auto [low, high] = range(numerator);
    if (high < divisor)
        return source;
    auto part = source;
    part.constant = 0;
    auto offset = source.constant;
    if (part.terms.empty())
        return Linear{width, {}, offset % divisor};
    bool power = !(divisor & (divisor - 1));
    auto top = largest(part, false);
    if (power || (top && *top <= mask(width) - offset)) {
        Linear rest{width, {}, 0};
        for (auto [atom, coefficient] : part.terms)
            if (coefficient % divisor &&
                !(power && product(lowest(coefficient, width), dividesAtom(atom), width) % divisor == 0))
                rest.terms.emplace_back(atom, coefficient);
        offset %= divisor;
        if (rest.terms.empty())
            return Linear{width, {}, offset};
        auto reach = largest(rest, false);
        if (reach && *reach < divisor - offset) {
            rest.constant = offset;
            return rest;
        }
        if (offset < std::gcd(divides(rest, false), divisor)) {
            auto result = atom({Kind::Remainder, {}, intern(rest), -1, divisor}, width);
            result.constant = offset;
            return result;
        }
        rest.constant = offset;
        return atom({Kind::Remainder, {}, intern(rest), -1, divisor}, width);
    }
    auto common = std::gcd(divides(part, false), divisor);
    if (low >= offset && common == divisor)
        return Linear{width, {}, offset % divisor};
    if (low < offset || offset % divisor >= common)
        return opaque(value, width);
    auto result = atom({Kind::Remainder, {}, intern(part), -1, divisor}, width);
    result.constant = offset % divisor;
    return result;
}

// x / d and x % d for a d known only through its range and a power of two `step` dividing it: x = base + c with base
// a multiple of step and c below step has base's quotient and base's remainder plus c, since base % d is a multiple of
// step and so at most d - step.
Facts::Linear Facts::divided(Value value, Value numerator, Value by, bool remainder) {
    auto source = linear(*form(numerator));
    auto width = source.width;
    auto denominator = *form(by);
    auto step = divides(linear(denominator));
    auto part = source;
    part.constant = 0;
    auto top = largest(part, false);
    if (!range(by).first || step < 2 || source.constant >= step || divides(part, false) < step || !top ||
        *top > mask(width) - source.constant)
        return opaque(value, width);
    auto result = atom({remainder ? Kind::Remainder : Kind::Quotient, {}, intern(part), denominator, 0}, width);
    if (remainder)
        result.constant = source.constant;
    return result;
}

Facts::Linear Facts::extend(Value value, Value input, unsigned width) {
    auto source = linear(*form(input));
    if (auto top = largest(source); top && *top <= mask(source.width)) {
        source.width = width;
        return source;
    }
    auto part = source;
    part.constant = 0;
    if (range(input).first >= source.constant) {
        auto result = atom({Kind::Extend, {}, intern(part), -1, 0}, width);
        result.constant = source.constant;
        return result;
    }
    return atom({Kind::Extend, {}, intern(source), -1, 0}, width);
}

Facts::Linear Facts::truncate(const Linear& input, unsigned width) {
    Linear result{width, {}, input.constant & mask(width)};
    for (auto [id, coefficient] : input.terms) {
        auto inner = atoms_[id];
        Linear term{width, {{id, 1}}, 0};
        if (inner.kind == Kind::Extend && linear(inner.left).width == width)
            term = linear(inner.left);
        result = combine(result, term, coefficient);
    }
    return result;
}

std::optional<uint64_t> Facts::largest(const Linear& linear, bool constant) const {
    unsigned __int128 total = constant ? linear.constant : 0;
    for (auto [atom, coefficient] : linear.terms) {
        auto top = largestAtom(atom);
        if (coefficient > mask(linear.width) / 2 || !top)
            return std::nullopt;
        total += (unsigned __int128)coefficient * *top;
        if (total > mask(linear.width))
            return std::nullopt;
    }
    return uint64_t(total);
}

std::optional<uint64_t> Facts::largestAtom(int id) const {
    const auto& atom = atoms_[id];
    switch (atom.kind) {
    case Kind::Value:
        return range(atom.value).second;
    case Kind::Quotient:
        return largest(linear(atom.left)).value_or(mask(linear(atom.left).width)) / std::max<uint64_t>(atom.divisor, 1);
    case Kind::Remainder:
        if (atom.right >= 0)
            return largest(linear(atom.right)).value_or(mask(linear(atom.right).width)) - 1;
        return atom.divisor - 1;
    case Kind::Product: {
        auto a = largest(linear(atom.left)), b = largest(linear(atom.right));
        if (!a || !b || (unsigned __int128)*a * *b > ~uint64_t(0))
            return std::nullopt;
        return *a * *b;
    }
    case Kind::Extend:
        return largest(linear(atom.left)).value_or(mask(linear(atom.left).width));
    }
    return std::nullopt;
}

uint64_t Facts::divides(const Linear& linear, bool constant) const {
    auto result = constant ? lowest(linear.constant, linear.width) : limit(linear.width);
    for (auto [atom, coefficient] : linear.terms)
        result = std::min(result, product(lowest(coefficient, linear.width), dividesAtom(atom), linear.width));
    return result;
}

uint64_t Facts::dividesAtom(int id) const {
    const auto& atom = atoms_[id];
    switch (atom.kind) {
    case Kind::Value:
        return divisor(atom.value);
    case Kind::Quotient: {
        auto inner = divides(linear(atom.left));
        bool power = atom.divisor && !(atom.divisor & (atom.divisor - 1));
        return power && inner >= atom.divisor ? inner / atom.divisor : 1;
    }
    case Kind::Remainder:
        return std::min(divides(linear(atom.left)),
                        atom.right >= 0 ? divides(linear(atom.right)) : lowest(atom.divisor, 64));
    case Kind::Product:
        return product(divides(linear(atom.left)), divides(linear(atom.right)), 64);
    case Kind::Extend:
        return divides(linear(atom.left));
    }
    return 1;
}

static std::optional<uint64_t> apply(Operation* operation, unsigned width, ArrayRef<uint64_t> inputs) {
    auto bits = [&](uint64_t result) { return result & mask(width); };
    if (isa<arith::ExtUIOp, arith::IndexCastUIOp>(operation))
        return inputs[0];
    if (isa<arith::TruncIOp>(operation))
        return bits(inputs[0]);
    if (auto compare = dyn_cast<arith::CmpIOp>(operation)) {
        auto operandWidth = widthOf(compare.getLhs().getType());
        auto sign = [&](uint64_t x) {
            return operandWidth >= 64 ? int64_t(x) : int64_t(x << (64 - operandWidth)) >> (64 - operandWidth);
        };
        auto [a, b] = std::pair(inputs[0], inputs[1]);
        using enum arith::CmpIPredicate;
        switch (compare.getPredicate()) {
        case eq: return a == b;
        case ne: return a != b;
        case ult: return a < b;
        case ule: return a <= b;
        case ugt: return a > b;
        case uge: return a >= b;
        case slt: return sign(a) < sign(b);
        case sle: return sign(a) <= sign(b);
        case sgt: return sign(a) > sign(b);
        case sge: return sign(a) >= sign(b);
        }
    }
    if (auto guarded = dyn_cast<GuardedIntegerOp>(operation)) {
        auto [a, b] = std::pair(inputs[0], inputs[1]);
        auto opcode = guarded.getOpcode();
        if (opcode == "floordiv") return b ? a / b : 0;
        if (opcode == "mod") return b ? a % b : 0;
        if (opcode == "shl") return b < 32 ? bits(a << b) : 0;
        if (opcode == "shr") return b < 32 ? a >> b : 0;
        return std::nullopt;
    }
    if (inputs.size() != 2)
        return std::nullopt;
    auto [a, b] = std::pair(inputs[0], inputs[1]);
    if (isa<arith::AddIOp>(operation)) return bits(a + b);
    if (isa<arith::SubIOp>(operation)) return bits(a - b);
    if (isa<arith::MulIOp>(operation)) return bits(a * b);
    if (isa<arith::DivUIOp>(operation)) return b ? std::optional<uint64_t>(a / b) : std::nullopt;
    if (isa<arith::RemUIOp>(operation)) return b ? std::optional<uint64_t>(a % b) : std::nullopt;
    if (isa<arith::AndIOp>(operation)) return a & b;
    if (isa<arith::OrIOp>(operation)) return a | b;
    if (isa<arith::XOrIOp>(operation)) return a ^ b;
    if (isa<arith::ShLIOp>(operation)) return b < width ? std::optional<uint64_t>(bits(a << b)) : std::nullopt;
    if (isa<arith::ShRUIOp>(operation)) return b < width ? std::optional<uint64_t>(a >> b) : std::nullopt;
    if (isa<arith::MinUIOp>(operation)) return std::min(a, b);
    if (isa<arith::MaxUIOp>(operation)) return std::max(a, b);
    return std::nullopt;
}

const ThreadValues::Values& ThreadValues::all(Value value) {
    if (auto found = known_.find(value); found != known_.end())
        return *found->second;
    auto values = std::make_unique<Values>(evaluate(value));
    return *known_.try_emplace(value, std::move(values)).first->second;
}

ThreadValues::Values ThreadValues::evaluate(Value value) {
    Values result(threads_);
    auto* operation = value.getDefiningOp();
    if (!operation || !value.getType().isIntOrIndex())
        return result;
    if (auto known = constant(value))
        return Values(threads_, *known);
    if (auto builtin = dyn_cast<BuiltinOp>(operation)) {
        auto name = builtin.getName();
        if (name == "lane" || name == "simd_group")
            for (int64_t thread = 0; thread < threads_; ++thread)
                result[thread] = name == "lane" ? thread % 32 : thread / 32;
        return result;
    }
    if (auto choice = dyn_cast<arith::SelectOp>(operation)) {
        const auto& condition = all(choice.getCondition());
        const auto& yes = all(choice.getTrueValue());
        const auto& no = all(choice.getFalseValue());
        for (int64_t thread = 0; thread < threads_; ++thread)
            if (condition[thread])
                result[thread] = *condition[thread] ? yes[thread] : no[thread];
        return result;
    }
    SmallVector<const Values*, 2> operands;
    for (auto operand : operation->getOperands())
        operands.push_back(&all(operand));
    auto width = widthOf(value.getType());
    SmallVector<uint64_t, 2> inputs(operands.size());
    for (int64_t thread = 0; thread < threads_; ++thread) {
        if (llvm::any_of(operands, [&](const Values* operand) { return !(*operand)[thread]; }))
            continue;
        for (auto [input, operand] : llvm::zip(inputs, operands))
            input = *(*operand)[thread];
        result[thread] = apply(operation, width, inputs);
    }
    return result;
}

}
