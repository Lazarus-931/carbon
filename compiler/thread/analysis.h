
#pragma once

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace carbon::msl {

class Facts {
  public:
    explicit Facts(mlir::Operation* kernel);
    ~Facts();

    mlir::DataFlowSolver& solver() { return *solver_; }
    /// Unsigned bounds of an integer value; its whole type range when nothing is known.
    std::pair<uint64_t, uint64_t> range(mlir::Value value) const;
    /// Power-of-two divisor: 1 if unknown; zero has divisor 2^min(width, 63).
    uint64_t divisor(mlir::Value value) const;
    /// A number equal for two values exactly when their linear forms are equal; none for non-integers.
    std::optional<int> form(mlir::Value value);
    /// (form without its constant, constant); equal forms differ only by their constants.
    std::optional<std::pair<int, uint64_t>> split(mlir::Value value);
    uint64_t divides(int id) const { return divides(linear(id)); }
    /// An upper bound on `value` wherever each `(other, top)` holds as `other <= top`.
    std::optional<uint64_t> bound(mlir::Value value, llvm::ArrayRef<std::pair<mlir::Value, uint64_t>> limits);
    /// (lane-independent form without its constant, remainder at each lane); none if separation fails.
    std::optional<std::pair<int, std::array<uint64_t, 32>>> lanes(mlir::Value value, mlir::Value lane);
    void leaves(int id, llvm::SmallVectorImpl<mlir::Value>& values) const;
    /// Integer ops at `builder` that compute form `id`; none when a term's width differs from the form's.
    mlir::Value materialize(int id, mlir::OpBuilder& builder, mlir::Location location);

  private:
    /// constant + sum of coefficient * atom over `terms`, modulo 2^width; terms are sorted by atom.
    struct Linear {
        unsigned width = 32;
        std::vector<std::pair<int, uint64_t>> terms;
        uint64_t constant = 0;
        bool operator<(const Linear& other) const {
            return std::tie(width, terms, constant) < std::tie(other.width, other.terms, other.constant);
        }
    };
    enum class Kind { Value, Quotient, Remainder, Product, Extend };
    /// Irreducible term over `value`, forms `left`/`right`, or `left` and constant `divisor`.
    struct Atom {
        Kind kind;
        mlir::Value value;
        int left = -1, right = -1;
        uint64_t divisor = 0;
    };

    /// a + factor * b, in a's width.
    static Linear combine(Linear a, const Linear& b, uint64_t factor);
    static Linear scaled(const Linear& linear, uint64_t factor);
    Linear build(mlir::Value value);
    Linear quotient(mlir::Value value, mlir::Value numerator, uint64_t divisor);
    Linear remainder(mlir::Value value, mlir::Value numerator, uint64_t divisor);
    Linear divided(mlir::Value value, mlir::Value numerator, mlir::Value by, bool remainder);
    Linear extend(mlir::Value value, mlir::Value input, unsigned width);
    Linear truncate(const Linear& input, unsigned width);
    Linear opaque(mlir::Value value, unsigned width);
    Linear atom(Atom atom, unsigned width);
    int intern(const Linear& linear);
    const Linear& linear(int id) const { return forms_[id]; }
    /// Upper bound; none for unbounded/subtracted terms or possible wrapping.
    std::optional<uint64_t> largest(const Linear& linear, bool constant = true) const;
    std::optional<uint64_t> largestAtom(int atom) const;
    uint64_t divides(const Linear& linear, bool constant = true) const;
    uint64_t dividesAtom(int atom) const;
    bool mentions(int atom, mlir::Value lane) const;
    /// The value of a form, or of an atom, with `lane` set to `index`; none when it depends on any other value.
    std::optional<uint64_t> at(const Linear& linear, mlir::Value lane, uint64_t index) const;
    std::optional<uint64_t> atomAt(int atom, mlir::Value lane, uint64_t index) const;

    std::unique_ptr<mlir::DataFlowSolver> solver_;
    llvm::DenseMap<mlir::Value, int> values_;
    std::vector<Atom> atoms_;
    std::map<std::tuple<int, void*, int, int, uint64_t>, int> atomIds_;
    std::vector<Linear> forms_;
    std::map<Linear, int> formIds_;
};

/// Evaluate, for every thread, values determined solely by the lane, SIMD group and constants.
class ThreadValues {
  public:
    explicit ThreadValues(int64_t threads) : threads_(threads) {}
    std::optional<uint64_t> at(mlir::Value value, int64_t thread) { return all(value)[thread]; }

  private:
    using Values = std::vector<std::optional<uint64_t>>;
    const Values& all(mlir::Value value);
    Values evaluate(mlir::Value value);
    int64_t threads_;
    llvm::DenseMap<mlir::Value, std::unique_ptr<Values>> known_;
};

}
