include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def Carbon_Dialect : Dialect {
  let summary = "Tile IR: typed tile operations over layouts and views";
  let description = [{
    A value is a scalar, a tile whose layout says which SIMD group, lane and register slot holds each element, or
    memory: a device binding, a view of one, or threadgroup scratch. The verifiers check ownership, scratch
    synchronization and uniform control flow; lowering to the `msl` dialect then writes out what one thread does.
  }];
  let name = "carbon";
  let cppNamespace = "::carbon::ir";
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

def Carbon_Mode : AttrDef<Carbon_Dialect, "Mode"> {
  let summary = "A tree of (extent, stride) digits taking an index to an element offset; stride 0 repeats";
  let mnemonic = "mode";
  let parameters = (ins "::mlir::Attribute":$shape, "::mlir::Attribute":$stride);
  let assemblyFormat = "`<` params `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    static ModeAttr of(::mlir::MLIRContext* context, int64_t extent, int64_t stride);
    int64_t size() const;
    /// The element offset of index value `index`: each of its digits, most significant first, times that stride.
    int64_t offset(int64_t index) const;
    /// The index value that holds element `index`, taking 0 on repeating digits.
    int64_t inverse(int64_t index) const;
    /// The index value that holds element `index`, taking `preferred`'s value on repeating digits.
    int64_t holder(int64_t index, int64_t preferred) const;
    /// How many index values hold each element: the product of the stride-0 extents.
    int64_t repeats() const;
    bool replicated() const;
    /// (extent, stride) per digit, most significant first.
    std::vector<std::pair<int64_t, int64_t>> digits() const;
    /// This mode with each digit split at the element `boundaries` inside its span, each piece's stride
    /// transform(the piece's first element); stride-0 digits stay. Null when a boundary does not split a digit evenly.
    /// Example: Mode(8, 2).remap({4}, identity) is Mode((4, 2), (4, 2)); both take index i to element 2i.
    ModeAttr remap(::llvm::ArrayRef<int64_t> boundaries, ::llvm::function_ref<int64_t(int64_t)> transform) const;
  }];
}

def Carbon_Layout : AttrDef<Carbon_Dialect, "Layout"> {
  let summary = "Which SIMD group, lane and slot hold each element of a tile's row-major shape";
  let mnemonic = "layout";
  let parameters = (ins "::mlir::DenseI64ArrayAttr":$shape,
                        "::carbon::ir::ModeAttr":$groups,
                        "::carbon::ir::ModeAttr":$lanes,
                        "::carbon::ir::ModeAttr":$slots);
  let assemblyFormat = "`<` $shape `,` qualified($groups) `,` qualified($lanes) `,` qualified($slots) `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    int64_t size() const;
    /// The number of (group, lane, slot) places, padding and repeats included.
    int64_t capacity() const;
    int64_t threads() const;
    int64_t groupOwner(int64_t index) const;
    /// The element at slot `slot` of lane `lane` in SIMD group `group`; size() or more is padding.
    int64_t index(int64_t group, int64_t lane, int64_t slot) const;
    bool replicated() const;

    /// Consecutive threads on consecutive elements, then the next slot.
    static LayoutAttr striped(::mlir::MLIRContext* context, ::llvm::ArrayRef<int64_t> shape, int64_t threads);
    /// Each thread on consecutive elements.
    static LayoutAttr blocked(::mlir::MLIRContext* context, ::llvm::ArrayRef<int64_t> shape, int64_t threads);
    /// The same elements in the same places under a new row-major `shape`; null when an ownership digit would split.
    LayoutAttr reshape(::llvm::ArrayRef<int64_t> shape) const;
    /// `count` copies of the tile along a new axis at `axis`, held in new slots; null when a digit would split.
    LayoutAttr stack(int64_t count, int64_t axis) const;
    /// The tile with its axes permuted, every element in the same place; null when a digit would split.
    LayoutAttr transpose(::llvm::ArrayRef<int64_t> axes) const;
    /// Without `axis`, each reduction root staying on its lane.
    LayoutAttr reduced(int64_t axis) const;
    /// This one-SIMD-group layout held whole by each of `groups` SIMD groups.
    LayoutAttr acrossGroups(int64_t groups) const;
  }];
}

def Carbon_Tile : TypeDef<Carbon_Dialect, "Tile"> {
  let summary = "Elements held in registers across the threadgroup as the layout places them";
  let mnemonic = "tile";
  let parameters = (ins "::mlir::Type":$elementType, "::carbon::ir::LayoutAttr":$layout);
  let assemblyFormat = "`<` $elementType `,` qualified($layout) `>`";
  let genVerifyDecl = 1;
}

def Carbon_View : TypeDef<Carbon_Dialect, "View"> {
  let summary = "Strided elements of device or threadgroup memory; a runtime view, of extent -1, takes its -1 "
                "dimensions, its strides when it has none, and an offset of -1 from the dispatch";
  let mnemonic = "view";
  let parameters = (ins "::mlir::Type":$elementType,
                        "::mlir::DenseI64ArrayAttr":$shape,
                        "::mlir::DenseI64ArrayAttr":$strides,
                        "int64_t":$offsetBytes, "int64_t":$extentBytes,
                        StringRefParameter<>:$addressSpace);
  let assemblyFormat = "`<` params `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    bool isDynamic() const;
  }];
}

def Carbon_Memory : AnyTypeOf<[AnyMemRef, Carbon_View]>;

class Carbon_Op<string name, list<Trait> traits = []> :
    Op<Carbon_Dialect, name, traits>;

def Carbon_ProgramIdOp : Carbon_Op<"program_id", [Pure]> {
  let summary = "One-dimensional threadgroup index";
  let results = (outs I32:$result);
}

def Carbon_SimdIdOp : Carbon_Op<"simd_id", [Pure]> {
  let summary = "SIMD-group index within the threadgroup";
  let results = (outs I32:$result);
}

def Carbon_IotaOp : Carbon_Op<"iota", [Pure]> {
  let summary = "Logical row-major indices of a distributed tile";
  let results = (outs Carbon_Tile:$result);
  let hasVerifier = 1;
}

def Carbon_LoadOp : Carbon_Op<"load", [MemoryEffects<[MemRead]>, AllTypesMatch<["fallback", "result"]>]> {
  let summary = "Masked tile load with explicit inactive-lane values, at flat indices or one coordinate per view axis";
  let description = [{
    Reads one element per slot of the result. `indices` is one tile of flat row-major indices, or one coordinate
    tile per axis of a view. A slot reads memory only where it holds a real element, its `mask` element is true and
    its index is inside the binding or view; otherwise it takes the matching `fallback` element. `mask`, `fallback`
    and the indices share one layout, which the result keeps.
  }];
  let arguments = (ins Carbon_Memory:$buffer, Carbon_Tile:$mask, Carbon_Tile:$fallback, Variadic<Carbon_Tile>:$indices);
  let results = (outs Carbon_Tile:$result);
  let hasVerifier = 1;
}

def Carbon_ScalarLoadOp : Carbon_Op<"load_scalar", [MemoryEffects<[MemRead]>,
                                                     AllTypesMatch<["fallback", "result"]>]> {
  let summary = "Bounded uniform scalar load from a read-only device binding";
  let arguments = (ins Carbon_Memory:$buffer, I32:$index, I1:$mask, AnyType:$fallback);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def Carbon_StoreOp : Carbon_Op<"store", [MemoryEffects<[MemWrite]>]> {
  let summary = "Masked tile store into a device binding, view or scratch, at flat indices or one coordinate per axis";
  let description = [{
    Writes one element per slot of `values` where its `mask` element is true and its index is inside the binding or
    view, with `indices` as for `carbon.load`. Where a layout repeats an element, only its first holder writes it.
    A scratch store needs a view without repeated addresses.
  }];
  let arguments = (ins Carbon_Memory:$buffer, Carbon_Tile:$values, Carbon_Tile:$mask, Variadic<Carbon_Tile>:$indices);
  let hasVerifier = 1;
}

def Carbon_AtomicOp : Carbon_Op<"atomic", [MemoryEffects<[MemRead, MemWrite]>, AllTypesMatch<["values", "result"]>]> {
  let summary = "Bounded relaxed device fetch-and-op: add, min or max on u32, add on f32; masked elements return zero";
  let description = [{
    Applies `opcode` to each element where `mask` is true and the index is inside the binding or view, and returns
    the previous values. Atomics act on device bindings and views only, with one tile of flat indices.
  }];
  let arguments = (ins Carbon_Memory:$buffer, Carbon_Tile:$indices,
                       Carbon_Tile:$values, Carbon_Tile:$mask, StrAttr:$opcode);
  let results = (outs Carbon_Tile:$result);
  let hasVerifier = 1;
}

def Carbon_AtomicStoreOp : Carbon_Op<"atomic_store", [MemoryEffects<[MemWrite]>]> {
  let summary = "Bounded relaxed device store of u32 or f32 elements, without reading the previous value";
  let arguments = (ins Carbon_Memory:$buffer, Carbon_Tile:$indices, Carbon_Tile:$values, Carbon_Tile:$mask);
  let hasVerifier = 1;
}

def Carbon_MakeViewOp : Carbon_Op<"view", [Pure]> {
  let summary = "A static view of a flat device binding, or of a scratch allocation within its extent";
  let arguments = (ins Carbon_Memory:$buffer);
  let results = (outs Carbon_View:$result);
  let hasVerifier = 1;
}

def Carbon_ScratchOp : Carbon_Op<"scratch", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Kernel-lifetime threadgroup storage";
  let results = (outs Carbon_View:$result);
  let hasVerifier = 1;
  let extraClassDeclaration = [{ int64_t allocationBytes(); }];
}

def Carbon_BarrierOp : Carbon_Op<"barrier"> {
  let summary = "Threadgroup execution barrier and threadgroup memory fence";
  let hasVerifier = 1;
}

def Carbon_DimOp : Carbon_Op<"dim", [Pure]> {
  let summary = "Logical dimension of a buffer or view";
  let arguments = (ins Carbon_Memory:$view, I64Attr:$axis);
  let results = (outs I32:$result);
  let hasVerifier = 1;
}

def Carbon_PartitionOp : Carbon_Op<"partition", [Pure, InferTypeOpAdaptor]> {
  let summary = "One bounded tile of `shape` from a row-major partition grid";
  let description = [{
    A device `view` cut into a row-major grid of tiles of `shape`; the result is tile `index`, clipped to the view,
    for a threadgroup-uniform `index`. Past the last tile the view is inactive and every access is masked.
  }];
  let arguments = (ins Carbon_View:$view, I32:$index, DenseI64ArrayAttr:$shape);
  let results = (outs Carbon_View:$result);
  let hasVerifier = 1;
}

def Carbon_SubviewOp : Carbon_Op<"subview", [Pure, AttrSizedOperandSegments, InferTypeOpAdaptor]> {
  let summary = "Clipped rank-preserving region of an existing device view";
  let description = [{
    The region of a device `view` from `offsets` with extents `sizes`, all threadgroup-uniform, clipped to the view
    and keeping its strides. A flat binding counts as a one-axis view.
  }];
  let arguments = (ins Carbon_Memory:$view, Variadic<I32>:$offsets, Variadic<I32>:$sizes);
  let results = (outs Carbon_View:$result);
  let hasVerifier = 1;
}

def Carbon_BroadcastOp : Carbon_Op<"broadcast", [Pure]> {
  let summary = "A tile with every element equal to a scalar of its element type";
  let arguments = (ins AnyType:$value);
  let results = (outs Carbon_Tile:$result);
  let hasVerifier = 1;
}

def Carbon_MapOp : Carbon_Op<"map", [Pure, InferTypeOpAdaptor]> {
  let summary = "Elementwise add, sub, mul, truediv, neg, comparisons, or u32 floordiv and mod by a positive constant";
  let description = [{
    `opcode` is add, sub or mul; truediv, or neg with one input, on f32; floordiv or mod on u32 by a positive
    constant; or a comparison lt, le, gt, ge, eq or ne, whose result is an i1 tile. The inputs share one f32 or u32
    tile type, and the result keeps their layout.
  }];
  let arguments = (ins Variadic<Carbon_Tile>:$inputs, StrAttr:$opcode);
  let results = (outs Carbon_Tile:$result);
  let hasVerifier = 1;
}

def Carbon_CastOp : Carbon_Op<"cast", [Pure]> {
  let summary = "Explicit float conversion, integer to f32, or u32 to and from 8- and 16-bit integers";
  let description = [{
    Floats convert to floats through f32; u32 and the 8- and 16-bit integers convert to f32 exactly or by
    rounding to nearest even; an 8- or 16-bit integer extends to u32 by its sign, and u32 truncates to one. Each
    element stays in its thread.
  }];
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def Carbon_MathOp : Carbon_Op<"math", [Pure, AllTypesMatch<["value", "result"]>]> {
  let summary = "Accurate f32 scalar or tile elementary function";
  let arguments = (ins AnyType:$value, StrAttr:$opcode);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def Carbon_BitwiseOp : Carbon_Op<"bitwise", [Pure, AllTypesMatch<["left", "right", "result"]>]> {
  let summary = "Unsigned or predicate bit operations; shifts of 32 or more yield zero";
  let arguments = (ins AnyType:$left, AnyType:$right, StrAttr:$opcode);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def Carbon_IntegerDivOp : Carbon_Op<"integer_div", [Pure, AllTypesMatch<["left", "right", "result"]>]> {
  let summary = "Unsigned quotient or remainder; a zero divisor produces zero";
  let arguments = (ins AnyType:$left, AnyType:$right, StrAttr:$opcode);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def Carbon_BitcastOp : Carbon_Op<"bitcast", [Pure]> {
  let summary = "Bit-preserving f32/u32 reinterpretation without ownership changes";
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def Carbon_MaximumOp : Carbon_Op<"maximum", [Pure, AllTypesMatch<["left", "right", "result"]>]> {
  let summary = "Elementwise maximum; for f32, NaN if either side is NaN, and +0 above -0";
  let arguments = (ins AnyType:$left, AnyType:$right);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def Carbon_MinimumOp : Carbon_Op<"minimum", [Pure, AllTypesMatch<["left", "right", "result"]>]> {
  let summary = "Elementwise minimum; for f32, NaN if either side is NaN, and -0 below +0";
  let arguments = (ins AnyType:$left, AnyType:$right);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def Carbon_FmaOp : Carbon_Op<"fma", [Pure, AllTypesMatch<["a", "b", "c", "result"]>]> {
  let summary = "f32 a * b + c with one rounding";
  let arguments = (ins AnyType:$a, AnyType:$b, AnyType:$c);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def Carbon_SelectOp : Carbon_Op<"select", [Pure, AllTypesMatch<["left", "right", "result"]>]> {
  let summary = "Bit-preserving scalar or elementwise tile selection";
  let arguments = (ins AnyType:$condition, AnyType:$left, AnyType:$right);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def Carbon_BroadcastInDimOp : Carbon_Op<"broadcast_in_dim"> {
  let summary = "Replicate tile coordinates with explicit axes and collective workspace";
  let description = [{
    Input axis i becomes output axis `axes[i]`, and an input extent of 1 repeats along its axis, as the input does
    along output axes no input axis names. Elements that move between SIMD groups go through `scratch`, a workspace
    of the input's shape and element type, u32 for predicates, and need control flow uniform across the threadgroup;
    otherwise control flow must be uniform across each SIMD group.
  }];
  let arguments = (ins Carbon_Tile:$value, Optional<Carbon_View>:$scratch, DenseI64ArrayAttr:$axes);
  let results = (outs Carbon_Tile:$result);
  let hasVerifier = 1;
}

def Carbon_ReshapeOp : Carbon_Op<"reshape", [Pure, InferTypeOpAdaptor]> {
  let summary = "Logical shape change to `shape` preserving flat indices and physical ownership";
  let arguments = (ins Carbon_Tile:$value, DenseI64ArrayAttr:$shape);
  let results = (outs Carbon_Tile:$result);
}

def Carbon_StackOp : Carbon_Op<"stack", [Pure, InferTypeOpAdaptor]> {
  let summary = "Insert a tile axis using per-thread slots without communication";
  let arguments = (ins Variadic<Carbon_Tile>:$inputs, I64Attr:$axis);
  let results = (outs Carbon_Tile:$result);
  let hasVerifier = 1;
}

def Carbon_TransposeOp : Carbon_Op<"transpose", [Pure, InferTypeOpAdaptor]> {
  let summary = "Permute tile coordinates without changing physical owners";
  let arguments = (ins Carbon_Tile:$value, DenseI64ArrayAttr:$axes);
  let results = (outs Carbon_Tile:$result);
}

def Carbon_ConvertLayoutOp : Carbon_Op<"convert_layout"> {
  let summary = "The same elements in another layout, using shuffles or explicit scratch";
  let description = [{
    Shape, element type and SIMD-group count stay. Without scratch, shuffles move elements within each SIMD group.
    With scratch, a store and barrier publish the input, then a load and barrier complete the readers. With `free`,
    no element may change thread and scratch is forbidden. Scratch requires threadgroup-uniform control flow;
    otherwise control flow must be uniform across each SIMD group.
  }];
  let arguments = (ins Carbon_Tile:$value, Optional<Carbon_View>:$scratch,
                       DefaultValuedAttr<BoolAttr, "false">:$free);
  let results = (outs Carbon_Tile:$result);
  let hasVerifier = 1;
}

class Carbon_Reduction<string name> : Carbon_Op<name> {
  let summary = "Sum or maximum along `axis`: `pairwise` order adds in a fixed tree, `fast` lets f32 sums take the "
                "hardware's order; values that cross SIMD groups need `scratch`";
  let description = [{
    The result is the input without `axis`, in its own layout, and neither layout repeats an element. Values that
    cross SIMD groups go through `scratch`, a workspace of the input's shape and type, and need control flow uniform
    across the threadgroup; otherwise control flow must be uniform across each SIMD group.
  }];
  let arguments = (ins Carbon_Tile:$value, Optional<Carbon_View>:$scratch,
                       I64Attr:$axis, StrAttr:$order);
  let results = (outs Carbon_Tile:$result);
  let hasVerifier = 1;
}

def Carbon_ReduceSumOp : Carbon_Reduction<"reduce_sum">;
def Carbon_ReduceMaxOp : Carbon_Reduction<"reduce_max">;

class Carbon_Scan<string name> : Carbon_Op<name, [AllTypesMatch<["value", "result"]>]> {
  let summary = "Inclusive logical-axis scan; pairwise order combines, at each distance 2^k, the element 2^k before";
  let description = [{
    A running sum or maximum along `axis`, in the input's layout. Partners in other SIMD groups exchange through
    `scratch`, a workspace of the input's shape and type; `order` and the control-flow rules are those of
    reductions.
  }];
  let arguments = (ins Carbon_Tile:$value, Optional<Carbon_View>:$scratch, I64Attr:$axis, StrAttr:$order);
  let results = (outs Carbon_Tile:$result);
  let hasVerifier = 1;
}

def Carbon_ScanSumOp : Carbon_Scan<"scan_sum">;
def Carbon_ScanMaxOp : Carbon_Scan<"scan_max">;

def Carbon_ItemOp : Carbon_Op<"item", [InferTypeOpAdaptor]> {
  let summary = "Broadcast one logical tile element to a threadgroup-uniform scalar";
  let description = [{
    A tile spread over several SIMD groups passes its element through `scratch`; control flow around the operation
    must be uniform across the threadgroup.
  }];
  let arguments = (ins Carbon_Tile:$value, Optional<Carbon_View>:$scratch);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def Carbon_MmaOp : Carbon_Op<"mma", [NoMemoryEffect, AllTypesMatch<["accumulator", "result"]>]> {
  let summary = "accumulator + left x right on one matrix unit in each SIMD group: `simdgroup` (8x8x8 SIMD-group "
                "matrices) or `tensor` (a Metal 4 matmul2d of `shape` m, n, k on cooperative tensors). Each operand "
                "is a tile stacking one block per SIMD group in the layout measured for that unit; a thread's slots "
                "are the unit's elements in its own order";
  let arguments = (ins Carbon_Tile:$left, Carbon_Tile:$right, Carbon_Tile:$accumulator, StrAttr:$unit,
                       DenseI64ArrayAttr:$shape, BoolAttr:$relaxed);
  let results = (outs Carbon_Tile:$result);
  let hasVerifier = 1;
}

def Carbon_TensorMatmulOp : Carbon_Op<"tensor_matmul", [MemoryEffects<[MemRead]>,
                                                        AllTypesMatch<["accumulator", "result"]>]> {
  let summary = "accumulator + left x right with every SIMD group of the threadgroup cooperating, the matrix unit "
                "reading left and right from memory itself (Metal 4 tensor ops); the accumulator is a tile in the "
                "layout measured for that destination";
  let description = [{
    `left` is (M, K) and `right` (K, N), both f16, bf16 or f32; each view's root is row- or column-major with fixed
    strides. `relaxed` sets matmul2d's relaxed-precision mode. Control flow must be uniform across the threadgroup.
  }];
  let arguments = (ins Carbon_View:$left, Carbon_View:$right, Carbon_Tile:$accumulator, BoolAttr:$relaxed);
  let results = (outs Carbon_Tile:$result);
  let hasVerifier = 1;
}
