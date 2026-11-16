include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Msl_Dialect : Dialect {
  let summary = "Per-thread Metal IR: what one thread executes once tile operations are lowered";
  let description = [{
    Index math is ordinary `arith` on integers; these operations cover what `arith` cannot say exactly in Metal
    terms, and the printer turns each into one MSL statement. SIMD-group collectives and barriers are not
    speculatable: no pass may move them across control flow, and verifyConvergence checks that every participant
    reaches them. Builtins and parameters carry the dispatch contract's bounds, which seed the integer facts that
    fold guards and form vector accesses.
  }];
  let name = "msl";
  let cppNamespace = "::carbon::msl";
  let useDefaultTypePrinterParser = 1;
}

def Msl_Pointer : TypeDef<Msl_Dialect, "Pointer"> {
  let summary = "A pointer to `elementType` elements in the `space` address space";
  let mnemonic = "ptr";
  let parameters = (ins StringRefParameter<>:$space, "::mlir::Type":$elementType);
  let assemblyFormat = "`<` $space `,` $elementType `>`";
}

class Msl_Op<string mnemonic, list<Trait> traits = []> : Op<Msl_Dialect, mnemonic, traits>;

def Msl_ArgumentOp : Msl_Op<"argument", [Pure]> {
  let summary = "Device pointer of one kernel binding";
  let arguments = (ins I64Attr:$binding);
  let results = (outs Msl_Pointer:$pointer);
}

def Msl_ScratchOp : Msl_Op<"scratch", [Pure]> {
  let summary = "Threadgroup pointer at a byte offset of the kernel's scratch block";
  let arguments = (ins I64Attr:$offset);
  let results = (outs Msl_Pointer:$pointer);
}

def Msl_ReinterpretOp : Msl_Op<"reinterpret", [Pure]> {
  let summary = "Same address, another element type";
  let arguments = (ins Msl_Pointer:$pointer);
  let results = (outs Msl_Pointer:$result);
}

def Msl_BuiltinOp : Msl_Op<"builtin",
    [Pure, DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>]> {
  let summary = "lane, simd_group or group (threadgroup position), below `bound`";
  let arguments = (ins StrAttr:$name, I64Attr:$bound);
  let results = (outs I32:$result);
}

def Msl_ParamOp : Msl_Op<"param", [Pure, DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>]> {
  let summary = "A field of the constant parameter block; integer bounds are inclusive and checked at dispatch";
  let arguments = (ins StrAttr:$field, DefaultValuedAttr<I64Attr, "0">:$minimum,
                       DefaultValuedAttr<I64Attr, "4294967295">:$maximum);
  let results = (outs AnyType:$result);
}

def Msl_LoadOp : Msl_Op<"load", [MemoryEffects<[MemRead]>]> {
  let summary = "pointer[index]; with several results, one vector access of that many elements";
  let arguments = (ins Msl_Pointer:$pointer, AnySignlessIntegerOrIndex:$index);
  let results = (outs Variadic<AnyType>:$results);
}

def Msl_GuardedLoadOp : Msl_Op<"guarded_load", [MemoryEffects<[MemRead]>]> {
  let summary = "condition ? pointer[index] : fallback, reading memory only when the condition holds; with several "
                "results, one vector access of that many elements";
  let arguments = (ins Msl_Pointer:$pointer, AnySignlessIntegerOrIndex:$index, I1:$condition,
                       Variadic<AnyType>:$fallbacks);
  let results = (outs Variadic<AnyType>:$results);
  let hasCanonicalizer = 1;
}

def Msl_StoreOp : Msl_Op<"store", [MemoryEffects<[MemWrite]>]> {
  let summary = "pointer[index] = value; with several values, one vector access";
  let arguments = (ins Msl_Pointer:$pointer, AnySignlessIntegerOrIndex:$index, Variadic<AnyType>:$values);
}

def Msl_GuardedStoreOp : Msl_Op<"guarded_store", [MemoryEffects<[MemWrite]>]> {
  let summary = "if (condition) pointer[index] = value; with several values, one vector access";
  let arguments = (ins Msl_Pointer:$pointer, AnySignlessIntegerOrIndex:$index, I1:$condition,
                       Variadic<AnyType>:$values);
  let hasCanonicalizer = 1;
}

def Msl_AtomicOp : Msl_Op<"atomic", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Relaxed u32 or f32 fetch-and-op when the condition holds, else zero";
  let arguments = (ins Msl_Pointer:$pointer, AnySignlessIntegerOrIndex:$index, AnyTypeOf<[I32, F32]>:$value,
                       I1:$condition, StrAttr:$opcode);
  let results = (outs AnyTypeOf<[I32, F32]>:$result);
}

def Msl_AtomicStoreOp : Msl_Op<"atomic_store", [MemoryEffects<[MemWrite]>]> {
  let summary = "Relaxed u32 or f32 atomic store when the condition holds";
  let arguments = (ins Msl_Pointer:$pointer, AnySignlessIntegerOrIndex:$index, AnyTypeOf<[I32, F32]>:$value,
                       I1:$condition);
}

def Msl_BarrierOp : Msl_Op<"barrier"> {
  let summary = "threadgroup_barrier(mem_flags::mem_threadgroup)";
}

def Msl_ShuffleOp : Msl_Op<"shuffle", [NoMemoryEffect]> {
  let summary = "simd_shuffle from a lane, or simd_shuffle_xor (butterfly) with a lane mask";
  let arguments = (ins AnyType:$value, I32:$lane, BoolAttr:$butterfly);
  let results = (outs AnyType:$result);
}

def Msl_SimdOp : Msl_Op<"simd", [NoMemoryEffect]> {
  let summary = "SIMD-group collective: sum, max, all, prefix_inclusive_sum or prefix_exclusive_sum; `uniform` marks "
                "one whose every SIMD group combines the same values";
  let arguments = (ins AnyType:$value, StrAttr:$kind, UnitAttr:$uniform);
  let results = (outs AnyType:$result);
}

def Msl_MathOp : Msl_Op<"math", [Pure]> {
  let summary = "precise:: math function; abs, floor, ceil and the reciprocal 1 / x are exact";
  let arguments = (ins AnyFloat:$value, StrAttr:$function);
  let results = (outs AnyFloat:$result);
}

def Msl_MaximumOp : Msl_Op<"maximum", [Pure]> {
  let summary = "Carbon f32 maximum: NaN if either side is NaN, and +0 above -0";
  let arguments = (ins F32:$lhs, F32:$rhs);
  let results = (outs F32:$result);
}

def Msl_MinimumOp : Msl_Op<"minimum", [Pure]> {
  let summary = "Carbon f32 minimum: NaN if either side is NaN, and -0 below +0";
  let arguments = (ins F32:$lhs, F32:$rhs);
  let results = (outs F32:$result);
}

def Msl_FmaOp : Msl_Op<"fma", [Pure]> {
  let summary = "metal::fma: a * b + c rounded once";
  let arguments = (ins F32:$a, F32:$b, F32:$c);
  let results = (outs F32:$result);
}

def Msl_ConvertOp : Msl_Op<"convert", [Pure]> {
  let summary = "T(float(value)): numeric conversion through f32";
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
}

def Msl_GuardedIntegerOp : Msl_Op<"guarded_integer",
    [Pure, DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>]> {
  let summary = "u32 floordiv/mod returning 0 for a zero divisor, shl/shr returning 0 for shifts of 32 or more";
  let arguments = (ins I32:$lhs, I32:$rhs, StrAttr:$opcode);
  let results = (outs I32:$result);
}

def Msl_TensorMatmulOp : Msl_Op<"tensor_matmul", [MemoryEffects<[MemRead, MemWrite]>, AttrSizedOperandSegments]> {
  let summary = "matmul2d of two tensors, each (pointer, element offset, columns, rows, leading stride), into this "
                "thread's elements of a cooperative destination, or into a third tensor with no accumulator; every "
                "SIMD group of the threadgroup takes part";
  let arguments = (ins Variadic<AnyType>:$tensors, Variadic<AnyType>:$accumulator, I64Attr:$m, I64Attr:$n,
                       I64Attr:$groups, BoolAttr:$relaxed, BoolAttr:$transposeLeft, BoolAttr:$transposeRight);
  let results = (outs Variadic<AnyType>:$results);
}

def Msl_MatrixLoadOp : Msl_Op<"matrix_load", [MemoryEffects<[MemRead]>]> {
  let summary = "One SIMD group's 8x8 block at pointer[index], rows `stride` elements apart (columns when "
                "`transposed`), as this thread's elements of a SIMD-group matrix in the hardware's order";
  let arguments = (ins Msl_Pointer:$pointer, AnySignlessIntegerOrIndex:$index, I64Attr:$stride, BoolAttr:$transposed);
  let results = (outs Variadic<AnyType>:$results);
}

def Msl_MmaOp : Msl_Op<"mma", [NoMemoryEffect, AttrSizedOperandSegments]> {
  let summary = "One SIMD group's multiply-accumulate on this thread's elements: `simdgroup` builds 8x8 SIMD-group "
                "matrices from them, `tensor` cooperative tensors of a matmul2d of shape m, n, k";
  let arguments = (ins Variadic<AnyType>:$left, Variadic<AnyType>:$right, Variadic<AnyType>:$accumulator,
                       StrAttr:$unit, I64Attr:$m, I64Attr:$n, I64Attr:$k, BoolAttr:$relaxed);
  let results = (outs Variadic<AnyType>:$results);
}
