find_package(MLIR CONFIG REQUIRED)
if(NOT LLVM_PACKAGE_VERSION VERSION_EQUAL "22.1.7")
  message(FATAL_ERROR "Carbon's compiler currently requires LLVM/MLIR 22.1.7")
endif()
get_filename_component(CARBON_LLVM_PREFIX "${MLIR_DIR}/../../.." REALPATH)
get_filename_component(CARBON_LLVM_CONFIG_PREFIX "${LLVM_DIR}/../../.." REALPATH)
if(NOT CARBON_LLVM_PREFIX STREQUAL CARBON_LLVM_CONFIG_PREFIX)
  message(FATAL_ERROR "LLVM_DIR and MLIR_DIR must use the same pinned distribution")
endif()
execute_process(
  COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/check_toolchain.py"
    --llvm "${CARBON_LLVM_PREFIX}" --cxx "${CMAKE_CXX_COMPILER}"
    --metal-cpp "${CARBON_METAL_CPP_INCLUDE_DIR}" --build-type "${CMAKE_BUILD_TYPE}"
    --output "${CMAKE_CURRENT_BINARY_DIR}/build-manifest.json"
  OUTPUT_VARIABLE CARBON_BUILD_ID OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/toolchain.json" CARBON_TOOLCHAIN_ID)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/toolchain.json" "${CMAKE_CURRENT_SOURCE_DIR}/check_toolchain.py")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}")
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_TARGET_DEFINITIONS tile/dialect.td)
mlir_tablegen(CarbonOps.h.inc -gen-op-decls)
mlir_tablegen(CarbonOps.cpp.inc -gen-op-defs)
mlir_tablegen(CarbonDialect.h.inc -gen-dialect-decls -dialect=carbon)
mlir_tablegen(CarbonDialect.cpp.inc -gen-dialect-defs -dialect=carbon)
mlir_tablegen(CarbonAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=carbon)
mlir_tablegen(CarbonAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=carbon)
mlir_tablegen(CarbonTypes.h.inc -gen-typedef-decls -typedefs-dialect=carbon)
mlir_tablegen(CarbonTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=carbon)
add_public_tablegen_target(CarbonIR)

set(LLVM_TARGET_DEFINITIONS thread/dialect.td)
mlir_tablegen(ThreadOps.h.inc -gen-op-decls)
mlir_tablegen(ThreadOps.cpp.inc -gen-op-defs)
mlir_tablegen(ThreadDialect.h.inc -gen-dialect-decls -dialect=msl)
mlir_tablegen(ThreadDialect.cpp.inc -gen-dialect-defs -dialect=msl)
mlir_tablegen(ThreadTypes.h.inc -gen-typedef-decls -typedefs-dialect=msl)
mlir_tablegen(ThreadTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=msl)
add_public_tablegen_target(CarbonThreadIR)

nanobind_add_module(_compiler
  tile/dialect.cpp tile/analysis.cpp tile/layout.cpp tile/memory.cpp tile/module.cpp
  thread/dialect.cpp thread/lowering.cpp thread/analysis.cpp thread/optimize.cpp thread/printer.cpp
  bindings/layouts.cpp bindings/builder.cpp bindings/module.cpp)
add_dependencies(_compiler CarbonIR CarbonThreadIR)
target_compile_features(_compiler PRIVATE cxx_std_17)
target_compile_definitions(_compiler PRIVATE
  CARBON_TOOLCHAIN_ID="${CARBON_TOOLCHAIN_ID}" CARBON_BUILD_ID="${CARBON_BUILD_ID}")
option(CARBON_STATIC_LLVM "Link the needed LLVM/MLIR components statically into the compiler module" ON)
if(CARBON_STATIC_LLVM)
  # The exported static targets name the LLVM and MLIR dylibs in their link interfaces, so walk them and
  # link archive files instead, skipping shared libraries.
  function(carbon_static_archives output)
    set(pending ${ARGN})
    set(seen "")
    set(archives "")
    while(pending)
      list(POP_FRONT pending item)
      string(REGEX REPLACE "^\\$<LINK_ONLY:(.*)>$" "\\1" item "${item}")
      if(item IN_LIST seen OR item STREQUAL "LLVM" OR item STREQUAL "MLIR" OR NOT TARGET "${item}")
        continue()
      endif()
      list(APPEND seen "${item}")
      get_target_property(kind "${item}" TYPE)
      if(kind STREQUAL "STATIC_LIBRARY")
        get_target_property(location "${item}" LOCATION)
        list(APPEND archives "${location}")
      endif()
      get_target_property(dependencies "${item}" INTERFACE_LINK_LIBRARIES)
      if(dependencies)
        list(APPEND pending ${dependencies})
      endif()
    endwhile()
    set(${output} ${archives} PARENT_SCOPE)
  endfunction()
  carbon_static_archives(CARBON_ARCHIVES
    MLIRParser MLIRBytecodeWriter MLIRPass MLIRTransforms MLIRTransformUtils MLIRArithDialect MLIRArithTransforms
    MLIRAnalysis
    MLIRFuncDialect MLIRSCFDialect LLVMSupport LLVMDemangle)
  target_link_libraries(_compiler PRIVATE ${CARBON_ARCHIVES} z)
  target_link_options(_compiler PRIVATE -Wl,-dead_strip)
else()
  target_link_libraries(_compiler PRIVATE MLIR LLVM)
  set_target_properties(_compiler PROPERTIES
    INSTALL_RPATH "${CARBON_LLVM_PREFIX}/lib" BUILD_WITH_INSTALL_RPATH ON)
  foreach(library MLIR LLVM)
    add_custom_command(TARGET _compiler POST_BUILD
      COMMAND /usr/bin/install_name_tool -change
        "/opt/homebrew/opt/llvm/lib/lib${library}.dylib"
        "${CARBON_LLVM_PREFIX}/lib/lib${library}.dylib" "$<TARGET_FILE:_compiler>")
  endforeach()
endif()
add_custom_command(TARGET _compiler POST_BUILD
  COMMAND /usr/bin/codesign --force --sign - "$<TARGET_FILE:_compiler>")
install(TARGETS _compiler LIBRARY DESTINATION carbon)
# Installation can strip the module after its build-time signature is applied.
install(CODE "
  execute_process(COMMAND /usr/bin/codesign --force --sign -
    \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/carbon/$<TARGET_FILE_NAME:_compiler>\"
    COMMAND_ERROR_IS_FATAL ANY)
")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/build-manifest.json" DESTINATION carbon)
