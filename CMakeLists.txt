cmake_minimum_required(VERSION 3.23)
project(carbon LANGUAGES C CXX)

if(NOT APPLE)
  message(FATAL_ERROR "The Metal host requires macOS")
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
find_package(nanobind CONFIG REQUIRED)
add_subdirectory(runtime)
option(CARBON_BUILD_COMPILER "Build the MLIR compiler slice" OFF)
if(CARBON_BUILD_COMPILER)
  add_subdirectory(compiler)
endif()
