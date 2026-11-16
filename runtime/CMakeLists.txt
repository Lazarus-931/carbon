find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
set(CARBON_METAL_CPP_DIR "" CACHE PATH "Directory containing Metal/Metal.hpp")
find_path(CARBON_METAL_CPP_INCLUDE_DIR Metal/Metal.hpp
  HINTS "${CARBON_METAL_CPP_DIR}" ENV CARBON_METAL_CPP_DIR)
if(NOT CARBON_METAL_CPP_INCLUDE_DIR)
  message(FATAL_ERROR "Set CARBON_METAL_CPP_DIR to Apple's metal-cpp headers")
endif()

nanobind_add_module(_metal metal.cpp)
target_compile_features(_metal PRIVATE cxx_std_20)
target_compile_options(_metal PRIVATE -fblocks -Wall -Wextra -Wpedantic)
target_include_directories(_metal SYSTEM PRIVATE "${CARBON_METAL_CPP_INCLUDE_DIR}")
target_link_libraries(_metal PRIVATE ${FOUNDATION_FRAMEWORK} ${METAL_FRAMEWORK})
install(TARGETS _metal LIBRARY DESTINATION carbon)

