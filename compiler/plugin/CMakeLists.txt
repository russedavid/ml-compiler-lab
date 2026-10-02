add_library(ml_compiler_lab_plugin STATIC Plugin.cpp)
target_include_directories(ml_compiler_lab_plugin PRIVATE
  ${IREE_SOURCE_DIR}/compiler/src
  ${IREE_BINARY_DIR}/compiler/src
  ${LLVM_SOURCE_DIR}/../mlir/include
  ${IREE_BINARY_DIR}/llvm-project/tools/mlir/include
  ${LLVM_BINARY_DIR}/include
  ${LLVM_SOURCE_DIR}/include)
target_link_libraries(ml_compiler_lab_plugin PUBLIC MLIRIR MLIRPass MLIRTransforms iree::compiler::PluginAPI)
target_compile_features(ml_compiler_lab_plugin PRIVATE cxx_std_17)
iree_compiler_register_plugin(PLUGIN_ID lab TARGET ml_compiler_lab_plugin)
if(NOT MSVC)
  target_compile_options(ml_compiler_lab_plugin PRIVATE -fno-rtti -fno-exceptions)
endif()
