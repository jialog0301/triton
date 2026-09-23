add_triton_library(VentusGPUToLLVM
    Membar.cpp
    SelectToBranch.cpp
    StripPointerHints.cpp
    TritonGPUToLLVM.cpp
    VentusLoadStoreOpToLLVM.cpp

    LINK_LIBS PUBLIC
    TritonAnalysis
    TritonGPUToLLVM
    MLIRArithToLLVM
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
    MLIRGPUDialect
    MLIRMathToLLVM
    MLIRReconcileUnrealizedCasts
    MLIRUBToLLVM
)

target_include_directories(VentusGPUToLLVM
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
