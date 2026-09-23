add_triton_library(VentusGPUToLLVM
    BoundedDivergence.cpp
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
    MLIRSCFDialect
    MLIRUBToLLVM
)

target_include_directories(VentusGPUToLLVM
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
