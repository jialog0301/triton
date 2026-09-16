add_subdirectory(include/Target)
add_subdirectory(lib/TritonVentusToLLVM)
add_subdirectory(lib/Target)

if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonVentus
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_ventus.cc
    LINK_LIBS VentusGPUToLLVM VentusArtifactImpl)
  target_include_directories(TritonVentus
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(TritonVentus PRIVATE Python3::Module nanobind-static)
  if(TARGET triton)
    target_sources(triton PRIVATE $<TARGET_OBJECTS:TritonVentus>)
    # The root build snapshots TRITON_PLUGINS before this subdirectory is
    # added, so the plugin's lowering library would never reach libtriton
    # through the global plugin list. Linking VentusGPUToLLVM by name here
    # expands its objects (and their MLIR/Triton link closure) into the
    # libtriton shared module.
    target_link_libraries(triton PRIVATE VentusGPUToLLVM VentusArtifactImpl)
  endif()
endif()

if(TRITON_BUILD_UT)
  add_executable(VentusArtifactRoundtrip test/Target/artifact-roundtrip.cpp)
  target_link_libraries(VentusArtifactRoundtrip PRIVATE VentusArtifactImpl)
  add_test(NAME VentusArtifactRoundtrip COMMAND VentusArtifactRoundtrip)
  add_dependencies(TritonUnitTests VentusArtifactRoundtrip)
endif()
