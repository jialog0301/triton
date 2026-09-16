#include "Target/TritonVentusToLLVM.h"
#include "Target/VentusArtifact.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace py = nanobind;

using triton::ventus::ArtifactReference;
using triton::ventus::ArtifactValidationContext;
using triton::ventus::CapabilityRecord;
using triton::ventus::CompatibilityEvidence;
using triton::ventus::ElfIdentity;
using triton::ventus::GridMetadata;
using triton::ventus::Identity;
using triton::ventus::KernelConstraints;
using triton::ventus::LinkEvidence;
using triton::ventus::RawResourceRecord;
using triton::ventus::ResourceMetadata;
using triton::ventus::ResourceUnits;
using triton::ventus::ToolchainIdentity;
using triton::ventus::ToolInvocation;
using triton::ventus::VentusKernelArgument;
using triton::ventus::VentusKernelMetadata;

void init_triton_ventus(py::module_ &m) {
  m.doc() = "Python bindings to the Ventus backend";
  // V1 has no target dialect of its own; Triton/TritonGPU dialects already
  // registered by the core loader are sufficient for the lowering pipeline.
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    (void)context;
    return false;
  });

  auto passes = m.def_submodule("passes");
  passes.def("register_passes", []() { return true; });
  passes.def("add_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::ventus::createConvertTritonGPUToVentusLLVMPass());
  });

  m.def("finalize_llir", [](llvm::Module *mod) {
    mlir::triton::ventus::finalizeLLVMModule(*mod);
  });

  m.def("registration_status", []() {
    py::dict status;
    status[py::str("dialect")] = py::bool_(false);
    status[py::str("passes")] = py::bool_(true);
    status[py::str("reason")] = py::str(
        "V1 uses the core Triton/TritonGPU dialects; the TritonGPU-to-LLVM "
        "conversion pass is registered");
    return status;
  });

  //------------------------------------------------------------------------//
  // Versioned kernel artifact manifest (V1 schema, see VentusArtifact.h).
  // The Python backend emits this during the ELF stage and validates it with
  // `validate()` before the artifact is handed to the launcher/runtime.
  //------------------------------------------------------------------------//

  auto identity = py::class_<Identity>(m, "ArtifactIdentity")
      .def(py::init<>())
      .def_rw("revision", &Identity::revision)
      .def_rw("content_hash", &Identity::contentHash);

  auto toolchain = py::class_<ToolchainIdentity>(m, "ArtifactToolchainIdentity")
      .def(py::init<>())
      .def_rw("identity", &ToolchainIdentity::identity)
      .def_rw("llvm", &ToolchainIdentity::llvm)
      .def_rw("driver", &ToolchainIdentity::driver)
      .def_rw("spike", &ToolchainIdentity::spike)
      .def_rw("cycle_sim", &ToolchainIdentity::cycleSim)
      .def_rw("rtl_simulator", &ToolchainIdentity::rtlSimulator);

  auto elf = py::class_<ElfIdentity>(m, "ArtifactElfIdentity")
      .def(py::init<>())
      .def_rw("content_hash", &ElfIdentity::contentHash)
      .def_rw("entry_point", &ElfIdentity::entryPoint)
      .def_rw("elf_class", &ElfIdentity::elfClass)
      .def_rw("endianness", &ElfIdentity::endianness)
      .def_rw("machine", &ElfIdentity::machine)
      .def_rw("validated", &ElfIdentity::validated);

  auto argument = py::class_<VentusKernelArgument>(m, "ArtifactKernelArgument")
      .def(py::init<>())
      .def_rw("kind", &VentusKernelArgument::kind)
      .def_rw("binding", &VentusKernelArgument::binding)
      .def_rw("offset", &VentusKernelArgument::offset)
      .def_rw("size", &VentusKernelArgument::size)
      .def_rw("alignment", &VentusKernelArgument::alignment);

  auto grid = py::class_<GridMetadata>(m, "ArtifactGridMetadata")
      .def(py::init<>())
      .def_rw("dimensions", &GridMetadata::dimensions)
      .def_rw("global_size", &GridMetadata::globalSize)
      .def_rw("local_size", &GridMetadata::localSize)
      .def_rw("global_offset", &GridMetadata::globalOffset)
      .def_rw("one_cta_per_work_group", &GridMetadata::oneCtaPerWorkGroup)
      .def_rw("calculation", &GridMetadata::calculation);

  auto constraints = py::class_<KernelConstraints>(m, "ArtifactKernelConstraints")
      .def(py::init<>())
      .def_rw("required_features", &KernelConstraints::requiredFeatures)
      .def_rw("operation", &KernelConstraints::operation)
      .def_rw("dtype", &KernelConstraints::dtype)
      .def_rw("layout", &KernelConstraints::layout)
      .def_rw("shape", &KernelConstraints::shape)
      .def_rw("full_active_warp", &KernelConstraints::fullActiveWarp)
      .def_rw("mma_profile_hash", &KernelConstraints::mmaProfileHash);

  auto resources = py::class_<ResourceMetadata>(m, "ArtifactResourceMetadata")
      .def(py::init<>())
      .def_rw("shared_bytes", &ResourceMetadata::sharedBytes)
      .def_rw("private_bytes", &ResourceMetadata::privateBytes)
      .def_rw("vgpr", &ResourceMetadata::vgpr)
      .def_rw("sgpr", &ResourceMetadata::sgpr)
      .def_rw("lds_bytes", &ResourceMetadata::ldsBytes)
      .def_rw("pds_bytes", &ResourceMetadata::pdsBytes)
      .def_rw("vgpr_limit", &ResourceMetadata::vgprLimit)
      .def_rw("sgpr_limit", &ResourceMetadata::sgprLimit)
      .def_rw("lds_limit_bytes", &ResourceMetadata::ldsLimitBytes)
      .def_rw("pds_limit_bytes", &ResourceMetadata::pdsLimitBytes)
      .def_rw("range_validated", &ResourceMetadata::rangeValidated);

  auto raw = py::class_<RawResourceRecord>(m, "ArtifactRawResourceRecord")
      .def(py::init<>())
      .def_rw("magic", &RawResourceRecord::magic)
      .def_rw("format_version", &RawResourceRecord::formatVersion)
      .def_rw("record_size", &RawResourceRecord::recordSize)
      .def_rw("values", &RawResourceRecord::values)
      .def_rw("parser_version", &RawResourceRecord::parserVersion)
      .def_rw("source", &RawResourceRecord::source)
      .def_rw("target_endian_decoded", &RawResourceRecord::targetEndianDecoded);

  auto units = py::class_<ResourceUnits>(m, "ArtifactResourceUnits")
      .def(py::init<>())
      .def_rw("lds", &ResourceUnits::lds)
      .def_rw("pds", &ResourceUnits::pds)
      .def_rw("sgpr", &ResourceUnits::sgpr)
      .def_rw("vgpr", &ResourceUnits::vgpr);

  auto capability = py::class_<CapabilityRecord>(m, "ArtifactCapabilityRecord")
      .def(py::init<>())
      .def_rw("version", &CapabilityRecord::version)
      .def_rw("identity", &CapabilityRecord::identity)
      .def_rw("simulator_address_convention",
              &CapabilityRecord::simulatorAddressConvention)
      .def_rw("rtl_profile_hash", &CapabilityRecord::rtlProfileHash)
      .def_rw("completion_contract_version",
              &CapabilityRecord::completionContractVersion)
      .def_rw("timeout_observable", &CapabilityRecord::timeoutObservable)
      .def_rw("completion_observable", &CapabilityRecord::completionObservable)
      .def_rw("cache_flush_observable", &CapabilityRecord::cacheFlushObservable);

  auto artifact_ref =
      py::class_<ArtifactReference>(m, "ArtifactReference")
          .def(py::init<>())
          .def_rw("kind", &ArtifactReference::kind)
          .def_rw("path", &ArtifactReference::path)
          .def_rw("content_hash", &ArtifactReference::contentHash);

  auto invocation = py::class_<ToolInvocation>(m, "ArtifactToolInvocation")
      .def(py::init<>())
      .def_rw("executable", &ToolInvocation::executable)
      .def_rw("argv", &ToolInvocation::argv)
      .def_rw("stdout", &ToolInvocation::stdoutText)
      .def_rw("stderr", &ToolInvocation::stderrText)
      .def_rw("exit_status", &ToolInvocation::exitStatus)
      .def_rw("tool_identity", &ToolInvocation::toolIdentity);

  auto compatibility =
      py::class_<CompatibilityEvidence>(m, "ArtifactCompatibilityEvidence")
          .def(py::init<>())
          .def_rw("llvm_ir_path", &CompatibilityEvidence::llvmIrPath)
          .def_rw("llvm_ir_hash", &CompatibilityEvidence::llvmIrHash)
          .def_rw("internal_check_passed",
                  &CompatibilityEvidence::internalCheckPassed)
          .def_rw("diagnostics", &CompatibilityEvidence::diagnostics)
          .def_rw("opt", &CompatibilityEvidence::opt)
          .def_rw("llc", &CompatibilityEvidence::llc)
          .def_rw("object_hash", &CompatibilityEvidence::objectHash);

  auto link = py::class_<LinkEvidence>(m, "ArtifactLinkEvidence")
      .def(py::init<>())
      .def_rw("lld", &LinkEvidence::lld)
      .def_rw("inputs", &LinkEvidence::inputs)
      .def_rw("elf_hash", &LinkEvidence::elfHash)
      .def_rw("validated", &LinkEvidence::validated);

  py::class_<VentusKernelMetadata>(m, "VentusKernelMetadata")
      .def(py::init<>())
      .def_rw("artifact_abi_version", &VentusKernelMetadata::artifactAbiVersion)
      .def_rw("entry_point", &VentusKernelMetadata::entryPoint)
      .def_rw("target_triple", &VentusKernelMetadata::targetTriple)
      .def_rw("mcpu", &VentusKernelMetadata::mcpu)
      .def_rw("pointer_width", &VentusKernelMetadata::pointerWidth)
      .def_rw("warp_size", &VentusKernelMetadata::warpSize)
      .def_rw("rtl_profile", &VentusKernelMetadata::rtlProfile)
      .def_rw("toolchain", &VentusKernelMetadata::toolchain)
      .def_rw("elf", &VentusKernelMetadata::elf)
      .def_rw("calling_convention", &VentusKernelMetadata::callingConvention)
      .def_rw("arguments", &VentusKernelMetadata::arguments)
      .def_rw("grid", &VentusKernelMetadata::grid)
      .def_rw("constraints", &VentusKernelMetadata::constraints)
      .def_rw("resources", &VentusKernelMetadata::resources)
      .def_rw("raw_resource", &VentusKernelMetadata::rawResource)
      .def_rw("resource_units", &VentusKernelMetadata::resourceUnits)
      .def_rw("capability", &VentusKernelMetadata::capability)
      .def_rw("artifacts", &VentusKernelMetadata::artifacts)
      .def_rw("result_records", &VentusKernelMetadata::resultRecords)
      .def_rw("compatibility", &VentusKernelMetadata::compatibility)
      .def_rw("link", &VentusKernelMetadata::link)
      .def_rw("hard_coded_resource_consumption_rejected",
              &VentusKernelMetadata::hardCodedResourceConsumptionRejected)
      .def_static("serialize", [](const VentusKernelMetadata &metadata) {
        return triton::ventus::serialize(metadata);
      })
      .def_static("deserialize",
                  [](const std::string &json) {
                    return triton::ventus::deserialize(json);
                  })
      .def_static("validate",
                  [](const VentusKernelMetadata &metadata,
                     const ArtifactValidationContext &context) {
                    return triton::ventus::validate(metadata, context);
                  });

  auto validation_context =
      py::class_<ArtifactValidationContext>(m, "ArtifactValidationContext")
          .def(py::init<>())
          .def_rw("rtl_profile", &ArtifactValidationContext::rtlProfile)
          .def_rw("toolchain", &ArtifactValidationContext::toolchain)
          .def_rw("capability_identity",
                  &ArtifactValidationContext::capabilityIdentity);

  m.def("parse_resource_record",
        [](py::bytes record, bool littleEndian) {
          std::string data(record.c_str(), record.size());
          return triton::ventus::parseResourceRecord(
              std::vector<uint8_t>(data.begin(), data.end()), littleEndian);
        },
        py::arg("record"), py::arg("little_endian"));
}
