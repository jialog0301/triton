#include "Target/VentusArtifact.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"

#include <tuple>
#include <limits>
#include <set>

using llvm::json::Array;
using llvm::json::Object;
using llvm::json::ObjectMapper;
using llvm::json::Path;
using llvm::json::Value;

namespace triton::ventus {

Array strings(const std::vector<std::string> &values) {
  Array result;
  for (const auto &value : values)
    result.push_back(value);
  return result;
}

Array integers(const std::vector<uint32_t> &values) {
  Array result;
  for (uint32_t value : values)
    result.push_back(value);
  return result;
}

Array triple(const std::array<uint32_t, 3> &values) {
  return Array{values[0], values[1], values[2]};
}

Array rawValues(const std::array<uint32_t, 4> &values) {
  return Array{values[0], values[1], values[2], values[3]};
}

template <typename T, size_t N>
bool mapFixedArray(const Value &value, std::array<T, N> &output, Path path) {
  auto *array = value.getAsArray();
  if (!array || array->size() != N)
    return path.report("expected fixed-size array"), false;
  for (size_t i = 0; i < N; ++i) {
    auto integer = (*array)[i].getAsInteger();
    if (!integer || *integer < 0 ||
        static_cast<uint64_t>(*integer) > std::numeric_limits<T>::max())
      return path.index(i).report("integer outside destination range"), false;
    output[i] = static_cast<T>(*integer);
  }
  return true;
}

Value toJSON(const Identity &v) {
  return Object{{"revision", v.revision}, {"content_hash", v.contentHash}};
}
bool fromJSON(const Value &v, Identity &o, Path p) {
  ObjectMapper m(v, p);
  return m && m.map("revision", o.revision) &&
         m.map("content_hash", o.contentHash);
}

Value toJSON(const ToolchainIdentity &v) {
  return Object{{"identity", v.identity}, {"llvm", toJSON(v.llvm)},
                {"driver", toJSON(v.driver)}, {"spike", toJSON(v.spike)},
                {"cyclesim", toJSON(v.cycleSim)},
                {"rtl_simulator", toJSON(v.rtlSimulator)}};
}
bool fromJSON(const Value &v, ToolchainIdentity &o, Path p) {
  ObjectMapper m(v, p);
  return m && m.map("identity", o.identity) && m.map("llvm", o.llvm) &&
         m.map("driver", o.driver) && m.map("spike", o.spike) &&
         m.map("cyclesim", o.cycleSim) &&
         m.map("rtl_simulator", o.rtlSimulator);
}

Value toJSON(const ElfIdentity &v) {
  return Object{{"content_hash", v.contentHash}, {"entry_point", v.entryPoint},
                {"class", v.elfClass},           {"endianness", v.endianness},
                {"machine", v.machine},          {"validated", v.validated}};
}
bool fromJSON(const Value &v, ElfIdentity &o, Path p) {
  ObjectMapper m(v, p);
  return m && m.map("content_hash", o.contentHash) &&
         m.map("entry_point", o.entryPoint) && m.map("class", o.elfClass) &&
         m.map("endianness", o.endianness) && m.map("machine", o.machine) &&
         m.map("validated", o.validated);
}

Value toJSON(const VentusKernelArgument &v) {
  return Object{{"kind", v.kind},       {"binding", v.binding},
                {"offset", v.offset},   {"size", v.size},
                {"alignment", v.alignment}};
}
bool fromJSON(const Value &v, VentusKernelArgument &o, Path p) {
  ObjectMapper m(v, p);
  return m && m.map("kind", o.kind) && m.map("binding", o.binding) &&
         m.map("offset", o.offset) && m.map("size", o.size) &&
         m.map("alignment", o.alignment);
}

Value toJSON(const GridMetadata &v) {
  return Object{{"dimensions", v.dimensions},
                {"global_size", triple(v.globalSize)},
                {"local_size", triple(v.localSize)},
                {"global_offset", triple(v.globalOffset)},
                {"one_cta_per_work_group", v.oneCtaPerWorkGroup},
                {"calculation", v.calculation}};
}
bool fromJSON(const Value &v, GridMetadata &o, Path p) {
  ObjectMapper m(v, p);
  const auto *object = v.getAsObject();
  const Value *global = object ? object->get("global_size") : nullptr;
  const Value *local = object ? object->get("local_size") : nullptr;
  const Value *offset = object ? object->get("global_offset") : nullptr;
  return m && global && local && offset && m.map("dimensions", o.dimensions) &&
         mapFixedArray(*global, o.globalSize, p.field("global_size")) &&
         mapFixedArray(*local, o.localSize, p.field("local_size")) &&
         mapFixedArray(*offset, o.globalOffset, p.field("global_offset")) &&
         m.map("one_cta_per_work_group", o.oneCtaPerWorkGroup) &&
         m.map("calculation", o.calculation);
}

Value toJSON(const KernelConstraints &v) {
  return Object{{"required_features", strings(v.requiredFeatures)},
                {"operation", v.operation}, {"dtype", v.dtype},
                {"layout", v.layout}, {"shape", integers(v.shape)},
                {"full_active_warp", v.fullActiveWarp},
                {"mma_profile_hash", v.mmaProfileHash}};
}
bool fromJSON(const Value &v, KernelConstraints &o, Path p) {
  ObjectMapper m(v, p);
  return m && m.map("required_features", o.requiredFeatures) &&
         m.map("operation", o.operation) && m.map("dtype", o.dtype) &&
         m.map("layout", o.layout) && m.map("shape", o.shape) &&
         m.map("full_active_warp", o.fullActiveWarp) &&
         m.map("mma_profile_hash", o.mmaProfileHash);
}

Value toJSON(const ResourceMetadata &v) {
  return Object{{"shared_bytes", v.sharedBytes}, {"private_bytes", v.privateBytes},
                {"vgpr", v.vgpr}, {"sgpr", v.sgpr}, {"lds_bytes", v.ldsBytes},
                {"pds_bytes", v.pdsBytes}, {"vgpr_limit", v.vgprLimit},
                {"sgpr_limit", v.sgprLimit}, {"lds_limit_bytes", v.ldsLimitBytes},
                {"pds_limit_bytes", v.pdsLimitBytes},
                {"range_validated", v.rangeValidated}};
}
bool fromJSON(const Value &v, ResourceMetadata &o, Path p) {
  ObjectMapper m(v, p);
  return m && m.map("shared_bytes", o.sharedBytes) &&
         m.map("private_bytes", o.privateBytes) && m.map("vgpr", o.vgpr) &&
         m.map("sgpr", o.sgpr) && m.map("lds_bytes", o.ldsBytes) &&
         m.map("pds_bytes", o.pdsBytes) && m.map("vgpr_limit", o.vgprLimit) &&
         m.map("sgpr_limit", o.sgprLimit) &&
         m.map("lds_limit_bytes", o.ldsLimitBytes) &&
         m.map("pds_limit_bytes", o.pdsLimitBytes) &&
         m.map("range_validated", o.rangeValidated);
}

Value toJSON(const RawResourceRecord &v) {
  return Object{{"magic", v.magic}, {"format_version", v.formatVersion},
                {"record_size", v.recordSize}, {"values", rawValues(v.values)},
                {"parser_version", v.parserVersion},
                {"source", v.source},
                {"target_endian_decoded", v.targetEndianDecoded}};
}
bool fromJSON(const Value &v, RawResourceRecord &o, Path p) {
  ObjectMapper m(v, p);
  const auto *object = v.getAsObject();
  const Value *values = object ? object->get("values") : nullptr;
  uint32_t formatVersion = 0, recordSize = 0;
  if (!m || !values || !m.map("magic", o.magic) ||
      !m.map("format_version", formatVersion) ||
      !m.map("record_size", recordSize) ||
      formatVersion > std::numeric_limits<uint16_t>::max() ||
      recordSize > std::numeric_limits<uint16_t>::max() ||
      !mapFixedArray(*values, o.values, p.field("values")) ||
      !m.map("parser_version", o.parserVersion) || !m.map("source", o.source) ||
      !m.map("target_endian_decoded", o.targetEndianDecoded))
    return false;
  o.formatVersion = static_cast<uint16_t>(formatVersion);
  o.recordSize = static_cast<uint16_t>(recordSize);
  return true;
}

Value toJSON(const ResourceUnits &v) {
  return Object{{"lds", v.lds}, {"pds", v.pds}, {"sgpr", v.sgpr}, {"vgpr", v.vgpr}};
}
bool fromJSON(const Value &v, ResourceUnits &o, Path p) {
  ObjectMapper m(v, p);
  return m && m.map("lds", o.lds) && m.map("pds", o.pds) &&
         m.map("sgpr", o.sgpr) && m.map("vgpr", o.vgpr);
}

Value toJSON(const CapabilityRecord &v) {
  return Object{{"version", v.version}, {"identity", v.identity},
                {"simulator_address_convention", v.simulatorAddressConvention},
                {"rtl_profile_hash", v.rtlProfileHash},
                {"completion_contract_version", v.completionContractVersion},
                {"timeout_observable", v.timeoutObservable},
                {"completion_observable", v.completionObservable},
                {"cache_flush_observable", v.cacheFlushObservable}};
}
bool fromJSON(const Value &v, CapabilityRecord &o, Path p) {
  ObjectMapper m(v, p);
  return m && m.map("version", o.version) && m.map("identity", o.identity) &&
         m.map("simulator_address_convention", o.simulatorAddressConvention) &&
         m.map("rtl_profile_hash", o.rtlProfileHash) &&
         m.map("completion_contract_version", o.completionContractVersion) &&
         m.map("timeout_observable", o.timeoutObservable) &&
         m.map("completion_observable", o.completionObservable) &&
         m.map("cache_flush_observable", o.cacheFlushObservable);
}

Value toJSON(const ArtifactReference &v) {
  return Object{{"kind", v.kind}, {"path", v.path},
                {"content_hash", v.contentHash}};
}
bool fromJSON(const Value &v, ArtifactReference &o, Path p) {
  ObjectMapper m(v, p);
  return m && m.map("kind", o.kind) && m.map("path", o.path) &&
         m.map("content_hash", o.contentHash);
}

Value toJSON(const ToolInvocation &v) {
  return Object{{"executable", v.executable}, {"argv", strings(v.argv)},
                {"stdout", v.stdoutText}, {"stderr", v.stderrText},
                {"exit_status", v.exitStatus}, {"tool_identity", v.toolIdentity}};
}
bool fromJSON(const Value &v, ToolInvocation &o, Path p) {
  ObjectMapper m(v, p);
  return m && m.map("executable", o.executable) && m.map("argv", o.argv) &&
         m.map("stdout", o.stdoutText) && m.map("stderr", o.stderrText) &&
         m.map("exit_status", o.exitStatus) &&
         m.map("tool_identity", o.toolIdentity);
}

Value toJSON(const CompatibilityEvidence &v) {
  return Object{{"llvm_ir_path", v.llvmIrPath}, {"llvm_ir_hash", v.llvmIrHash},
                {"internal_check_passed", v.internalCheckPassed},
                {"diagnostics", v.diagnostics}, {"opt", toJSON(v.opt)},
                {"llc", toJSON(v.llc)}, {"object_hash", v.objectHash}};
}
bool fromJSON(const Value &v, CompatibilityEvidence &o, Path p) {
  ObjectMapper m(v, p);
  return m && m.map("llvm_ir_path", o.llvmIrPath) &&
         m.map("llvm_ir_hash", o.llvmIrHash) &&
         m.map("internal_check_passed", o.internalCheckPassed) &&
         m.map("diagnostics", o.diagnostics) && m.map("opt", o.opt) &&
         m.map("llc", o.llc) && m.map("object_hash", o.objectHash);
}

Value toJSON(const LinkEvidence &v) {
  Array inputs;
  for (const auto &input : v.inputs)
    inputs.push_back(toJSON(input));
  return Object{{"lld", toJSON(v.lld)}, {"inputs", std::move(inputs)},
                {"elf_hash", v.elfHash}, {"validated", v.validated}};
}
bool fromJSON(const Value &v, LinkEvidence &o, Path p) {
  ObjectMapper m(v, p);
  return m && m.map("lld", o.lld) && m.map("inputs", o.inputs) &&
         m.map("elf_hash", o.elfHash) && m.map("validated", o.validated);
}

Value toJSON(const VentusKernelMetadata &v) {
  Array arguments, artifacts;
  for (const auto &argument : v.arguments)
    arguments.push_back(toJSON(argument));
  for (const auto &artifact : v.artifacts)
    artifacts.push_back(toJSON(artifact));
  return Object{
      {"artifact_abi_version", v.artifactAbiVersion},
      {"entry_point", v.entryPoint}, {"target_triple", v.targetTriple},
      {"mcpu", v.mcpu}, {"pointer_width", v.pointerWidth},
      {"warp_size", v.warpSize}, {"rtl_profile", toJSON(v.rtlProfile)},
      {"toolchain", toJSON(v.toolchain)}, {"elf", toJSON(v.elf)},
      {"calling_convention", v.callingConvention},
      {"arguments", std::move(arguments)}, {"grid", toJSON(v.grid)},
      {"constraints", toJSON(v.constraints)}, {"resources", toJSON(v.resources)},
      {"raw_resource", toJSON(v.rawResource)},
      {"resource_units", toJSON(v.resourceUnits)},
      {"capability", toJSON(v.capability)}, {"artifacts", std::move(artifacts)},
      {"result_records", strings(v.resultRecords)},
      {"compatibility", toJSON(v.compatibility)},
      {"link", toJSON(v.link)},
      {"hard_coded_resource_consumption_rejected",
       v.hardCodedResourceConsumptionRejected}};
}
bool fromJSON(const Value &v, VentusKernelMetadata &o, Path p) {
  ObjectMapper m(v, p);
  return m && m.map("artifact_abi_version", o.artifactAbiVersion) &&
         m.map("entry_point", o.entryPoint) &&
         m.map("target_triple", o.targetTriple) && m.map("mcpu", o.mcpu) &&
         m.map("pointer_width", o.pointerWidth) && m.map("warp_size", o.warpSize) &&
         m.map("rtl_profile", o.rtlProfile) && m.map("toolchain", o.toolchain) &&
         m.map("elf", o.elf) && m.map("calling_convention", o.callingConvention) &&
         m.map("arguments", o.arguments) && m.map("grid", o.grid) &&
         m.map("constraints", o.constraints) && m.map("resources", o.resources) &&
         m.map("raw_resource", o.rawResource) &&
         m.map("resource_units", o.resourceUnits) &&
         m.map("capability", o.capability) && m.map("artifacts", o.artifacts) &&
         m.map("result_records", o.resultRecords) &&
         m.map("compatibility", o.compatibility) &&
         m.map("link", o.link) &&
         m.map("hard_coded_resource_consumption_rejected",
               o.hardCodedResourceConsumptionRejected);
}

#define VENTUS_EQ(Type, Lhs, Rhs)                                             \
  bool Type::operator==(const Type &o) const {                                \
    return std::tie Lhs == std::tie Rhs;                                      \
  }

VENTUS_EQ(Identity, (revision, contentHash), (o.revision, o.contentHash))
VENTUS_EQ(ToolchainIdentity,
          (identity, llvm, driver, spike, cycleSim, rtlSimulator),
          (o.identity, o.llvm, o.driver, o.spike, o.cycleSim, o.rtlSimulator))
VENTUS_EQ(ElfIdentity,
          (contentHash, entryPoint, elfClass, endianness, machine, validated),
          (o.contentHash, o.entryPoint, o.elfClass, o.endianness, o.machine,
           o.validated))
VENTUS_EQ(VentusKernelArgument, (kind, binding, offset, size, alignment),
          (o.kind, o.binding, o.offset, o.size, o.alignment))
VENTUS_EQ(GridMetadata,
          (dimensions, globalSize, localSize, globalOffset, oneCtaPerWorkGroup,
           calculation),
          (o.dimensions, o.globalSize, o.localSize, o.globalOffset,
           o.oneCtaPerWorkGroup, o.calculation))
VENTUS_EQ(KernelConstraints,
          (requiredFeatures, operation, dtype, layout, shape, fullActiveWarp,
           mmaProfileHash),
          (o.requiredFeatures, o.operation, o.dtype, o.layout, o.shape,
           o.fullActiveWarp, o.mmaProfileHash))
VENTUS_EQ(ResourceMetadata,
          (sharedBytes, privateBytes, vgpr, sgpr, ldsBytes, pdsBytes, vgprLimit,
           sgprLimit, ldsLimitBytes, pdsLimitBytes, rangeValidated),
          (o.sharedBytes, o.privateBytes, o.vgpr, o.sgpr, o.ldsBytes, o.pdsBytes,
           o.vgprLimit, o.sgprLimit, o.ldsLimitBytes, o.pdsLimitBytes,
           o.rangeValidated))
VENTUS_EQ(RawResourceRecord,
          (magic, formatVersion, recordSize, values, parserVersion, source,
           targetEndianDecoded),
          (o.magic, o.formatVersion, o.recordSize, o.values, o.parserVersion,
           o.source, o.targetEndianDecoded))
VENTUS_EQ(ResourceUnits, (lds, pds, sgpr, vgpr),
          (o.lds, o.pds, o.sgpr, o.vgpr))
VENTUS_EQ(CapabilityRecord,
          (version, identity, simulatorAddressConvention, rtlProfileHash,
           completionContractVersion, timeoutObservable, completionObservable,
           cacheFlushObservable),
          (o.version, o.identity, o.simulatorAddressConvention, o.rtlProfileHash,
           o.completionContractVersion, o.timeoutObservable,
           o.completionObservable, o.cacheFlushObservable))
VENTUS_EQ(ArtifactReference, (kind, path, contentHash),
          (o.kind, o.path, o.contentHash))
VENTUS_EQ(ToolInvocation,
          (executable, argv, stdoutText, stderrText, exitStatus, toolIdentity),
          (o.executable, o.argv, o.stdoutText, o.stderrText, o.exitStatus,
           o.toolIdentity))
VENTUS_EQ(CompatibilityEvidence,
          (llvmIrPath, llvmIrHash, internalCheckPassed, diagnostics, opt, llc,
           objectHash),
          (o.llvmIrPath, o.llvmIrHash, o.internalCheckPassed, o.diagnostics,
           o.opt, o.llc, o.objectHash))
VENTUS_EQ(LinkEvidence, (lld, inputs, elfHash, validated),
          (o.lld, o.inputs, o.elfHash, o.validated))

#undef VENTUS_EQ

VentusKernelArgument VentusKernelArgument::buffer(uint32_t binding,
                                                  uint32_t offset,
                                                  uint32_t size,
                                                  uint32_t alignment) {
  return {"buffer", binding, offset, size, alignment};
}
VentusKernelArgument VentusKernelArgument::scalarI32(uint32_t offset) {
  return {"i32", 0, offset, 4, 4};
}

bool VentusKernelMetadata::operator==(const VentusKernelMetadata &o) const {
  return std::tie(artifactAbiVersion, entryPoint, targetTriple, mcpu,
                  pointerWidth, warpSize, rtlProfile, toolchain, elf,
                  callingConvention, arguments, grid, constraints, resources,
                  rawResource, resourceUnits, capability, artifacts,
                  resultRecords, compatibility, link,
                  hardCodedResourceConsumptionRejected) ==
         std::tie(o.artifactAbiVersion, o.entryPoint, o.targetTriple, o.mcpu,
                  o.pointerWidth, o.warpSize, o.rtlProfile, o.toolchain, o.elf,
                  o.callingConvention, o.arguments, o.grid, o.constraints,
                  o.resources, o.rawResource, o.resourceUnits, o.capability,
                  o.artifacts, o.resultRecords, o.compatibility, o.link,
                  o.hardCodedResourceConsumptionRejected);
}

namespace {
bool complete(const Identity &identity) {
  return !identity.revision.empty() && !identity.contentHash.empty();
}

bool successful(const ToolInvocation &tool) {
  return !tool.executable.empty() && tool.executable.front() == '/' &&
         !tool.argv.empty() && tool.argv.front() == tool.executable &&
         tool.exitStatus == 0 && !tool.toolIdentity.empty();
}

std::optional<std::string> validationError(const VentusKernelMetadata &m) {
  if (m.artifactAbiVersion != 1)
    return "unsupported artifact ABI version";
  if (m.entryPoint.empty() || m.targetTriple != "riscv32" ||
      m.mcpu != "ventus-gpgpu" || m.pointerWidth != 32 || m.warpSize != 32)
    return "missing or invalid target identity";
  if (!m.elf.validated || m.elf.contentHash.empty() ||
      m.elf.entryPoint != m.entryPoint || m.elf.elfClass != 32 ||
      m.elf.endianness != "little" || m.elf.machine != "EM_RISCV")
    return "invalid ELF identity";
  if (m.callingConvention != "ventus_kernel")
    return "invalid calling convention";
  if (!complete(m.rtlProfile) || m.toolchain.identity.empty() ||
      !complete(m.toolchain.llvm) || !complete(m.toolchain.driver) ||
      !complete(m.toolchain.spike) || !complete(m.toolchain.cycleSim) ||
      !complete(m.toolchain.rtlSimulator))
    return "missing toolchain or RTL identity";
  if (m.grid.dimensions != 1 || !m.grid.oneCtaPerWorkGroup ||
      (m.grid.localSize != std::array<uint32_t, 3>{32, 1, 1} &&
       m.grid.localSize != std::array<uint32_t, 3>{64, 1, 1}) ||
      m.grid.globalSize[0] == 0 ||
      m.grid.globalSize[0] % m.grid.localSize[0] != 0 ||
      m.grid.globalSize[1] != 1 || m.grid.globalSize[2] != 1 ||
      m.grid.globalOffset != std::array<uint32_t, 3>{0, 0, 0})
    return "invalid V1 launch geometry";
  if (m.grid.calculation.empty() || m.constraints.operation.empty() ||
      m.constraints.dtype.empty() || m.constraints.layout.empty() ||
      m.constraints.shape.empty())
    return "missing geometry or operation constraints";
  if (m.arguments.empty())
    return "missing packed arguments";
  uint64_t packedEnd = 0;
  std::set<uint32_t> bufferBindings;
  for (const auto &argument : m.arguments) {
    if ((argument.kind != "buffer" && argument.kind != "i32") ||
        argument.size == 0 || argument.alignment == 0 ||
        argument.offset % argument.alignment != 0 ||
        argument.offset < packedEnd)
      return "invalid packed argument layout";
    if (argument.kind == "buffer" &&
        !bufferBindings.insert(argument.binding).second)
      return "duplicate buffer binding";
    packedEnd = static_cast<uint64_t>(argument.offset) + argument.size;
  }
  if (!m.resources.rangeValidated || m.resources.vgpr > m.resources.vgprLimit ||
      m.resources.sgpr > m.resources.sgprLimit ||
      m.resources.ldsBytes > m.resources.ldsLimitBytes ||
      m.resources.pdsBytes > m.resources.pdsLimitBytes)
    return "resource range validation failed";
  if (m.rawResource.magic != 0x53455256 || m.rawResource.formatVersion != 1 ||
      m.rawResource.recordSize != 24 || m.rawResource.parserVersion == 0 ||
      m.rawResource.source.empty() ||
      !m.rawResource.targetEndianDecoded || m.resourceUnits.lds.empty() ||
      m.resourceUnits.pds.empty() || m.resourceUnits.sgpr.empty() ||
      m.resourceUnits.vgpr.empty())
    return "missing resource parser or unit contract";
  if (m.resources.vgpr > std::numeric_limits<uint32_t>::max() ||
      m.resources.sgpr > std::numeric_limits<uint32_t>::max() ||
      m.resources.ldsBytes > std::numeric_limits<uint32_t>::max() ||
      m.resources.pdsBytes > std::numeric_limits<uint32_t>::max())
    return "normalized resource value exceeds VRES v1 field width";
  if (m.rawResource.values !=
          std::array<uint32_t, 4>{static_cast<uint32_t>(m.resources.vgpr),
                                  static_cast<uint32_t>(m.resources.sgpr),
                                  static_cast<uint32_t>(m.resources.ldsBytes),
                                  static_cast<uint32_t>(m.resources.pdsBytes)} ||
      m.resourceUnits.lds != "bytes_per_cta" ||
      m.resourceUnits.pds != "bytes_per_work_item" ||
      m.resourceUnits.sgpr != "32_bit_slots_per_wavefront" ||
      m.resourceUnits.vgpr != "wavefront_wide_slots_per_wavefront" ||
      m.resources.sharedBytes > m.resources.ldsBytes ||
      m.resources.privateBytes > m.resources.pdsBytes)
    return "resource record and normalized metadata mismatch";
  if (m.capability.version == 0 || m.capability.identity.empty() ||
      m.capability.simulatorAddressConvention.empty() ||
      m.capability.completionContractVersion == 0 ||
      !m.capability.timeoutObservable || !m.capability.completionObservable ||
      !m.capability.cacheFlushObservable ||
      !m.hardCodedResourceConsumptionRejected)
    return "invalid capability or resource-consumption contract";
  const std::set<std::string> requiredArtifacts = {
      "ttir", "ttgir", "llvm_ir", "assembly", "elf", "launcher_input",
      "test_result"};
  std::set<std::string> observedArtifacts;
  for (const auto &artifact : m.artifacts) {
    if (artifact.kind.empty() || artifact.path.empty() ||
        artifact.contentHash.empty() ||
        !observedArtifacts.insert(artifact.kind).second)
      return "invalid artifact reference";
  }
  if (observedArtifacts != requiredArtifacts || m.resultRecords.empty())
    return "missing artifact or result records";
  if (!m.compatibility.internalCheckPassed ||
      m.compatibility.llvmIrPath.empty() || m.compatibility.llvmIrHash.empty() ||
      !successful(m.compatibility.opt) || !successful(m.compatibility.llc) ||
      m.compatibility.objectHash.empty())
    return "missing compatibility evidence";
  const std::set<std::string> requiredLinkInputs = {
      "linker_script", "crt0", "libclc", "workitem"};
  std::set<std::string> observedLinkInputs;
  for (const auto &input : m.link.inputs) {
    if (input.kind.empty() || input.path.empty() || input.path.front() != '/' ||
        input.contentHash.empty() || !observedLinkInputs.insert(input.kind).second)
      return "invalid link input";
  }
  if (!m.link.validated || !successful(m.link.lld) ||
      observedLinkInputs != requiredLinkInputs ||
      m.link.elfHash != m.elf.contentHash)
    return "missing or invalid link evidence";
  bool requiresMma = false;
  for (const auto &feature : m.constraints.requiredFeatures)
    requiresMma |= feature == "VentusMmaProfileA";
  if (requiresMma &&
      (m.grid.localSize != std::array<uint32_t, 3>{32, 1, 1} ||
       !m.constraints.fullActiveWarp || m.constraints.mmaProfileHash.empty() ||
       m.constraints.mmaProfileHash != m.rtlProfile.contentHash ||
       m.capability.rtlProfileHash != m.rtlProfile.contentHash))
    return "MMA simulator/profile identity mismatch";
  return std::nullopt;
}
} // namespace

bool validate(const VentusKernelMetadata &metadata,
              const ArtifactValidationContext &context) {
  return !validationError(metadata) && metadata.rtlProfile == context.rtlProfile &&
         metadata.toolchain == context.toolchain &&
         metadata.capability.identity == context.capabilityIdentity;
}

std::optional<RawResourceRecord>
parseResourceRecord(const std::vector<uint8_t> &bytes, bool littleEndian) {
  if (!littleEndian || bytes.size() != 24)
    return std::nullopt;
  auto read16 = [&](size_t offset) {
    return static_cast<uint16_t>(bytes[offset]) |
           (static_cast<uint16_t>(bytes[offset + 1]) << 8);
  };
  auto read32 = [&](size_t offset) {
    return static_cast<uint32_t>(bytes[offset]) |
           (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
           (static_cast<uint32_t>(bytes[offset + 3]) << 24);
  };
  RawResourceRecord record;
  record.magic = read32(0);
  record.formatVersion = read16(4);
  record.recordSize = read16(6);
  if (record.magic != 0x53455256 || record.formatVersion != 1 ||
      record.recordSize != bytes.size())
    return std::nullopt;
  for (size_t i = 0; i < record.values.size(); ++i)
    record.values[i] = read32(8 + i * 4);
  record.parserVersion = 1;
  record.source = "VRES v1 ELF section";
  record.targetEndianDecoded = true;
  return record;
}

std::string serialize(const VentusKernelMetadata &metadata) {
  return llvm::formatv("{0:2}", toJSON(metadata)).str();
}

std::optional<VentusKernelMetadata> deserialize(const std::string &json) {
  auto value = llvm::json::parse(json);
  if (!value)
    return std::nullopt;
  VentusKernelMetadata metadata;
  Path::Root root("VentusKernelMetadata");
  if (!fromJSON(*value, metadata, root) || validationError(metadata))
    return std::nullopt;
  return metadata;
}

} // namespace triton::ventus
