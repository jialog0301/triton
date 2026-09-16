#ifndef TRITON_THIRD_PARTY_VENTUS_TARGET_VENTUSARTIFACT_H
#define TRITON_THIRD_PARTY_VENTUS_TARGET_VENTUSARTIFACT_H

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace triton::ventus {

struct Identity {
  std::string revision;
  std::string contentHash;
  bool operator==(const Identity &) const;
};

struct ToolchainIdentity {
  std::string identity;
  Identity llvm, driver, spike, cycleSim, rtlSimulator;
  bool operator==(const ToolchainIdentity &) const;
};

struct ElfIdentity {
  std::string contentHash, entryPoint;
  uint32_t elfClass = 0;
  std::string endianness, machine;
  bool validated = false;
  bool operator==(const ElfIdentity &) const;
};

struct VentusKernelArgument {
  std::string kind;
  uint32_t binding = 0, offset = 0, size = 0, alignment = 0;
  static VentusKernelArgument buffer(uint32_t binding, uint32_t offset,
                                     uint32_t size, uint32_t alignment);
  static VentusKernelArgument scalarI32(uint32_t offset);
  bool operator==(const VentusKernelArgument &) const;
};

struct GridMetadata {
  uint32_t dimensions = 0;
  std::array<uint32_t, 3> globalSize{}, localSize{}, globalOffset{};
  bool oneCtaPerWorkGroup = false;
  std::string calculation;
  bool operator==(const GridMetadata &) const;
};

struct KernelConstraints {
  std::vector<std::string> requiredFeatures;
  std::string operation, dtype, layout;
  std::vector<uint32_t> shape;
  bool fullActiveWarp = false;
  std::string mmaProfileHash;
  bool operator==(const KernelConstraints &) const;
};

struct ResourceMetadata {
  uint64_t sharedBytes = 0, privateBytes = 0;
  uint64_t vgpr = 0, sgpr = 0, ldsBytes = 0, pdsBytes = 0;
  uint64_t vgprLimit = 0, sgprLimit = 0, ldsLimitBytes = 0, pdsLimitBytes = 0;
  bool rangeValidated = false;
  bool operator==(const ResourceMetadata &) const;
};

struct RawResourceRecord {
  uint32_t magic = 0;
  uint16_t formatVersion = 0, recordSize = 0;
  std::array<uint32_t, 4> values{};
  uint32_t parserVersion = 0;
  std::string source;
  bool targetEndianDecoded = false;
  bool operator==(const RawResourceRecord &) const;
};

struct ResourceUnits {
  std::string lds, pds, sgpr, vgpr;
  bool operator==(const ResourceUnits &) const;
};

struct CapabilityRecord {
  uint32_t version = 0;
  std::string identity, simulatorAddressConvention, rtlProfileHash;
  uint32_t completionContractVersion = 0;
  bool timeoutObservable = false, completionObservable = false,
       cacheFlushObservable = false;
  bool operator==(const CapabilityRecord &) const;
};

struct ArtifactReference {
  std::string kind, path, contentHash;
  bool operator==(const ArtifactReference &) const;
};

struct ToolInvocation {
  std::string executable;
  std::vector<std::string> argv;
  std::string stdoutText, stderrText;
  int32_t exitStatus = -1;
  std::string toolIdentity;
  bool operator==(const ToolInvocation &) const;
};

struct CompatibilityEvidence {
  std::string llvmIrPath, llvmIrHash;
  bool internalCheckPassed = false;
  std::string diagnostics;
  ToolInvocation opt, llc;
  std::string objectHash;
  bool operator==(const CompatibilityEvidence &) const;
};

struct LinkEvidence {
  ToolInvocation lld;
  std::vector<ArtifactReference> inputs;
  std::string elfHash;
  bool validated = false;
  bool operator==(const LinkEvidence &) const;
};

struct VentusKernelMetadata {
  uint32_t artifactAbiVersion = 0;
  std::string entryPoint, targetTriple, mcpu;
  uint32_t pointerWidth = 0, warpSize = 0;
  Identity rtlProfile;
  ToolchainIdentity toolchain;
  ElfIdentity elf;
  std::string callingConvention;
  std::vector<VentusKernelArgument> arguments;
  GridMetadata grid;
  KernelConstraints constraints;
  ResourceMetadata resources;
  RawResourceRecord rawResource;
  ResourceUnits resourceUnits;
  CapabilityRecord capability;
  std::vector<ArtifactReference> artifacts;
  std::vector<std::string> resultRecords;
  CompatibilityEvidence compatibility;
  LinkEvidence link;
  bool hardCodedResourceConsumptionRejected = false;
  bool operator==(const VentusKernelMetadata &) const;
};

struct ArtifactValidationContext {
  Identity rtlProfile;
  ToolchainIdentity toolchain;
  std::string capabilityIdentity;
};

bool validate(const VentusKernelMetadata &metadata,
              const ArtifactValidationContext &context);
std::optional<RawResourceRecord>
parseResourceRecord(const std::vector<uint8_t> &bytes, bool littleEndian);
std::string serialize(const VentusKernelMetadata &metadata);
std::optional<VentusKernelMetadata> deserialize(const std::string &json);

} // namespace triton::ventus

#endif
