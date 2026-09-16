#include "Target/VentusArtifact.h"

#include <iostream>
#include <string>

using namespace triton::ventus;

static bool check(bool condition, const char *message) {
  if (!condition)
    std::cerr << "artifact-roundtrip: " << message << '\n';
  return condition;
}

static VentusKernelMetadata makeMetadata() {
  VentusKernelMetadata metadata;
  metadata.artifactAbiVersion = 1;
  metadata.entryPoint = "vector_add";
  metadata.targetTriple = "riscv32";
  metadata.mcpu = "ventus-gpgpu";
  metadata.pointerWidth = 32;
  metadata.warpSize = 32;
  metadata.rtlProfile = {"rtl-v1", "rtl-hash"};
  metadata.toolchain = {"toolchain-v1",
                        {"llvm-rev", "llvm-hash"},
                        {"driver-rev", "driver-hash"},
                        {"spike-rev", "spike-hash"},
                        {"cyclesim-rev", "cyclesim-hash"},
                        {"rtl-sim-rev", "rtl-sim-hash"}};
  metadata.elf = {"elf-hash", "vector_add", 32, "little", "EM_RISCV", true};
  metadata.callingConvention = "ventus_kernel";
  metadata.arguments = {
      VentusKernelArgument::buffer(0, 0, 4, 4),
      VentusKernelArgument::buffer(1, 4, 4, 4),
      VentusKernelArgument::buffer(2, 8, 4, 4),
      VentusKernelArgument::scalarI32(12),
  };
  metadata.grid = {1, {32, 1, 1}, {32, 1, 1}, {0, 0, 0}, true,
                   "global_x=grid_x*local_x"};
  metadata.constraints = {{"masked_as1"}, "vector_add", "f32", "blocked",
                          {32}, true, ""};
  metadata.resources = {0, 0, 8, 12, 0, 0, 128, 128, 131072, 131072, true};
  metadata.rawResource = {0x53455256, 1, 24, {8, 12, 0, 0}, 1,
                          "Ventus LLVM AsmPrinter", true};
  metadata.resourceUnits = {"bytes_per_cta", "bytes_per_work_item",
                            "32_bit_slots_per_wavefront",
                            "wavefront_wide_slots_per_wavefront"};
  metadata.capability = {1, "capability-hash", "physical_device_values",
                         "rtl-hash", 1, true, true, true};
  metadata.artifacts = {{"ttir", "kernel.ttir", "ttir-hash"},
                        {"ttgir", "kernel.ttgir", "ttgir-hash"},
                        {"llvm_ir", "kernel.ventus.ll", "llvm-hash"},
                        {"assembly", "kernel.s", "asm-hash"},
                        {"elf", "kernel.elf", "elf-hash"},
                        {"launcher_input", "launcher.json", "launcher-hash"},
                        {"test_result", "result.json", "test-hash"}};
  metadata.resultRecords = {"fallback=none", "skip=none", "tolerance=exact"};
  metadata.compatibility = {
      "kernel.ventus.ll", "ll-hash", true, "checker=pass",
      {"/opt", {"/opt", "-passes=verify", "kernel.ventus.ll"}, "", "", 0,
       "opt-hash"},
      {"/llc", {"/llc", "-mcpu=ventus-gpgpu", "kernel.ventus.ll"}, "", "", 0,
       "llc-hash"},
      "object-hash"};
  metadata.link = {{"/ld.lld", {"/ld.lld", "-T", "/link.ld", "kernel.o"},
                    "", "", 0, "lld-hash"},
                   {{"linker_script", "/link.ld", "link-hash"},
                    {"crt0", "/crt0.o", "crt0-hash"},
                    {"libclc", "/riscv32clc.o", "libclc-hash"},
                    {"workitem", "/libworkitem.a", "workitem-hash"}},
                   "elf-hash", true};
  metadata.hardCodedResourceConsumptionRejected = true;
  return metadata;
}

int main() {
  const auto metadata = makeMetadata();
  const ArtifactValidationContext context = {
      metadata.rtlProfile, metadata.toolchain, metadata.capability.identity};
  const auto serialized = serialize(metadata);
  if (!check(serialized == serialize(metadata),
             "serialization is not deterministic"))
    return 1;

  auto parsed = deserialize(serialized);
  if (!check(parsed && *parsed == metadata, "round-trip changed metadata"))
    return 1;
  if (!check(validate(metadata, context), "trusted identity was rejected"))
    return 1;

  auto badContext = context;
  badContext.toolchain.llvm.contentHash = "foreign-llvm";
  if (!check(!validate(metadata, badContext), "foreign toolchain was accepted"))
    return 1;

  auto missing = metadata;
  missing.entryPoint.clear();
  if (!check(!validate(missing, context), "missing entry point was accepted"))
    return 1;

  missing = metadata;
  missing.compatibility.objectHash.clear();
  if (!check(!validate(missing, context), "missing object hash was accepted"))
    return 1;

  missing = metadata;
  missing.arguments.clear();
  if (!check(!validate(missing, context), "missing arguments were accepted"))
    return 1;

  missing = metadata;
  missing.toolchain.llvm.contentHash.clear();
  if (!check(!validate(missing, context), "missing toolchain identity was accepted"))
    return 1;

  auto badArguments = metadata;
  badArguments.arguments[1].offset = 2;
  if (!check(!validate(badArguments, context), "overlapping arguments were accepted"))
    return 1;

  auto badGeometry = metadata;
  badGeometry.grid.localSize = {48, 1, 1};
  if (!check(!validate(badGeometry, context), "illegal local size was accepted"))
    return 1;

  badGeometry = metadata;
  badGeometry.grid.globalSize = {33, 1, 1};
  if (!check(!validate(badGeometry, context), "inconsistent global size was accepted"))
    return 1;

  auto badAbi = metadata;
  badAbi.artifactAbiVersion = 2;
  if (!check(!validate(badAbi, context), "unknown ABI version was accepted"))
    return 1;

  auto badElf = metadata;
  badElf.elf.validated = false;
  if (!check(!validate(badElf, context), "invalid ELF identity was accepted"))
    return 1;

  auto overflow = metadata;
  overflow.resources.ldsBytes = overflow.resources.ldsLimitBytes + 1;
  if (!check(!validate(overflow, context), "resource overflow was accepted"))
    return 1;

  auto resourceMismatch = metadata;
  resourceMismatch.rawResource.values[0] += 1;
  if (!check(!validate(resourceMismatch, context), "raw resource mismatch was accepted"))
    return 1;

  resourceMismatch = metadata;
  resourceMismatch.resourceUnits.lds = "words_per_cta";
  if (!check(!validate(resourceMismatch, context), "invalid resource units were accepted"))
    return 1;

  auto failedGate = metadata;
  failedGate.compatibility.llc.exitStatus = 1;
  if (!check(!validate(failedGate, context), "failed llc gate was accepted"))
    return 1;

  auto missingLink = metadata;
  missingLink.link.validated = false;
  if (!check(!validate(missingLink, context), "missing link evidence was accepted"))
    return 1;

  auto mmaMismatch = metadata;
  mmaMismatch.constraints.requiredFeatures.push_back("VentusMmaProfileA");
  mmaMismatch.constraints.mmaProfileHash = "stale-profile";
  mmaMismatch.rtlProfile.contentHash = "rtl-hash";
  if (!check(!validate(mmaMismatch, context), "stale MMA profile was accepted"))
    return 1;

  mmaMismatch = metadata;
  mmaMismatch.constraints.requiredFeatures.push_back("VentusMmaProfileA");
  mmaMismatch.constraints.mmaProfileHash = "rtl-hash";
  mmaMismatch.grid.localSize = {64, 1, 1};
  if (!check(!validate(mmaMismatch, context), "64-lane MMA geometry was accepted"))
    return 1;

  mmaMismatch = metadata;
  mmaMismatch.constraints.requiredFeatures.push_back("VentusMmaProfileA");
  mmaMismatch.constraints.mmaProfileHash = "rtl-hash";
  mmaMismatch.capability.rtlProfileHash = "other-rtl";
  if (!check(!validate(mmaMismatch, context), "mismatched RTL capability was accepted"))
    return 1;

  auto malformed = serialized;
  const auto rawValue = malformed.find("\"values\": [");
  if (!check(rawValue != std::string::npos, "raw resource JSON not found"))
    return 1;
  const auto firstValue = malformed.find('8', rawValue);
  malformed.replace(firstValue, 1, "4294967296");
  if (!check(!deserialize(malformed), "overflowing uint32 was accepted"))
    return 1;

  const std::vector<uint8_t> resourceBytes = {
      0x56, 0x52, 0x45, 0x53, 0x01, 0x00, 0x18, 0x00,
      0x08, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  auto raw = parseResourceRecord(resourceBytes, true);
  if (!check(raw && raw->values == metadata.rawResource.values,
             "resource bytes did not parse"))
    return 1;
  auto truncated = resourceBytes;
  truncated.pop_back();
  if (!check(!parseResourceRecord(truncated, true),
             "truncated resource record was accepted"))
    return 1;
  auto unknownVersion = resourceBytes;
  unknownVersion[4] = 2;
  if (!check(!parseResourceRecord(unknownVersion, true),
             "unknown resource version was accepted"))
    return 1;
  if (!check(!parseResourceRecord(resourceBytes, false),
             "wrong-endian resource record was accepted"))
    return 1;

  return 0;
}
