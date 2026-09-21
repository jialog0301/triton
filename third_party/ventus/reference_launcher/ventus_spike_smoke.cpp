#include "ventus.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

constexpr uint64_t kKernelRunBase = 0x80000000;
constexpr size_t kMetadataWords = 14;

struct MetaData {
  uint64_t kernel_id;
  uint64_t kernel_size[3];
  uint64_t wf_size;
  uint64_t wg_size;
  uint64_t metaDataBaseAddr;
  uint64_t ldsSize;
  uint64_t pdsSize;
  uint64_t sgprUsage;
  uint64_t vgprUsage;
  uint64_t pdsBaseAddr;
};

struct BufferSpec {
  std::string path;
  uint64_t size = 0;
};

struct Config {
  std::string elf;
  uint64_t entry = 0;
  uint64_t grid[3] = {0, 0, 0};
  uint64_t local[3] = {0, 0, 0};
  uint64_t lds = 0;
  uint64_t pds = 0;
  uint64_t sgpr = 32;
  uint64_t vgpr = 32;
  uint64_t timeout_ms = 60000;
  std::string log;
  std::string args_log;
  std::vector<BufferSpec> inputs;
  BufferSpec output;
  BufferSpec expected;
};

void usage(const char *argv0) {
  std::cout << "usage: " << argv0
            << " --elf PATH --entry ADDR --grid X,Y,Z --local 16,1,1"
               " --lds-size BYTES --pds-size BYTES"
               " --input PATH:BYTES [--input PATH:BYTES ...]"
               " --output PATH:BYTES --expected PATH:BYTES"
               " [--sgpr N] [--vgpr N] [--timeout-ms N]"
               " [--log PATH] [--args-log PATH]\n"
            << "legacy-8x2 launch shape only (8 lanes/warp x 2 warps).\n"
               "For V1 Triton kernels use third_party/ventus/backend/"
               "launcher.py with --profile v1-32 or v1-64.\n";
}

uint64_t parse_u64(const std::string &value, const char *name) {
  size_t consumed = 0;
  int base = value.compare(0, 2, "0x") == 0 || value.compare(0, 2, "0X") == 0
                 ? 16
                 : 10;
  uint64_t result = std::stoull(value, &consumed, base);
  if (consumed != value.size())
    throw std::runtime_error(std::string("invalid ") + name + ": " + value);
  return result;
}

std::array<uint64_t, 3> parse_triplet(const std::string &value,
                                      const char *name) {
  std::array<uint64_t, 3> result{};
  std::stringstream stream(value);
  std::string part;
  for (size_t i = 0; i < result.size(); ++i) {
    if (!std::getline(stream, part, ','))
      throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    result[i] = parse_u64(part, name);
  }
  if (std::getline(stream, part, ','))
    throw std::runtime_error(std::string("invalid ") + name + ": " + value);
  return result;
}

BufferSpec parse_buffer(const std::string &value, const char *name) {
  const size_t separator = value.rfind(':');
  if (separator == std::string::npos || separator == 0 ||
      separator + 1 == value.size())
    throw std::runtime_error(std::string("invalid ") + name + ": " + value);
  return {value.substr(0, separator),
          parse_u64(value.substr(separator + 1), "buffer size")};
}

Config parse_args(int argc, char **argv) {
  Config config;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    auto value = [&](const char *name) {
      if (i + 1 >= argc)
        throw std::runtime_error(std::string("missing value for ") + name);
      return std::string(argv[++i]);
    };
    if (option == "--help" || option == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else if (option == "--elf") {
      config.elf = value("--elf");
    } else if (option == "--entry") {
      config.entry = parse_u64(value("--entry"), "entry");
    } else if (option == "--grid") {
      auto triplet = parse_triplet(value("--grid"), "grid");
      std::copy(triplet.begin(), triplet.end(), config.grid);
    } else if (option == "--local") {
      auto triplet = parse_triplet(value("--local"), "local");
      std::copy(triplet.begin(), triplet.end(), config.local);
    } else if (option == "--lds-size") {
      config.lds = parse_u64(value("--lds-size"), "lds-size");
    } else if (option == "--pds-size") {
      config.pds = parse_u64(value("--pds-size"), "pds-size");
    } else if (option == "--sgpr") {
      config.sgpr = parse_u64(value("--sgpr"), "sgpr");
    } else if (option == "--vgpr") {
      config.vgpr = parse_u64(value("--vgpr"), "vgpr");
    } else if (option == "--input") {
      config.inputs.push_back(parse_buffer(value("--input"), "input"));
    } else if (option == "--output") {
      config.output = parse_buffer(value("--output"), "output");
    } else if (option == "--expected") {
      config.expected = parse_buffer(value("--expected"), "expected");
    } else if (option == "--timeout-ms") {
      config.timeout_ms = parse_u64(value("--timeout-ms"), "timeout-ms");
    } else if (option == "--log") {
      config.log = value("--log");
    } else if (option == "--args-log") {
      config.args_log = value("--args-log");
    } else {
      throw std::runtime_error("unknown option: " + option);
    }
  }
  if (config.elf.empty() || config.entry == 0 || config.grid[0] == 0 ||
      config.local[0] == 0 || config.lds == 0 || config.pds == 0 ||
      config.inputs.empty() || config.output.path.empty() ||
      config.expected.path.empty())
    throw std::runtime_error("missing required launcher option");
  if (config.entry > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("entry is not a valid RV32 address");
  // This tool implements the legacy Shape: 8 lanes per warp. Its
  // `warp_count` is derived as `local[0] / 8`, so accepting a 32- or 64-lane
  // local size would launch the same lane count split into more, smaller
  // warps than a V1 kernel was compiled for -- a silent divergence that only
  // shows up as an output mismatch. V1 kernels belong on the Python
  // reference launcher (`third_party/ventus/backend/launcher.py`), which
  // selects a named launch profile and reads the kernel's own resource
  // record.
  if (config.local[0] != 16)
    throw std::runtime_error(
        "this legacy smoke tool launches 8-lane warps and supports only "
        "--local 16,1,1; use third_party/ventus/backend/launcher.py with a "
        "named profile (v1-32/v1-64) for V1 Triton kernels");
  if (config.local[1] != 1 || config.local[2] != 1)
    throw std::runtime_error("local must be 16,1,1");
  if (config.grid[1] != 1 || config.grid[2] != 1)
    throw std::runtime_error("only one-dimensional grid is supported");
  if (config.output.size != config.expected.size)
    throw std::runtime_error("output and expected sizes differ");
  for (const auto &input : config.inputs)
    if (input.size == 0)
      throw std::runtime_error("input size must be nonzero");
  return config;
}

std::vector<uint8_t> read_file(const BufferSpec &spec) {
  std::ifstream file(spec.path, std::ios::binary);
  if (!file)
    throw std::runtime_error("cannot open " + spec.path);
  std::vector<uint8_t> data(spec.size);
  file.read(reinterpret_cast<char *>(data.data()), data.size());
  if (file.gcount() != static_cast<std::streamsize>(data.size()))
    throw std::runtime_error("file size mismatch: " + spec.path);
  return data;
}

std::string sha256(const std::string &path) {
  std::string command = "sha256sum " + path;
  FILE *pipe = popen(command.c_str(), "r");
  if (!pipe)
    throw std::runtime_error("cannot calculate ELF hash");
  char buffer[256] = {};
  std::string output;
  if (fgets(buffer, sizeof(buffer), pipe))
    output = buffer;
  int status = pclose(pipe);
  if (status != 0 || output.size() < 64)
    throw std::runtime_error("sha256sum failed for " + path);
  return output.substr(0, 64);
}

void write_log(const Config &config, const std::string &status,
               const std::string &detail) {
  if (config.log.empty())
    return;
  std::ofstream log(config.log);
  log << "{\n"
      << "  \"status\": \"" << status << "\",\n"
      << "  \"detail\": \"" << detail << "\",\n"
      << "  \"elf\": \"" << config.elf << "\",\n"
      << "  \"elf_sha256\": \"" << sha256(config.elf) << "\",\n"
      << "  \"entry\": " << config.entry << ",\n"
      << "  \"grid\": [" << config.grid[0] << ", " << config.grid[1] << ", "
      << config.grid[2] << "],\n"
      << "  \"local\": [" << config.local[0] << ", " << config.local[1] << ", "
      << config.local[2] << "],\n"
      << "  \"lds_size\": " << config.lds << ",\n"
      << "  \"pds_size\": " << config.pds << ",\n"
      << "  \"timeout_ms\": " << config.timeout_ms << "\n"
      << "}\n";
}

int run_kernel(const Config &config) {
  if (config.entry == 0)
    return 2;
  if (config.entry != 0x800000b8)
    throw std::runtime_error(
        "installed driver only supports entry metadata 0x800000b8");
  for (const auto &input : config.inputs)
    (void)read_file(input);
  (void)read_file(config.expected);

  vt_device_h device = nullptr;
  if (vt_dev_open(&device) != 0)
    throw std::runtime_error("vt_dev_open failed");
  std::vector<uint64_t> addresses;
  std::vector<uint64_t> input_addresses;
  std::vector<std::vector<uint8_t>> input_data;
  auto allocate = [&](uint64_t size) {
    uint64_t address = 0;
    if (vt_buf_alloc(device, size, &address, 0, 0, 0) != 0)
      throw std::runtime_error("vt_buf_alloc failed");
    addresses.push_back(address);
    return address;
  };
  try {
    // The current baremetal spike driver expects the first allocation to
    // reserve its 256 MiB program/global mapping, as in its reference test.
    (void)allocate(0x10000000);
    uint64_t pds_base = 0;
    if (vt_buf_alloc(device, config.pds * config.local[0] * config.grid[0],
                     &pds_base, 0, 0, 0) != 0)
      throw std::runtime_error("PDS allocation failed");
    for (const auto &input : config.inputs) {
      input_data.push_back(read_file(input));
      uint64_t address = allocate(input.size);
      input_addresses.push_back(address);
      if (vt_copy_to_dev(device, address, input_data.back().data(), input.size,
                         0, 0) != 0)
        throw std::runtime_error("vt_copy_to_dev failed");
    }
    if (config.inputs.size() != 2)
      throw std::runtime_error(
          "the initial smoke kernel requires exactly two inputs");
    // vecadd.riscv is an in-place A += B kernel. The output path names the
    // host-side copy of the first input rather than a third device argument.
    const uint64_t output_address = input_addresses[0];
    const std::vector<uint64_t> &kernel_arguments = input_addresses;
    const uint64_t warp_size = 8;
    const uint64_t warp_count = config.local[0] / warp_size;
    const uint64_t metadata_address =
        allocate(kMetadataWords * sizeof(uint32_t));
    const uint64_t buffer_base_address =
        allocate(kernel_arguments.size() * sizeof(uint32_t));
    std::vector<uint32_t> buffer_base;
    for (uint64_t address : kernel_arguments)
      buffer_base.push_back(static_cast<uint32_t>(address));
    if (vt_copy_to_dev(device, buffer_base_address, buffer_base.data(),
                       buffer_base.size() * sizeof(uint32_t), 0, 0) != 0)
      throw std::runtime_error("buffer-base upload failed");
    const uint64_t print_buffer_address = allocate(0x10000000);
    std::vector<uint32_t> metadata(kMetadataWords, 0);
    metadata[0] = static_cast<uint32_t>(config.entry);
    metadata[1] = static_cast<uint32_t>(buffer_base_address);
    metadata[2] = static_cast<uint32_t>(config.grid[0]);
    metadata[6] = static_cast<uint32_t>(warp_size);
    metadata[12] = static_cast<uint32_t>(print_buffer_address);
    metadata[13] = 0x10000000;
    if (vt_copy_to_dev(device, metadata_address, metadata.data(),
                       metadata.size() * sizeof(uint32_t), 0, 0) != 0)
      throw std::runtime_error("metadata upload failed");

    uint64_t groups[3] = {config.grid[0], config.grid[1], config.grid[2]};
    MetaData driver_metadata{0,
                             {groups[0], groups[1], groups[2]},
                             warp_size,
                             warp_count,
                             metadata_address,
                             config.lds,
                             config.pds,
                             config.sgpr,
                             config.vgpr,
                             pds_base};
    if (vt_upload_kernel_file(device, config.elf.c_str(), 0) != 0)
      throw std::runtime_error("vt_upload_kernel_file failed");
    if (vt_start(device, &driver_metadata, 0) != 0)
      throw std::runtime_error("vt_start failed");
    if (vt_ready_wait(device, config.timeout_ms) != 0)
      throw std::runtime_error("vt_ready_wait failed");
    std::vector<uint8_t> output(config.output.size);
    if (vt_copy_from_dev(device, output_address, output.data(), output.size(),
                         0, 0) != 0)
      throw std::runtime_error("output download failed");
    std::vector<uint8_t> expected = read_file(config.expected);
    if (output != expected)
      throw std::runtime_error("output mismatch");
    std::ofstream output_file(config.output.path, std::ios::binary);
    output_file.write(reinterpret_cast<const char *>(output.data()),
                      static_cast<std::streamsize>(output.size()));
    if (!output_file)
      throw std::runtime_error("cannot write output file");
    vt_dev_close(device);
    return 0;
  } catch (...) {
    vt_dev_close(device);
    throw;
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    Config config = parse_args(argc, argv);
    if (!config.args_log.empty()) {
      std::ofstream args(config.args_log);
      args << "elf=" << config.elf << "\nentry=0x" << std::hex << config.entry
           << std::dec << "\ngrid=" << config.grid[0] << "," << config.grid[1]
           << "," << config.grid[2] << "\nlocal=" << config.local[0] << ","
           << config.local[1] << "," << config.local[2]
           << "\nlds_size=" << config.lds << "\npds_size=" << config.pds
           << "\n";
    }
    const pid_t child = fork();
    if (child < 0)
      throw std::runtime_error("fork failed");
    if (child == 0) {
      try {
        const int result = run_kernel(config);
        _exit(result);
      } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        _exit(1);
      }
    }
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(config.timeout_ms);
    int status = 0;
    while (waitpid(child, &status, WNOHANG) == 0) {
      if (std::chrono::steady_clock::now() >= deadline) {
        kill(child, SIGKILL);
        waitpid(child, &status, 0);
        write_log(config, "timeout", "kernel execution exceeded timeout");
        return 124;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
      write_log(config, "pass", "output matches expected");
      return 0;
    }
    write_log(config, "fail", "driver or output validation failed");
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << "\n";
    return 2;
  }
}
