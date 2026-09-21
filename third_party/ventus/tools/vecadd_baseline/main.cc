/* Host side of the OpenCL comparison arm.
 *
 * Runs the same workload shape as the Triton reference kernel: `n` elements,
 * a work-group of `local` work-items, `ceil(n / local)` work-groups, and
 * `gid < n` guarding the tail. Verifies against the CPU reference and prints a
 * single machine-readable RESULT line so a runner can collect it.
 *
 * Build (the installed ICD and headers):
 *   clang++ -O2 -std=c++11 main.cc -o vecadd_baseline -I$V/include -L$V/lib
 * -lOpenCL
 */
#include <CL/cl.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void check(cl_int err, const char *what) {
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "OpenCL error %d at %s\n", err, what);
    std::exit(1);
  }
}

static std::string read_file(const char *path) {
  FILE *f = std::fopen(path, "rb");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path);
    std::exit(1);
  }
  std::string out;
  char buf[4096];
  size_t got;
  while ((got = std::fread(buf, 1, sizeof(buf), f)) > 0)
    out.append(buf, got);
  std::fclose(f);
  return out;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <kernel.cl> <n> <local> [items]\n",
                 argv[0]);
    return 2;
  }
  const char *kernel_path = argv[1];
  const int n = std::atoi(argv[2]);
  const int local = std::atoi(argv[3]);
  const int items = argc > 4 ? std::atoi(argv[4]) : 1;
  // One work-group covers `local * items` elements, so the grid follows the
  // tile and not the work-group size -- the same relation the Triton arm uses.
  const size_t per_group = (size_t)local * (size_t)items;
  const size_t global = ((size_t)(n + per_group - 1) / per_group) * local;

  cl_platform_id platform;
  cl_device_id device;
  check(clGetPlatformIDs(1, &platform, NULL), "clGetPlatformIDs");
  check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL),
        "clGetDeviceIDs");
  char name[256] = {0};
  clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);

  cl_int err;
  cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  check(err, "clCreateContext");
  cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);
  check(err, "clCreateCommandQueue");

  std::string src = read_file(kernel_path);
  const char *src_ptr = src.c_str();
  size_t src_len = src.size();
  cl_program program =
      clCreateProgramWithSource(ctx, 1, &src_ptr, &src_len, &err);
  check(err, "clCreateProgramWithSource");
  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t log_len = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_len);
    std::vector<char> log(log_len + 1, 0);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_len,
                          log.data(), NULL);
    std::fprintf(stderr, "build failed:\n%s\n", log.data());
    return 1;
  }
  cl_kernel kernel = clCreateKernel(program, "vecadd", &err);
  check(err, "clCreateKernel");

  const size_t bytes = sizeof(float) * (size_t)n;
  std::vector<float> a(n), b(n), c(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    a[i] = (float)(i + 1);
    b[i] = (float)(2 * i);
  }
  cl_mem abuf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               bytes, a.data(), &err);
  check(err, "clCreateBuffer(a)");
  cl_mem bbuf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               bytes, b.data(), &err);
  check(err, "clCreateBuffer(b)");
  cl_mem cbuf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
  check(err, "clCreateBuffer(c)");

  check(clSetKernelArg(kernel, 0, sizeof(cl_mem), &abuf), "arg a");
  check(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bbuf), "arg b");
  check(clSetKernelArg(kernel, 2, sizeof(cl_mem), &cbuf), "arg c");
  check(clSetKernelArg(kernel, 3, sizeof(cl_int), &n), "arg n");
  check(clSetKernelArg(kernel, 4, sizeof(cl_int), &items), "arg items");

  const size_t gws[1] = {global};
  const size_t lws[1] = {(size_t)local};
  check(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, gws, lws, 0, NULL, NULL),
        "clEnqueueNDRangeKernel");
  check(clFinish(queue), "clFinish");
  check(clEnqueueReadBuffer(queue, cbuf, CL_TRUE, 0, bytes, c.data(), 0, NULL,
                            NULL),
        "clEnqueueReadBuffer");

  int mismatches = 0;
  for (int i = 0; i < n; ++i) {
    const float expect = a[i] + b[i];
    if (c[i] != expect) {
      if (mismatches < 4)
        std::fprintf(stderr, "mismatch at %d: got %f expect %f\n", i, c[i],
                     expect);
      ++mismatches;
    }
  }

  clReleaseMemObject(abuf);
  clReleaseMemObject(bbuf);
  clReleaseMemObject(cbuf);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  std::printf("RESULT device=%s n=%d local=%d items=%d global=%zu grid=%zu "
              "num_mismatches=%d\n",
              name, n, local, items, global, global / (size_t)local,
              mismatches);
  return mismatches == 0 ? 0 : 1;
}
