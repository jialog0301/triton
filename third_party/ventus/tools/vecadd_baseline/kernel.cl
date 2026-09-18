// OpenCL arm of the cross-implementation comparison.
//
// Semantics deliberately mirror the Triton reference kernel in
// python/test/unit/ventus/test_pipeline_vector_add.py: a `gid < n` bound instead
// of a separate mask tensor, and a[i] = i + 1 / b[i] = 2 * i as the input
// pattern.
//
// `items` elements per work-item, taken strided by the work-group size, which is
// the shape Triton produces when a program's tile is larger than the warp
// (`BLOCK = local * items` with `sizePerThread = 1`): work-item l of work-group
// w owns w*local*items + l + k*local for k in [0, items). With `items = 1` this
// is the plain one-element-per-work-item kernel.
__kernel void vecadd(__global const float *a, __global const float *b,
                     __global float *c, const int n, const int items) {
  const int lid = get_local_id(0);
  const int lsize = get_local_size(0);
  const int base = get_group_id(0) * lsize * items;
  for (int k = 0; k < items; ++k) {
    const int idx = base + lid + k * lsize;
    if (idx < n)
      c[idx] = a[idx] + b[idx];
  }
}
