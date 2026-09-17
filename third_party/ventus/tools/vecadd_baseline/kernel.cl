// OpenCL arm of the cross-implementation comparison.
//
// Semantics deliberately mirror the Triton reference kernel in
// python/test/unit/ventus/test_pipeline_vector_add.py: one element per
// work-item, a `gid < n` bound instead of a separate mask tensor, and
// a[i] = i + 1 / b[i] = 2 * i as the input pattern.
__kernel void vecadd(__global const float *a, __global const float *b,
                     __global float *c, const int n) {
  const int gid = get_global_id(0);
  if (gid < n)
    c[gid] = a[gid] + b[gid];
}
