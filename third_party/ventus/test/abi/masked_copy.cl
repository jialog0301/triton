__kernel void masked_copy(__global const float *src,
                          __global float *dst,
                          int n) {
  int i = get_global_id(0);
  if (i < n)
    dst[i] = src[i];
}
