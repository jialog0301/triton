__kernel void barrier_local(__global const float *src,
                            __global float *dst) {
  __local float values[64];
  int local_id = get_local_id(0);
  int global_id = get_global_id(0);
  values[local_id] = src[global_id];
  barrier(CLK_LOCAL_MEM_FENCE);
  dst[global_id] = values[(local_id + 1) % get_local_size(0)];
}
