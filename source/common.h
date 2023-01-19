#include <cuda_runtime.h>

enum struct ChannelMode { Y, UV, YUV, RGB };

extern cudaError_t nlmeans(
    void * d_dst,
    void * d_src,
    float * d_buffer,
    float * d_buffer_bwd,
    float * d_buffer_fwd,
    float * d_wdst,
    float * d_weight,
    float * d_max_weight,
    bool is_float,
    int bits_per_sample,
    int width, int height, int image_stride, int buffer_stride,
    int radius, int spatial_radius, int block_radius, float h2_inv_norm,
    ChannelMode channels, int wmode, float wref, bool has_ref,
    cudaStream_t stream
);
