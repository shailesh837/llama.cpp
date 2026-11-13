#pragma once

#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_sycl_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

// GGML_SYCL_GDN_STATE_DIRECT: the kernel reads its input state from the cache rows ids[sequence]
// instead of a gathered copy; row_stride in elements
struct ggml_sycl_gated_delta_net_direct_state {
    const float *   base;
    const int32_t * ids;
    int64_t         row_stride;
};

void ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_sycl_try_gdn_cache_fusion)
void ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache);

// the op with either or both of the fusions above; null pointers select the plain behaviour
void ggml_sycl_op_gated_delta_net_ex(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                     const ggml_sycl_gated_delta_net_fused_cache *  cache,
                                     const ggml_sycl_gated_delta_net_direct_state * direct);
