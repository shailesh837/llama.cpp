#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "ggml.h"
#include "gated_delta_net.hpp"
#include <cmath>


// GGML_SYCL_GDN_LAZY: state update of one element, the same expression in the token loop and in the replays
static __dpct_inline__ float gdn_state_upd(float g, float s, float k, float d) {
    return g * s + k * d;
}

// GGML_SYCL_GDN_LAZY: the cache keeps the state before the last step (base) and, per token of that step, the
// update terms k, delta and exp(g) (stash). The kernel replays the tokens that were kept, stores the result as
// the new base and stashes the terms of its own tokens instead of writing a state snapshot per token.
// A stash token is [k: H*S_v][delta: H*S_v][exp(g): H*S_v] (exp(g) at [h*S_v] only when g is one value per head).
struct gdn_lazy_args {
    const float * base_in;     // state row to start from
    float *       base_out;    // row that receives the state after the replay
    const float * stash_prev;  // terms of the previous step
    float *       stash_new;   // terms of this step
    int           n_replay;    // tokens of the previous step to apply first
    int           write_base;
};

template <int S_v, bool KDA, bool keep_rs_t, bool LAZY>
void gated_delta_net_sycl(const float *     q,
                          const float *     k,
                          const float *     v,
                          const float *     g,
                          const float *     beta,
                          const float *     curr_state,
                          float *           dst,
                          float *           state,
                          int64_t           H,
                          int64_t           n_tokens,
                          int64_t           sq1,
                          int64_t           sq2,
                          int64_t           sq3,
                          int64_t           sv1,
                          int64_t           sv2,
                          int64_t           sv3,
                          int64_t           sb1,
                          int64_t           sb2,
                          int64_t           sb3,
                          const sycl::uint3 neqk1_magic,
                          const sycl::uint3 rq3_magic,
                          float             scale,
                          int64_t           state_slot_stride,
                          int               K,
                          const int32_t *   s_ids,
                          int64_t           s_row_stride,
                          const gdn_lazy_args lazy) {
    auto           item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const uint32_t h_idx    = item_ct1.get_group(2);
    const uint32_t sequence = item_ct1.get_group(1);
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = item_ct1.get_local_id(2);
    const int      col      = item_ct1.get_group(0) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    // s_ids != nullptr (GGML_SYCL_GDN_STATE_DIRECT): curr_state is the cache, the state is its row s_ids[sequence].
    // Every work-item loads its part of the state below before it writes a snapshot, and it writes
    // only the elements it loaded, so the cache may also be the output.
    const int64_t state_in_offset      = (s_ids != nullptr ? (int64_t) s_ids[sequence] * s_row_stride : (int64_t) sequence * H * S_v * S_v) +
                                         h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_sycl_get_physical_warp_size() < S_v ? ggml_sycl_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    const int64_t hs_tok = 3 * H * S_v;   // floats per stash token
    if constexpr (LAZY) {
        const int64_t el = (int64_t) h_idx * S_v * S_v + (int64_t) col * S_v;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            s_shard[r]  = lazy.base_in[el + i];
        }
        for (int t = 0; t < lazy.n_replay; t++) {
            const float * st = lazy.stash_prev + t * hs_tok + (int64_t) h_idx * S_v;
            const float   d  = st[H * S_v + col];
            const float * gp = st + 2 * H * S_v;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = gdn_state_upd(KDA ? gp[i] : gp[0], s_shard[r], st[i], d);
            }
        }
        if (lazy.write_base) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                lazy.base_out[el + i] = s_shard[r];
            }
        }
    } else {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            s_shard[r]  = curr_state[i];
        }
    }

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        if constexpr (!KDA) {
            const float g_val = sycl::native::exp(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += s_shard[r] * k_t[i];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = gdn_state_upd(g_val, s_shard[r], k_t[i], delta_col);
                attn_partial += s_shard[r] * q_t[i];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
            if constexpr (LAZY) {
                float * st = lazy.stash_new + t * hs_tok + (int64_t) h_idx * S_v;
                if (lane == 0) {
                    st[H * S_v + col] = delta_col;
                }
                if (col == 0) {
                    if (lane == 0) {
                        st[2 * H * S_v] = g_val;
                    }
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        st[i]       = k_t[i];
                    }
                }
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += sycl::native::exp(g_t[i]) * s_shard[r] * k_t[i];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = gdn_state_upd(sycl::native::exp(g_t[i]), s_shard[r], k_t[i], delta_col);
                attn_partial += s_shard[r] * q_t[i];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if constexpr (LAZY) {
                float * st = lazy.stash_new + t * hs_tok + (int64_t) h_idx * S_v;
                if (lane == 0) {
                    st[H * S_v + col] = delta_col;
                }
                if (col == 0) {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i           = r * warp_size + lane;
                        st[i]                 = k_t[i];
                        st[2 * H * S_v + i]   = sycl::native::exp(g_t[i]);
                    }
                }
            }

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;


    // Write state back to global memory
        if constexpr (keep_rs_t && !LAZY) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t && !LAZY) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <bool KDA, bool keep_rs_t, bool LAZY = false>
static void launch_gated_delta_net(const float *   q_d,
                                   const float *   k_d,
                                   const float *   v_d,
                                   const float *   g_d,
                                   const float *   b_d,
                                   const float *   s_d,
                                   float *         dst_d,
                                   float *         state_d,
                                   int64_t         S_v,
                                   int64_t         H,
                                   int64_t         n_tokens,
                                   int64_t         n_seqs,
                                   int64_t         sq1,
                                   int64_t         sq2,
                                   int64_t         sq3,
                                   int64_t         sv1,
                                   int64_t         sv2,
                                   int64_t         sv3,
                                   int64_t         sb1,
                                   int64_t         sb2,
                                   int64_t         sb3,
                                   int64_t         neqk1,
                                   int64_t         rq3,
                                   float           scale,
                                   int64_t         state_slot_stride,
                                   int             K,
                                   const int32_t * s_ids,
                                   int64_t         s_row_stride,
                                   dpct::queue_ptr stream,
                                   const gdn_lazy_args lazy = {}) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_size;

    const int num_warps = 4;
    dpct::dim3 grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dpct::dim3 block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const sycl::uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const sycl::uint3 rq3_magic   = init_fastdiv_values(rq3);

    switch (S_v) {
        case 16:
            {
                constexpr int sv = 16;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t, LAZY>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride, lazy);
                                     });
            }
            break;
        case 32:
            {
                constexpr int sv = 32;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t, LAZY>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride, lazy);
                                     });
            }
            break;
        case 64: {
            {
                constexpr int sv = 64;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t, LAZY>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride, lazy);
                                        });
            }
            break;
        }
        case 128: {
            {
                constexpr int sv = 128;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t, LAZY>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride, lazy);
                                        });
            }
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_sycl_op_gated_delta_net_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              const ggml_sycl_gated_delta_net_fused_cache * cache,
                                              const ggml_sycl_gated_delta_net_direct_state * direct = nullptr,
                                              const ggml_sycl_gated_delta_net_lazy * lazy = nullptr) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float *   s_d          = direct != nullptr ? direct->base : (const float *) src_state->data;
    const int32_t * s_ids        = direct != nullptr ? direct->ids : nullptr;
    const int64_t   s_row_stride = direct != nullptr ? direct->row_stride : 0;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    dpct::queue_ptr stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> dst tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (lazy != nullptr) {
        // GGML_SYCL_GDN_LAZY: no snapshot per token, see gdn_lazy_args
        GGML_ASSERT(n_seqs == 1);
        const gdn_lazy_args la = { lazy->base_in, lazy->base_out, lazy->stash_prev, lazy->stash_new, lazy->n_replay,
                                   lazy->write_base ? 1 : 0 };
        if (kda) {
            launch_gated_delta_net<true, false, true>(q_d, k_d, v_d, g_d, b_d, nullptr, dst_d, nullptr,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, 0, K, nullptr, 0, stream, la);
        } else {
            launch_gated_delta_net<false, false, true>(q_d, k_d, v_d, g_d, b_d, nullptr, dst_d, nullptr,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, 0, K, nullptr, 0, stream, la);
        }
        return;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, s_ids, s_row_stride, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, s_ids, s_row_stride, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, s_ids, s_row_stride, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, s_ids, s_row_stride, stream);
        }
    }
}

void ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net(ctx, dst);
}

void ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, &cache);
}

void ggml_sycl_op_gated_delta_net_ex(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                     const ggml_sycl_gated_delta_net_fused_cache *  cache,
                                     const ggml_sycl_gated_delta_net_direct_state * direct) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, cache, direct);
}

void ggml_sycl_op_gated_delta_net_lazy(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                       const ggml_sycl_gated_delta_net_lazy & lazy) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, nullptr, nullptr, &lazy);
}

// GGML_SYCL_GDN_LAZY: turn a lazy cell back into state snapshots. Applies the n_prev stashed tokens to the base
// state and writes the state after token t (1-based) to slot n_prev - t, as the token loop of the kernel does;
// with n_prev < K the base itself goes to slot n_prev. Every work-item reads its elements of the base before it
// writes, so slot 0 may be the base row.
template <int S_v, bool KDA>
static void gdn_materialize_sycl(const float * base, const float * stash, float * planes, int64_t slot_stride,
                                 int64_t H, int n_prev, int K) {
    auto           item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const uint32_t h_idx    = item_ct1.get_group(2);
    const int      lane     = item_ct1.get_local_id(2);
    const int      col      = item_ct1.get_group(0) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    constexpr int warp_size = ggml_sycl_get_physical_warp_size() < S_v ? ggml_sycl_get_physical_warp_size() : S_v;
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    const int64_t hs_tok = 3 * H * S_v;
    const int64_t el     = (int64_t) h_idx * S_v * S_v + (int64_t) col * S_v;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = base[el + i];
    }
    if (n_prev < K) {
        float * out = planes + (int64_t) n_prev * slot_stride + el;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            out[i]      = s_shard[r];
        }
    }
    for (int t = 0; t < n_prev; t++) {
        const float * st = stash + t * hs_tok + (int64_t) h_idx * S_v;
        const float   d  = st[H * S_v + col];
        const float * gp = st + 2 * H * S_v;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            s_shard[r]  = gdn_state_upd(KDA ? gp[i] : gp[0], s_shard[r], st[i], d);
        }
        const int target_slot = n_prev - 1 - t;
        if (target_slot < K) {
            float * out = planes + (int64_t) target_slot * slot_stride + el;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                out[i]      = s_shard[r];
            }
        }
    }
}

template <int S_v, bool KDA>
static void gdn_materialize_launch(dpct::queue_ptr stream, const sycl::nd_range<3> & range, const float * base,
                                   const float * stash, float * planes, int64_t slot_stride, int64_t H, int n_prev,
                                   int K) {
    stream->parallel_for(range, [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        gdn_materialize_sycl<S_v, KDA>(base, stash, planes, slot_stride, H, n_prev, K);
    });
}

bool ggml_sycl_gdn_materialize(dpct::queue_ptr stream, int64_t S_v, int64_t H, bool kda, const float * base,
                               const float * stash, float * planes, int64_t slot_stride, int n_prev, int K) {
    const int warp_size = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_size;
    const int num_warps = 4;
    dpct::dim3 grid_dims(H, 1, (S_v + num_warps - 1) / num_warps);
    dpct::dim3 block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);
    const sycl::nd_range<3> range(grid_dims * block_dims, block_dims);
#define GDN_MAT(SV)                                                                                         \
    case SV:                                                                                                \
        if (kda) {                                                                                          \
            gdn_materialize_launch<SV, true>(stream, range, base, stash, planes, slot_stride, H, n_prev, K);  \
        } else {                                                                                            \
            gdn_materialize_launch<SV, false>(stream, range, base, stash, planes, slot_stride, H, n_prev, K); \
        }                                                                                                   \
        return true;
    switch (S_v) {
        GDN_MAT(16)
        GDN_MAT(32)
        GDN_MAT(64)
        GDN_MAT(128)
        default:
            return false;
    }
#undef GDN_MAT
}
