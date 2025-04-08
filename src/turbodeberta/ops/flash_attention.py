import math
import torch
import triton
import triton.language as tl


def cdiv(a, b):
    return (a + b - 1) // b

@triton.jit
def _fwd_kernel_deberta_disentangled_attention(
    Q, K, V,
    K_POS, Q_POS,
    L, O,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_pk0, stride_pk1, stride_pk2, stride_pk3,
    stride_pq0, stride_pq1, stride_pq2, stride_pq3,
    Z, H, M, N, P_SEQ,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr, LARGER_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
    HAS_C2P: tl.constexpr, HAS_P2C: tl.constexpr,
    ATT_SPAN: tl.constexpr,
    NUM_BUCKETS: tl.constexpr, MAX_DISTANCE: tl.constexpr
):
    input_dtype = Q.dtype.element_ty

    start_m = tl.program_id(0)
    off_h   = tl.program_id(1)
    off_z   = tl.program_id(2)

    log2e: tl.constexpr = 1.4426950408889634

    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    O += off_z * stride_oz + off_h * stride_oh
    L += (off_z * H + off_h) * M  # L is of shape (B*H, M)

    if HAS_C2P:
        K_POS += off_z*stride_pk0 + off_h*stride_pk1
    if HAS_P2C:
        Q_POS += off_z*stride_pq0 + off_h*stride_pq1

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)  # (BLOCK_M, BLOCK_DMODEL)
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)  # (BLOCK_M, BLOCK_DMODEL)
    l_ptrs = L + offs_m

    mask_m = offs_m < M
    if DIVISIBLE_M:
        q = tl.load(q_ptrs, cache_modifier=".cg")
    else:
        q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier=".cg")

    if BLOCK_DMODEL < 128:
        I = tl.where(offs_k[:, None] == offs_k,
                     tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 1.0, dtype=q.dtype),
                     tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 0.0, dtype=q.dtype))
        q = tl.dot(q, I).to(q.dtype)

    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    offs_n_init = offs_n_base
    k_ptrs = K + (offs_k[:, None] * stride_kk + offs_n_init[None, :] * stride_kn)  # (BLOCK_DMODEL, BLOCK_N)
    v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk)  # (BLOCK_N, BLOCK_DMODEL)

    if IS_CAUSAL:
        hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(0, hi)
    else:
        hi = N

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base

        mask_n = offs_n < N
        if DIVISIBLE_N:
            k = tl.load(k_ptrs, cache_modifier=".cg")
            v = tl.load(v_ptrs, cache_modifier=".cg")
        else:
            k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
            v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")

        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=input_dtype)
        s += tl.dot(q, k) * sm_scale

        relative_positions = offs_m[:, None]-offs_n[None, :]  # shape: (BLOCK_M, BLOCK_N)

        sign = tl.where(relative_positions > 0.0, 1.0, tl.where(relative_positions < 0.0, -1.0, 0.0))

        mid_val = NUM_BUCKETS // 2

        abs_relative = tl.abs(relative_positions)
        condition = (relative_positions < mid_val) & (relative_positions > -mid_val)
        abs_pos = tl.where(condition, mid_val - 1.0, abs_relative)

        log_numer = tl.log(abs_pos / mid_val)
        log_denom = tl.log((MAX_DISTANCE - 1) / mid_val)
        log_scaled = log_numer / log_denom * (mid_val - 1.0)
        log_pos = tl.ceil(log_scaled) + mid_val

        bucket_pos = tl.where(abs_pos <= mid_val, relative_positions, log_pos * sign)

        if HAS_C2P:
            c2p_index = tl.minimum(tl.maximum(bucket_pos + ATT_SPAN, 0), 2 * ATT_SPAN - 1).to(tl.int32)

            k_pos_ptrs = K_POS+offs_m[:, None]*stride_pk2 + c2p_index*stride_pk3

            c2p_bias = tl.load(k_pos_ptrs, mask=mask_m[:, None] & (c2p_index < 2*ATT_SPAN), other=0.0)

            s += c2p_bias * sm_scale

        if HAS_P2C:
            p2c_index = tl.minimum(tl.maximum(-bucket_pos + ATT_SPAN, 0), 2 * ATT_SPAN - 1).to(tl.int32)

            q_pos_ptrs = Q_POS + (offs_n[:, None] * stride_pq2 + p2c_index * stride_pq3)

            p2c_bias = tl.load(q_pos_ptrs, mask=mask_n[:, None] & (p2c_index < 2*ATT_SPAN), other=0.0).trans(1, 0)
            s += p2c_bias * sm_scale

        if not DIVISIBLE_N:
            s = tl.where(mask_n[None, :], s, float("-inf"))
        if IS_CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2((m_i - m_i_new) * log2e)
        p = tl.math.exp2((s - m_i_new[:, None]) * log2e)
        acc *= alpha[:, None]
        acc += tl.dot(p.to(q.dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    if IS_CAUSAL and LARGER_M:
        is_empty_line = (offs_m + P_SEQ) < 0
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        l = tl.where(is_empty_line, float("-inf"), m_i + tl.log(l_i))
    else:
        acc = acc * (1.0 / l_i[:, None])
        l = m_i + tl.log(l_i)

    if DIVISIBLE_M:
        tl.store(l_ptrs, l, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(q.dtype), cache_modifier=".cg")
    else:
        tl.store(l_ptrs, l, mask=mask_m, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(q.dtype), mask=mask_m[:, None], cache_modifier=".cg")

def get_fwd_config(B, H, M, N, D, causal, disentangled=False):
    """
    Determine optimal kernel configuration parameters.

    Args:
        B, H, M, N, D: Batch, head, query length, key length, per-head dimension.
        causal (bool): Whether causal masking is applied.
        disentangled (bool): Whether to use the DeBERTa-style disentangled attention kernel.
                              This flag allows for small tweaks in configuration.

    Returns:
        Tuple (BLOCK_M, BLOCK_N, num_stages, num_warps)
    """
    capability = torch.cuda.get_device_capability()
    if capability == (8, 0):
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
            else:
                if M <= 1024:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 3, 4
                else:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 3, 8
        else:  # causal
            if D <= 64:
                # When using disentangled attention, we may lower num_stages slightly.
                if disentangled:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
                else:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 4, 4
            else:
                if M <= 1024:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
                else:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 3, 8
    elif capability == (8, 6):
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
        else:  # causal
            if D <= 64:
                # For (8,6) devices we boost BLOCK_M for disentangled relative attention.
                if disentangled:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
                else:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 1, 4
    return (BLOCK_M, BLOCK_N, num_stages, num_warps)


def flash_attn_v2_fwd_dise(q, k, v, pos_key, pos_query, causal, sm_scale, BLOCK_M, BLOCK_N,
                           position_buckets, max_relative_distance, num_warps, num_stages):
    """
    Performs the forward pass of FlashAttention with DeBERTa-style disentangled relative attention.

    This function computes the attention output `o` and log-normalizer `L` for the input query (q),
    key (k), and value (v) tensors. It supports disentangled relative attention using optional
    positional projection matrices for content-to-position (C2P) and position-to-content (P2C) biases.

    Args:
        q (Tensor): Query tensor of shape (B, H, M, D) where
            B = batch size, H = number of heads, M = query sequence length, D = head dimension.
        k (Tensor): Key tensor of shape (B, H, N, D) where
            N = key sequence length.
        v (Tensor): Value tensor of shape (B, H, N, D).
        pos_key (Tensor or None): Relative position embedding tensor for C2P bias with shape (2 * max_distance, D),
            or None to disable content-to-position bias.
        pos_query (Tensor or None): Relative position embedding tensor for P2C bias with shape (2 * max_distance, D),
            or None to disable position-to-content bias.
        causal (bool): If True, applies causal (autoregressive) masking to the attention weights.
        sm_scale (float): Scaling factor applied to the dot-product attention scores.
        BLOCK_M (int): Block size for splitting the query sequence dimension.
        BLOCK_N (int): Block size for splitting the key sequence dimension.
        position_buckets (int): Number of relative position buckets. If > 0, bucketing is applied.
        max_relative_distance (int): Maximum relative distance used in bucketing or span window size.
        num_warps (int): Number of warps used in the Triton kernel (hardware-specific parallelism).
        num_stages (int): Number of pipeline stages in the Triton kernel.

    Returns:
        o (Tensor): Output attention tensor of shape (B, H, M, D), same shape as `q`.
        L (Tensor): Log-sum-exp normalizer tensor of shape (B, H, M), used for numerically stable softmax.

    Notes:
        - This function utilizes a custom Triton kernel to efficiently compute block-sparse FlashAttention
          with optional relative position biasing (both C2P and P2C).
        - The relative attention mechanism supports DeBERTa's disentangled attention formulation, where
          the attention bias is computed separately for position-query and key-position interactions.
        - The number of relative position buckets and max distance determines the size and behavior
          of the relative bias.
    """
    B, H, M, D = q.shape
    N = k.shape[2]
    P_SEQ = N - M
    larger_m = M > N

    divisible_m = (M % BLOCK_M) == 0
    divisible_n = (N % BLOCK_N) == 0

    # Determine if each bias term is present.
    has_c2p = pos_key is not None
    has_p2c = pos_query is not None

    # Determine ATT_SPAN from pos_key: assume shape is (2*ATT_SPAN, D)
    if position_buckets>0:
        ATT_SPAN = position_buckets
    else:
        ATT_SPAN = max_relative_distance

    # Setup grid: use a 3D grid (query blocks, heads, batch)
    grid = (cdiv(M, BLOCK_M), H, B)
    o = torch.empty_like(q)
    L = torch.empty((B, H, M), device=q.device, dtype=torch.float32)

    if has_c2p:
        stride_pk0, stride_pk1, stride_pk2, stride_pk3 = pos_key.stride()
    else:
        stride_pk0 = stride_pk1 = stride_pk2 = stride_pk3 = 0
    if has_p2c:
        stride_pq0, stride_pq1, stride_pq2, stride_pq3 = pos_query.stride()
    else:
        stride_pq0 = stride_pq1 = stride_pq2 = stride_pq3 = 0

    with torch.cuda.device(q.device.index):
        _fwd_kernel_deberta_disentangled_attention[grid](
            q, k, v,
            pos_key, pos_query,
            L, o,
            sm_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            stride_pk0, stride_pk1, stride_pk2, stride_pk3,
            stride_pq0, stride_pq1, stride_pq2, stride_pq3,
            B, H, M, N, P_SEQ,
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N,
            IS_CAUSAL=causal, LARGER_M=larger_m,
            DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
            HAS_C2P=has_c2p, HAS_P2C=has_p2c,
            ATT_SPAN=ATT_SPAN,
            NUM_BUCKETS=position_buckets,
            MAX_DISTANCE=max_relative_distance,
            num_warps=num_warps, num_stages=num_stages,
        )

    return o, L


def flash_attention_with_disentangled(q, k, v, q_pos, k_pos, causal=False, sm_scale=None,
                                      position_buckets=0, max_relative_distance=0):
    """
    An implementation of FlashAttention v2 with DeBERTa-style disentangled relative attention.

    Args:
        q (Tensor): Queries of shape (B, H, M, D).
        k (Tensor): Keys of shape (B, H, N, D).
        v (Tensor): Values of shape (B, H, N, D).
        q_pos (Tensor): Relative projection tensor for content→position bias
            (shape: (2*ATT_SPAN, D)).
        k_pos (Tensor): Relative projection tensor for position→content bias
            (shape: (2*ATT_SPAN, D)).
        causal (bool): Whether to apply causal masking (default: False).
        sm_scale (float or None): Scaling factor for attention logits; if None, uses 1 / sqrt(D).
        position_buckets (int): Number of relative position buckets to use. If > 0, bucketing is applied.
        max_relative_distance (int): Maximum relative distance supported by position projections.

    Returns:
        out (Tensor): Output tensor of shape (B, H, M, D).

    Note:
        - This implementation supports only the forward pass.
        - The backward pass is not yet implemented or may not be functional.
          If used in a differentiable pipeline, gradients will not be computed.
    """

class FlashAttentionDisentangled(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, q_pos, k_pos, causal,
                sm_scale, position_buckets, max_relative_distance):

        Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Dq == Dk == Dv, "Query, key, and value must have the same head dimension"
        
        B, H, M, D = q.shape
        N = k.shape[2]
        if sm_scale is None:
            sm_scale = 1. / math.sqrt(D)
        
        config = get_fwd_config(B, H, M, N, D, causal, disentangled=True)
        BLOCK_M, BLOCK_N, num_stages, num_warps = config
        
        o, L = flash_attn_v2_fwd_dise(q, k, v, q_pos, k_pos, causal, sm_scale,
                                      BLOCK_M, BLOCK_N, position_buckets,
                                      max_relative_distance, num_warps, num_stages)
        return o

    @staticmethod
    def backward(ctx, grad_output):
        # Exclude backward capabilities by raising an error.
        raise RuntimeError("Backward pass is not implemented for FlashAttentionDisentangled")

def flash_attention_with_disentangled(q, k, v, q_pos, k_pos, causal=False, sm_scale=None,
                                      position_buckets=0, max_relative_distance=0):
    """
    An implementation of FlashAttention v2 with DeBERTa-style disentangled relative attention.
    This version does not support backward propagation.

    Args:
        q (Tensor): Queries of shape (B, H, M, D).
        k (Tensor): Keys of shape (B, H, N, D).
        v (Tensor): Values of shape (B, H, N, D).
        q_pos (Tensor): Relative projection tensor for content→position bias.
        k_pos (Tensor): Relative projection tensor for position→content bias.
        causal (bool): Whether to apply causal masking.
        sm_scale (float): Scaling factor for softmax (if None, uses 1/sqrt(D)).
        position_buckets (int): Number of position buckets.
        max_relative_distance (int): Maximum relative distance.

    Returns:
        out (Tensor): Output tensor of shape (B, H, M, D).

    Note:
        The backward pass is not implemented, so this function only supports forward propagation.
    """
    return FlashAttentionDisentangled.apply(q, k, v, q_pos, k_pos, causal, sm_scale,
                                            position_buckets, max_relative_distance)
