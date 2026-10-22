"""简单的 Triton Flash Attention 前向示例。

标准 attention:
    S = Q @ K^T / sqrt(d)
    P = softmax(S)
    O = P @ V

Flash Attention 的核心：沿序列维分块（tiling），用 online softmax
在 SRAM 里维护行最大值 m 与归一化因子 l，避免物化完整 [N, N] 的 S/P。

本文件实现多 batch、多 head 的 forward（无 causal mask，便于学习）。
输入形状均为 [B, H, N, D]。
"""

import math

import torch
import triton
import triton.language as tl

# TRITON_INTERPRET=1 python flash_atten.py

@triton.jit
def flash_attn_fwd_kernel(
    q_ptr,  # [B*H, N, D]
    k_ptr,
    v_ptr,
    o_ptr,
    stride_q_bh, stride_q_n, stride_q_d,
    stride_k_bh, stride_k_n, stride_k_d,
    stride_v_bh, stride_v_n, stride_v_d,
    stride_o_bh, stride_o_n, stride_o_d,
    N,  # 序列长度
    D,  # 真实 head dim
    scale,  # 1 / sqrt(D)
    BLOCK_M: tl.constexpr,  # 每个 program 处理的 query 行数
    BLOCK_N: tl.constexpr,  # 每次扫过的 key/value 块大小
    BLOCK_D: tl.constexpr,  # >= D 的 2 次幂
):
    # grid = (B*H * cdiv(N, BLOCK_M),)
    # 每个 program：某一个 (batch, head) 上的一段 query 行
    pid = tl.program_id(axis=0)
    num_q_blocks = tl.cdiv(N, BLOCK_M)
    bh = pid // num_q_blocks
    start_m = (pid % num_q_blocks) * BLOCK_M

    offs_m = start_m + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_d = tl.arange(0, BLOCK_D)  # [BLOCK_D]

    q_bh = q_ptr + bh * stride_q_bh
    k_bh = k_ptr + bh * stride_k_bh
    v_bh = v_ptr + bh * stride_v_bh
    o_bh = o_ptr + bh * stride_o_bh

    # 加载本段 Q，形状 [BLOCK_M, BLOCK_D]，乘 scale
    q_ptrs = q_bh + offs_m[:, None] * stride_q_n + offs_d[None, :] * stride_q_d
    q_mask = (offs_m[:, None] < N) & (offs_d[None, :] < D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32) * scale

    # online softmax 状态（按 query 行）
    # m_i: 已见过的最大 logit；l_i: softmax 分母；acc: 加权 V 的累加
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # 沿 key/value 序列维 tiling
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # K 读成 [BLOCK_D, BLOCK_N]，便于 Q @ K^T
        k_ptrs = k_bh + offs_n[None, :] * stride_k_n + offs_d[:, None] * stride_k_d
        k_mask = (offs_n[None, :] < N) & (offs_d[:, None] < D)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # V: [BLOCK_N, BLOCK_D]
        v_ptrs = v_bh + offs_n[:, None] * stride_v_n + offs_d[None, :] * stride_v_d
        v_mask = (offs_n[:, None] < N) & (offs_d[None, :] < D)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)

        # S = Q @ K^T -> [BLOCK_M, BLOCK_N]
        s = tl.dot(q, k)
        s = tl.where(offs_n[None, :] < N, s, -float("inf"))

        # ----- online softmax 更新 -----
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(s - m_new[:, None])  # [BLOCK_M, BLOCK_N]
        l_ij = tl.sum(p, axis=1)

        # 旧结果按新的 max 重新缩放
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        l_i = l_i * alpha + l_ij
        m_i = m_new

    # 越界 query 行的 l_i 可能为 0，避免除零
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    o = acc / l_i[:, None]

    o_ptrs = o_bh + offs_m[:, None] * stride_o_n + offs_d[None, :] * stride_o_d
    o_mask = (offs_m[:, None] < N) & (offs_d[None, :] < D)
    tl.store(o_ptrs, o, mask=o_mask)


def flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Flash Attention 前向。

    Args:
        q, k, v: [B, H, N, D]，同一 device（建议 float32）

    Returns:
        o: [B, H, N, D]
    """
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    assert q.shape == k.shape == v.shape
    B, H, N, D = q.shape
    assert D <= 128, "本教学实现要求 head dim <= 128"

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o = torch.empty_like(q)

    # 展成 [B*H, N, D]，kernel 用一维 bh 索引
    q2 = q.view(B * H, N, D)
    k2 = k.view(B * H, N, D)
    v2 = v.view(B * H, N, D)
    o2 = o.view(B * H, N, D)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(D)
    scale = 1.0 / math.sqrt(D)

    grid = (B * H * triton.cdiv(N, BLOCK_M),)
    flash_attn_fwd_kernel[grid](
        q2, k2, v2, o2,
        q2.stride(0), q2.stride(1), q2.stride(2),
        k2.stride(0), k2.stride(1), k2.stride(2),
        v2.stride(0), v2.stride(1), v2.stride(2),
        o2.stride(0), o2.stride(1), o2.stride(2),
        N, D, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return o


def attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """朴素 attention，用作正确性对照。"""
    scale = 1.0 / math.sqrt(q.shape[-1])
    s = torch.matmul(q, k.transpose(-2, -1)) * scale
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v)


if __name__ == "__main__":
    # Mac: TRITON_INTERPRET=1 python flash_atten.py
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, H, N, D = 1, 2, 128, 32

    q = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

    out_triton = flash_attn(q, k, v)
    out_torch = attention_ref(q, k, v)

    diff = (out_triton - out_torch).abs().max().item()
    print(f"device: {device}")
    print(f"max abs diff: {diff:.2e}")
    print("ok" if torch.allclose(out_triton, out_torch, atol=1e-2, rtol=1e-2) else "mismatch")
