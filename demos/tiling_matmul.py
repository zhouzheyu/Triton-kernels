"""简单的 Triton 矩阵乘法 Kernel 示例（分块 tiling）。

C[M, N] = A[M, K] @ B[K, N]
每个 block 负责输出矩阵 C 上的一块 [BLOCK_M, BLOCK_N]，
沿 K 维循环累加。
"""

import torch
import triton
import triton.language as tl

# TRITON_INTERPRET=1 python tiling_matmul.py

@triton.jit
def matmul_kernel(
    a_ptr,  # *指针* A，形状 [M, K]，行主序
    b_ptr,  # *指针* B，形状 [K, N]，行主序
    c_ptr,  # *指针* C，形状 [M, N]，行主序
    M,  # 标量
    N,  # 标量
    K,  # 标量
    stride_am,  # A 沿 M 方向步长（通常 = K）
    stride_ak,  # A 沿 K 方向步长（通常 = 1）
    stride_bk,  # B 沿 K 方向步长（通常 = N）
    stride_bn,  # B 沿 N 方向步长（通常 = 1）
    stride_cm,  # C 沿 M 方向步长（通常 = N）
    stride_cn,  # C 沿 N 方向步长（通常 = 1）
    BLOCK_M: tl.constexpr,  # 每个 block 负责的输出行数
    BLOCK_N: tl.constexpr,  # 每个 block 负责的输出列数
    BLOCK_K: tl.constexpr,  # 沿 K 维每次累加的块大小
):
    # 2D grid：pid_m / pid_n 定位本 block 负责 C 的哪一块
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # 本 block 覆盖的输出行/列下标，形状分别为 [BLOCK_M]、[BLOCK_N]
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # 沿 K 的局部下标，形状 [BLOCK_K]
    offs_k = tl.arange(0, BLOCK_K)

    # 广播off，找到对应的地址
    # 找到A块的指针区域：形状上 [BLOCK_M, BLOCK_K]
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # 找到B块的指针区域：形状上 [BLOCK_K, BLOCK_N]
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # 累加器：形状 [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 沿 K 维 tiling：每次加载 A 的一竖条、B 的一横条，做小块 matmul 再累加
    for k in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
        b_mask = (offs_k[:, None] + k < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, BLOCK_K]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, b)  # [BLOCK_M, BLOCK_N]

        # 指针沿 K 方向前进一块
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # 写回 C 的对应块
    c = acc.to(tl.float32)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B，A: [M, K]，B: [K, N]，均需 CUDA、连续、float32。"""
    # assert a.is_cuda and b.is_cuda
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0]
    assert a.dtype == torch.float32 and b.dtype == torch.float32

    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

    # grid: 覆盖 C 需要多少个 [BLOCK_M, BLOCK_N] 的 block
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


if __name__ == "__main__":
    M, K, N = 512, 256, 512
    # a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    # b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    a = torch.randn(M, K, dtype=torch.float32)
    b = torch.randn(K, N, dtype=torch.float32)

    out_triton = matmul(a, b)
    out_torch = a @ b

    print(f"max abs diff: {(out_triton - out_torch).abs().max().item():.2e}")
    print("ok" if torch.allclose(out_triton, out_torch, atol=1e-2, rtol=1e-2) else "mismatch")
