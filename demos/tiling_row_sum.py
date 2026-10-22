"""当一行的列数大于 BLOCK_SIZE 时，怎么按行求和。

输入 x 是二维矩阵 [n_rows, n_cols]。每个 block 处理一行，
用 for 循环按 BLOCK_SIZE 分块加载并累加。
"""

# TRITON_INTERPRET=1 python tiling_row_sum.py

import torch
import triton
import triton.language as tl


@triton.jit
def row_sum_kernel(
    x_ptr,  # 输入矩阵 [n_rows, n_cols]
    out_ptr,  # 输出向量 [n_rows]
    n_cols,  # 每行的列数（可大于 BLOCK_SIZE）
    BLOCK_SIZE: tl.constexpr,  # 每次循环处理的元素数
):
    row_id = tl.program_id(axis=0)
    row_start = row_id * n_cols
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for off in range(0, n_cols, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        acc += x

    tl.store(out_ptr + row_id, tl.sum(acc, axis=0))


def row_sum(x: torch.Tensor) -> torch.Tensor:
    """对二维 CUDA 矩阵按最后一维求和，返回形状 [n_rows] 的向量。"""
    # assert x.is_cuda and x.ndim == 2

    n_rows, n_cols = x.shape
    out = torch.empty(n_rows, device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 1024  # 故意小于很长的行，靠循环覆盖

    row_sum_kernel[(n_rows,)](x, out, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


if __name__ == "__main__":
    n_rows, n_cols = 2048, 2048
    # x = torch.randn(n_rows, n_cols, device="cuda")
    x = torch.randn(n_rows, n_cols)

    out_triton = row_sum(x)
    out_torch = x.sum(dim=-1)

    print(f"max abs diff: {(out_triton - out_torch).abs().max().item():.2e}")
    print("ok" if torch.allclose(out_triton, out_torch, atol=1e-3, rtol=1e-3) else "mismatch")
