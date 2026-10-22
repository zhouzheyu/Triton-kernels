import torch
import triton
import triton.language as tl


@triton.jit
def tiling_row_sum_kernel(
    # TODO: 填写 kernel 参数（input/output pointer、n_cols、BLOCK_SIZE 等）
    x_ptr,
    out_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # TODO: 填写 Softmax kernel 实现
    pid = tl.program_id(0)
    row_start = pid * n_cols
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for j in range(0, n_cols, BLOCK_SIZE):
        offsets = j + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_start + offsets, mask=mask)
        acc = acc + x 
    tl.store(out_ptr + pid, tl.sum(acc))



def tiling_row_sum(x: torch.Tensor) -> torch.Tensor:
    """
    Softmax 的 Python 封装：分配输出、计算 grid、启动 kernel。
    """
    # TODO: 填写封装逻辑（shape 检查、y = empty_like、launch softmax_kernel）
    BLOCK_SIZE = 128
    n_rows, n_cols = x.shape
    out = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    tiling_row_sum_kernel[(n_rows,)](x, out, n_cols, BLOCK_SIZE)
    return out


if __name__ == "__main__":
    x = torch.randn(128, 256, dtype=torch.float32)
    y = tiling_row_sum(x)

    # 可选：和 PyTorch 对照
    ref = torch.sum(x, -1)
    print(torch.allclose(y, ref, atol=1e-5))
    print(y)

# TRITON_INTERPRET=1 python test.py