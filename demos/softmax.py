import torch
import triton
import triton.language as tl

# TRITON_INTERPRET=1 python softmax.py

@triton.jit
def softmax_kernel(
    # TODO: 填写 kernel 参数（input/output pointer、n_cols、BLOCK_SIZE 等）
    x_ptr,
    out_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # TODO: 填写 Softmax kernel 实现
    pid = tl.program_id(0)
    mask = tl.arange(0, BLOCK_SIZE) < n_cols
    offset = pid * n_cols + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset, mask=mask)
    x_max = tl.max(x)
    x = x - x_max 
    x_exp = tl.exp(x)
    x_exp_sum = tl.sum(x_exp)
    out = x_exp / x_exp_sum
    tl.store(out_ptr + offset, out, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Softmax 的 Python 封装：分配输出、计算 grid、启动 kernel。
    """
    # TODO: 填写封装逻辑（shape 检查、y = empty_like、launch softmax_kernel）
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    out = torch.empty_like(x)
    softmax_kernel[(n_rows,)](x, out, n_cols, BLOCK_SIZE = BLOCK_SIZE)
    return out


if __name__ == "__main__":
    x = torch.randn(128, 256, dtype=torch.float32)
    y = softmax(x)

    # 可选：和 PyTorch 对照
    ref = torch.softmax(x, dim=-1)
    print(torch.allclose(y, ref, atol=1e-5))
    print(y)

