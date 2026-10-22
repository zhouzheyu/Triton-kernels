"""简单的 Triton GeLU Kernel 示例。"""

import torch
import triton
import triton.language as tl

# TRITON_INTERPRET=1 python gelu.py

@triton.jit
def gelu_kernel(
    x_ptr,  # 输入向量
    out_ptr,  # 输出向量
    n_elements,  # 向量长度
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # [0, 1, 2, ..., BLOCK_SIZE-1]
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    

    x = tl.load(x_ptr + offsets, mask=mask)

    # GeLU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    out = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
    tl.store(out_ptr + offsets, out, mask=mask)


def gelu(x: torch.Tensor) -> torch.Tensor:
    # assert x.is_cuda

    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024

    gelu_kernel[(n_elements // BLOCK_SIZE + 1,)](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


if __name__ == "__main__":
    size = 98432
    # x = torch.randn(size, device="cuda")
    x = torch.randn(size)

    out_triton = gelu(x)
    out_torch = torch.nn.functional.gelu(x)

    print(f"max abs diff: {(out_triton - out_torch).abs().max().item():.2e}")
    print("ok" if torch.allclose(out_triton, out_torch, atol=1e-5) else "mismatch")



