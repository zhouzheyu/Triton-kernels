"""简单的 Triton 向量加法 Kernel 示例。"""

import torch
import triton
import triton.language as tl

# TRITON_INTERPRET=1 python vector_add.py

@triton.jit
def add_kernel(
    x_ptr,  # *指针* 输入向量 x
    y_ptr,  # *指针* 输入向量 y
    out_ptr,  # *指针* 输出向量
    n_elements,  # 标量，向量总长度
    BLOCK_SIZE: tl.constexpr,  # 编译期常量：每个 block 处理多少个元素
):

    pid = tl.program_id(axis=0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
 
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape

    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
 
    # grid：需要多少个 block 才能覆盖整个向量
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


if __name__ == "__main__":
    size = 98432
    # x = torch.rand(size, device="cuda")  
    # y = torch.rand(size, device="cuda")
    x, y = torch.rand(size), torch.rand(size)  

    out_triton = add(x, y)
    out_torch = x + y

    print(f"max abs diff: {(out_triton - out_torch).abs().max().item():.2e}")
    print("ok" if torch.allclose(out_triton, out_torch) else "mismatch")
