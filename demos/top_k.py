"""MoE Top-K：对每个 token 的 expert logits 做 gating + 选出 top-k experts。

输入:
  logits: [num_tokens, num_experts]  router 打分（投影后的结果）

输出:
  topk_weights: [num_tokens, TOP_K]  选中 expert 的 gating 权重
  topk_indices: [num_tokens, TOP_K]  选中 expert 的下标（int32）

常见流程（Mixtral 风格）:
  1. 对每个 token 的 logits 做 stable softmax（或 sigmoid）
  2. 迭代选 top-k：argmax → 存下 weight/index → 把该位置 mask 掉再选下一个
     注意：mask 成 -1.0（不要 mask 成 0.0），否则专家很多时分数接近 0，会重复选中

用法:
  TRITON_INTERPRET=1 python top_k.py
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel：请自己实现核心逻辑
# ---------------------------------------------------------------------------

@triton.jit
def moe_topk_kernel(
    logits_ptr,  # [num_tokens, num_experts]
    weights_ptr,  # [num_tokens, TOP_K]
    indices_ptr,  # [num_tokens, TOP_K]，存 int32
    num_tokens,
    stride_logits_t,  # logits 行步长，通常 = num_experts
    stride_w_t,  # weights 行步长，通常 = TOP_K
    stride_i_t,  # indices 行步长，通常 = TOP_K
    NUM_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # >= NUM_EXPERTS 的 2 次幂，方便 tl.arange
):
    """每个 program 处理一个 token。"""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < NUM_EXPERTS

    # 加载一行 logits，越界位置用 -inf，不影响 max / softmax
    logits = tl.load(
        logits_ptr + pid * stride_logits_t + offs,
        mask=mask,
        other=-float("inf"),
    )

    # Stable softmax
    logits = logits - tl.max(logits, axis=0)
    probs = tl.exp(logits)
    probs = probs / tl.sum(probs, axis=0)
    probs = tl.where(mask, probs, -1.0)

    # 迭代 top-k：每次取最大，再 mask 成 -1.0
    for k in range(TOP_K):
        top_val = tl.max(probs, axis=0)
        top_idx = tl.argmax(probs, axis=0)
        tl.store(weights_ptr + pid * stride_w_t + k, top_val)
        tl.store(indices_ptr + pid * stride_i_t + k, top_idx.to(tl.int32))
        probs = tl.where(offs == top_idx, -1.0, probs)


# ---------------------------------------------------------------------------
# 辅助：Python 封装 / PyTorch 参考 / 测试
# ---------------------------------------------------------------------------

def moe_topk(logits: torch.Tensor, top_k: int):
    """对 logits 做 softmax gating + top-k，返回 (weights, indices)。

    logits: float32, shape [num_tokens, num_experts]，连续内存
    """
    assert logits.ndim == 2
    assert logits.dtype == torch.float32
    assert logits.is_contiguous()
    num_tokens, num_experts = logits.shape
    assert 1 <= top_k <= num_experts

    weights = torch.empty((num_tokens, top_k), device=logits.device, dtype=torch.float32)
    indices = torch.empty((num_tokens, top_k), device=logits.device, dtype=torch.int32)

    BLOCK_SIZE = triton.next_power_of_2(num_experts)
    grid = (num_tokens,)

    moe_topk_kernel[grid](
        logits,
        weights,
        indices,
        num_tokens,
        logits.stride(0),
        weights.stride(0),
        indices.stride(0),
        NUM_EXPERTS=num_experts,
        TOP_K=top_k,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return weights, indices


def moe_topk_ref(logits: torch.Tensor, top_k: int):
    """PyTorch 参考实现：softmax + topk。"""
    probs = torch.softmax(logits, dim=-1)
    weights, indices = torch.topk(probs, k=top_k, dim=-1)
    return weights, indices.to(torch.int32)


if __name__ == "__main__":
    torch.manual_seed(0)
    num_tokens, num_experts, top_k = 128, 8, 2
    # logits = torch.randn(num_tokens, num_experts, device="cuda", dtype=torch.float32)
    logits = torch.randn(num_tokens, num_experts, dtype=torch.float32)

    w_ref, i_ref = moe_topk_ref(logits, top_k)
    w_tri, i_tri = moe_topk(logits, top_k)

    # top-k 的顺序一般按权重大小降序；indices 也应一致
    w_ok = torch.allclose(w_tri, w_ref, atol=1e-5, rtol=1e-5)
    i_ok = torch.equal(i_tri, i_ref)

    print(f"tokens={num_tokens}, experts={num_experts}, top_k={top_k}")
    print(f"weights match: {w_ok}  (max abs diff={(w_tri - w_ref).abs().max().item():.2e})")
    print(f"indices match: {i_ok}")
    if not i_ok:
        print("ref indices[0]:", i_ref[0])
        print("tri indices[0]:", i_tri[0])
        print("ref weights[0]:", w_ref[0])
        print("tri weights[0]:", w_tri[0])
    print("ok" if (w_ok and i_ok) else "mismatch / kernel 尚未实现")
