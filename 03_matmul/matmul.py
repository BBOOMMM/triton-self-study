import torch
import triton
import triton.language as tl
DEVICE = torch.device('cuda:0')


autotune_configs = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2)
]

@triton.autotune(configs=autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    A_stride_M, A_stride_K,
    B_stride_K, B_stride_N,
    C_stride_M, C_stride_N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # [0,  2, |  4,  6]      前两行是一个大 group
    # [1,  3, |  5,  7]
    # --------|--------
    # [8, 10, | 12, 14]
    # [9, 11, | 13, 15] 
    # 人为想要对应 pid 处理对应行和列，共享 SM 的 SRAM
    PID = tl.program_id(0)   # = NUM_BLOCK_ALONG_M * NUM_BLOCK_ALONG_N
    NUM_BLOCK_ALONG_M = tl.cdiv(M, BLOCK_SIZE_M)
    NUM_BLOCK_ALONG_N = tl.cdiv(N, BLOCK_SIZE_N)
    NUM_PID_PER_GROUP = GROUP_SIZE * NUM_BLOCK_ALONG_N
    group_id = PID // NUM_PID_PER_GROUP
    group_start_M = group_id * GROUP_SIZE   # group 开始的行号
    
    edge_length = min(GROUP_SIZE, NUM_BLOCK_ALONG_M - group_start_M)
    
    PID_M = group_start_M + (PID % NUM_PID_PER_GROUP) % edge_length   # 该 PID 对应的 BLOCK_SIZE_M 那一行
    
    PID_N = (PID % NUM_PID_PER_GROUP) // edge_length
    
    # A_block 行
    offset_M = PID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)   # 真实的行号偏移
    mask_M = offset_M < M
    offset_M *= A_stride_M  # [BM]
    
    # B_block 列
    offset_N = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_N = offset_N < N
    offset_N *= B_stride_N  # [BN]
    
    offset_K = tl.arange(0, BLOCK_SIZE_K)  # [BK]
    
    offset_M = offset_M[:, None] + offset_K[None, :]*A_stride_K      # [BM, 1] + [1, BK] = [BM, BK]
    offset_N = offset_K[:, None]*B_stride_K + offset_N[None, :]      # [BK, 1] + [1, BN] = [BK, BN]
    
    bC = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k_idx in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask_K = offset_K + k_idx * BLOCK_SIZE_K < K
        
        bA = tl.load(A_ptr + offset_M, mask= (mask_M[:, None] & mask_K[None, :]) , other=0)
        bB = tl.load(B_ptr + offset_N, mask= (mask_K[:, None] & mask_N[None, :]) , other=0)
        
        # bC += tl.dot(bA, bB)
        bC = tl.dot(bA, bB, acc=bC)
        
        # 移动到下一个 block
        offset_M = offset_M + BLOCK_SIZE_K*A_stride_K
        offset_N = BLOCK_SIZE_K*B_stride_K + offset_N
    
    bC = bC.to(tl.bfloat16)
    offset_C1 = PID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) # [BM]
    offset_C2 = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) # [BN]
    offset_C = offset_C1[:, None]*C_stride_M + offset_C2[None, :]*C_stride_N
    tl.store(C_ptr + offset_C, bC, mask=(mask_M[:, None] & mask_N[None, :]))


def matmul(A, B):
    assert A.ndim == B.ndim == 2
    assert A.shape[1] == B.shape[0]
    
    M, K, N = *A.shape, B.shape[1]
    
    C = torch.empty(M, N, device=A.device, dtype=torch.bfloat16)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    _matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    
    return C
    



def test_matmul_kernel(size, atol=1e-3, rtol=1e-3, device=DEVICE):
    torch.manual_seed(21)
    assert len(size) == 2
    
    A = torch.randn(*size, device=DEVICE, dtype=torch.bfloat16)
    B = torch.randn(*size, device=DEVICE, dtype=torch.bfloat16)
    
    C_tri = matmul(A, B)
    C_ref = A @ B
    
    torch.testing.assert_close(C_tri, C_ref, atol=atol, rtol=rtol)
    
    print('passed')
    


configs = [
    triton.testing.Benchmark(
        x_names = ["M", "N", "K"], # we can increase multiple dimensions simultaneously while benchmarking
        x_vals = [128 * i for i in range(2, 33)],
        line_arg = "provider", 
        line_vals = ["torch", "triton"],
        line_names = ["PyTorch", "Triton"],
        styles = [("green", "-"), ("blue", "-")],
        ylabel = "TFLOPS", 
        plot_name = "matmul-performance",
        args={},
    )
]
@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.bfloat16)
    quantiles = [0.5, 0.05, 0.95]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 3 * M * N * K * 1e-12 / (ms * 1e-3)
        # 3 = number of memory operations (2 read + 1 write)
        # M * N * K = number of elements per memory op
        # 1e-12 converts flops to Teraflops
        # 1e-3 converts milliseconds to seconds
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    # always run unit-tests
    test_matmul_kernel(size=(1024, 1024))

    # Only run benchmark if explicitly requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)