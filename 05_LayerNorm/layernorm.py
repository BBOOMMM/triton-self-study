import torch
import triton
import triton.language as tl

DEVICE = torch.device('cuda:0')


@triton.jit
def _layernorm_fwd_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    mean_ptr, rstd_ptr,
    stride_M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = x_ptr + row*stride_M
    y_ptr = y_ptr + row*stride_M
    
    acc1 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in tl.range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr+cols, mask=mask, other=0.).to(tl.float32)
        acc1 += x
    mean = tl.sum(acc1) / N
    tl.store(mean_ptr + row, mean)
        
    acc2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in tl.range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr+cols, mask=mask, other=0.).to(tl.float32)
        diff = tl.where(mask, x-mean, 0.)
        acc2 += (diff * diff)
    var = tl.sum(acc2) / N
    rstd = tl.rsqrt(var + eps)
    tl.store(rstd_ptr + row, rstd)
    
    for offset in tl.range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        
        x = tl.load(x_ptr+cols, mask=mask, other=0.)
        w = tl.load(weight_ptr+cols, mask=mask, other=0.)
        b = tl.load(bias_ptr+cols, mask=mask, other=0.)
        
        x_normed = (x - mean) * rstd
        y = x_normed * w + b
        tl.store(y_ptr + cols, y, mask=mask)
        


@triton.jit
def _layernorm_backward_dLdx(
    x_ptr, dLdx_ptr, dLdy_ptr,                              # pointers to first entries of tensors of shape (M, N)
    w_ptr,                                                  # pointers to first entries of tensors of shape (N)
    dLdw_intermediate_ptr, dLdb_intermediate_ptr,           # pointers to first entries of tensors of shape (GROUP_SIZE, N)
    mean_ptr, rstd_ptr,                                     # pointers to first entries of tensors of shape (M)
    locks_ptr,                                              # pointers to first entry of tensor of shape (2 * GROUP_SIZE)
    stride, N,                                              # dynamic variables determined at run-time
    GROUP_SIZE: tl.constexpr, BLOCK_SIZE_N: tl.constexpr    # static variables determined at compile-time
):
    """
    there's a weird grouping strategy being used here for the _dLdw and _dLdb
    the idea is that each pid is assigned some subset of rows (which are interleaved rather than next to each other)
    and it's that pid's job to accumulate the gradients over all of the rows it has been assigned
    then once each pid is done, in the next kernel we'll accumulate all of those individiual partial sums
    """
    PID = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    x_ptr += PID * stride
    dLdx_ptr += PID * stride
    dLdy_ptr += PID * stride

    # Load data to SRAM
    x = tl.load(x_ptr + cols, mask=mask, other=0).to(tl.float32)            # shape (BLOCK_SIZE_N)
    dLdy = tl.load(dLdy_ptr + cols, mask=mask, other=0).to(tl.float32)      # shape (BLOCK_SIZE_N)
    w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)                     # shape (BLOCK_SIZE_N)
    mean = tl.load(mean_ptr + PID)                                          # shape (1)
    rstd = tl.load(rstd_ptr + PID)                                          # shape (1)

    # Compute dLdx
    x_normalized = tl.where(mask, (x - mean) * rstd, 0.)        # shape (BLOCK_SIZE_N)
    dydx_normed = tl.where(mask, w * dLdy, 0.)                  # shape (BLOCK_SIZE_N)
    # c1 and c2 are just intermediary labels; the names don't have any real meaning
    c1 = tl.sum(x_normalized * dydx_normed, axis=0) / N         # shape (1)
    c2 = tl.sum(dydx_normed, axis=0) / N                        # shape (1)
    dLdx = (dydx_normed - (x_normalized * c1 + c2)) * rstd      # shape (BLOCK_SIZE_N)

    # Write dLdx back to DRAM
    tl.store(dLdx_ptr + cols, dLdx, mask=mask)

    # Here we'll accumulate partial sums for dLdw and dLdb, meaning these are only the single rows of 
    #  the dLdw and dLdb gradients that this PID had the job of calculating
    dLdw_contribution = (dLdy * x_normalized).to(w.dtype)
    dLdb_contribution = (dLdy).to(w.dtype)

    # To start we figure out which lock ID corresponds to our PID and move our pointers accordingly
    lock_id = PID % GROUP_SIZE # so there are GROUP_SIZE number of locks
    # the first GROUP_SIZE entries in Lock hold the state of that lock in the entry locks_ptr for each pid
    locks_ptr += lock_id
    # the next GROUP_SIZE entries hold the count of how many accumulations have already happened on that lock
    count_ptr = locks_ptr + GROUP_SIZE
    # then we figre out which row of dLdw_intermediate and dLdb_intermediate we're meant to point to
    dLdw_intermediate_ptrs = dLdw_intermediate_ptr + lock_id * N + cols 
    dLdb_intermediate_ptrs = dLdb_intermediate_ptr + lock_id * N + cols 
        # we can use N in place of a .stride() here since these tensors are generated specifically for 
        #  this purpose and therefore guaranteed to be contiguous in memory
    
    # atomic_cas() compares the contents of a memory location with a given value and, 
    #  only if they are the same, modifies the contents of that memory location to a new given value.
    # cas: compare and swap, 永远返回旧值, 第二个参数cmp, 第三个参数val, 如果旧值等于cmp, 则将val写入内存位置, 否则不修改内存位置
    while tl.atomic_cas(locks_ptr, 0, 1) == 1:
        pass
        # so here, we're looking at the location locks_ptr_ptr and:
        # - If it's 0 (unlocked), change it to 1 (locked) and return 0 (False) to exit the while loop
        # - If it's 1 (already locked), leave it as 1 and return 1 (True) so that we stay in the while loop
    
    # then here we grab the number of times this lock position has already been accumulated into
    count = tl.load(count_ptr) # shape (1)
    if count == 0: # if this PID is the first one to access the lock
        # then no need to do the memory reads & flops; we can just set the row of dLdw_intermediate & 
        #  dLdB_intermediate equal to dLdw_contribution and dLdb_contribution (done below, outside the if/else)
        # atomic_xchg() sets the value at Count_ptr equal to 1 so the next PID knows we've been here
        tl.atomic_xchg(count_ptr, 1)
    else: # but if this is not the first pid in the accumulation process,
        # then we've actually gotta accumulate by grabbing the values already there in 
        #  DRAM and adding them to the rows of dLdw_contribution and dLdb_contribution that our PID generated
        dLdw_contribution += tl.load(dLdw_intermediate_ptrs, mask=mask) # we load and add in one step (+= operator)
        dLdb_contribution += tl.load(dLdb_intermediate_ptrs, mask=mask) #  so as not to consume unnecessary SRAM
    
    # now we get to store our accumulated values back to DRAM
    tl.store(dLdw_intermediate_ptrs, dLdw_contribution, mask=mask)
    tl.store(dLdb_intermediate_ptrs, dLdb_contribution, mask=mask)

    # and finally release the lock so that any pids waiting in their while loop can take their turn
    tl.atomic_xchg(locks_ptr, 0) # we set the value at our lock equal to 0
    # whichever pid gets to the 0 value first with its .atomic_cas() will get to go next



@triton.jit
def _layernorm_backward_dLdw_dLdb(
    dLdw_intermediate_ptr,  dLdb_intermediate_ptr, # pointers to first entries of tensors of shape (GROUP_SIZE, N)
    dLdw_ptr, dLdb_ptr,                            # pointers to first entries of tensors of shape (N)
    GROUP_SIZE,  N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    # our PIDs are split up within the N dimension
    PID = tl.program_id(0)
    col_ptrs = PID * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # here is where we'll accumulate the stored group values into as we read them
    dLdw_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    dLdb_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate through the rows of _dLdw and _dLdb to sum them up
    for i in range(0, GROUP_SIZE, BLOCK_SIZE_M):
        row_ptrs = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (row_ptrs[:, None] < GROUP_SIZE) & (col_ptrs[None, :] < N)
        offsets = row_ptrs[:, None] * N + col_ptrs[None, :]

        # load the partial sums from all that group locking nonsense earlier and add them to our final output
        dLdw_acc += tl.load(dLdw_intermediate_ptr + offsets, mask=mask, other=0.) 
        dLdb_acc += tl.load(dLdb_intermediate_ptr + offsets, mask=mask, other=0.)
            # masked-out values get set to 0 so they don't affect sum

    # sum along our BLOCK_SIZE_M dimension to get the final BLOCK_SIZE_N chunk of dLdw & dLdb that this 
    #  PID was assigned to
    sum_dLdw = tl.sum(dLdw_acc, axis=0) # shape (BLOCK_SIZE_N)
    sum_dLdb = tl.sum(dLdb_acc, axis=0)

    # Write the final sum to the output.
    tl.store(dLdw_ptr + col_ptrs, sum_dLdw, mask=col_ptrs < N)
    tl.store(dLdb_ptr + col_ptrs, sum_dLdb, mask=col_ptrs < N)



class LayerNorm(torch.autograd.Function):
    @staticmethod   # 不依赖类的实例和属性
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        M, N = x.reshape(-1, x.shape[-1]).shape
        
        y = torch.empty_like(x)
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
        
        MAX_FUSED_SIZE = 65536 // x.element_size()
        # 一个 SRAM 上可以存 64KB(65536B)
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB")
        
        num_warps = min(8, max(1, BLOCK_SIZE // 256))  # [1, 8]之间, 希望为 BLOCK_SIZE // 256
        # 相当于想让一个 warp 处理 256 大小
        
        _layernorm_fwd_kernel[(M,)](
            x, y, weight, bias,
            mean, rstd,
            x.stride(0),
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,  # triton 里有这个参数, 自定义里不需要再写
        )
        
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps

        return y
    
    @staticmethod
    def backward(ctx, dy):
        x, w, b, mean, rstd = ctx.saved_tensors
        M, N = x.reshape(-1, x.shape[-1]).shape

        dw = torch.empty_like(w)
        db = torch.empty_like(b)
        dx = torch.empty_like(x)
        
        GROUP_SIZE = 64
        if N <= 8192: GROUP_SIZE = 96
        if N <= 4096: GROUP_SIZE = 128
        if N <= 1024: GROUP_SIZE = 256
        
        dw_intermediate = torch.zeros((GROUP_SIZE, N), dtype=torch.float32, device=w.device)
        db_intermediate = torch.zeros((GROUP_SIZE, N), dtype=torch.float32, device=b.device)
        
        locks = torch.zeros((GROUP_SIZE * 2), dtype=torch.int32, device=w.device)
        
        _layernorm_backward_dLdx[(M, )](  # parallelize across rows
            x, dx, dy, 
            w, dw_intermediate, db_intermediate, 
            mean, rstd, 
            locks,  
            x.stride(0), N,  # dynamic run-time variables
            GROUP_SIZE = GROUP_SIZE, BLOCK_SIZE_N = ctx.BLOCK_SIZE, num_warps = ctx.num_warps)
        
        grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])] # parallelize within rows
        _layernorm_backward_dLdw_dLdb[grid](
            dw_intermediate, db_intermediate, dw, db, 
            min(GROUP_SIZE, M), N,  # run-time integer values
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, # heuristically chosen compile-time values
        )
        
        return dx, None, dw, db, None 
        

layernorm = LayerNorm.apply


def test_layernorm_kernel(M, N, dtype, eps=1e-5, device=DEVICE):
    x = -2.3 + 0.5 * torch.randn((M, N), dtype=dtype, device=device)
    weight = torch.rand((N, ), dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand((N, ), dtype=dtype, device=device, requires_grad=True)
    dLdy = 0.1 * torch.randn_like(x)
    # setting requires_grad to True here instead of x's initial definition means the graph doesn't have to move through 
    #  the -2.3 and 0.5 operations. That's not a big deal here for testing but if we didn't do it in the benchmark then
    #  those results would be confounded by the kernels pytorch implements for entry-wise multiplication and addition
    x.requires_grad_(True)
    # forward pass
    y_tri = layernorm(x, (N,), weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, (N,), weight, bias, eps).to(dtype)
    torch.testing.assert_close(y_tri, y_ref, atol=1e-2, rtol=0) 
    print("Passed fwd")
    # backward pass (triton)
    y_tri.backward(dLdy, retain_graph=True) # this writes directly to x.grad, weight.grad and bias.grad
        # retain_graph is used to control whether the computation graph should be kept in memory after the backward pass. 
        # Setting retain_graph=True allows you to perform multiple backward passes on the same graph, but it can increase 
        # memory usage, so it's generally recommended to use it only when necessary for a scenario like this
    # This detaches our gradients so that we can run pytorch on the same input tensors and test against each other later
    dLdx_tri, dLdw_tri, dLdb_tri = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None
    y_ref.backward(dLdy, retain_graph=True)
    dLdx_ref, dLdw_ref, dLdb_ref = [_.grad.clone() for _ in [x, weight, bias]]

    torch.testing.assert_close(dLdx_tri, dLdx_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdb_tri, dLdb_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdw_tri, dLdw_ref, atol=1e-2, rtol=0)
        # rtol=0 means we don't use relative tolerance 
    print("Passed bwd")
    
    
    
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)], # if you increase past 32 the kernel will break since features become larger than 64kb
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='layer-norm-backward',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'backward'}, # so we're actually only benchmarking the backward pass
    ))
def benchmark(M, N, dtype, provider, mode='backward', eps=1e-5, device=DEVICE):
    # create data
    x_shape = (M, N)
    w_shape = (N, )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)#, requires_grad=True)
    dLdy = .1 * torch.randn_like(x)
    x.requires_grad_(True) 
    quantiles = [0.5, 0.05, 0.95]

    def y_fwd():
        if provider == "triton":
            return layernorm(x, w_shape, weight, bias, eps) 
        if provider == "torch":
            return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps) 

    # forward pass
    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    # backward pass
    if mode == 'backward':
        y = y_fwd()
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)  # noqa: F811, E704
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dLdy, retain_graph=True), quantiles=quantiles,
                                                     grad_to_none=[x], rep=500)
    return gbps(ms), gbps(max_ms), gbps(min_ms)



if __name__ == "__main__":
    # always run unit-tests
    test_layernorm_kernel(1151, 8192, torch.float16)

    # Only run benchmark if explicitly requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)