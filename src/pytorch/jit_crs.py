import torch
PROFILE = False

if 1:
    import torch.cuda.profiler as profiler
    import pyprof2
    pyprof2.init()
    PROFILE = True

'''
    # CRS --strategy random
    if strategy == 'random':
        # sample k random indexes!
        indexes_all = torch.randperm(in_size, device=w.device)
        indexes = indexes_all[:k]
    # CRS --strategy det_top_k
    # CRS --strategy first_k
    elif strategy == 'first_k':
        indexes = torch.arange(k, device=w.device)
    # CRS --strategy single_norm
    elif strategy == 'single_norm':
        col_norms_A = torch.norm(x, p=None, dim=0)
        _, indexes = torch.topk(col_norms_A, k)
    else:
        pass
'''

def crs_det_top_k_profile(w, x, dy, k):
    ###
    # FORWARD PASS
    ###
    col_norms_A = torch.norm(x, p=None, dim=0)  # compute in_size norms, each with b elements
    row_norms_B = torch.norm(w.t(), p=None, dim=1)  # compute in_size norms, each with out_size elements
    assert col_norms_A.shape == row_norms_B.shape
    norm_products = col_norms_A * row_norms_B
    assert norm_products.shape == col_norms_A.shape
    # same device as norm_products
    _, indexes = torch.topk(norm_products, k)  # topk over in_size elements

    # index_select_ like
    x_sampled = x[:, indexes]
    w_t_sampled = w.t()[indexes, :]
    # no expansion
    y = torch.mm(x_sampled, w_t_sampled)  # (b, k) @ (k, out_size) = (b, out_size)

    ###
    # BACKWARD PASS
    # dw
    ###
    # index_select
    sampled_x = x[:, indexes]
    # mm
    partial_dw = torch.mm(dy.t(), sampled_x)  # (out, b) @ (b, k) = (out, k)  # save over the in dimension here. Need a good `k/in` ratio.
    # Can do a second CRS here over b dim. Can save further on large b
    # alloc
    dw = torch.zeros_like(w)  # (out, in)
    # expand
    dw[:, indexes] = partial_dw  # alternative to scatter_ or index_copy_

    ###
    # BACKWARD PASS
    # dx
    ###
    partial_dx = torch.mm(dy, w[:, indexes])  # (b, out) @ (out, k) = (b, k)  # save over the in dimension. Need a good `k/in` ratio.
    dx = torch.zeros_like(x)
    dx[:, indexes] = partial_dx  # (b, in)

    return y, dw, dx


def crs_det_top_k_jit(w, x, dy, k):
    ###
    # FORWARD PASS
    ###
    col_norms_A = torch.norm(x, p=None, dim=0)  # compute in_size norms, each with b elements
    row_norms_B = torch.norm(w.t(), p=None, dim=1)  # compute in_size norms, each with out_size elements
    assert col_norms_A.shape == row_norms_B.shape
    norm_products = col_norms_A * row_norms_B
    assert norm_products.shape == col_norms_A.shape
    # same device as norm_products
    _, indexes = torch.topk(norm_products, k)  # topk over in_size elements

    # index_select_ like
    x_sampled = x[:, indexes]
    w_t_sampled = w.t()[indexes, :]
    # no expansion
    y = torch.mm(x_sampled, w_t_sampled)  # (b, k) @ (k, out_size) = (b, out_size)

    ###
    # BACKWARD PASS
    # dw
    ###
    # index_select
    sampled_x = x[:, indexes]
    # mm
    # (out, b) @ (b, k) = (out, k)  # save over the in dimension here. Need a good `k/in` ratio.
    partial_dw = torch.mm(dy.t(), sampled_x)
    # Can do a second CRS here over b dim. Can save further on large b
    # alloc
    dw = torch.zeros_like(w)  # (out, in)
    # expand
    dw[:, indexes] = partial_dw  # alternative to scatter_ or index_copy_

    ###
    # BACKWARD PASS
    # dx
    ###
    # (b, out) @ (out, k) = (b, k)  # save over the in dimension. Need a good `k/in` ratio.
    partial_dx = torch.mm(dy, w[:, indexes])
    dx = torch.zeros_like(x)
    dx[:, indexes] = partial_dx  # (b, in)

    return y, dw, dx


def full_calc(w, x, dy):
    # full forward mm()
    # (b, in_size) @ (in_size, out_size) = (b, out_size)
    full_y = torch.mm(x, w.t())

    # full backward mm dw
    # (out, b) @ (b, in) = (out, in)
    full_dw = torch.mm(dy.t(), x)

    # full backward mm dx
    # (b, out) @ (out, in) = (b, in)
    full_dx = torch.mm(dy, w)
    
    return full_y, full_dw, full_dx


# jit_crs_det_top_k = torch.jit.script(crs_det_top_k_jit)
jit_crs_det_top_k = torch.jit.script(crs_det_top_k_profile)

STEPS = 100

batch_sizes = [2**i for i in range(12)]
dim_sizes = [2**i for i in range(6, 12)]
k_values = [64, 256]
print('steps,profiled_steps,batch,k,input_dim,output_dim,mean runtime_jit (ms),mean runtime_nojit (ms),mean runtime_full (ms)')
for b in batch_sizes:
    for k in k_values:
        for in_size in dim_sizes:
            if k > in_size:
                continue
            for out_size in dim_sizes:
                profiled_steps = 0
                for step in range(1, STEPS+1):
                    out_size, in_size, b, k = map(torch.tensor, (out_size, in_size, b, k))
                    w = torch.rand(out_size, in_size, device='cuda')
                    x = torch.rand(b, in_size, device='cuda')
                    dy = torch.rand(b, out_size, device='cuda')

                    # timers
                    start = torch.cuda.Event(True)
                    end = torch.cuda.Event(True)
                    runtime_jit = []
                    runtime_nojit = []
                    runtime_full = []

                    if PROFILE and step % STEPS == 0:
                        print('starting profiling now...')
                        profiler.start()

                    start.record()
                    jit_crs_det_top_k(w, x, dy, k)
                    end.record()
                    end.synchronize()
                    runtime_jit.append(start.elapsed_time(end))

                    start.record()
                    crs_det_top_k_profile(w, x, dy, k)
                    end.record()
                    end.synchronize()
                    runtime_nojit.append(start.elapsed_time(end))

                    start.record()
                    full_calc(w, x, dy)
                    end.record()
                    end.synchronize()
                    runtime_full.append(start.elapsed_time(end))

                    if PROFILE and step % STEPS == 0:
                        profiler.stop()
                        print('ended profiling now...')
                        profiled_steps += 1

                print('{},{},{},{},{},{},{},{},{}'.format(
                        STEPS,
                        profiled_steps,
                        b,
                        k,
                        in_size,
                        out_size,
                        torch.mean(torch.tensor(runtime_jit)).item(),
                        torch.mean(torch.tensor(runtime_nojit)).item(),
                        torch.mean(torch.tensor(runtime_full)).item()
                    )
                )
