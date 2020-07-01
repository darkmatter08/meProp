'''
Benchmark norm operation only.
'''

import torch
PROFILE = False

if 0:
    import torch.cuda.profiler as profiler
    import pyprof2
    pyprof2.init()
    PROFILE = True

STEPS = 1
k = 80
for step in range(STEPS):
    # data, target = data.cuda(), target.view(-1).cuda()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)

    runtime = []

    # if PROFILE and step % 50 == 0:  # arbitrary step to profile.
    if PROFILE:
        print('starting profiling now...')
        profiler.start()
    start.record()

    '''
    # Actual compute to be profiled...
    data = torch.rand(8192, 8192, device='cuda')
    
    # Similar to:
    # norm reduce_kernel T=(256,8192), fp32,	72671	W1	col_norms	        col_norms_A = torch.norm(A, dim=0)
    # except A has fewer rows (32x fewer)
    # norm reduce_kernel T=(8192, 8192), fp32,	1816295	1888966	row_norms	        row_norms_B = torch.norm(B, dim=1)
    col_norms = torch.norm(data, dim=0)
    row_norms = torch.norm(data, dim=1)

    # Similar to:
    # __matmul__	maxwell_sgemm_128x128_nn	A=(256,80),B=(80,8192),fp32,	73311		Compute D (i.e. y=x@w.T+b)	        D = cols_A @ rows_B	crs_mm -- det_top_k
    torch.mm(data[:256, :k], data[:k, :])

    # full matmul
    torch.mm(data, data)

    # Alloc an empty tensor on CUDA.
    empty_tensor = torch.empty(8192, 8192, device='cuda')
    # Zero in place.
    empty_tensor.zero_()

    # alloc zero tensor
    zero_tensor = torch.zeros(8192, 8192, device='cuda')
    '''
    
    ''' 
    # dw.shape == w.shape == (out, in)
    # x.shape = (b, in)
    # y.shape = (b, out)
    # dy.T.shape = (out, b)
    # full_dw = dy.T @ x
    # full_dw = (out, b) @ (b, in)

    # partial_dw = (out, b) @ (b, k) = (out, k)
    partial_dw = dy.T @ x[:, indexes]
    # dw = (out, in)
    dw = torch.zeros_like(w)
    # copy into dw a matrix of (out, k)
    dw[:, indexes] = partial_dw  # alternative to scatter_ or index_copy_
    '''
    # indexes_all = torch.randperm(1000, device='cuda')  # not profiled.
    # indexes = indexes_all[:k]  # not profiled

    if 0:
        def crs_profile(out_size, in_size, b, k):
            w = torch.rand(out_size, in_size, device='cuda')
            x = torch.rand(b, in_size, device='cuda')

            ###
            # FORWARD PASS
            ###
            # CRS --strategy random
            if 0:
                # sample k random indexes!
                indexes_all = torch.randperm(in_size, device=w.device)
                indexes = indexes_all[:k]
            # CRS --strategy det_top_k
            if 1:
                col_norms_A = torch.norm(x, dim=0)  # compute in_size norms, each with b elements
                row_norms_B = torch.norm(w.T, dim=1)  # compute in_size norms, each with out_size elements
                assert col_norms_A.shape == row_norms_B.shape
                norm_products = col_norms_A * row_norms_B
                assert norm_products.shape == col_norms_A.shape
                # same device as norm_products
                _, indexes = torch.topk(norm_products, k)  # topk over in_size elements
            # CRS --strategy first_k
            if 0:
                indexes = torch.arange(k, device=w.device)
            # CRS --strategy single_norm
            if 0:
                col_norms_A = torch.norm(x, dim=0)
                _, indexes = torch.topk(col_norms_A, k)

            # index_select_ like
            x_sampled = x[:, indexes]
            w_t_sampled = w.T[indexes, :]
            # no expansion
            y = torch.mm(x_sampled, w_t_sampled)  # (b, k) @ (k, out_size) = (b, out_size)

            # full forward mm()
            full_y = torch.mm(x, w.T)  # (b, in_size) @ (in_size, out_size) = (b, out_size)

            ###
            # BACKWARD PASS
            # dw
            ###
            dy = torch.rand(b, out_size, device='cuda')
            # index_select
            sampled_x = x[:, indexes]
            # mm
            partial_dw = torch.mm(dy.T, sampled_x)  # (out, b) @ (b, k) = (out, k)  # save over the in dimension here. Need a good `k/in` ratio.
            # Can do a second CRS here over b dim. Can save further on large b
            # alloc
            dw = torch.zeros_like(w)  # (out, in)
            # expand
            dw[:, indexes] = partial_dw  # alternative to scatter_ or index_copy_

            # full backward mm dw
            full_dw = torch.mm(dy.T, x)  # (out, b) @ (b, in) = (out, in)

            ###
            # BACKWARD PASS
            # dx
            ###
            # partial_dx = dy @ w[:, indexes]  # (b, out) @ (out, k) = (b, k)  # save over the in dimension. Need a good `k/in` ratio.
            # dx = torch.zeros_like(x)
            # dx[:, indexes] = partial_dx  # (b, in)

            # full backward mm dx
            # dx = dy @ w  # (b, out) @ (out, in) = (b, in)

        
        n_trials = 10000
        batch_sizes = [2**i for i in range(12)]
        dim_sizes = [2**i for i in range(6, 12)]
        k_values = [64, 256]
        # k_values = [64]

        print('batch, k, input_dim, output_dim')
        for batch in batch_sizes:
            for k in k_values:
                for input_dim in dim_sizes:
                    if k > input_dim:
                        continue
                    for output_dim in dim_sizes:
                        print('{},{},{},{}'.format(batch, k, input_dim, output_dim))
                        crs_profile(out_size=output_dim, in_size=input_dim, b=batch, k=k)
                        # print('{},{},{},{},{},{}'.format(batch, k, input_dim, output_dim, total_forward_time, total_forward_time / n_trials))


    ### TODO: layer_norm
    # x = self.final_layer_norm(x)
    # x = self.self_attn_layer_norm(x)
    # The dim of x is torch.Size([21, 192, 512])

    end.record()
    end.synchronize()
    runtime.append(start.elapsed_time(end))

    # if PROFILE and step % 50 == 0:  # arbitrary step to profile.
    if PROFILE:
        profiler.stop()
        print('ended profiling now...')
        # NOTE: don't profile stepping.

print('STEPS executed:', STEPS)
print('mean runtime (ms):', torch.mean(torch.tensor(runtime)).item())
print('done')
