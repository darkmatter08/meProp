'''
Benchmark norm operation only.
'''

import torch
PROFILE = False

if 1:
    import torch.cuda.profiler as profiler
    import pyprof2
    pyprof2.init()
    PROFILE = True

k = 80
for step in range(1000):
    # data, target = data.cuda(), target.view(-1).cuda()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)

    runtime = []

    if PROFILE and step % 50 == 0:  # arbitrary step to profile.
        print('starting profiling now...')
        profiler.start()
    start.record()

    # Actual compute to be profiled...
    data = torch.rand(256, 256, device='cuda')
    
    # Similar to:
    # norm reduce_kernel T=(256,8192), fp32,	72671	W1	col_norms	        col_norms_A = torch.norm(A, dim=0)
    # except A has fewer rows (32x fewer)
    # norm reduce_kernel T=(8192, 8192), fp32,	1816295	1888966	row_norms	        row_norms_B = torch.norm(B, dim=1)
    col_norms = torch.norm(data, dim=0)
    row_norms = torch.norm(data, dim=1)

    # Similar to:
    # __matmul__	maxwell_sgemm_128x128_nn	A=(256,80),B=(80,8192),fp32,	73311		Compute D (i.e. y=x@w.T+b)	        D = cols_A @ rows_B	crs_mm -- det_top_k
    torch.mm(data[:256, :k], data[:k, :])

    end.record()
    end.synchronize()
    runtime.append(start.elapsed_time(end))

    if PROFILE and step % 50 == 0:  # arbitrary step to profile.
        profiler.stop()
        print('ended profiling now...')
        # NOTE: don't profile stepping.

print('done')
