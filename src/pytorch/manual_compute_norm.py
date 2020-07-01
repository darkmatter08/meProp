# manual_compute_norm.py
import torch
PROFILE = False

if 0:
    import torch.cuda.profiler as profiler
    import pyprof2
    pyprof2.init()
    PROFILE = True

'''
# using %timeit; not sure if accurate because of CUDA async issues.
In [1]: import torch

In [2]: device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

In [3]: print(device)
cuda:0

In [4]: in_size, out_size = (2048, 2048)

In [5]: A = torch.rand(in_size, out_size, device=device)

In [6]: %timeit A_row_norm_gt = torch.norm(A, dim=-1)
135 µs ± 41.7 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

In [7]: %timeit A_manual = torch.sqrt(torch.sum(torch.pow(A, 2), dim=-1))  # expect a shape of (2048,1)
110 µs ± 27.1 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
'''

def manual_norm_computation(A):
    return torch.sqrt(torch.sum(torch.pow(A, 2), dim=-1))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
dims = [2**i for i in range(6, 12)]  # (128, 512, 2048)
print('steps,profiled_steps,input_dim,output_dim,mean runtime_baseline (ms),mean runtime_experiment (ms)')
for dim in dims:
    in_size, out_size = (dim, dim)
    A = torch.rand(in_size, out_size, device=device)

    STEPS = 100
    profiled_steps = 0
    for step in range(1, STEPS+1):

        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)

        runtime_baseline = []
        runtime_experiment = []

        # if PROFILE and step % 50 == 0:  # arbitrary step to profile.
        if PROFILE and step % STEPS == 0:
            print('starting profiling now...')
            profiler.start()
        start.record()

        A_row_norm_gt = torch.norm(A, dim=-1)  # expect a shape of (2048, 1)

        end.record()
        end.synchronize()
        runtime_baseline.append(start.elapsed_time(end))

        start.record()
        # A_manual = manual_norm_computation(A)  # expect a shape of (2048,1)
        A_manual = torch.sqrt(torch.sum(torch.pow(A, 2), dim=-1))  # expect a shape of (2048,1)
        end.record()
        end.synchronize()
        runtime_experiment.append(start.elapsed_time(end))
        
        # do two experiments, profile with timing and profile w/o timing. compare them, are they different?

        # if PROFILE and step % 50 == 0:  # arbitrary step to profile.
        if PROFILE and step % STEPS == 0:
            profiler.stop()
            print('ended profiling now...')
            profiled_steps += 1

        if not torch.allclose(A_row_norm_gt, A_manual):
            print('mismatch!')
            import ipdb; ipdb.set_trace()

    # print('A.shape:', A.shape)
    # print('STEPS executed:', STEPS)
    # print('profiled_steps:', profiled_steps)
    # print('mean runtime_baseline (ms):', torch.mean(torch.tensor(runtime_baseline)).item())
    # print('mean runtime_experiment (ms):', torch.mean(torch.tensor(runtime_experiment)).item())
    # print('steps,profiled_steps,input_dim,output_dim,mean runtime_baseline (ms),mean runtime_experiment (ms)')
    print('{},{},{},{},{},{}'.format(
        STEPS,
        profiled_steps,
        in_size,
        out_size,
        torch.mean(torch.tensor(runtime_baseline)).item(),
        torch.mean(torch.tensor(runtime_experiment)).item()
        )
    )

print('done')
