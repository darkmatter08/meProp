# manual_compute_norm.py
import torch
PROFILE = False

if 0:
    import torch.cuda.profiler as profiler
    import pyprof2
    pyprof2.init()
    PROFILE = True


def manual_norm_computation(A):
    return torch.sqrt(torch.sum(torch.pow(A, 2), dim=-1))

def manual_norm_computation_square(A):
    return torch.sqrt(torch.sum(A * A, dim=-1))

jit_manual_norm_computation = torch.jit.script(manual_norm_computation)

jit_manual_norm_computation_square = torch.jit.script(manual_norm_computation_square)

if 1:  # validate all methods
    w = torch.rand(512, 512, device='cuda')
    gt = torch.norm(w, dim=-1)
    tmp0 = manual_norm_computation(w)
    assert torch.allclose(gt, tmp0)
    tmp1 = jit_manual_norm_computation(w)
    assert torch.allclose(gt, tmp1)
    tmp2 = jit_manual_norm_computation_square(w)
    assert torch.allclose(gt, tmp2)
    tmp3 = manual_norm_computation_square(w)
    assert torch.allclose(gt, tmp3)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

STEPS = 10000

dims = [2**i for i in range(6, 12)]  # (128, 512, 2048)
print('steps,profiled_steps,input_dim,output_dim,mean runtime_jit (ms),mean runtime_nojit (ms),mean runtime_full (ms),mean runtime_square_jit (ms),mean runtime_square_nojit (ms)')
for dim in dims:
    profiled_steps = 0
    runtime_jit = []
    runtime_nojit = []
    runtime_full = []
    runtime_square_jit = []
    runtime_square_nojit = []

    for step in range(1, STEPS+1):
        out_size, in_size = map(torch.tensor, (dim, dim))
        w = torch.rand(out_size, in_size, device='cuda')

        # timers
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)

        if PROFILE and step % STEPS == 0:
            print('starting profiling now...')
            profiler.start()

        start.record()
        jit_manual_norm_computation(w)
        end.record()
        end.synchronize()
        runtime_jit.append(start.elapsed_time(end))

        start.record()
        manual_norm_computation(w)
        end.record()
        end.synchronize()
        runtime_nojit.append(start.elapsed_time(end))

        start.record()
        torch.norm(w, dim=-1)
        end.record()
        end.synchronize()
        runtime_full.append(start.elapsed_time(end))

        start.record()
        jit_manual_norm_computation_square(w)
        end.record()
        end.synchronize()
        runtime_square_jit.append(start.elapsed_time(end))

        start.record()
        manual_norm_computation_square(w)
        end.record()
        end.synchronize()
        runtime_square_nojit.append(start.elapsed_time(end))

        if PROFILE and step % STEPS == 0:
            profiler.stop()
            print('ended profiling now...')
            profiled_steps += 1

    print('{},{},{},{},{},{},{},{},{}'.format(
        STEPS,
        profiled_steps,
        in_size,
        out_size,
        torch.mean(torch.tensor(runtime_jit)).item(),
        torch.mean(torch.tensor(runtime_nojit)).item(),
        torch.mean(torch.tensor(runtime_full)).item(),
        torch.mean(torch.tensor(runtime_square_jit)).item(),
        torch.mean(torch.tensor(runtime_square_nojit)).item(),
    )
    )

print('done')
