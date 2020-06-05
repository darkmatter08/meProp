'''
Train MLPs for MNIST using meProp
'''
import os
import sys
from argparse import ArgumentParser

import torch

from data import get_mnist
from util import TestGroup

# Other profilng options:
# https://pytorch.org/docs/stable/bottleneck.html
# python -m torch.utils.bottleneck main_joint.py [args]

# https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.profile
# torch.autograd.profiler.profile()

def get_args():
    # a simple use example (not unified)
    parser = ArgumentParser()
    parser.add_argument(
        '--n_epoch', type=int, default=20, help='number of training epochs')
    parser.add_argument(
        '--d_hidden', type=int, default=512, help='dimension of hidden layers')
    parser.add_argument(
        '--n_layer',
        type=int,
        default=3,
        help='number of layers, including the output layer')
    parser.add_argument(
        '--d_minibatch', type=int, default=10, help='size of minibatches')
    parser.add_argument(
        '--dropout', type=float, default=0, help='dropout rate')
    parser.add_argument(
        '--k',
        type=int,
        default=80,
        help='k in meProp (if invalid, e.g. 0, do not use meProp)')
    parser.add_argument(
        '--layer_type',
        default='unified',
        choices=['meProp', 'meProp_unified', 'shawn_unified', 'crs', 'pyTorch'],
        help='which layer type to use.'
    )
    parser.add_argument(
        '--strategy',
        default='det_top_k',
        choices=('random', 'det_top_k', 'nps'),
        help='CRS sampling strategy. Only used if `--layer_type crs` is set.'
    )
    parser.add_argument(
        '--random_seed', type=int, default=12976, help='random seed')
    parser.add_argument(
        '--profile',
        action='store_true',
        default=False,
        help='do GPU profiling of the code')
    return parser.parse_args()


def main():
    args = get_args()
    print('main() args=', args)
    if args.profile:
        import torch.cuda.profiler as profiler
        import pyprof2
        pyprof2.init()

    trn, dev, tst = get_mnist()

    group = TestGroup(
        args,
        trn,
        args.d_minibatch,
        args.d_hidden,
        args.n_layer,
        args.dropout,
        devset=dev,
        tstset=tst,
        # cudatensor=True,
        file=sys.stdout,
        layer_type=args.layer_type,
        strategy=args.strategy)

    if args.profile:
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        with torch.autograd.profiler.emit_nvtx():
            group.run(0, args.n_epoch)
        # print('profiler result:')
        # print('sort_by="self_cpu_time_total"')
        # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        # print('sort_by="cuda_time_total"')
        # print(prof.key_averages().table(sort_by="cuda_time_total"))
    else:
        group.run(0, args.n_epoch)

    if args.profile:
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        with torch.autograd.profiler.emit_nvtx():
            group.run(args.k, args.n_epoch)
        # print('profiler result:')
        # print('sort_by="self_cpu_time_total"')
        # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        # print('sort_by="cuda_time_total"')
        # print(prof.key_averages().table(sort_by="cuda_time_total"))
    else:
        group.run(args.k, args.n_epoch)


if __name__ == '__main__':
    main()
