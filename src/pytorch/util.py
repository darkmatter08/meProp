'''
Helper class to facilitate experiments with different k
'''
import sys
import time
from statistics import mean

import torch
import torch.cuda
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable  # TODO move away from Variable.

from model import NetLayer


class TestGroup(object):
    '''
    A network and k in meprop form a test group.
    Test groups differ in minibatch size, hidden features, layer number and dropout rate.
    '''

    def __init__(self,
                 args,
                 trnset,
                 mb,
                 hidden,
                 layer,
                 dropout,
                 unified,
                 devset=None,
                 tstset=None,
                 cudatensor=False,
                 file=sys.stdout,
                 crs=False,
                 strategy=None):
        self.args = args
        self.mb = mb  # mini-batch size
        self.hidden = hidden  # hidden dimension
        self.layer = layer  #???
        self.dropout = dropout  # dropout rate
        self.file = file  # output file (default=stdout)
        self.trnset = trnset  # training set
        self.unified = unified  # unified -- refers to how batching is implemented
        self.crs = crs
        self.strategy = strategy

        if cudatensor:  # dataset is on GPU
            self.trainloader = torch.utils.data.DataLoader(
                trnset, batch_size=mb, num_workers=0)
            if tstset:
                self.testloader = torch.utils.data.DataLoader(
                    tstset, batch_size=mb, num_workers=0)
            else:
                self.testloader = None
        else:  # dataset is on CPU, using prefetch and pinned memory to shorten the data transfer time
            self.trainloader = torch.utils.data.DataLoader(
                trnset,
                batch_size=mb,
                shuffle=True,
                num_workers=1,
                pin_memory=True)
            if devset:
                self.devloader = torch.utils.data.DataLoader(
                    devset,
                    batch_size=mb,
                    shuffle=False,
                    num_workers=1,
                    pin_memory=True)
            else:
                self.devloader = None
            if tstset:
                self.testloader = torch.utils.data.DataLoader(
                    tstset,
                    batch_size=mb,
                    shuffle=False,
                    num_workers=1,
                    pin_memory=True)
            else:
                self.testloader = None
        self.basettime = None
        self.basebtime = None

    def reset(self):
        '''
        Reinit the trainloader at the start of each run,
        so that the traning examples is in the same random order
        '''
        torch.manual_seed(self.args.random_seed)
        self.trainloader = torch.utils.data.DataLoader(
            self.trnset,
            batch_size=self.mb,
            shuffle=True,
            num_workers=1,
            pin_memory=True)

    def _train(self, model, opt):
        '''
        Train the given model using the given optimizer
        Record the time and loss
        '''
        model.train()
        ftime = []
        btime = []
        utime = []
        tloss = 0
        for bid, (data, target) in enumerate(self.trainloader):
            data, target = Variable(data).cuda(), Variable(
                target.view(-1)).cuda()
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)

            start.record()
            opt.zero_grad()
            end.record()
            end.synchronize()
            # utime.append(start.elapsed_time(end))

            start.record()
            output = model(data)
            loss = F.nll_loss(output, target)
            end.record()
            end.synchronize()
            ftime.append(start.elapsed_time(end))  # time for forward propogation

            start.record()
            loss.backward()
            end.record()
            end.synchronize()
            btime.append(start.elapsed_time(end))  # time for backward propogation

            start.record()
            opt.step()
            end.record()
            end.synchronize()
            utime.append(start.elapsed_time(end))  # time to step optimizer/apply weight updates

            tloss += loss.data.item()
        tloss /= len(self.trainloader)
        return tloss, ftime, btime, utime

    def _evaluate(self, model, loader, name='test'):
        '''
        Use the given model to classify the examples in the given data loader
        Record the loss and accuracy.
        '''
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in loader:
            data, target = Variable(
                data, volatile=True).cuda(), Variable(target).cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target).data.item()
            pred = output.data.max(1)[
                1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

        test_loss = test_loss
        test_loss /= len(
            loader)  # loss function already averages over batch size
        print(
            '{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                name, test_loss, correct,
                len(loader.dataset), 100. * correct / len(loader.dataset)),
            file=self.file,
            flush=True)
        return 100. * correct / len(loader.dataset)

    def run(self, k=None, epoch=None):
        '''
        Run a training loop.
        '''
        if k is None:
            k = self.args.k
        if epoch is None:
            epoch = self.args.n_epoch
        print(
            'mini-batch size: {}, hidden size: {}, number_layers: {}, dropout: {}, k: {}'.
            format(self.mb, self.hidden, self.layer, self.dropout, k),
            file=self.file)
        # Init the model, the optimizer and some structures for logging
        self.reset()

        model = NetLayer(self.hidden, k, self.layer, self.dropout,
                         self.unified, crs=self.crs, strategy=self.strategy)
        print('==='*10)
        print('model:')
        print(model)
        print('==='*10)
        model.reset_parameters()
        model.cuda()

        opt = optim.Adam(model.parameters())  # default lr=0.001, betas=(0.9, 0.999), eps=1e-08,

        acc = 0  # best dev. acc.
        accc = 0  # test acc. at the time of best dev. acc.
        e = -1  # best dev iteration/epoch

        times = []
        losses = []
        ftime = []
        btime = []
        utime = []

        # training loop
        for t in range(epoch):
            print('{}:'.format(t), end='', file=self.file, flush=True)
            # train
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)
            start.record()
            loss, ft, bt, ut = self._train(model, opt)
            end.record()
            end.synchronize()
            ttime = start.elapsed_time(end)

            times.append(ttime)
            losses.append(loss)
            ftime += ft
            btime += bt
            utime += ut
            # predict
            curacc = self._evaluate(model, self.devloader, 'dev')
            if curacc > acc:  # only eval on test set if dev accuracy has improved.
                e = t
                acc = curacc
                accc = self._evaluate(model, self.testloader, '    test')
        etime = [sum(t) for t in zip(ftime, btime, utime)]  # overall execution time
        print('Best performance (by dev_acc):')
        print(
            'dev_acc={:.2f} | test_acc={:.2f} at epoch={}'.format(acc, accc, e),
            file=self.file,
            flush=True)
        means_stds = [(torch.mean(times), torch.std(times)) for times in map(torch.tensor, (ftime, btime, utime))]
        # print('mean__std\nfwd_time:{:.4f}__{:.4f}\tbw_time:{:.4f}__{:.4f}\tstep_time:{:.4f}__{:.4f}'.format(
        #     *means_stds[0], *means_stds[1], *means_stds[2]))
        print('(mean,std)  fwd_time, bw_time, step_time\n{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}'.format(
            *means_stds[0], *means_stds[1], *means_stds[2]))
        print('', file=self.file)

    def _stat(self, name, t, agg=mean):
        return '{:<5}:\t{:8.3f}; {}'.format(
            name, agg(t), ', '.join(['{:8.2f}'.format(x) for x in t]))
