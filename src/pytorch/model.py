'''
The MLP model for MNIST
'''

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import Linear, LinearCRS, LinearShawn


class NetLayer(nn.Module):
    '''
    A complete network(MLP) for MNSIT classification.
    
    Input feature is 28*28=784
    Output feature is 10
    Hidden features are of hidden size
    
    Activation is ReLU
    '''

    def __init__(self, hidden, k, layer, dropout=None,
        layer_type=None, strategy=None,
        ):
        # layer: number of layers in this network.
        super(NetLayer, self).__init__()
        self.k = k
        self.layer = layer
        self.dropout = dropout
        if layer_type is None:
            raise ValueError('need to specify layer_type')
        if layer_type == 'crs':
            assert strategy in ('random', 'det_top_k', 'nps')
        self.layer_type = layer_type
        self.strategy = strategy
        self.model = nn.Sequential(self._create(hidden, k, layer, dropout))
        
    def _create(self, hidden, k, layer, dropout=None):
        if layer == 1:
            return OrderedDict([Linear(784, 10, 0)])
        d = OrderedDict()
        for i in range(layer):
            if i == 0:  # input layer case
                if self.layer_type == 'crs':
                    d['linearCRS' + str(i)] = LinearCRS(784, hidden, k, strategy=self.strategy)
                elif self.layer_type == 'shawn_unified':
                    d['linear_shawn_unified' +str(i)] = LinearShawn(784, hidden, k)
                elif self.layer_type == 'meProp_unified':
                    d['linear_meProp_unified' + str(i)] = Linear(784, hidden, k, unified=True)
                elif self.layer_type == 'meProp':
                    d['linear_meProp' + str(i)] = Linear(784, hidden, k, unified=False)
                elif self.layer_type == 'pyTorch':
                    d['linear_pyTorch' + str(i)] = torch.nn.Linear(784, hidden)
                else:
                    raise ValueError('invalid layer type! {}'.format(self.layer_type))
                d['relu' + str(i)] = nn.ReLU()
                if dropout:
                    d['dropout' + str(i)] = nn.Dropout(p=dropout)
            elif i == layer - 1:  # final layer/readout layer.
                # Do not sample on last layer
                if self.layer_type in ('meProp_unified', 'shawn_unified'):
                    d['linear' + str(i)] = Linear(hidden, 10, 0, unified=True)
                elif self.layer_type in ('crs', 'meProp'):
                    d['linear' + str(i)] = Linear(hidden, 10, 0, unified=False)
                elif self.layer_type in ('pyTorch'):
                    d['linear_pyTorch' + str(i)] = torch.nn.Linear(hidden, 10)
                # unified=True for shawnunified, linear_meProp_unified
                # unified=False for linearCRS, linear_meProp
                # pytorch linear layer for linear_pyTorch
            else:  # standard middle layer
                if self.layer_type == 'crs':
                    d['linearCRS' + str(i)] = LinearCRS(hidden, hidden, k, strategy=self.strategy)
                elif self.layer_type == 'shawn_unified':
                    d['linear_shawn_unified' + str(i)] = LinearShawn(hidden, hidden, k)
                elif self.layer_type == 'meProp_unified':
                    d['linear_meProp_unified' + str(i)] = Linear(hidden, hidden, k, unified=True)
                elif self.layer_type == 'meProp':
                    d['linear_meProp' + str(i)] = Linear(hidden, hidden, k, unified=False)
                elif self.layer_type == 'pyTorch':
                    d['linear_pyTorch' + str(i)] = torch.nn.Linear(hidden, hidden)
                else:
                    raise ValueError('invalid layer type! {}'.format(self.layer_type))
                d['relu' + str(i)] = nn.ReLU()
                if dropout:
                    d['dropout' + str(i)] = nn.Dropout(p=dropout)
        return d

    def forward(self, x):
        return F.log_softmax(self.model(x.view(-1, 784)))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, type(Linear)):
                m.reset_parameters()
