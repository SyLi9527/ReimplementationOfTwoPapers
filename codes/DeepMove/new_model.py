# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
# ############# simple rnn model ####################### #


class TrajPreSimple(object):
    """baseline rnn model"""

    def __init__(self, parameters):
        super(TrajPreSimple, self).__init__()
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.hidden_size = parameters.hidden_size
        self.use_cuda = parameters.use_cuda
        self.rnn_type = parameters.rnn_type

        self.loc_emb = np.random.normal(
            0, 1, size=(self.loc_size, self.loc_emb_size))
        self.tim_emb = np.random.normal(
            0, 1, size=(self.tim_size, self.tim_emb_size))

        self.input_size = self.loc_emb_size + self.tim_emb_size
        self.hidden_weights = np.random.uniform(-np.sqrt(1 / self.hidden_size),
                                                np.sqrt(1 / self.hidden_size), (self.hidden_size, self.hidden_size))
        # if self.rnn_type == 'GRU':
        #     self.rnn = nn.GRU(input_size, self.hidden_size, 1)
        # elif self.rnn_type == 'LSTM':
        #     self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
        if self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, self.hidden_size, 1)
        self.init_weights()

        self.fc = nn.Linear(self.hidden_size, self.loc_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)

    def init_weight_input(self, loc, tim):
        S = np.zeros((self.loc_emb_size, self.hidden_size))
    # def init_weights(self):
    #     """
    #     Here we reproduce Keras default initialization weights for consistency with Keras version
    #     """
    #     ih = (param.data for name, param in self.named_parameters()
    #           if 'weight_ih' in name)
    #     hh = (param.data for name, param in self.named_parameters()
    #           if 'weight_hh' in name)
    #     b = (param.data for name, param in self.named_parameters() if 'bias' in name)

    #     for t in ih:
    #         nn.init.xavier_uniform_(t)
    #     for t in hh:
    #         nn.init.orthogonal_(t)
    #     for t in b:
    #         nn.init.constant_(t, 0)

    # def forward(self, loc, tim):
    #     h1 = Variable(torch.zeros(1, 1, self.hidden_size))
    #     c1 = Variable(torch.zeros(1, 1, self.hidden_size))

    #     loc_emb = self.emb_loc(loc)
    #     tim_emb = self.emb_tim(tim)
    #     x = torch.cat((loc_emb, tim_emb), 2)
    #     x = self.dropout(x)

    #     if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
    #         out, h1 = self.rnn(x, h1)
    #     elif self.rnn_type == 'LSTM':
    #         out, (h1, c1) = self.rnn(x, (h1, c1))
    #     out = out.squeeze(1)
    #     out = F.selu(out)
    #     out = self.dropout(out)

    #     y = self.fc(out)
    #     score = F.log_softmax(y, dim=1)  # calculate loss by NLLoss
    #     # print(score.shape)
    #     # print(sum(score[:, 0]))
    #     return score
        def forward(self):


def forward(self, input, hx=None):
    batch_sizes = None  # is not packed, batch_sizes = None
    max_batch_size = input.size(
        0) if self.batch_first else input.size(1)  # batch_size

    if hx is None:  # 使用者可以不传输hidden, 自动创建全0的hidden
        num_directions = 2 if self.bidirectional else 1
        hx = torch.autograd.Variable(input.data.new(self.num_layers *
                                                    num_directions,
                                                    max_batch_size,
                                                    self.hidden_size).zero_())
        if self.mode == 'LSTM':  # h_0, c_0
            hx = (hx, hx)

    flat_weight = None  # if cpu

    func = self._backend.RNN(  # self._backend = thnn_backend # backend = THNNFunctionBackend(), FunctionBackend
        self.mode,
        self.input_size,
        self.hidden_size,
        num_layers=self.num_layers,
        batch_first=self.batch_first,
        dropout=self.dropout,
        train=self.training,
        bidirectional=self.bidirectional,
        batch_sizes=batch_sizes,
        dropout_state=self.dropout_state,
        flat_weight=flat_weight
    )
    output, hidden = func(input, self.all_weights, hx)

    return output, hidden
