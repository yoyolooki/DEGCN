# -*- coding: utf-8 -*-
# @Time    : 2024/5/4 19:07
# @Author  : Li Yu
# @File    : DenseGCN_model.py
from torch import nn
import torch.nn.functional as F
from Layer import GraphConvolution


class DEGCN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, dropout=None):
        super(DEGCN, self).__init__()
        self.gc1 = GraphConvolution(n_in, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_hid)
        self.gc3 = GraphConvolution(n_hid, n_hid)
        self.gc4 = GraphConvolution(n_hid, n_hid)
        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)
        self.dp3 = nn.Dropout(dropout)
        self.dp4 = nn.Dropout(dropout)
        # self.fc1 = nn.Linear(n_hid, n_hid)
        self.fc = nn.Linear(n_hid, n_out)
        self.dropout = dropout
        self.linear_gc1_gc2 = nn.Linear(n_hid, n_hid)
        self.linear_gc1_gc3 = nn.Linear(n_hid, n_hid)
        self.linear_gc1_gc4 = nn.Linear(n_hid, n_hid)
        self.linear_gc2_gc3 = nn.Linear(n_hid, n_hid)
        self.linear_gc2_gc4 = nn.Linear(n_hid, n_hid)
        self.linear_gc3_gc4 = nn.Linear(n_hid, n_hid)
        self.in_gc1 = nn.Linear(n_in, n_hid)
        self.in_gc2 = nn.Linear(n_in, n_hid)
        self.in_gc3 = nn.Linear(n_in, n_hid)
        self.in_gc4 = nn.Linear(n_in, n_hid)

    def forward(self, input, adj):
        l_in_gc1 = self.in_gc1(input)
        l_in_gc2 = self.in_gc2(input)
        l_in_gc3 = self.in_gc3(input)
        l_in_gc4 = self.in_gc4(input)

        x_gc1 = self.gc1(input, adj)
        xl_1_2 = self.linear_gc1_gc2(x_gc1)
        xl_1_3 = self.linear_gc1_gc3(x_gc1)
        xl_1_4 = self.linear_gc1_gc4(x_gc1)

        x = F.elu(x_gc1 + l_in_gc1)
        x = self.dp1(x)

        xl_2_3 = self.linear_gc2_gc3(x)
        xl_2_4 = self.linear_gc2_gc4(x)

        x_gc2 = self.gc2(x, adj)

        x = F.elu(x_gc2 + xl_1_2 + l_in_gc2)
        x = self.dp2(x)

        x_gc3 = self.gc3(x, adj)
        x = F.elu(x_gc3 + xl_1_3 + xl_2_3 + l_in_gc3)
        x = self.dp3(x)

        xl_3_4 = self.linear_gc3_gc4(x)

        x_gc4 = self.gc4(x, adj)
        x = F.elu(x_gc4 + xl_1_4 + xl_2_4 + xl_3_4 + l_in_gc4)
        x = self.dp4(x)

        x = self.fc(x)

        return x
