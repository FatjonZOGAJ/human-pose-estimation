from __future__ import absolute_import

from functools import reduce

import torch.nn as nn
from lib.models.sem_graph_conv import SemGraphConv
from lib.models.graph_non_local import GraphNonLocal


class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = SemGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class _GraphNonLocal(nn.Module):
    def __init__(self, hid_dim, grouped_order, restored_order, group_size, inter_channels=None):
        super(_GraphNonLocal, self).__init__()

        self.nonLocal = GraphNonLocal(hid_dim, inter_channels=inter_channels, sub_sample=group_size)
        self.grouped_order = grouped_order
        self.restored_order = restored_order

    def forward(self, x):
        out = x[:, self.grouped_order, :]
        out = self.nonLocal(out.transpose(1, 2)).transpose(1, 2)
        out = out[:, self.restored_order, :]
        return out


class SemGCN(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4, nodes_group=None, p_dropout=None,
                 sub_sample_pooling=2, move_neck_nose_to_end=False):
        super(SemGCN, self).__init__()

        _gconv_input = [_GraphConv(adj, coords_dim[0], hid_dim, p_dropout=p_dropout)]
        _gconv_layers = []

        if nodes_group is None:
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
        else:
            group_size = len(nodes_group[0])
            assert group_size > 1

            grouped_order = list(reduce(lambda x, y: x + y, nodes_group))
            restored_order = [0] * len(grouped_order)
            if move_neck_nose_to_end:
                grouped_order = [i + 1 if i >= 9 else i  # fix nodes_group not taking into account Neck/Nose
                                 for i in grouped_order]
                grouped_order.append(9)  # add Neck/Nose to end
                sub_sample_pooling = 1  # was group_size; 1 means do not downsample joints
            for i in range(len(restored_order)):
                for j in range(len(grouped_order)):
                    if grouped_order[j] == i:  # restored_order[0] = idx of 0 where it appears in grouped_order
                        restored_order[i] = j
                        break

            # has to be after above for loop, while adding Neck/Nose has to be done before
            if not move_neck_nose_to_end:  # Add Neck/Nose at normal position
                grouped_order.append(16)
                restored_order.append(16)

            # defining inter_channels allows us to set size of feature downpooling (default = input (aka hid_dim) / 2)
            inter_channels = None
            _gconv_input.append(
                _GraphNonLocal(hid_dim, grouped_order, restored_order, sub_sample_pooling,
                               inter_channels=inter_channels))
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
                _gconv_layers.append(
                    _GraphNonLocal(hid_dim, grouped_order, restored_order, sub_sample_pooling,
                                   inter_channels=inter_channels))

        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        self.gconv_output = SemGraphConv(hid_dim, coords_dim[1], adj)

    def forward(self, x):
        out = self.gconv_input(x)
        out = self.gconv_layers(out)
        out = self.gconv_output(out)
        return out
