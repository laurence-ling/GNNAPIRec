import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):

    def __init__(self, in_feats, out_feats, bias=True):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        # input size(n, d)  adj size(n, n)
        x = torch.mm(inputs, self.weight)
        x = torch.sparse.mm(adj, x)
        if self.bias is not None:
            x += self.bias
        out = F.relu(x)
        out = F.normalize(out, p=2, dim=1)
        return out
