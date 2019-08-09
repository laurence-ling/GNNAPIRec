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
        self.Q = nn.Parameter(torch.Tensor(2*in_feats, out_feats))
        self.q = nn.Parameter(torch.Tensor(out_feats))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.Q.data.uniform_(-stdv, stdv)
        self.q.data.uniform_(-stdv, stdv)

    def gcn(self, inputs, adj):
        x = torch.mm(inputs, self.weight)
        x = torch.sparse.mm(adj, x)
        if self.bias is not None:
            x += self.bias
        return F.relu(x)

    def graph_sage(self, inputs, adj):
        x = self.gcn(inputs, adj)
        x = torch.cat([x, inputs], dim=1)
        x = torch.mm(x, self.Q) + self.q
        return F.relu(x)

    def forward(self, inputs, adj):
        # input size(n, d)  adj size(n, n)
        out = self.gcn(inputs, adj)
        out = F.normalize(out, p=2, dim=1)
        return out
