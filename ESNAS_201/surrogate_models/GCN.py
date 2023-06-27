import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)  # 构建第一层 GCN
        self.gc2 = GraphConvolution(nhid, nclass)  # 构建第二层 GCN
        self.dropout = dropout
        self.fc = nn.Linear(nhid)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        pred = self.fc(x).view(-1)
        return pred

