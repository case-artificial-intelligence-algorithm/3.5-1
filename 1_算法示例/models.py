import torch
import torch.nn as nn

from layers import GraphConvolution
import keshihua as kh


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        # 1 layer
        self.gc1 = GraphConvolution(nfeat, nhid)
        # 2 layer
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = nn.Dropout(p=dropout)
        self.nums = 0

    def forward(self, x, adj):
        # 正向传播
        x = torch.relu(self.gc1(x, adj))

        # 可视化中间节点变化过程
        # if self.nums == 50 or self.nums == 100 or self.nums == 150 or self.nums == 200:
        #     kh.pic(x.data.cpu().numpy(), 'layer1-%-epoch.png' % self.nums)
        x = self.dropout(x)
        x = self.gc2(x, adj)
        # if self.nums == 50 or self.nums == 100 or self.nums == 150 or self.nums == 200:
        #     kh.pic(x.data.cpu().numpy(), 'layer2-%-epoch.png' % self.nums)
        # self.nums += 1
        return torch.log_softmax(x, dim=1)
