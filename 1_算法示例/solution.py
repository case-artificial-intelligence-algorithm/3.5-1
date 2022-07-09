import scipy.sparse as sp
import networkx as nx
from dgl.data import CoraGraphDataset

import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch
import torch.nn as nn

import random
import argparse
import numpy as np
import torch.optim as optim

# 加载数据集
def load_data(dataset):
    if dataset == 'cora':
        data = CoraGraphDataset()

    g = data[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    nxg = g.to_networkx()
    adj = nx.to_scipy_sparse_matrix(nxg, dtype=np.float)

    adj = preprocess_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels, train_mask, val_mask, test_mask

# 邻接矩阵预处理
def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized

# 对称归一化连接矩阵
def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

# 准确率计算
def accuracy(pred, targ):
    pred = torch.max(pred, 1)[1]
    ac = ((pred == targ).float()).sum().item() / targ.size()[0]
    return ac

# 稀疏矩阵转换到稀疏张量
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# 定义GCN层
class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 权重初始化
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            # 设置偏置
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# 定义GCN网络
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
        x = torch.relu(self.gc1(x, adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)
        return torch.log_softmax(x, dim=1)

# 参数设置
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="cora",
                    help='dataset for training')

parser.add_argument('--times', type=int, default=1,
                    help='times of repeat training')

parser.add_argument('--seed', type=int, default=33, help='Random seed.')

parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')

parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')

parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')

parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')

parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()

# 定义损失函数为交叉熵函数, 此外还有L1Loss, MSELoss
criterion = torch.nn.NLLLoss()

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

# 模型训练
def train(epoch, model, optimizer, adj, features, labels, idx_train, idx_val):

    model.train()

    optimizer.zero_grad()

    output = model(features, adj)

    loss_train = criterion(output[idx_train], labels[idx_train])

    loss_train.backward()

    optimizer.step()

    with torch.no_grad():
        model.eval()
        output = model(features, adj)
        loss_val = criterion(output[idx_val], labels[idx_val])

    return loss_val

def main(dataset, times):

    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset)

    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)

    nclass = labels.max().item() + 1

    for seed in random.sample(range(0, 100000), times):

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,   # 16个神经元
                    nclass=nclass,  # 7个类
                    dropout=args.dropout)

        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)

        model.to(device)

        for epoch in range(args.epochs):
            train(epoch, model, optimizer, adj,
                  features, labels, idx_train, idx_val)

        print("模型训练完成。")
        ind = random.sample(range(1, 100), 10)
        out = torch.argmax(model(features, adj), dim=1)

        print("从验证集中随机抽取10个节点的结果进行对比")
        print("节点索引 ", ind)
        print("真实类别编号 ", labels[idx_val][ind].tolist())
        print("预测类别编号 ", out[idx_val][ind].tolist())


if __name__ == '__main__':
    main(dataset=args.dataset, times=args.times)
