# GCN算法。
# 本代码以cora数据集为例，实现基于GCN的节点分类算法。
# GCN是谱图卷积的一阶局部近似，是一个多层的图卷积神经网络。
# 每一个卷积层仅处理一阶邻域信息，通过叠加若干卷积层可以实现多阶邻域的信息传递。
# 通过信息传递，每一个节点都能够获取其局部结构信息，产生带有局部结构信息以及自身特征的嵌入。
# 对嵌入进行降维，得到每个节点对应各分类的概率。
# 取最大概率的分类为节点的预测结果。


import random
import time
import argparse

import numpy as np
import torch
import torch.optim as optim
from utils import accuracy
from models import GCN
from utils import load_data

# 参数设置
parser = argparse.ArgumentParser()

# 设置数据集参数, 可用数据集 cora, citeseer, pubmed
parser.add_argument('--dataset', type=str, default="cora",
                    help='dataset for training')

# 训练次数
parser.add_argument('--times', type=int, default=1,
                    help='times of repeat training')

# 随机种子
parser.add_argument('--seed', type=int, default=33, help='Random seed.')

# 批处理次数
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')

# 学习率
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')

# 权重衰减, 避免模型过拟合
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')

# 隐藏层神经元个数
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')

# 丢失率, 避免模型过拟合
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

# 获取模型输入参数
args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 定义损失函数为交叉熵函数, 此外还有L1Loss, MSELoss
criterion = torch.nn.NLLLoss()

# 获取设备
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


# 模型训练
def train(epoch, model, optimizer, adj, features, labels, idx_train, idx_val):
    """
    模型训练函数
    :param epoch: 训练轮次
    :param model: 输入模型
    :param optimizer: 训练优化器
    :param adj: 边
    :param features: 节点特征
    :param labels: 节点标记
    :param idx_train: 训练集节点索引
    :param idx_val: 验证集节点索引
    :return:
    """

    # 获取当前时间
    t = time.time()

    # 模型训练，权重参与更新
    model.train()

    # 梯度设置为0
    optimizer.zero_grad()

    # 计算模型输出
    output = model(features, adj)

    # 计算误差
    loss_train = criterion(output[idx_train], labels[idx_train])

    # 计算准确率
    acc_train = accuracy(output[idx_train], labels[idx_train])

    # 反向传播
    loss_train.backward()

    # 更新参数
    optimizer.step()

    # 计算验证集的损失值以及准确率，并不参与权重更新
    with torch.no_grad():
        model.eval()
        output = model(features, adj)
        loss_val = criterion(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

    # 输出训练过程中，训练集的损失值、准确率，验证集的损失值、准确率，打印时间开销
    print(f'Epoch: {epoch + 1:04d}',
          f'loss_train: {loss_train.item():.4f}',
          f'acc_train: {acc_train:.4f}',
          f'loss_val: {loss_val.item():.4f}',
          f'acc_val: {acc_val:.4f}',
          f'time: {time.time() - t:.4f}s')

    # 返回验证集损失值
    return loss_val


# 利用模型进行预测
@torch.no_grad()  # 不进行权重的更新
def predict(model, adj, features, labels, idx_test):
    """
    预测函数
    :param model: 训练完成的模型
    :param adj: 边数据
    :param features: 节点特征
    :param labels: 节点标记
    :param idx_test: 测试集节点索引
    :return:
    """

    # 模型执行，权重不更新
    model.eval()

    # 输入测试数据
    output = model(features, adj)

    # 计算测试集损失值
    loss_test = criterion(output[idx_test], labels[idx_test])

    # 计算测试集准确率
    acc_test = accuracy(output[idx_test], labels[idx_test])

    # 打印结果
    print(f"Test set results:",
          f"loss= {loss_test.item():.4f}",
          f"accuracy= {acc_test:.4f}")

    # 返回测试集准确率和预测结果
    return acc_test, output


def main(dataset, times):

    # 加载数据
    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset)

    # to(device) 将数据复制到GPU上面
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    # 分类总数
    nclass = labels.max().item() + 1

    acc_lst = list()

    # 模型训练次数
    for seed in random.sample(range(0, 100000), times):

        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # 初始化模型
        model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,   # 16个神经元
                    nclass=nclass,  # 7个类
                    dropout=args.dropout)

        # 初始化迭代器
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)

        model.to(device)

        # 训练
        t_total = time.time()
        for epoch in range(args.epochs):

            # 训练模型
            train(epoch, model, optimizer, adj,
                  features, labels, idx_train, idx_val)

        print(f"Total time elapsed: {time.time() - t_total:.4f}s")

        # 保存模型
        torch.save(model, 'model/model.pkl')
        print("模型训练结束, 模型保存到model/")


if __name__ == '__main__':
    main(dataset=args.dataset, times=args.times)
