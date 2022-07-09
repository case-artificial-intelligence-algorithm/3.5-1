import train
import torch
from utils import load_data
import keshihua as ks
import numpy as np


if __name__ == '__main__':
    """
        测试例子说明：
            本例子使用cora数据集中自带测试例子进行测试，集合中包含1000个节点并对节点进行分类预测。
            结果记录在out.txt文件中，并将测试集中前20个节点进行可视化
    """
    # 加载数据集，本样例使用cora数据集进行测试
    adj, features, labels, idx_train, idx_val, idx_test = load_data("cora")
    # 获取测试数据中前20个样本用于可视化展示
    ind = [i for i, v in enumerate(idx_test) if v]
    ind = ind[:20]
    # 获取特征矩阵
    f = features[ind]
    # 获取标签矩阵
    l = labels[ind]

    # 原始数据可视化
    ks.pic(feature=f, label=l, name='result/raw.png', mode='raw')

    #获取设备
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # 数据输入到设备
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    idx_test = idx_test.to(device)

    # 加载完成训练的模型
    model = torch.load('model/model.pkl')
    acc_test, output = train.predict(model=model, adj=adj, features=features, labels=labels, idx_test=idx_test)

    # 预测后进行数据可视化
    output = output[ind]
    out = torch.argmax(output, dim=1)
    ks.pic(feature=f, label=out, name='result/predict.png', mode='predict')

    # 随机选取两个样本的结果进行对比
    import random
    ran = [random.randint(0,20) for i in range(2)]
    ran_ind = [ind[i] for i in ran]

    # 记录结果
    with open('result/output.txt', 'a', encoding='utf-8')as f:
        f.write("\n测试集准确率为：%s \n" % acc_test)
        f.write("--------predict-------\n")
        for i in range(2):
            f.write("节点编号为 %s 的真实分类为 %s，预测分类为 %s \n" % (ran_ind[i], labels[ind[ran[i]]].item(), out[ran[i]].item()))
    print("测试集准确率为：%s " % acc_test)
    

