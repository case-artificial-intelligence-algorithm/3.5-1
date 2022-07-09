# Graph Convolutional Networks in PyTorch

PyTorch 1.6 and Python 3.7 implementation of Graph Convolutional Networks (GCNs) for semi-supervised classification [1].

Tested on the cora/pubmed/citeseer data set, the code on this repository can achieve the effect of the paper.

## Benchmark

| dataset       | Citeseea | Cora | Pubmed |
|---------------|----------|------|--------|
| GCN(official) | 70.3     | 81.5 | 79.0   |
| This repo.    | 70.7     | 81.2 | 79.2   |

NOTE: The result of the experiment is to repeat the run 10 times, and then take the average of accuracy.

## Requirements
* PyTorch==1.6.0
* Python==3.7
* dgl==0.5.2 
* scipy==1.5.2 
* numpy==1.19.1 
* networkx==2.5

## Usage

train

```shell
python train.py --dataset=cora --times=1
```

predict

```shell
python predict.py
```

## help docs
cora dataset intro : https://linqs.soe.ucsc.edu/data
cora original data download ： https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz
dgl help docs : https://docs.dgl.ai/#stanford-sentiment-treebank-dataset

## others
data/labels记录了所有数据的真实标签，方便读者对比预测结果与真实结果。  
其中行数表示节点的编号，如labels文件下第1716行数值为2，表示节点编号1716的真实标签为2  
注意：程序中从dgl公开库中加载的cora数据集并不是cora的原始数据集。为了方便神经网络模型的训练，dgl已经对原始数据进行处理。  

## References
[1] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](http://xxx.itp.ac.cn/pdf/1609.02907.pdf)
