# Readme

[下载](https://cloud.tsinghua.edu.cn/d/ea00e70945cc431d866d/)完整版程序。

## 数据来源

[THUNews](http://thuctc.thunlp.org/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews)的一个子集，从10类新闻中随机抽取了20万条新闻标题，每类各2万条。按照18：1：1划分训练、验证、测试集。


## Performance

| Model    | Test Acc |
| -------- | ------ |
| Naive Bayes | 90.93% |
| fastText (3-gram) | 92.54% |
| TextCNN | 91.48% |
| DPCNN    | 92.00% |
| BiLSTM | 91.58% |
| BiLSTM with Attention | 91.60% |
| TextRCNN | 91.79% |
| BERT-wwm-ext ([来源](https://github.com/ymcui/Chinese-BERT-wwm)) | 94.61% |
| RoBERTa-wwm-ext | 94.89% |

内存有限，GCN的测试仅取数据集的十分之一进行，即20000训练/验证集，1000测试集。

| Model         | Acc    |
| ------------- | ------ |
| Naive Bayes | 85.60% |
| TextGCN       | 83.30% |

## Usage

在src文件夹下运行脚本。

```shell
bash test.sh
```

改变json文件中的`load`和`num_epochs`，可以选择是否加载已训练模型和训练的epoch数。

对于gcn，在`text_gcn`文件夹下运行

```shell
python train.py r8
```

进行训练和测试。

## 文件结构

`data/` 各种数据

`config/` 模型超参数

`src/` 源代码

`model/` 保存模型和日志

[word2vec预训练词向量](https://pan.baidu.com/s/1pUqyn7mnPcUmzxT64gGpSw)，生成的词表已经保存在data文件夹下。
