# Bert-based Text Bi-classification Model

### Requirements

* tqdm
* numpy >= 1.19.4
* pytorch >= 1.7.0
* transformers >= 4.2.1

### 文件树

```
bert4task1
├── config.py					# 模型超参数
├── data						# 保存训练验证以及测试数据
│   ├── dev.json
│   ├── test.json
│   ├── train.json
│   ├── united_dev.json
│   └── united_test.json
├── data_loader.py				# 输入模型前的预处理
├── experiments					# 保存训练好的模型和数据
│   ├── config.json				# 模型配置文件
│   ├── pytorch_model.bin		# 模型参数
|	└── train.log				# 训练日志
├── metrics.py					# 计算评测指标
├── pretrained_bert_models		# 保存预训练参数
│   └── chinese-wwm-ext
│       ├── config.json
│       ├── pytorch_model.bin
│       └── vocab.txt
├── run.py						# 程序入口
├── train.py					# 训练代码
└── utils.py					# 保存训练日志的工具
```

### 模型

task1和task2的模型均采用了bert+fintuning

* 具体实现：使用了[huggingface](https://huggingface.co/)开发的transformers工具包中的[BertForSequenceClassification](https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification)直接作为训练使用的模型（该模型相当于是在基本的bert编码结构后接上了分类器用以token序列分类）
* 预训练数据则使用了`BERT-wwm-ext`（可以点[此处](https://github.com/ymcui/Chinese-BERT-wwm)了解更多或下载，解压完成后放入`pretrained_bert_models`文件夹下，并将`bert_config.json`重命名为`config.json`即可）

### 数据预处理

这里为了处理方便，直接使用了transformers包中的BertTokenizer工具，实例化tokenizer后，可以直接向其传入待处理的原始文本获得分词完并转化后词在词表中的索引。比如：

```python
from transformers import BertTokenizer

bert_model = './pretrained_bert_models/chinese-wwm-ext/'
tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
result = tokenizer.encode("在城门口经过一阵可怕的拥挤后，我终于到了郊外。")
print(result)
```

```
[101, 1762, 1814, 100, 5307, 6814, 100, 7347, 1377, 2586, 4638, 2881, 2915, 1400, 8024, 2769, 5303, 754, 1168, 749, 6946, 1912, 511, 102]
```

上面的列表是处理后的token索引，首尾的`101`和`102`分别是增加的两个特殊token`[CLS]`和`[SEP]`的索引。上述过程处理之后再经过padding和转成torch张量之后就可以输入到模型了。

由于task1中所有句子的长度有限（不超过510），所以不用进行截断处理；而task2中由于需要将句子和reason进行拼接，存在token长度超过509的情况，这里采用了先保证reason原长度不截断，如需截断则截断前面的句子的策略，所以任务二处理后的token序列可形式化的表示为`[CLS]`+context+`[SEP]`+reason+`[SEP]`。

### 训练

训练时设置了早停机制，当模型在验证集上的Accuracy在一定轮次内不再提升则停止训练，并使用在验证集上表现最好的模型作为最终模型。

### 实验结果

#### Task1

| 数据集 | Accuracy | Precision | Recall   | F1       |
| ------ | -------- | --------- | -------- | -------- |
| dev    | 0.652393 | 0.609453  | 0.673077 | 0.639687 |
| test   | 0.619107 | 0.612457  | 0.475806 | 0.535552 |

#### Task2

| 数据集 | Accuracy | Precision | Recall   | F1       |
| ------ | -------- | --------- | -------- | -------- |
| dev    | 0.781575 | 0.781479  | 0.783567 | 0.782522 |
| test   | 0.728865 | 0.701142  | 0.743590 | 0.721742 |

#### 联合任务

| 数据集 | Macro-f1    | 新指标   |
| ------ | ----------- | -------- |
| dev    | 0.547977    | 0.517037 |
| test   | f1:0.548180 | 0.545240 |

