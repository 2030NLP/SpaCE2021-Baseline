# Bert-based Text Bi-classification Model

### Requirements

* tqdm
* numpy >= 1.19.4
* pytorch >= 1.7.0
* transformers >= 4.2.1

### 文件树

如下仅展示了bert4task1的文件树，bert4task2与之格式上完全相同（内容上不同之处主要是数据预处理以及评价指标部分）

```
bert4task1
├── config.py					# 模型超参数
├── data						# 保存训练和测试数据
│   ├── task1-train-with-answer.json
│   └── task1-dev.json
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

task1和task2的模型均采用了bert完成文本编码

* 使用了transformers工具包中[BertForSequenceClassification](https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification)直接作为训练使用的模型（该模型相当于是在基本的bert结构后接上了分类器用以token序列分类）
* bert的预训练数据则使用了`BERT-wwm-ext`（可以点[此处](https://github.com/ymcui/Chinese-BERT-wwm)了解更多并下载，解压完成后放入`pretrained_bert_models`文件夹下，并将`bert_config.json`重命名为`config.json`即可）

### 数据预处理

这里为了处理方便，使用了transformers包中的BertTokenizer类，实例化后，可以调用`encode()`方法并向其传入待处理的原始文本进行分词并获得token在词表中的索引。比如：

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

上面的列表是处理后的token索引，首尾的`101`和`102`分别是增加的两个特殊token`[CLS]`和`[SEP]`的索引。上述过程处理之后再经过padding并转成torch张量之后就可直接输入模型。

task1中所有句子的长度有限（不超过510），不用进行截断处理；而task2中因需要将context和reason进行拼接（`[CLS]`+context+`[SEP]`+reason+`[SEP]`），存在token长度超过509的情况，这里采用了保证reason不截断，如需截断则截断context的策略。

### 训练

为了防止过拟合，模型训练前会从原始训练集中划分出占比为0.1的验证集，并且训练时设置了早停机制，当模型在验证集上的Accuracy在一定轮次内不再提升则停止训练，最后使用在验证集上表现最好的模型作为最终模型。

### 测试

task1和task2训练完成之后，会自动在`task1(2)-dev.json`上进行测试，测试结果保存到相应目录下的`predicion.json`文件中。对于task3，可将task3-dev.json置于根目录，并运行`test4task3.py`得到名为`model2_prediction.json`的测试结果。

