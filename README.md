# Bert-based Text Bi-classification Model

### 环境

* Ubuntu 18.04, CUDA 10.2

* Python 3.6.x

* 其他

  ```
  pip install -r requirements.txt
  ```

### 文件树

```
SpaCE2021-Baseline/
├── bert4task1
│   ├── config.py				# 模型超参设置
│   ├── data_loader.py			# 数据预处理和成批
│   ├── experiments				# 保存训好的模型和训练日志
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── train.log
│   ├── metrics.py				# 计算评测指标
│   ├── run.py					# 训练程序运行入口
│   ├── test.py					# 模型测试
│   ├── train.py				# 模型训练
│   └── utils.py				# 保存训练日志、验证集分割
├── bert4task2					# 同bert4task1
│   └── *
├── data						# 保存训练和测试数据
│   ├── task1-train-with-answer.json
│   ├── task1-dev.json
│   ├── task2-train-with-answer.json
│   ├── task2-dev.json
│   └── task3-dev.json
├── pretrained_bert_models		# 保存预训练模型
│   └── chinese-wwm-ext
│       ├── config.json
│       ├── pytorch_model.bin
│       └── vocab.txt
├── README.md
├── requirements.txt
└── test4task3.sh				# task3测试脚本

```

### 模型

task1和task2的模型均采用了bert完成文本编码

* 使用了transformers工具包中[BertForSequenceClassification](https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification)直接作为训练使用的模型（该模型相当于是在基本的bert结构后接上了分类器用以token序列分类）
* bert的预训练数据则使用了`BERT-wwm-ext`（可以点[此处](https://github.com/ymcui/Chinese-BERT-wwm)了解更多并下载，解压完成后放入`pretrained_bert_models`文件夹下，并将`bert_config.json`重命名为`config.json`即可）

### 数据预处理

首先需要将三个任务用到的数据集置于`data`文件夹下。

task1中所有句子的长度有限（不超过510），不用进行截断处理；而task2中因需要将context和reason拼接后（`[CLS]`+context+`[SEP]`+reason+`[SEP]`）输入到Bert中，存在token长度超过509的情况，这里采用了保证reason不截断，如需截断则截断context的策略。

### 训练

为了防止过拟合，模型训练前会从原始训练集中划分出占比为0.1的验证集（将`task(1|2)-train-with-answer.json`在训练前划分成`task(1|2)-splited-train.json`和`task(1|2)-splited-dev.json`），并且训练时设置了早停机制，当模型在验证集上的Accuracy在一定轮次内不再提升则停止训练，最后使用在验证集上表现最好的模型作为最终模型。

### 测试

task1和task2训练完成之后，会自动在`task(1|2)-dev.json`上进行测试，测试结果保存到相应目录下的`task(1|2)-dev-result.json`文件中。对于task3，可在根目录下运行`test4task3.sh`得到名为`task3-dev-result.json`的测试结果。

