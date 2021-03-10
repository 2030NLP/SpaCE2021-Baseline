## 模型

task1和task2的模型均采用了bert+fintuning

* 具体实现：使用了[huggingface](https://huggingface.co/)开发的transformers工具包中的[BertForSequenceClassification](https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification)直接作为训练使用的模型（该模型相当于是在基本的bert编码结构后接上了分类器用以token序列分类）

## 数据预处理

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

由于task1中所有句子的长度有限（不超过510），所以不用进行截断处理；而task2中由于需要将句子和reason进行拼接，存在token长度超过509的情况，因此这里采用了先保证reason原长度不截断，如需截断则截断前面的句子的策略，所以任务二处理后的token序列可形式化的表示为`[CLS]`+context+`[SEP]`+reason+`[SEP]`。

### 训练

训练时设置了早停机制，当模型在验证集上的准确率在一定轮次内不再提升则停止训练，并使用在验证集上表现最好的模型作为最终模型
