## 20210307 Task1数据预处理


```python
import json
from sklearn.model_selection import train_test_split

def save_dataset(dataset, flag):
    with open(flag + ".json", "w") as fw:
        json.dump(dataset, fw, indent=2, ensure_ascii=False)
        

# 原始数据集文件名
raw_file_name = "./task1(1).json"
with open(raw_file_name, "r") as fr:
    data = json.load(fr)
    question = data["questions"]

# 数据集划分
train = [item for item in question if item["set"] == "train" ]
test = [item for item in question if item["set"] == "test"]
dev = [item for item in question if item["set"] == "dev"]

# 保存
save_dataset(train, "train")
save_dataset(test, "test")
save_dataset(dev, "dev")

# 输出统计值
print("size of datasets train:{}, test:{}, dev:{}".format(len(train),len(test),len(dev)))
```

    size of datasets train:4237, test:806, dev:794


## 20210307 Task2数据预处理


```python
import json
from sklearn.model_selection import train_test_split

def save_dataset(dataset, flag):
    with open(flag + ".json", "w") as fw:
        json.dump(dataset, fw, indent=2, ensure_ascii=False)
        

# 原始数据集文件名
raw_file_name = "./task3(4).json"
with open(raw_file_name, "r") as fr:
    data = json.load(fr)
    question = data["questions"]

# 数据集划分
train = [item for item in question if item["set"] == "train" ]
test = [item for item in question if item["set"] == "test"]
dev = [item for item in question if item["set"] == "dev"]

# 保存
save_dataset(train, "train")
save_dataset(test, "test")
save_dataset(dev, "dev")

# 输出统计值
print("size of datasets train:{}, test:{}, dev:{}".format(len(train),len(test),len(dev)))
```

    size of datasets train:8534, test:2985, dev:2969


### Task3数据集生成


```python
# create datasets for Task2
import json
import re
import random

def save_dataset(dataset, flag):
    with open(flag + ".json", "w") as fw:
        json.dump(dataset, fw, indent=2, ensure_ascii=False)

        
def get_sample(dataset):
    ret = []
    for item in dataset:
        
        ori = item["content"]
        extraction = re.findall(r'【【([\u4e00-\u9fa5]+)→([\u4e00-\u9fa5]+)】】',ori)
        pre, substitute = extraction[0]
        res = re.sub(r'【【[\u4e00-\u9fa5]+→[\u4e00-\u9fa5]+】】', "[MASK]" * len(substitute), ori)
        item["prd_context"] = res
        item["pre"] = pre
        item["substitute"] = substitute
        item["label"] = 1 if item["judge_e"] == "变化不大" else 0
        ret.append(item)
    
    return ret
        

raw_file_name = "./task2(1).json"
with open(raw_file_name, "r") as fr:
    raw = json.load(fr)
    data = raw["questions"]

train = [d for d in data if d["set"] == "train"]
dev = [d for d in data if d["set"] == "dev"]
test = [d for d in data if d["set"] == "test"]
print("final num: train:{}, test:{}, dev:{}".format(len(train), len(test), len(dev)))

train = get_sample(train)
dev = get_sample(dev)
test = get_sample(test)
print("final num: train:{}, test:{}, dev:{}".format(len(train), len(test), len(dev)))

save_dataset(train, "train")
save_dataset(dev, "dev")
save_dataset(dev, "test")
```

    final num: train:1531, test:300, dev:289
    final num: train:1531, test:300, dev:289


## Task4测试集生成


```python
import json

raw_file_name = "./task4(3).json"
with open(raw_file_name, "r") as fr:
    raw = json.load(fr)
    data = raw["questions"]
    
dev = [item for item in data if item["set"] == "dev"]
test = [item for item in data if item["set"] == "test"]

with open("./united_dev.json","w") as fw:
    json.dump(dev, fw, indent=2, ensure_ascii=False)
    
with open("./united_test.json","w") as fw:
    json.dump(test,fw,indent=2, ensure_ascii=False)
    
print("num dev:{},test:{}".format(len(dev), len(test)))
```

    num dev:1305,test:1360



```python
# 统计最长句子
def get_max_length():
    max_length = 0
    for d in data:
        if len(d['context']) > max_length:
            max_length = len(d['text'])
    return max_length

def get_data(file_path):
    with open(file_path, "r") as fr:
        data = json.load(fr)
    return data
    
def get_statistics():
    max_text_len = 0
    max_ans_len = 0
    max_total_len = 0
    
    data = []
    data.extend(get_data("./train.json"))
    data.extend(get_data("./dev.json"))
    data.extend(get_data("./test.json"))
    
    for d in data:
        max_text_len = max(len(d['text']), max_text_len)
        max_ans_len = max(len(d['answer']), max_ans_len)
        max_total_len = max(len(d['text'])+ len(d['answer']), max_total_len)
    print("text:{}, answer:{}, total:{}".format(max_text_len, max_ans_len, max_total_len))
get_statistics()
```

    text:364, answer:107, total:415



```python

```
