## 联合任务准确率统计（macro-f1）


```python
import json
def get_macro_f1():
    with open("./BERT-Linear2/data/united_test.json", "r") as fr:
        data = json.load(fr)
    # tf ff ft
    gt_tf = 0
    gt_ff = 0
    gt_ft = 0

    p_tf = 0
    p_ff = 0
    p_ft = 0

    c_tf = 0
    c_ff = 0
    c_ft = 0
    for item in data:

        if item["judge1"] == True:
            gt_tf += 1
            gt = "tf"
        else:
            if item["judge2"] == True:
                gt_ft += 1
                gt = "ft"
            else:
                gt_ff += 1
                gt = "ff"

        if item["pred1"] == 1:
            p_tf += 1
            p = "tf"
        else:
            if item["pred2"] == 1:
                p_ft += 1
                p = "ft"
            else:
                p_ff += 1
                p = "ff"

        if gt == p:
            if gt == "tf":
                c_tf += 1
            elif gt == "ft":
                c_ft += 1
            else:
                c_ff += 1

    precision_tf = c_tf / p_tf
    recall_tf = c_tf / gt_tf
    f1_tf = 2 * precision_tf * recall_tf / (precision_tf + recall_tf)

    precision_ft = c_ft / p_ft
    recall_ft = c_ft / gt_ft
    f1_ft = 2 * precision_ft * recall_ft / (precision_ft + recall_ft)

    precision_ff = c_ff / p_ff
    recall_ff = c_ff /gt_ff
    f1_ff = 2 * precision_ff * recall_ff / (precision_ff + recall_ff)

    print("Macro-f1:{:.6f}".format((f1_tf + f1_ft + f1_ff) / 3))
    
get_macro_f1()
```

    Macro-f1:0.517037


## 联合任务准确率计算（micro-f1？）


```python
import json
def get_new_metrics():
    with open("./BERT-Linear2/data/united_dev.json", "r") as fr:
        data = json.load(fr)
    # tf ff ft
    gt_tf = 0
    gt_ff = 0
    gt_ft = 0

    p_tf = 0
    p_ff = 0
    p_ft = 0

    c_tf = 0
    c_ff = 0
    c_ft = 0

    tp = fp = tn = fn = 0
    for item in data:


        if item["judge1"] == True:
            gt_tf += 1
            gt = "tf"
        else:
            if item["judge2"] == True:
                gt_ft += 1
                gt = "ft"
            else:
                gt_ff += 1
                gt = "ff"

        if item["pred1"] == 1:
            p_tf += 1
            p = "tf"
        else:
            if item["pred2"] == 1:
                p_ft += 1
                p = "ft"
            else:
                p_ff += 1
                p = "ff"

        if gt == p:

            if gt == "tf":
                c_tf += 1
            elif gt == "ft":
                c_ft += 1
            else:
                c_ff += 1

    # precision：归因正确的/自己认为需要归因的
    #     自己认为需要归因的：模型1预测为负的案例（模型做了归因的）
    # recall：归因正确的/真正需要归因的
    #     真正需要归因的：真正为负的案例
    precision = ((c_ft + c_ff)/(p_ft + p_ff)) if (p_ft + p_ff)!=0 else 0
    recall = ((c_ft + c_ff)/(gt_ft + gt_ff)) if (gt_ft + gt_ff) !=0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall)!=0 else 0
    print("united test precision:{:.6f}, recall:{:.6f}, f1:{:.6f}".format(precision, recall, f1))
get_new_metrics()
```

    united test precision:0.588055, recall:0.513369, f1:0.548180



```python

```
