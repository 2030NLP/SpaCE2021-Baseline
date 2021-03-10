import config
import json

def get_metrics(true_tags, pred_tags, mode='dev'):

    result = {}
    tp = fp = fn = tn = 0
    for idx in range(len(pred_tags)):
        # print(pred_tags[idx], true_tags[idx])
        if int(pred_tags[idx]) == 1:
            # 正样本，预测为正--正确
            if int(true_tags[idx]) == 1:
                tp += 1
            # 负样本，预测为正--错误
            else:
                fp += 1
        else:
            # 正样本，预测为负--错误
            if int(true_tags[idx]) == 1:
                fn += 1
            # 负样本，预测为负--正确
            else:
                tn += 1
    # total
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = (tp / (tp + fp)) if (tp + fp) != 0 else 0
    recall = (tp / (tp + fn)) if (tp + fn) != 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) != 0 else 0
    result['accuracy'] = accuracy
    result['precision'] = precision
    result['recall'] = recall
    result['f1'] = f1
    # print(metrics)
    if mode == 'test':
        with open(config.test_dir, 'r') as fr:
            items = json.load(fr)

        for idx, item in enumerate(items):
            item['pred1'] = int(pred_tags[idx])

        with open(config.result_dir, 'w') as fw:
            json.dump(items, fw, indent=2, ensure_ascii=False)

    return result

if __name__ == "__main__":
    true = [0,1,0]
    pred = [1,1,0]
    print(acc(true, pred))