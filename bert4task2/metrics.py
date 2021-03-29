import json

def get_metrics(true_tags, pred_tags, mode='dev', test_dir=None, result_dir=None):

    result = {}
    if mode == 'test':
        with open(test_dir, 'r') as fr:
            items = json.load(fr)

        for idx, item in enumerate(items):
            item['pred2'] = int(pred_tags[idx])
            # item['groundTruth'] = int(true_tags[idx])

        with open(result_dir, 'w') as fw:
            json.dump(items, fw, indent=2, ensure_ascii=False)
    tp = fp = fn = tn = 0
    score_dict = {}
    # n(pred_tags) == len(true_tags)
    for idx in range(len(true_tags)):
        if pred_tags[idx] == 1:
            # 正样本，预测为正--正确
            if true_tags[idx] == 1:
                tp += 1
            # 负样本，预测为正--错误
            else:
                fp += 1
        else:
            # 正样本，预测为负--错误
            if true_tags[idx] == 1:
                fn += 1
            # 负样本，预测为负--正确
            else:
                tn += 1
    # total
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = (tp / (tp + fp)) if (tp + fp) != 0 else 0
    recall = (tp / (tp + fn)) if(tp + fn) != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    result['accuracy'] = accuracy
    result['precision'] = precision
    result['recall'] = recall
    result['f1'] = f1

    score_dict['total'] = [tp, fp, fn, tn]
    result['score_dict'] = score_dict

    return result
