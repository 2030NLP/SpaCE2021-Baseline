import json
import config

def get_metrics(true_tags, pred_tags, mode='dev'):

    result = {}
    if mode == 'test':
        with open(config.test_dir, 'r') as fr:
            items = json.load(fr)

        for idx, item in enumerate(items):
            item['pred2'] = int(pred_tags[idx])
            # item['groundTruth'] = int(true_tags[idx])

        with open(config.result_dir, 'w') as fw:
            json.dump(items, fw, indent=2, ensure_ascii=False)
    else:
        with open(config.dev_dir, 'r') as fr:
            items = json.load(fr)

    type_list = ['total', 'A', 'B', 'C', 'D', 'E']
    # 保存每种不同type的tp,tn,fp,fn
    score_dict = {}
    for t in type_list:
        score_dict[t] = [0, 0, 0, 0]
    tp = fp = fn = tn = 0
    # n(pred_tags) == len(true_tags)
    for idx, item in enumerate(items):
        t = item["type"][0]
        if pred_tags[idx] == 1:
            # 正样本，预测为正--正确
            if true_tags[idx] == 1:
                tp += 1
                score_dict[t][0] += 1
            # 负样本，预测为正--错误
            else:
                fp += 1
                score_dict[t][1] += 1
        else:
            # 正样本，预测为负--错误
            if true_tags[idx] == 1:
                fn += 1
                score_dict[t][2] += 1
            # 负样本，预测为负--正确
            else:
                tn += 1
                score_dict[t][3] += 1
    # total
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = (tp / (tp + fp)) if (tp + fp) != 0 else 0
    recall = (tp / (tp + fn)) if(tp + fn) != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    result['accuracy'] = accuracy
    result['precision'] = precision
    result['recall'] = recall
    result['f1'] = f1

    # each type
    type_statistics = {}
    for k, v in score_dict.items():
        accuracy = ((v[0] + v[3]) / sum(v)) if (sum(v) != 0) else 0
        precision = (v[0] / (v[0] + v[1])) if (v[0] +v[1]) != 0 else 0
        recall = (v[0] / (v[0] + v[2])) if (v[0] + v[2]) != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        type_statistics[k] = {'accuracy':accuracy,'precision':precision,'recall':recall,'f1':f1}

    score_dict['total'] = [tp, fp, fn, tn]
    result['score_dict'] = score_dict
    result['type_statistics'] = type_statistics

    return result

if __name__ == "__main__":
    true = [0,1,0]
    pred = [1,1,0]
    print(acc(true, pred))