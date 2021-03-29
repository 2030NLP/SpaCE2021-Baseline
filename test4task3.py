import os
import json
from bert4task1.run import test as test1
from bert4task2.run import test as test2

def get_new_metrics(result_dir):
    with open(result_dir, "r") as fr:
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

    precision = ((c_ft + c_ff)/(p_ft + p_ff)) if (p_ft + p_ff)!=0 else 0
    recall = ((c_ft + c_ff)/(gt_ft + gt_ff)) if (gt_ft + gt_ff) !=0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall)!=0 else 0
    print("Test3 precision:{:.6f}, recall:{:.6f}, f1:{:.6f}".format(precision, recall, f1))

if __name__ == "__main__":
    model1_dir = os.getcwd() + '/bert4task1/experiments'
    model2_dir = os.getcwd() + '/bert4task2/experiments'
    task3_test_dir = os.getcwd() + '/task3-dev.json'
    model1_prediction_dir = os.getcwd() + '/model1_prediction.json'
    model2_prediction_dir = os.getcwd() + '/model2_prediction.json'
    test1(model1_dir, task3_test_dir, model1_prediction_dir)
    test2(model2_dir, model1_prediction_dir, model2_prediction_dir)
    # get_new_metrics(model2_prediction_dir)