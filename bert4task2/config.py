import os
import torch

data_dir = os.getcwd() + '/data'
train_dir = data_dir + '/train.json'
dev_dir = data_dir + '/test.json'
test_dir = data_dir + '/dev.json'
files = ['train', 'dev', 'test']
bert_model = 'pretrained_bert_models/chinese-wwm-ext/'
roberta_model = 'pretrained_bert_models/chinese_roberta_wwm_large_ext/'
model_dir = os.getcwd() + '/experiments/clue/'
log_dir = model_dir + 'train.log'
case_dir = os.getcwd() + '/case/bad_case.txt'
result_dir = data_dir + '/prediction.json'

# 训练集、验证集划分比例
# dev_split_size = 0.1

# 是否加载训练好的NER模型
load_before = False

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
learning_rate = 3e-5
weight_decay = 0.01
clip_grad = 5

batch_size = 8
epoch_num = 50
min_epoch_num = 5
patience = 0.0002
patience_num = 10

gpu = '0'

if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")
