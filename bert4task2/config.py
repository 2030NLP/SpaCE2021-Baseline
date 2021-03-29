import os
import torch

data_dir = os.getcwd() + '/data'
ori_train_dir = data_dir + '/task2-train-with-answer.json'
train_dir = data_dir + '/task2-splited-train.json'
dev_dir = data_dir + '/task2-splited-dev.json'
test_dir = data_dir + '/task2-dev.json'
bert_model = 'pretrained_bert_models/chinese-wwm-ext/'
model_dir = os.getcwd() + '/experiments/'
log_dir = model_dir + 'train.log'
result_dir = data_dir + '/prediction.json'

# hyper-parameter
learning_rate = 1e-5
weight_decay = 0.01
clip_grad = 5

batch_size = 8
epoch_num = 30
min_epoch_num = 5
patience = 0.0002
patience_num = 5

gpu = '2'

if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")
