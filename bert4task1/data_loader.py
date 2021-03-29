import torch
import json
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class SpaCEDataset(Dataset):
    def __init__(self, file_path, config, test_flag=False):
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True)
        self.mode = test_flag
        self.dataset = self.preprocess(file_path)
        self.device = config.device

    def preprocess(self, file_path):
        with open(file_path, "r") as fr:
            items = json.load(fr)

        data = []
        for item in items:
            text = item['context']
            tokens = self.tokenizer.encode(text)
            if self.mode == True:
                data.append([tokens])
            else:
                label = 1 if item['judge1'] == True else 0
                data.append([tokens, label])

        return data

    def __getitem__(self, idx):
        """sample data to get batch"""
        tokens = self.dataset[idx][0]
        if self.mode == True:
            return [tokens]
        else:
            label = self.dataset[idx][1]
            return [tokens, label]

    def __len__(self):
        """get dataset size"""
        return len(self.dataset)

    def collate_fn(self, batch):
        """
        process batch data, including:
            1. padding: 将每个batch的data padding到同一长度（batch中最长的data长度）
            2. tensor：转化为tensor
        """
        sentences = [x[0] for x in batch]
        # print(sentences)
        # print(sentences)
        batch_data = pad_sequence([torch.from_numpy(np.array(s)) for s in sentences], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch_data = torch.tensor(batch_data, dtype=torch.long)
        # shift tensors to GPU if available
        batch_data = batch_data.to(self.device)

        if self.mode == True:
            return [batch_data]
        else:
            labels = [x[1] for x in batch]
            batch_label = labels
            batch_label = torch.tensor(batch_label, dtype=torch.long)
            batch_label = batch_label.to(self.device)
            return [batch_data, batch_label]
