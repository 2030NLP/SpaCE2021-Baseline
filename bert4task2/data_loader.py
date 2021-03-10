import torch
import json
import config
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence

torch.set_printoptions(profile="full")

class NERDataset(Dataset):
    def __init__(self, file_path, config):
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True)
        self.dataset = self.preprocess(file_path)
        self.device = config.device

    def preprocess(self, file_path):
        with open(file_path, "r") as fr:
            items = json.load(fr)

        data = []
        for item in items:
            text = item['context']
            answer = item['reason']
            label = 1 if item['judge2'] == True else 0

            text_tokens = self.tokenizer.encode(text)[1:-1]     # 恰头去尾好处理
            ans_tokens = self.tokenizer.encode(answer)[1:-1]

            # 优先保证answer完整
            if (len(text_tokens) + len(ans_tokens)) > 509:
                res = 509-len(ans_tokens)
            else:
                res = len(text_tokens)
            tokens = [self.tokenizer.cls_token_id] + text_tokens[:res] + \
                     [self.tokenizer.sep_token_id] + ans_tokens + [self.tokenizer.sep_token_id]
            segment = [0] * (res + 2) + [1] * len(ans_tokens)

            # res+2表示answer的起始位置
            data.append((tokens, label, segment))
            assert len(tokens) <= 512

        return data

    def __getitem__(self, idx):
        """sample data to get batch"""
        tokens = self.dataset[idx][0]
        label = self.dataset[idx][1]
        res = self.dataset[idx][2]
        return [tokens, label, res]

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
        labels = [x[1] for x in batch]
        segments = [x[2] for x in batch]

        batch_data = pad_sequence([torch.from_numpy(np.array(s)) for s in sentences], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch_label = labels
        batch_segment = pad_sequence([torch.from_numpy(np.array(s)) for s in segments], batch_first=True, padding_value=1)


        batch_data = torch.tensor(batch_data, dtype=torch.long)
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        batch_segment = torch.tensor(batch_segment, dtype=torch.long)

        # shift tensors to GPU if available
        batch_data = batch_data.to(self.device)
        batch_label = batch_label.to(self.device)
        batch_segment = batch_segment.to(self.device)
        return [batch_data, batch_label, batch_segment]

if __name__ == "__main__":
    dev_dataset = NERDataset('./data/dev.json',config)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                                shuffle=False, collate_fn=dev_dataset.collate_fn)
    for idx, batch in enumerate(dev_loader):
        print(batch[1])
        if idx > 0:
            break