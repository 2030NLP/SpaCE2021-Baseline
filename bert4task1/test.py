import config
import logging
import sys
import json
import torch
import numpy as np

from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
from data_loader import SpaCEDataset

def test(model_dir, test_dir, result_dir):
    # utils.set_logger(config.log_dir)
    test_dataset = SpaCEDataset(test_dir, config, True)
    logging.info("--------Dataset Build!--------")
    # build data_loader
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, collate_fn=test_dataset.collate_fn)
    logging.info("--------Get Data-loader!--------")
    # Prepare model
    if model_dir is not None:
        model = BertForSequenceClassification.from_pretrained(model_dir)
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(model_dir))
    else:
        logging.info("--------No model to test !--------")
        return

    model.eval()
    pred_tags = []
    with torch.no_grad():
        for idx, batch_samples in enumerate(test_loader):
            batch_data = batch_samples[0]
            batch_mask = batch_data.gt(0)
            # batch_segment = batch_data.lt(-1)

            outputs = model(batch_data, batch_mask)

            batch_output = outputs.logits   # shape: (batch_size, num_labels)

            batch_output = batch_output.detach().cpu().numpy()

            pred_tags.extend(np.argmax(batch_output, axis=-1))

    with open(test_dir, 'r') as fr:
        items = json.load(fr)

    for idx, item in enumerate(items):
        item['judge1'] = int(pred_tags[idx])

    with open(result_dir, 'w') as fw:
        json.dump(items, fw, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    test(sys.argv[1], sys.argv[2], sys.argv[3])