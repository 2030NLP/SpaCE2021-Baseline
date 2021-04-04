# from . import utils
# from . import config


import torch
import logging
import json
import numpy as np
import config
from utils import split_training_set, set_logger
from data_loader import SpaCEDataset
from train import train, evaluate

from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW
from transformers import BertForSequenceClassification

import warnings

warnings.filterwarnings('ignore')


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
        item['pred1'] = int(pred_tags[idx])

    with open(result_dir, 'w') as fw:
        json.dump(items, fw, indent=2, ensure_ascii=False)

    # val_metrics = evaluate(test_loader, model, 'test',test_dir, result_dir)
    # logging.info("test Accuracy: {:.6f}, Precision: {:.6f}, Recall: {:.6f}, f1:{:.6f}".format(val_metrics['accuracy'],
    #                                                                                                 val_metrics['precision'],
    #                                                                                                 val_metrics['recall'],
    #                                                                                                 val_metrics['f1']))

def run():
    """train the model"""
    # set the logger
    split_training_set()
    set_logger(config.log_dir)
    logging.info("device: {}".format(config.device))
    train_dataset = SpaCEDataset(config.train_dir, config)
    dev_dataset = SpaCEDataset(config.dev_dir, config)
    logging.info("--------Dataset Build!--------")
    train_size = len(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=dev_dataset.collate_fn)

    logging.info("--------Get Dataloader!--------")
    # Prepare model
    device = config.device
    model = BertForSequenceClassification.from_pretrained(config.bert_model, num_labels=2)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, correct_bias=False)
    train_steps_per_epoch = train_size // config.batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=train_steps_per_epoch,
                                                num_training_steps=config.epoch_num * train_steps_per_epoch)
    # Train the model
    logging.info("--------Start Training!--------")
    train(train_loader, dev_loader, model, optimizer, scheduler, config.model_dir)
    test(config.model_dir, config.test_dir, config.result_dir)

if __name__ == '__main__':
    # run()
    test(config.model_dir, config.test_dir, config.result_dir)
