import utils
import config
import logging
from data_loader import NERDataset
from train import train, evaluate
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW
from transformers import BertForSequenceClassification
import warnings

warnings.filterwarnings('ignore')

def test():
    utils.set_logger(config.log_dir)
    test_dataset = NERDataset(config.test_dir, config)
    logging.info("--------Dataset Build!--------")
    # build data_loader
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, collate_fn=test_dataset.collate_fn)
    logging.info("--------Get Data-loader!--------")
    # Prepare model
    if config.model_dir is not None:
        model = BertForSequenceClassification.from_pretrained(config.model_dir)
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(config.model_dir))
    else:
        logging.info("--------No model to test !--------")
        return
    val_metrics = evaluate(test_loader, model, mode='test')
    print_text = "test Accuracy: {:.6f}, Precision: {:.6f}, Recall: {:.6f}, f1:{:.6f}"
    logging.info(print_text.format(val_metrics['accuracy'], val_metrics['precision'], val_metrics['recall'],
                                   val_metrics['f1']))

    type_statistics = val_metrics['type_statistics']
    for k, v in type_statistics.items():
        if k == "total": continue
        print_text = "type {} \t accuracy:{:.6f},\t precision:{:.6f},\t recall:{:.6f},\t f1:{:.6f}"
        logging.info(print_text.format(k, v['accuracy'], v['precision'],
                                v['recall'], v['f1']))

    score_dict = val_metrics['score_dict']
    logging.info("         \t tp \t fp \t fn \t tn \t sum")
    for k, v in score_dict.items():
        logging.info("{} \t \t {} \t {} \t {} \t {} \t {}".format(k, v[0], v[1], v[2], v[3], sum(v)))

def run():
    """train the model"""
    # set the logger
    utils.set_logger(config.log_dir)
    logging.info("device: {}".format(config.device))
    train_dataset = NERDataset(config.train_dir, config)
    dev_dataset = NERDataset(config.dev_dir, config)
    test_dataset = NERDataset(config.test_dir, config)
    logging.info("--------Dataset Build!--------")
    train_size = len(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=dev_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=test_dataset.collate_fn)

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


if __name__ == '__main__':
    run()
    test()
