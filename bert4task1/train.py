import torch
import config
import logging
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from metrics import get_metrics
from transformers import BertTokenizer
from transformers import BertForSequenceClassification


def train_epoch(train_loader, model, optimizer, scheduler, epoch):
    # set model to training mode
    model.train()
    train_losses = 0
    for idx, batch_sample in enumerate(tqdm(train_loader)):
        batch_data, batch_label = batch_sample
        batch_mask = batch_data.gt(0)  # get padding mask

        # compute model output and loss
        outputs = model(batch_data, batch_mask, labels=batch_label)
        loss = outputs.loss
        train_losses += loss
        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
        # performs updates using calculated gradients
        optimizer.step()
        scheduler.step()
    train_loss = float(train_losses) / len(train_loader)
    logging.info("Epoch: {}, train loss: {}".format(epoch, train_loss))


def train(train_loader, dev_loader, model, optimizer, scheduler, model_dir):
    """train the model and test model performance"""
    # reload weights from restore_dir if specified
    if model_dir is not None and config.load_before:
        model = BertForSequenceClassification.from_pretrained(model_dir)
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(model_dir))
    best_val_acc = 0.0
    patience_counter = 0
    # start training
    for epoch in range(1, config.epoch_num + 1):
        train_epoch(train_loader, model, optimizer, scheduler, epoch)
        val_metrics = evaluate(dev_loader, model)
        val_acc = val_metrics['accuracy']
        logging.info("Epoch: {}, Accuracy: {:.6f}, Precision: {:.6f}, Recall: {:.6f}, f1:{:.6f}".format(epoch,val_metrics['accuracy'],
                                                                                                        val_metrics['precision'],val_metrics['recall'],
                                                                                                        val_metrics['f1']))
        improve_acc = val_acc - best_val_acc
        if improve_acc > 1e-5:
            best_val_acc = val_acc
            model.save_pretrained(model_dir)
            logging.info("--------Save best model!--------")
            if improve_acc < config.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1
        # Early stopping and logging best acc
        if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
            logging.info("Best val Accuracy: {}".format(best_val_acc))
            break
    logging.info("Training Finished!")


def evaluate(dev_loader, model, mode='dev'):
    # set model to evaluation mode
    model.eval()
    true_tags = []
    pred_tags = []
    dev_losses = 0

    with torch.no_grad():
        for idx, batch_samples in enumerate(dev_loader):
            batch_data, batch_label = batch_samples
            batch_mask = batch_data.gt(0)
            # batch_segment = batch_data.lt(-1)

            outputs = model(batch_data, batch_mask, labels=batch_label)
            dev_losses += outputs.loss.item()
            batch_output = outputs.logits   # shape: (batch_size, num_labels)

            batch_output = batch_output.detach().cpu().numpy()
            batch_tags = batch_label.to('cpu').numpy()

            pred_tags.extend(np.argmax(batch_output, axis=-1))
            true_tags.extend(batch_tags)

    assert len(pred_tags) == len(true_tags)

    # logging loss, acc, precision, recall, f1 and report
    metrics = get_metrics(true_tags, pred_tags, mode)
    metrics['loss'] = float(dev_losses) / len(dev_loader)
    return metrics


if __name__ == "__main__":
    a = [101, 679, 6814, 8024, 517, 2208, 3360, 2208, 1957, 518, 7027, 4638,
                             1957, 4028, 1447, 3683, 6772, 4023, 778, 8024, 6844, 1394, 3173, 4495,
                             807, 4638, 6225, 830, 5408, 8024, 5445, 3300, 1126, 1767, 3289, 3471,
                             4413, 4638, 2767, 738, 976, 4638, 3683, 6772, 1962, 511, 0, 0,
                             0, 0, 0]
    t = torch.tensor(a, dtype=torch.long)
    tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True, skip_special_tokens=True)
    word = tokenizer.convert_ids_to_tokens(t[1].item())
    sent = tokenizer.decode(t.tolist())
    print(word)
    print(sent)