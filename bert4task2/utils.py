import logging
import config
import json
from sklearn.model_selection import train_test_split


def split_training_set():
    with open(config.ori_train_dir,'r') as fr:
        origin_train = json.load(fr)

    splited_train, splited_dev = train_test_split(origin_train, train_size=0.9)

    with open(config.train_dir, 'w') as fw:
        json.dump(splited_train, fw, indent=2, ensure_ascii=False)
    with open(config.dev_dir, 'w') as fw:
        json.dump(splited_dev, fw, indent=2, ensure_ascii=False)

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)