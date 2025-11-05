import logging
import os
from datetime import datetime

def get_logger(log_dir="logs",log_file=None,level=logging.INFO):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    if log_file is None:
        log_file = datetime.now().strftime('log_%Y%m%d_%H%M%S.txt')
    
    log_path = os.path.join(log_dir,log_file)

    logger = logging.getLogger()
    logger.setLevel(level)

    if not logger.handlers:
        # 文件 handler
        fh = logging.FileHandler(log_path)
        fh.setLevel(level)
        fh_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

        # 控制台 handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)
    
    return logger