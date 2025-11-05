import yaml
import os
import torch

class Config:
    def __init__(self,cfg_dict):
        for key,value in cfg_dict.items():
            if isinstance(value,dict):
                value = Config(value)
            setattr(self,key,value)

def load_config(config_path: str="config/train_config.yaml"):
    with open(config_path,"r",encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = Config(cfg_dict)

    cfg.paths.save_dir = os.path.join(cfg.project.project_dir,cfg.paths.save_dir)
    cfg.paths.log_dir = os.path.join(cfg.project.project_dir,cfg.paths.log_dir)

    os.makedirs(cfg.paths.save_dir,exist_ok=True)
    os.makedirs(cfg.paths.log_dir,exist_ok=True)

    if not torch.cuda.is_available():
        cfg.runtime.device = "cpu"
        print(f"Cuda is not available, use CPU")
    
    return cfg