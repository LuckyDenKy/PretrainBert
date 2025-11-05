# 通用起步配置（FP16 + ZeRO stage2 + AdamW）
import os
import yaml
import argparse
import time
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
import deepspeed

from src.utils.config_parser import load_config
from src.data.dataset_loader import get_tokenizer,get_preprocessed_dataset,split_dataset
from src.data.dataloader_builder import get_dataloader
from src.models.model_builder import get_model
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.utils.save_load import save_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str,default="config/ds_train_config.yaml")
    parser.add_argument('--deepspeed_config',type=str,default="deepspeed_config/ds_config.json")
    parser.add_argument('--debug',action='store_true')
    parser.add_argument('--resume_from',type=str,default=None,help="Path to a checkpoint directory to resume training from.")
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def train():
    args = parse_args()

    # ---------------------- 加载Yaml配置文件 ----------------------
    cfg = {}
    cfg_path = args.config
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path,"r",encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # project paths
    ROOT_DIR = cfg["project"]["root_dir"]
    PROJECT_DIR = cfg["project"]["project_dir"]
    log_dir_rel = cfg["project"].get("log_dir","ds_logs")
    ckpt_dir_rel = cfg["project"].get("ckpt_dir","ds_checkpoints")
    deepspeed_config_path = args.deepspeed_config or cfg["project"].get("deepspeed_config")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(PROJECT_DIR,log_dir_rel,f"run_{timestamp}")
    ckpt_dir = os.path.join(PROJECT_DIR,ckpt_dir_rel)
    os.makedirs(log_dir,exist_ok=True)
    os.makedirs(ckpt_dir,exist_ok=True)

    # ---------------------- init seed and logging ----------------------
    set_seed(cfg.get("seed",42))

    # logger only writes out to files (but created on all ranks is fine); we'll only print/write TB from rank 0
    logger = get_logger(log_dir)
    rank = 0
    if deepspeed.comm.is_initialized():
        rank = deepspeed.comm.get_rank()
    logger.info(f"Process rank: {rank}")

    # TensorBoard writer only on rank 0
    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir=log_dir)
    
    logger.info(f"Loaded config from {cfg_path}")
    logger.info(f"DeepSpeed JSON: {deppspeed_config_path}")

    # -------- tokenizer + dataset --------
    tokenizer = get_tokenizer(PROJECT_DIR)
    tokenized_dataset = get_preprocessed_dataset(PROJECT_DIR)
    train_dataset,valid_dataset,test_dataset = split_dataset(tokenized_dataset)

    # debug subset (either from args or yaml)
    debug_samples = cfg["data"].get("debug_samples", 0)
    if args.debug:
        debug_samples = max(debug_samples, 200)
    if debug_samples and debug_samples > 0:
        train_dataset = train_dataset.select(range(min(debug_samples, len(train_dataset))))
        valid_dataset = valid_dataset.select(range(min(max(1, debug_samples // 5), len(valid_dataset))))
        logger.info(f"[DEBUG] Using small subset: {len(train_dataset)} train / {len(valid_dataset)} valid")
    
    # -------- dataloaders (per-device micro-batch size) --------
    per_device_batch = cfg['model'].get("per_device_batch_size",8)
    num_workers = cfg['data'].get("num_workers",2)
    pin_memory = cfg['data'].get("pin_memory",True)

    train_loader = get_dataloader(train_dataset,per_device_batch,tokenizer,is_train=True,num_workers=num_workers,pin_memory=pin_memory)
    valid_loader = get_dataloader(valid_dataset,per_device_batch,tokenizer,is_train=False,num_workers=num_workers,pin_memory=pin_memory)

    # -------- build model --------------
    model = get_model()

    # -------- DeepSpeed initialize --------
    model_params = filter(lambda p:p.requires_grad,model.parameters())

    # merge deepspeed json path from args / yaml
    ds_json = deepspeed_config_path
    if not os.path.exists(ds_json):
        logger.warning(f"DeepSpeed config JSON not found at {ds_json}. deepspeed.initialize may still accept args.")
        ds_json = None
    
    # call initialize with args so launcher-provided args are honored
    logger.info("Calling deepspeed.initialize(...)")
    model_engine,optimizer,_,_ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model_params,
        config=ds_json
    )

    logger.info(f"DeepSpeed initialized. local_rank={model_engine.local_rank}, global_rank={deepspeed.comm.get_rank()}")

    # if resume path provided via CLI, try load checkpoint
    if args.resume_from:
        if os.path.exists(args.resume_from):
            logger.info(f"Resuming from checkpoint: {args.resume_from}")
            # deepspeed load_checkpoint returns (load_success, client_state)
            load_success, client_state = model_engine.load_checkpoint(args.resume_from)
            logger.info(f"load_checkpoint returned: {load_success}, client_state keys: {list(client_state.keys()) if client_state else None}")
        else:
            logger.warning(f"resume_from path does not exist: {args.resume_from}")
    
    # ensure model_engine is in train mode
    model_engine.train()

    # 训练超参数
    num_epochs = int(cfg['model'].get("num_epochs",3))
    log_interval = int(cfg['model'].get('log_interval',10))
    eval_stpes = int(cfg['model'].get('eval_steps'),500)
    save_every_epoch = bool(cfg['model'].get("save_every_epoch",True))

    # 训练循环！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(1,num_epochs+1):
        logger.info(f"===== Epoch {epoch}/{num_epochs} =====")
        epoch_loss = 0.0
        steps = 0

        for step,batch in enumerate(train_loader):
            batch = {k:v.to(model_engine.device) for k,v in batch.items()}

            outputs = model_engine(**batch)
            loss = outputs.loss

            model_engine.backward(loss)
            model_engine.step()  # step handles accum/optimizer/zero_grad via DS

            epoch_loss += loss.item()
            steps += 1
            global_step += 1

            # logging on rank 0
            if rank == 0 and global_step % log_interval == 0:
                avg_loss = epoch_loss / steps
                logger.info(f"[Epoch {epoch}] Step {step+1}/{len(train_loader)} GlobalStep {global_step} AvgLoss {avg_loss:.4f}")
                if writer:
                    writer.add_scalar("train/loss",avg_loss,global_step)
            
            # eval
            if eval_steps > 0 and global_step % eval_steps == 0:
                val_loss = None
                model_engine.eval()
                total_val_loss = 0.0
                val_steps = 0
                with torch.no_grad():
                    for v_step,v_batch in enumerate(valid_dataloader):
                        v_batch = {k: v.to(model_engine.device) for k, v in v_batch.items()}
                        v_out = model_engine(**v_batch)
                        total_val_loss += v_out.loss.item()
                        val_steps += 1
                        if v_step >= 1000:  # safety cap
                            break
                if val_steps > 0:
                    val_loss = total_val_loss / val_steps
                model_engine.train()

                if rank == 0:
                    logger.info(f"[Eval] GlobalStep {global_step} Validation Loss: {val_loss:.4f}")
                    if writer:
                        writer.add_scalar("valid/loss", val_loss, global_step)
                
                if val_loss is not None and val_loss < best_val_loss and rank == 0:
                    best_val_loss = val_loss
                    tag = f"best_step{global_step}"
                    model_engine.save_checkpoint(ckpt_dir,tag)
                    save_model(model_engine.module,os.path.join(ckpt_dir,"best_model.pt"))
                    logger.info(f"Saved new best model at step {global_step} (val_loss={val_loss:.4f})")
                
        # 一个epoch结束
        epoch_avg = epoch_loss / max(1,steps)
        if rank == 0:
            logger.info(f"Epoch {epoch} finished. Avg Loss: {epoch_avg:.4f}")
            if writer:
                writer.add_scalar("train/epoch_loss",epoch_avg,epoch)

        # epoch_level validation & checkpoint
        model_engine.eval()
        val_loss = None
        total_val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for v_step, v_batch in enumerate(valid_loader):
                v_batch = {k: v.to(model_engine.device) for k, v in v_batch.items()}
                v_out = model_engine(**v_batch)
                total_val_loss += v_out.loss.item()
                val_steps += 1
                if v_step >= 1000:
                    break
        if val_steps > 0:
            val_loss = total_val_loss / val_steps

        if rank == 0:
            logger.info(f"[Epoch Eval] Epoch {epoch} Validation Loss: {val_loss:.4f}")
            if writer:
                writer.add_scalar("valid/epoch_loss", val_loss, epoch)

            # save checkpoint via DeepSpeed (handles sharded files)
            tag = f"epoch{epoch}"
            save_ok, client_state = model_engine.save_checkpoint(ckpt_dir, tag)
            logger.info(f"Saved checkpoint {tag} -> success={save_ok}")

            # optional plain save of module for convenience
            if save_every_epoch:
                save_model(model_engine.module, os.path.join(ckpt_dir, f"model_epoch{epoch}.pt"))      
        
        # restore train mode
        model_engine.train()

    if writer:
        writer.close()
    logger.info("Training finished.")          

if __name__ == '__main__':
    train()