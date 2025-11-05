import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.config_parser import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.utils.save_load import save_model
from src.data.dataset_loader import get_tokenizer, get_preprocessed_dataset, split_dataset
from src.data.dataloader_builder import get_dataloader
from src.models.model_builder import get_model

def train_one_epoch(model,dataloader,optimizer,tokenizer,epoch,writer,logger,device,log_interval):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader,desc=f"Training Epoch {epoch}")

    for step,batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (step+1)
        pbar.set_postfix({"loss": avg_loss})

        if (step+1)%log_interval == 0:
            logger.info(f"Epoch [{epoch}] Step [{step+1}] Loss:{avg_loss:.4f}")
            writer.add_scalar("Train/Loss",avg_loss,epoch*len(dataloader)+step)
    
    return avg_loss

@torch.no_grad()
def evaluate(model,dataloader,tokenizer,epoch,writer,logger,device):
    model.eval()
    total_loss = 0.0
    pbar = tqdm(dataloader,desc=F"Validating Epoch {epoch}")

    for step,batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        avg_loss = total_loss / (step+1)
        pbar.set_postfix({"val_loss": avg_loss})
    
    logger.info(f"[Validation] Epoch {epoch} | Avg Loss: {avg_loss:.4f}")
    writer.add_scalar("Valid/Loss",avg_loss,epoch)
    return avg_loss

def main():
    cfg = load_config("config/train_config.yaml")

    set_seed(cfg.training.seed)
    logger = get_logger(cfg.paths.log_dir)
    writer = SummaryWriter(log_dir=cfg.paths.log_dir)
    logger.info("==== PretrainBert Training Started ====")

    tokenizer = get_tokenizer(cfg.project.project_dir)
    dataset = get_preprocessed_dataset(cfg.project.project_dir)
    train_dataset,valid_dataset,_ = split_dataset(dataset)

    # Debug模式
    if cfg.training.debug_mode:
        train_dataset = train_dataset.select(range(100))
        valid_dataset = valid_dataset.select(range(20))
        cfg.training.epochs = 1
        print(f"[Debug mode] Using {len(train_dataset)} train samples and {len(valid_dataset)} valid samples")

    train_loader = get_dataloader(train_dataset,cfg.training.batch_size,tokenizer,is_train=True,num_workers=cfg.runtime.num_workers,pin_memory=cfg.runtime.pin_memory)
    valid_loader = get_dataloader(valid_dataset,cfg.training.batch_size,tokenizer)

    model = get_model().to(cfg.runtime.device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=float(cfg.training.lr))

    best_loss = float("inf")
    for epoch in range(1,cfg.training.epochs + 1):
        logger.info(f"===== Epoch {epoch} =====")
        train_loss = train_one_epoch(model,train_loader,optimizer,tokenizer,epoch,writer,logger,cfg.runtime.device,cfg.training.log_interval)
        val_loss = evaluate(model,valid_loader,tokenizer,epoch,writer,logger,cfg.runtime.device)

        if val_loss < best_loss:
            best_loss = val_loss
            save_model(model,cfg.paths.save_dir,model_name=f"bert_epoch{epoch}_loss{val_loss:.4f}.pt") 
    
    logger.info("Training Completed")
    writer.close()

if __name__ == "__main__":
    main()