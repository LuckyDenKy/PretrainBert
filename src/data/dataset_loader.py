from transformers import BertTokenizer
from datasets import load_dataset,load_from_disk
import os

def get_tokenizer(project_dir):
    tokenizer_dir = os.path.join(project_dir,"tokenizer/bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
    return tokenizer

def get_raw_dataset(root_dir="./"):
    data_dir = os.path.join(root_dir,"data/ag_news")
    dataset = load_dataset(data_dir)
    return dataset

def preprocess_and_save(tokenizer,root_dir="./"):
    data_dir = os.path.join(root_dir,"data/ag_news")
    dataset = load_dataset(data_dir)

    def preprocess(examples):
        return tokenizer(examples["text"],truncation = True,padding="max_length",max_length=128)
    
    tokenized = dataset.map(preprocess,batched=True,remove_columns=["label","text"])
    save_dir = os.path.join(root_dir,"data/ag_news_tokenized")
    if not os.path.exists(save_dir):
        tokenized.save_to_disk(save_dir)
    return tokenized

def get_preprocessed_dataset(root_dir="./"):
    data_dir = os.path.join(root_dir,"data/ag_news_tokenized")
    tokenized_data = load_from_disk(data_dir)
    return tokenized_data

def split_dataset(tokenized_dataset):
    dataset_split = tokenized_dataset["train"].train_test_split(test_size=0.05,seed=42)
    train_set = dataset_split["train"]
    valid_set = dataset_split["test"]
    test_set = tokenized_dataset["test"]
    return train_set,valid_set,test_set
