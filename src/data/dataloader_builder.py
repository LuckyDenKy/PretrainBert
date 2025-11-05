import torch
from torch.utils.data import DataLoader

def mask_tokens(inputs,tokenizer,mlm_probability=0.15):
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape,mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val,already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask,dtype=torch.bool),value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    # 80% [MASK], 10% 随机token，10% 保留原样
    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    indices_random = torch.bernoulli(torch.full(labels.shape,0.1)).bool() & masked_indices
    random_words = torch.randint(len(tokenizer),labels.shape,dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    return inputs,labels

def get_dataloader(dataset,batch_size,tokenizer,is_train=False,num_workers=0,pin_memory=False):
    def collate_fn(batch):
        input_ids = torch.tensor([item["input_ids"] for item in batch])
        attention_mask = torch.tensor([item["attention_mask"] for item in batch])
        token_type_ids = torch.tensor([item["token_type_ids"] for item in batch])
        masked_input_ids,labels = mask_tokens(input_ids,tokenizer)
        return {
            'input_ids': masked_input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

def test_mask_tokens(tokenizer):
    text = "Wall St. Bears Claw Back Into the"
    encoding = tokenizer(text)
    input_ids = torch.tensor([encoding["input_ids"]])
    masked_inputs,labels = mask_tokens(input_ids.clone(),tokenizer,mlm_probability=0.15)
    print("text:", text)
    print("original:", input_ids)
    print("masked:", masked_inputs)
    print("labels:", labels)
    tokens = tokenizer.convert_ids_to_tokens(masked_inputs[0])
    print("decoded tokens:", tokens)
    print("decoded text:", tokenizer.decode(masked_inputs[0], skip_special_tokens=False))
