# src/models/model_builder.py
from transformers import BertConfig, BertForMaskedLM

def get_model(config=None):
    if config is None:
        config = BertConfig(
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072
        )

    model = BertForMaskedLM(config)
    total_params = sum(p.numel() for p in model.parameters())
    if total_params >= 1e9:
        print(f"Model parameters: {total_params / 1e9:.2f}B")
    elif total_params >= 1e6:
        print(f"Model parameters: {total_params / 1e6:.2f}M")
    else:
        print(f"Model parameters: {total_params}")
    return model
