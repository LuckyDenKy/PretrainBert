import os 
import torch

def save_model(model,save_dir,model_name="best_model.pt"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir,model_name)
    torch.save(model.state_dict(),save_path)
    print(f"Model saved at {save_path}")
    return save_path

def load_model(model,load_path,device="cpu"):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"{load_path} not found!")
    state_dict = torch.load(load_path,map_location=device)
    model.load_state_dict(state_dict)
    print(f"Model loaded from {load_path}")
    return model