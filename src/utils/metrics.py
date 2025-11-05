import torch

def compute_loss(predictions,labels,criterion=None):
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = criterion(predictions.view(-1,predictions.size(-1)),labels.view(-1))
    return loss