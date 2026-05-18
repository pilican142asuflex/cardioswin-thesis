import torch
from .utils import calculate_accuracy

def train_one_epoch(model, loader, criterion, optimizer, device):
    
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    
    for inputs, labels in loader:
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        acc = calculate_accuracy(outputs, labels)
        
        running_loss += loss.item()
        running_acc += acc
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = running_acc / len(loader)
    
    return epoch_loss, epoch_acc