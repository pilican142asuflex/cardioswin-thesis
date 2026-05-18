import torch
from .utils import calculate_accuracy

def validate(model, loader, criterion, device):
    
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    
    with torch.no_grad():
        for inputs, labels in loader:
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            acc = calculate_accuracy(outputs, labels)
            
            running_loss += loss.item()
            running_acc += acc
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = running_acc / len(loader)
    
    return epoch_loss, epoch_acc