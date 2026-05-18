import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from heart_disease.configs.config import Config
from heart_disease.data_loader.acdc_dataset import ACDCDataset
from heart_disease.models.full_model import HeartDiseaseModel
from heart_disease.training.train import train_one_epoch
from heart_disease.training.validate import validate
from heart_disease.training.utils import save_checkpoint

"""def main():
    
    config = Config()
    
    print("Using device:", config.DEVICE)
    
    # Dataset
    train_dataset = ACDCDataset(config.DATA_ROOT, split="training")
    val_dataset = ACDCDataset(config.DATA_ROOT, split="testing")
    
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False)
    
    # Model
    model = HeartDiseaseModel(config).to(config.DEVICE)
    
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    best_val_acc = 0.0
    
    for epoch in range(config.NUM_EPOCHS):
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE
        )
        
        val_loss, val_acc = validate(
            model, val_loader, criterion, config.DEVICE
        )
        
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, "weights/best_model.pth")
            print("Model saved.")

if __name__ == "__main__":
    main()

class Config:
    MODEL_NAME = "swin_tiny_patch4_window7_224"
    NUM_CLASSES = 5
    TEMPORAL_MODE = "avg"
    HIDDEN_DIM = 256
    PRETRAINED = True
def main():
    
    device = torch.device("cpu")

    dataset = ACDCDataset("data/ACDC", split="training")
    x, y = dataset[0]

    print("Input shape:", x.shape)

    model = HeartDiseaseModel(Config).to(device)

    x = x.unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        out = model(x)

    print("Output shape:", out.shape)

if __name__ == "__main__":
    main()"""

#main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from heart_disease.configs.config import Config
from heart_disease.data_loader.acdc_dataset import ACDCDataset
from heart_disease.models.full_model import HeartDiseaseModel
import os
import sys

DATA_ROOT = "/content/drive/MyDrive/ACDC"
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    
    if not os.path.exists(DATA_ROOT):
        print(f"ERROR: Directory not found at {DATA_ROOT}. Check your Drive mount.")
        return

    
    train_dataset = ACDCDataset(DATA_ROOT, stage='train', transform=True)
    val_dataset = ACDCDataset(DATA_ROOT, stage='val', transform=False)

    
    print(f"Files found for training: {len(train_dataset)}")
    print(f"Files found for validation: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("ERROR: Dataset is empty. Check the file-finding logic in acdc_dataset.py")
        sys.exit()
    

    model = HeartDiseaseModel(Config).to(DEVICE)

    
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': head_params, 'lr': 1e-4}
    ], weight_decay=0.05)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    best_acc = 0
    for epoch in range(30):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = model(x)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        val_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/30] | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

if __name__ == "__main__":
    train_model()
