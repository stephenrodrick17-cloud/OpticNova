import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse

from datasets.dataset_loader import GlaucomaClassificationDataset
from preprocessing.augmentation import get_train_transforms, get_val_transforms
from classification.efficientnet import GlaucomaEfficientNet
from classification.resnet import GlaucomaResNet

def train_classification(data_dir, model_type='efficientnet', epochs=20, batch_size=8, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup Datasets
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_dir):
        print(f"Error: {train_dir} does not exist. Please organize your dataset into train/ and val/ folders, with class subfolders inside.")
        return
        
    train_dataset = GlaucomaClassificationDataset(train_dir, transforms=get_train_transforms())
    val_dataset = GlaucomaClassificationDataset(val_dir, transforms=get_val_transforms())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Setup Model
    if model_type == 'efficientnet':
        model = GlaucomaEfficientNet(num_classes=1, pretrained=True).to(device)
    else:
        model = GlaucomaResNet(num_classes=1, pretrained=True).to(device)
        
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            total_train += labels.size(0)
            
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss /= total_train
        train_acc = train_correct / total_train
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                total_val += labels.size(0)
                
        val_loss = val_loss / total_val if total_val > 0 else 0.0
        val_acc = val_correct / total_val if total_val > 0 else 0.0
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_{model_type}_classification.pth")
            print("Saved new best model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Classification Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to classification dataset directory (must contain 'train' and 'val' subfolders)")
    parser.add_argument("--model", type=str, default="efficientnet", choices=["efficientnet", "resnet"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    train_classification(args.data_dir, args.model, args.epochs, args.batch_size, args.lr)
