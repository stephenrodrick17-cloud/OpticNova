import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse

from datasets.dataset_loader import FundusSegmentationDataset
from preprocessing.augmentation import get_train_transforms, get_val_transforms
from segmentation.unet import UNet

def dice_loss(pred, target, smooth=1e-6):
    """Calculate Dice Loss"""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def train_segmentation(data_dir, epochs=30, batch_size=4, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup Datasets
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_dir):
        print(f"Error: {train_dir} does not exist. Please organize your dataset into train/ and val/ folders, each containing images/ and masks/ subfolders.")
        return
        
    train_dataset = FundusSegmentationDataset(train_dir, transforms=get_train_transforms())
    val_dataset = FundusSegmentationDataset(val_dir, transforms=get_val_transforms())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Setup Model
    model = UNet(n_channels=3, n_classes=2).to(device)
    
    # BCE + Dice Loss
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Combine BCE and Dice Loss
            loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss /= len(train_loader.dataset)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)
                
                val_loss += loss.item() * images.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_unet_segmentation.pth")
            print("Saved new best model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Segmentation Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to segmentation dataset directory (must contain 'train' and 'val' subfolders)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    train_segmentation(args.data_dir, args.epochs, args.batch_size, args.lr)
