import os, sys, json, time, math, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from model import GWSatModel

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        
        # GWSat expects [B, 8, 64, 64]
        preds = model(X)
        loss = criterion(preds, y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            _, predicted = torch.max(preds.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=60)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=3e-4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training on: {device}")

    # ─── DATA LOADING (UPDATED FOR REAL DATA) ───
    # Ensure you have run 'python build_real_dataset.py' first!
    if not os.path.exists('data/train.pt'):
        print("❌ Error: data/train.pt not found. Run build_real_dataset.py first.")
        sys.exit(1)

    print("📦 Loading real Sentinel-2 tensors...")
    train_data = torch.load('data/train.pt', weights_only=False)
    val_data   = torch.load('data/val.pt', weights_only=False)

    train_loader = DataLoader(
        TensorDataset(train_data['X'], train_data['y']), 
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_data['X'], val_data['y']), 
        batch_size=args.batch_size
    )

    # ─── MODEL SETUP ───
    model = GWSatModel(device=device)
    # Note: Encoder is frozen by default in model.py
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    
    best_acc = 0
    os.makedirs("checkpoints", exist_ok=True)

    print(f"开始 (Start) training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc  = evaluate(model, val_loader, device)
        
        if acc > best_acc:
            best_acc = acc
            model.save_head("checkpoints/best_head.pth")
            
        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val Acc: {acc:.4f}")

    print(f"\n✅ Training Complete. Best Val Accuracy: {best_acc:.4f}")
    print("Head saved to: checkpoints/best_head.pth")