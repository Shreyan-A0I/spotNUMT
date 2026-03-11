import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score
import numpy as np
import torch.nn.functional as F

from dataset import get_dataloaders
from model import HybridNuMTModel

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs are logits. compute bce with logits
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(X)
        
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
        
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred_probs = []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            
            total_loss += loss.item() * X.size(0)
            
            probs = torch.sigmoid(logits)
            y_true.extend(y.cpu().numpy())
            y_pred_probs.extend(probs.cpu().numpy())
            
    avg_loss = total_loss / len(loader.dataset)
    
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs >= 0.5).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    # Check if there are no positive predictions to avoid warnings
    if sum(y_pred) == 0:
        prec = 0.0
    else:
        prec = precision_score(y_true, y_pred, zero_division=0)
        
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    try:
        auroc = roc_auc_score(y_true, y_pred_probs)
        auprc = average_precision_score(y_true, y_pred_probs)
    except ValueError:
        auroc = 0.5 # In case only one class is present in the batch
        auprc = 0.0
        
    metrics = {
        'loss': avg_loss,
        'acc': acc,
        'prec': prec,
        'rec': rec,
        'auroc': auroc,
        'auprc': auprc
    }
    
    return metrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=64)
    
    # Calculate pos_weight for Imbalanced Learning
    # For focal loss, alpha acts as the weighting. alpha=0.75 means we care more about the positive class.
    # standard alpha is 0.25 (meaning neg class is weighted higher), but since we want to find positives, we invert.
    model = HybridNuMTModel().to(device)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 25
    best_val_loss = float('inf')
    
    print("\n--- Starting Training ---")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val AUROC: {val_metrics['auroc']:.4f} | "
              f"Val AUPRC: {val_metrics['auprc']:.4f}")
              
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), 'best_model.pt')
            print("  --> Saved new best model!")
            
    print("\n--- Evaluating on Test Set ---")
    model.load_state_dict(torch.load('best_model.pt', weights_only=True))
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"Test Loss:      {test_metrics['loss']:.4f}")
    print(f"Test Accuracy:  {test_metrics['acc']:.4f}")
    print(f"Test Precision: {test_metrics['prec']:.4f}")
    print(f"Test Recall:    {test_metrics['rec']:.4f}")
    print(f"Test AUROC:     {test_metrics['auroc']:.4f}")
    print(f"Test AUPRC:     {test_metrics['auprc']:.4f}")

if __name__ == "__main__":
    main()
