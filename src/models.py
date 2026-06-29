import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def evaluate_predictions(y_true, y_pred_proba, train_time=0.0, infer_time=0.0):
    """
    Evaluates predictions using key credit risk metrics:
    - AUC-ROC
    - Precision-Recall AUC (Average Precision)
    - F1-Score (optimized over thresholds)
    - Brier Score (calibration check)
    - Latency (training and inference time)
    """
    # AUC-ROC
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    
    # PR-AUC
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    # Optimal F1 and threshold
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    # Brier Score
    brier = brier_score_loss(y_true, y_pred_proba)
    
    return {
        'auc_roc': auc_roc,
        'pr_auc': pr_auc,
        'f1_score': best_f1,
        'optimal_threshold': best_threshold,
        'brier_score': brier,
        'train_time_sec': train_time,
        'infer_time_sec': infer_time
    }

class SimpleMLP(nn.Module):
    """
    Standard PyTorch Multi-Layer Perceptron (MLP) architecture
    used as a direct architectural baseline comparison to KAN.
    """
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.2):
        super(SimpleMLP, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_mlp_baseline(X_train, y_train, X_val, y_val, hidden_dims=[128, 64], 
                       lr=1e-3, batch_size=512, epochs=50, dropout=0.2, 
                       weight_decay=1e-5, device='cpu'):
    """
    Trains the PyTorch SimpleMLP baseline with early stopping, 
    class weights, and returns predictions and training/inference latency.
    """
    # Convert data to PyTorch Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = SimpleMLP(X_train.shape[1], hidden_dims, dropout).to(device)
    
    # Class weights for loss function (credit default imbalance: ~92/8)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    pos_weight = torch.tensor([neg_count / (pos_count + 1e-5)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 5
    patience_counter = 0
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        with torch.no_grad():
            val_X_device = X_val_t.to(device)
            val_y_device = y_val_t.to(device)
            val_outputs = model(val_X_device)
            val_loss = criterion(val_outputs, val_y_device).item()
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    train_time = time.time() - start_time
    
    # Load best model for inference
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    model.eval()
    with torch.no_grad():
        start_infer = time.time()
        val_outputs = torch.sigmoid(model(X_val_t.to(device))).cpu().numpy().flatten()
        infer_time = time.time() - start_infer
        
    return val_outputs, train_time, infer_time
