"""
train.py
Chest CT Multi-label Classification using Fine-tuned BERT
- Binary classification for pneumonia(pn), embolism(em), cancer(ca)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, hamming_loss
from tqdm import tqdm
import os

# ==========================================
# Configuration
# ==========================================
DATA_PATH = "chest_ct_labeled_data.csv"
MODEL_NAME = "model path" 
OUTPUT_DIR = "chest_ct_classification_results"
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "saved_models")

EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
MAX_LENGTH = 256

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ==========================================
# Dataset Class
# ==========================================
class ChestCTDataset(Dataset):
    """Dataset for chest CT report classification"""
    def __init__(self, df, tokenizer, max_length=256):
        self.texts = df['report'].tolist()
        self.labels = df[['pn', 'em', 'ca']].values.astype(float)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

# ==========================================
# Model Training
# ==========================================
def train_model(train_loader, val_loader, model, device, epochs=20):
    """Train multi-label classification model"""
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    log_history = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        all_labels, all_preds = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask).logits
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
        
        all_labels = np.vstack(all_labels)
        all_preds = np.vstack(all_preds)
        
        # Calculate metrics
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='samples', zero_division=0
        )
        macro_f1 = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )[2]
        hamming = hamming_loss(all_labels, all_preds)
        
        # Per-label metrics
        per_label = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        
        # Log results
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'macro_f1': macro_f1,
            'hamming_loss': hamming,
            'pn_f1': per_label[2][0],
            'em_f1': per_label[2][1],
            'ca_f1': per_label[2][2]
        }
        log_history.append(log_entry)
        
        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={acc:.4f}, "
              f"F1={f1:.4f}, Macro-F1={macro_f1:.4f}")
        print(f"  PN F1: {per_label[2][0]:.4f}, "
              f"EM F1: {per_label[2][1]:.4f}, "
              f"CA F1: {per_label[2][2]:.4f}")
        
        # Save model checkpoint
        model_path = os.path.join(MODEL_SAVE_DIR, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), model_path)
    
    return log_history

# ==========================================
# Main Pipeline
# ==========================================
def main():
    """Main training pipeline"""
    
    print("=" * 60)
    print("CHEST CT MULTI-LABEL CLASSIFICATION")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    
    # Split data (assuming dataset column exists)
    train_df = df[df['dataset'] == 'train']
    val_df = df[df['dataset'] == 'val']
    test_df = df[df['dataset'] == 'test']
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val: {len(val_df)}")
    print(f"  Test: {len(test_df)}")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=3,
        problem_type="multi_label_classification"
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"\nUsing device: {device}")
    
    # Create datasets and loaders
    train_dataset = ChestCTDataset(train_df, tokenizer, MAX_LENGTH)
    val_dataset = ChestCTDataset(val_df, tokenizer, MAX_LENGTH)
    test_dataset = ChestCTDataset(test_df, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Train model
    print(f"\nStarting training for {EPOCHS} epochs...")
    log_history = train_model(train_loader, val_loader, model, device, EPOCHS)
    
    # Save training log
    log_df = pd.DataFrame(log_history)
    log_df.to_csv(os.path.join(OUTPUT_DIR, 'training_log.csv'), index=False)
    
    # Test evaluation
    print("\nEvaluating on test set...")
    model.eval()
    test_preds = []
    test_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask).logits
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            test_preds.append(preds.cpu().numpy())
            test_probs.append(probs.cpu().numpy())
    
    test_preds = np.vstack(test_preds)
    test_probs = np.vstack(test_probs)
    
    # Save test predictions
    test_results = pd.DataFrame({
        'report': test_df['report'].values,
        'pn_pred': test_preds[:, 0],
        'em_pred': test_preds[:, 1],
        'ca_pred': test_preds[:, 2],
        'pn_prob': test_probs[:, 0],
        'em_prob': test_probs[:, 1],
        'ca_prob': test_probs[:, 2]
    })
    test_results.to_csv(os.path.join(OUTPUT_DIR, 'test_predictions.csv'), index=False)
    
    print(f"\nTraining completed!")
    print(f"Results saved in: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()