import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

class SimCLRTrainer:
    def __init__(self, model, optimizer, scheduler, criterion, device, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.args = args
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        scaler = GradScaler(enabled=self.args.get('fp16', False))
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for images, _ in pbar:
            images = torch.cat(images, dim=0).to(self.device)
            
            with autocast(enabled=self.args.get('fp16', False)):
                features = self.model(images)
                features = F.normalize(features, dim=1)
                
                batch_size = len(images) // 2
                logits, labels = self.compute_loss(features, batch_size)
                loss = self.criterion(logits, labels)
            
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        if self.scheduler:
            self.scheduler.step()
        
        return total_loss / len(train_loader)
    
    def compute_loss(self, features, batch_size):
        temperature = self.args.get('temperature', 0.5)
        
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)
        
        similarity_matrix = torch.matmul(features, features.T)
        
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        logits = logits / temperature
        
        return logits, labels
    
    def save_checkpoint(self, save_path, epoch, best_loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': best_loss
        }, save_path)
