import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

class RetrievalTrainer:
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
        correct = 0
        total = 0
        
        scaler = GradScaler(enabled=self.args.get('fp16', False))
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            with autocast(enabled=self.args.get('fp16', False)):
                features = self.model(images)
                features = F.normalize(features, dim=1)
                
                similarity = torch.matmul(features, features.T)
                loss = self.compute_retrieval_loss(similarity, labels)
            
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            predictions = torch.argmax(similarity, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
        
        if self.scheduler:
            self.scheduler.step()
        
        return total_loss / len(train_loader), correct / total
    
    def compute_retrieval_loss(self, similarity, labels):
        targets = labels.unsqueeze(0) == labels.unsqueeze(1)
        targets = targets.float().to(self.device)
        
        loss = F.binary_cross_entropy_with_logits(similarity, targets)
        return loss
    
    def save_checkpoint(self, save_path, epoch, best_metric):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': best_metric
        }, save_path)
