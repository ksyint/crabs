import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


class MatchingTrainer:
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
        for img1, img2, labels in pbar:
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            labels = labels.to(self.device).float()

            with autocast(enabled=self.args.get('fp16', False)):
                feat1 = self.model(img1)
                feat2 = self.model(img2)

                feat1 = F.normalize(feat1, dim=1)
                feat2 = F.normalize(feat2, dim=1)

                similarity = F.cosine_similarity(feat1, feat2)
                loss = F.binary_cross_entropy_with_logits(similarity, labels)

            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            total_loss += loss.item()

            predictions = (torch.sigmoid(similarity) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({'loss': loss.item(), 'acc': correct / total})

        if self.scheduler:
            self.scheduler.step()

        return total_loss / len(train_loader), correct / total

    def save_checkpoint(self, save_path, epoch, best_acc):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': best_acc
        }, save_path)
