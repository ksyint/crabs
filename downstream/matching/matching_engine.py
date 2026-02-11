import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class MatchingEngine:
    def __init__(self, model, device, threshold=0.5):
        self.model = model
        self.device = device
        self.threshold = threshold
        self.model.eval()
    
    def compute_similarity(self, img1, img2):
        with torch.no_grad():
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            
            feat1 = self.model(img1)
            feat2 = self.model(img2)
            
            feat1 = F.normalize(feat1, dim=1)
            feat2 = F.normalize(feat2, dim=1)
            
            similarity = F.cosine_similarity(feat1, feat2)
            
        return similarity
    
    def match(self, img1, img2):
        similarity = self.compute_similarity(img1, img2)
        is_match = (similarity > self.threshold).float()
        return is_match, similarity
    
    def evaluate(self, dataloader):
        correct = 0
        total = 0
        
        for img1, img2, labels in tqdm(dataloader, desc='Evaluating'):
            is_match, similarity = self.match(img1, img2)
            
            predictions = (similarity > self.threshold).float()
            correct += (predictions.cpu() == labels.float()).sum().item()
            total += labels.size(0)
        
        accuracy = correct / total
        return accuracy
