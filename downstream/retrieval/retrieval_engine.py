import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class RetrievalEngine:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def extract_features(self, dataloader):
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc='Extracting features'):
                images = images.to(self.device)
                features = self.model(images)
                features = F.normalize(features, dim=1)
                
                features_list.append(features.cpu())
                labels_list.append(labels)
        
        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        return features, labels
    
    def retrieve(self, query_features, gallery_features, top_k=5):
        similarity = torch.matmul(query_features, gallery_features.T)
        top_k_indices = torch.topk(similarity, k=top_k, dim=1)[1]
        return top_k_indices
    
    def evaluate(self, query_features, query_labels, gallery_features, gallery_labels, top_k=5):
        top_k_indices = self.retrieve(query_features, gallery_features, top_k)
        
        correct = 0
        total = len(query_labels)
        
        for i in range(total):
            query_label = query_labels[i]
            retrieved_labels = gallery_labels[top_k_indices[i]]
            
            if query_label in retrieved_labels:
                correct += 1
        
        accuracy = correct / total
        return accuracy
