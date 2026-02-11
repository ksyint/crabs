import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class MatchingDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        self.pairs = []
        self.labels = []
        
        labelnames = sorted(os.listdir(root_dir))
        self.label_to_images = {}
        
        for label_idx, label in enumerate(labelnames):
            folderpath = os.path.join(root_dir, label)
            if not os.path.isdir(folderpath):
                continue
            
            namelist = sorted(os.listdir(folderpath))
            images = []
            for name in namelist:
                if not (name.endswith('.png') or name.endswith('.jpg') or name.endswith('.jpeg')):
                    continue
                imgpath = os.path.join(folderpath, name)
                images.append(imgpath)
            
            self.label_to_images[label_idx] = images
        
        self._create_pairs()
    
    def _create_pairs(self):
        all_labels = list(self.label_to_images.keys())
        
        for label in all_labels:
            images = self.label_to_images[label]
            
            for i in range(len(images)):
                for j in range(i+1, len(images)):
                    self.pairs.append((images[i], images[j]))
                    self.labels.append(1)
            
            num_negative = len(images) * 2
            for _ in range(num_negative):
                img1 = np.random.choice(images)
                neg_label = np.random.choice([l for l in all_labels if l != label])
                img2 = np.random.choice(self.label_to_images[neg_label])
                self.pairs.append((img1, img2))
                self.labels.append(0)
    
    def __getitem__(self, index):
        img1_path, img2_path = self.pairs[index]
        label = self.labels[index]
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, label
    
    def __len__(self):
        return len(self.pairs)
