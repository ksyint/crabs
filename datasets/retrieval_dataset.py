import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class RetrievalDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        self.image_paths = []
        self.labels = []
        
        labelnames = sorted(os.listdir(root_dir))
        for label_idx, label in enumerate(labelnames):
            folderpath = os.path.join(root_dir, label)
            if not os.path.isdir(folderpath):
                continue
            
            namelist = sorted(os.listdir(folderpath))
            for name in namelist:
                if not (name.endswith('.png') or name.endswith('.jpg') or name.endswith('.jpeg')):
                    continue
                imgpath = os.path.join(folderpath, name)
                self.image_paths.append(imgpath)
                self.labels.append(label_idx)
    
    def __getitem__(self, index):
        imgpath = self.image_paths[index]
        label = self.labels[index]
        
        image = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    
    def __len__(self):
        return len(self.image_paths)
