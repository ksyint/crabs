import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class ImagePairDataset(Dataset):
    def __init__(self, root_dir, pairs_file=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform or T.Compose([
            T.Resize((480, 640)),
            T.ToTensor(),
        ])

        self.pairs = []
        self.labels = []

        if pairs_file and os.path.exists(pairs_file):
            with open(pairs_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        self.pairs.append((parts[0], parts[1]))
                        self.labels.append(int(parts[2]))
        else:
            self._auto_generate_pairs()

    def _auto_generate_pairs(self):
        label_dirs = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])

        label_images = {}
        for idx, label in enumerate(label_dirs):
            folder = os.path.join(self.root_dir, label)
            imgs = sorted([
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            label_images[idx] = imgs

        all_labels = list(label_images.keys())
        for label in all_labels:
            imgs = label_images[label]
            for i in range(len(imgs)):
                for j in range(i + 1, min(i + 3, len(imgs))):
                    self.pairs.append((imgs[i], imgs[j]))
                    self.labels.append(1)

            for _ in range(len(imgs)):
                img1 = np.random.choice(imgs)
                neg_label = np.random.choice([l for l in all_labels if l != label])
                img2 = np.random.choice(label_images[neg_label])
                self.pairs.append((img1, img2))
                self.labels.append(0)

    def __getitem__(self, index):
        path0, path1 = self.pairs[index]
        label = self.labels[index]

        img0 = Image.open(path0).convert('RGB')
        img1 = Image.open(path1).convert('RGB')

        img0 = self.transform(img0)
        img1 = self.transform(img1)

        return img0, img1, label

    def __len__(self):
        return len(self.pairs)
