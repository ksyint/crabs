import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class SimCLRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        self.image_label_dict = {}
        self.label_image_dict = {}

        self.labelnames = sorted(os.listdir(root_dir))
        for label in self.labelnames:
            folderpath = os.path.join(root_dir, label)
            if not os.path.isdir(folderpath):
                continue

            namelist = sorted(os.listdir(folderpath))
            imgpaths = []
            for name in namelist:
                if not (name.endswith('.png') or name.endswith('.jpg') or name.endswith('.jpeg')):
                    continue
                imgpath = os.path.join(folderpath, name)
                imgpaths.append(imgpath)
                self.image_paths.append(imgpath)
                self.image_label_dict[imgpath] = label
            self.label_image_dict[label] = imgpaths

    def __getitem__(self, index):
        label_index = index
        label = self.labelnames[label_index]
        imgpath_list = self.label_image_dict[label]

        num_img_choices = len(imgpath_list)
        idx_a = np.random.randint(num_img_choices)
        idx_b = np.random.randint(num_img_choices)
        imgpath_a = imgpath_list[idx_a]
        imgpath_b = imgpath_list[idx_b]

        img1 = Image.open(imgpath_a).convert('RGB')
        img2 = Image.open(imgpath_b).convert('RGB')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return [img1, img2], label_index

    def __len__(self):
        return len(self.labelnames)
