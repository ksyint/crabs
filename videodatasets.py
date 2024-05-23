import os
import json
import numpy as np
from PIL import Image

import cv2
import time

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from tqdm import tqdm

class VideoDataset(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, video_dir, transform=None):
        
        self.video_dir = video_dir
        self.transform = transform
        
        self.images_name, self.images = self.get_video(self.video_dir)

    def __getitem__(self, index):
        cv2_img = self.images[index]
        img_name = self.images_name[index]
        
        img = Image.fromarray(cv2_img).convert('RGB')        
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img_name, img, index
        
    def __len__(self):
        return self.images.shape[0]
    
    def get_video(self, video_path):
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_data = []
        image_data_name = []
        frame = 0
        video_name_list = video_path.split('/')
        
        # while success:
        #     if frame % 10 == 0:
        #         image_data.append(image)
        #         image_data_name.append(video_name_list[-1][:-4] + '_' + str(frame))
        #     success, image = vidcap.read()
        #     if success == True:
        #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #     frame += 1
        while success:
            image_data.append(image)
            image_data_name.append(video_name_list[-1][:-4] + '_' + '{:08d}'.format(frame))
            
            success, image = vidcap.read()
            if success == True:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       
            frame += 1
            
        image_data_numpy = np.array(image_data)
        
        return image_data_name, image_data_numpy

class ImageDataset(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, image_dir, transform=None):

        self.image_dir = image_dir
        self.transform = transform
        
        self.image_path = os.listdir(self.image_dir)

    def __getitem__(self, index):
        imgpath = os.path.join(self.image_dir, self.image_path[index])
        img = Image.open(imgpath).convert('RGB')
        img_name = self.image_path[index][:-4]
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img_name, img, index

    def __len__(self):
        return len(self.image_path)

# v1 = VideoDataset('./crabs_dataset/crabs_videos/Celebshop9_07:20-11:40.mp4')
# v_list = list(v1)
# print(v_list)