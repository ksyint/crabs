import os
import json
import torch
import torch.nn as nn
import time
import numpy as np
import torch.nn.functional as F
import cv2

# keyworkd extractors
from PIL import Image
from PIL import ImageDraw, ImageOps
from tqdm import tqdm
from torchvision import transforms

class CrabsPipeline():
    
    def __init__(self, query_dir, frame_dir, device):

        self.query_dir = query_dir
        self.frame_dir = frame_dir

        self.query_names, self.query_paths = self._load_image_names_and_paths(query_dir)
        self.frame_names, self.frame_paths = self._load_image_names_and_paths(frame_dir)
        self.device = device

    def _load_image_names_and_paths(self, directory):
        namelist = sorted(os.listdir(directory))
        imgnames = []
        imgpaths = []
        for name in namelist:
            if '.png' in name:
                imgpath = os.path.join(directory, name)
                imgpaths.append(imgpath)
                imgnames.append(name[:-4])

        return imgnames, imgpaths

    def draw_label(self, image, results):

        h,w,_ = image.shape

        xmin, ymin, xmax, ymax = 0, 0, w, h
        scores = results["scores"]
        labels = results["labels"]
        score = scores[0]
        label = labels[0]

        text=f"{label}: {round(float(score),4)}"
        font=cv2.FONT_HERSHEY_SIMPLEX
        org = (xmin, 20)
        org_max = (xmax, ymin)
        org_text = (xmin+5, 15)
        fontsize = 0.5
        color = (255,255,255)
        thick = 1

        cv2.rectangle(image,org,org_max,(0,0,255), -1)
        cv2.putText(image,text,org_text,font,fontsize,color,thick, cv2.LINE_AA)
        
        return image
    
    def initialize_matching_model(self, model):
        self.matching_model = model
        self.matching_model.to(self.device)
        self.matching_model.eval()

    def _save_query_features(self, transform=transforms.ToTensor()):
        query_inputs = []
        for query_path in self.query_paths:
            query_img = Image.open(query_path).convert('RGB')
            query_input = transform(query_img)
            query_inputs.append(query_input)

        query_inputs = torch.stack(query_inputs)
        query_inputs = query_inputs.to(self.device)
        self.query_features = self.matching_model(query_inputs)
        
    def inference_one_frame(self, frame_path, transform=transforms.ToTensor()):

        new_result_dict = dict()
        new_result_dict['scores'] = []
        new_result_dict['labels'] = []

        frame_img = Image.open(frame_path).convert('RGB')
        frame_input = torch.unsqueeze(transform(frame_img), dim=0).to(self.device)
        frame_feature = self.matching_model(frame_input)
        scores = F.cosine_similarity(frame_feature, self.query_features)
        scores = np.array(scores.data.cpu())
                          
        best_idx = np.argmax(scores)
        best_score = str(scores[best_idx])
        best_label = self.query_names[best_idx]
        new_result_dict['scores'].append(best_score)
        new_result_dict['labels'].append(best_label)

        return new_result_dict, frame_img
    

    def inference_all_frames(self, savedir, transform=transforms.ToTensor()):
        self._save_query_features(transform)

        json_savedir = os.path.join(savedir, 'json_results')
        os.makedirs(json_savedir, exist_ok=True)
        draw_savedir = os.path.join(savedir, 'draw')
        os.makedirs(draw_savedir, exist_ok=True)            

        for i, frame_path in enumerate(tqdm(self.frame_paths)):
            frame_name = self.frame_names[i]
            result_dict, frame_image = self.inference_one_frame(frame_path, transform=transform)
            savepath = os.path.join(json_savedir, frame_name + '.json')
            with open(savepath, 'w') as f:
                json.dump(result_dict, f)

            frame_image = np.array(frame_image, dtype=np.uint8)
            frame_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
            drawimg = self.draw_label(frame_image, result_dict)
            savepath = os.path.join(draw_savedir, frame_name + '.png')
            cv2.imwrite(savepath, drawimg)
            
