import os
import json
import torch
import torch.nn as nn
import time
import numpy as np
import torch.nn.functional as F
import cv2
import imutils
import pickle

from PIL import Image
from tqdm import tqdm
from torchvision import transforms

class CrabsVideoPipeline():
    
    def __init__(self, video_path, query_dir, device, frame_size):

        self.video_path = video_path
        self.query_dir = query_dir
        self.device = device
        self.size = frame_size

        self.query_names, self.query_paths = self._load_image_names_and_paths(query_dir)
        self.intervals = None

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

    def draw_label(self, image, query_idxes, query_scores):

        h,w,_ = image.shape

        xmin, ymin, xmax, ymax = 0, 0, w, h

        text = ""
        for i, idx in enumerate(query_idxes):
            label = self.query_names[idx]
            score = query_scores[i]
            text += f"top {i+1}: {label} {round(float(score),4)}, "
        font=cv2.FONT_HERSHEY_SIMPLEX
        org = (xmin, 20)
        org_max = (xmax, ymin)
        org_text = (xmin+5, 15)
        fontsize = 0.3
        color = (255,255,255)
        thick = 1

        cv2.rectangle(image,org,org_max,(0,0,255), -1)
        cv2.putText(image,text,org_text,font,fontsize,color,thick, cv2.LINE_AA)
        
        return image
    
    def initialize_matching_model(self, model):
        self.matching_model = model
        self.matching_model.to(self.device)
        self.matching_model.eval()

        size = self.size
        test_transform = transforms.Compose([transforms.Resize(size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(0.5, 0.5)
                                            ])
        query_inputs = []
        for query_path in self.query_paths:
            query_img = Image.open(query_path).convert('RGB')
            query_input = test_transform(query_img)
            query_inputs.append(query_input)

        query_inputs = torch.stack(query_inputs)
        query_inputs = query_inputs.to(self.device)
        self.query_features = self.matching_model(query_inputs)
        
    def detect_scene(self):

        print('spliting intervals..')

        upper_bound = 0.6
        lower_bound = 0.45
        captured = False

        # initialize the background subtractor
        fgbg = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=3) 
        # note that detect_scene not works for the frame in scene change whose index is smaller thant initializationFrames

        vs = cv2.VideoCapture(self.video_path)
        self.frame_count = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(vs.get(cv2.CAP_PROP_FPS))

        (W, H) = (None, None)
        
        scene_bdy = []

        for idx in tqdm(range(self.frame_count)):
            # ret: whether the frame is read correctly or not
            ret, frame = vs.read()
            if frame is None:
                break
            if idx%2==0:
                frame = imutils.resize(frame, width=64, height=128)
                mask = fgbg.apply(frame)

                # remove noise
                mask = cv2.erode(mask, None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=2)

                if W is None or H is None:
                    (W, H) = mask.shape[:2]

                # black: 0, white/changing part: 1
                p = (cv2.countNonZero(mask)/float(W * H))

                if p > upper_bound and not captured:
                    captured = True
                    scene_bdy.append(idx)
                elif captured and p < lower_bound:
                    # return to capturing mode
                    captured = False

        # When everything done, release the capture
        scene_bdy.append(idx)
        scene_bdy.sort()
        scene_bdy = scene_bdy[1:]
        scene_bdy.append(0)
        scene_bdy.sort()

        return scene_bdy
    
    def video_inference(self):

        if self.intervals==None:
            self.intervals = self.detect_scene()

        size = self.size

        test_transform = transforms.Compose([transforms.Resize(size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(0.5, 0.5)
                                            ])
        vs = cv2.VideoCapture(self.video_path)
        
        interval_results = np.zeros((len(self.intervals)-1, len(self.query_features)))
        interval_scores = np.zeros(len(self.query_features))
        interval_count = 0
        interval_idx = 1

        for frame_idx in tqdm(range(self.frame_count)):

            ret, frame = vs.read()

            if self.intervals[interval_idx] < frame_idx:
                interval_results[interval_idx-1] = interval_scores/interval_count
                interval_scores = np.zeros([len(self.query_features)])
                interval_count = 0
                interval_idx+=1

            if frame_idx%5==0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame).convert('RGB')  
                frame_input = torch.unsqueeze(test_transform(frame), dim=0).to(self.device)
                frame_feature = self.matching_model(frame_input)
                scores = F.cosine_similarity(frame_feature, self.query_features)
                scores = np.array(scores.data.cpu())
                interval_scores += scores
                interval_count+=1

        self.interval_results = interval_results
        return interval_results
    
    
    def video_inference2(self):

        size = (224,224)
        test_transform = transforms.Compose([transforms.Resize(size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(0.5, 0.5)
                                            ])
        scores = np.zeros((self.query_features.size()[0],len(self.frames)))
        
        for i in tqdm(range(self.query_features.size()[0])):
            temp_query_feature = self.query_features[i]
            temp = np.zeros(len(self.frames))
            for idx in range(len(self.frames)):
                frame_input = torch.unsqueeze(test_transform(self.frames[idx]), dim=0).to(self.device)
                frame_feature = self.matching_model(frame_input)
                score = F.cosine_similarity(frame_feature, temp_query_feature)
                score = np.array(score.data.cpu())
                temp[idx] = score
                
            scores[i] = temp
            
        scores_idx = np.zeros((self.query_features.size()[0], 50), dtype=np.int64)
        for i in tqdm(range(scores.shape[0])):
            temp_idx = np.argsort(scores[i])[::-1][:50]
            scores_idx[i] = temp_idx

        sorted_scores_idx=np.zeros((scores_idx.shape[0], scores_idx.shape[1]), dtype=np.int64)

        for i in range(scores_idx.shape[0]):
            sorted_scores_idx[i] = sorted(scores_idx[i])
        self.total_scores = scores
        self.scores_idx = scores_idx
        self.sorted_scores_idx = sorted_scores_idx
        
        return sorted_scores_idx
    

    def parse_output(self, output_path, k=3):

        os.makedirs(output_path, exist_ok=True)
        topk_scores = []
        sorted_idx = np.flip(np.argsort(self.interval_results), axis=1)
        self.topk = sorted_idx[:,:k]
        for i in range(len(self.interval_results)):
            topk_score = self.interval_results[i][self.topk[i]]
            topk_scores.append(topk_score)
        self.topk_scores = np.array(topk_scores, dtype=np.float32)

        self.time_intervals = []
        for idx in self.intervals:
            time = idx/self.fps
            self.time_intervals.append(time)

        interval_idx = 1
        vs = cv2.VideoCapture(self.video_path)
        # width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))//2
        height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))//2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        outvideo_path = os.path.join(output_path, 'output.mp4')
        outvideo = cv2.VideoWriter(outvideo_path, fourcc, self.fps, (width, height))

        for idx in tqdm(range(self.frame_count)):
            ret, frame = vs.read()
            frame = cv2.resize(frame, (width,height))
            
            if self.intervals[interval_idx] < idx:
                interval_idx+=1
            
            newframe = self.draw_label(frame, self.topk[interval_idx-1], self.topk_scores[interval_idx-1])
            outvideo.write(newframe)
        outvideo.release()
        
        results = dict()
        results['intervals'] = self.intervals
        results['time_intervals'] = self.time_intervals
        results['query_names'] = self.query_names
        results['topk'] = self.topk
        results['topk_scores'] = self.topk_scores
        save_dict = os.path.join(output_path, 'result.pickle')
        with open(save_dict, 'wb') as f:
            pickle.dump(results, f)


    def export_result_video(self, video_path, output_path, k=3):
        
        result_path = os.path.join(output_path, 'result.pickle')

        with open(result_path, 'rb') as f:
            results = pickle.load(f)

        intervals = results['intervals']
        # time_intervals = results['time_intervals']
        # query_names = results['query_names']
        topk = results['topk']
        topk_scores = results['topk_scores']

        interval_idx = 1
        vs = cv2.VideoCapture(video_path)
        width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))//2
        height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))//2
        fps = int(vs.get(cv2.CAP_PROP_FPS))
        frame_count = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        outvideo_path = os.path.join(output_path, 'output.mp4')
        outvideo = cv2.VideoWriter(outvideo_path, fourcc, fps, (width, height))

        for idx in tqdm(range(frame_count)):
            ret, frame = vs.read()
            frame = cv2.resize(frame, (width,height))

            if intervals[interval_idx] < idx:
                interval_idx+=1
            
            newframe = self.draw_label(frame, topk[interval_idx-1], topk_scores[interval_idx-1])
            outvideo.write(newframe)
        outvideo.release()
        