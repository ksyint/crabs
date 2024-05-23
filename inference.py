import os
import json
import numpy as np
import argparse
import torch
import cv2
import time

from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from resnet_simclr import ResNetSimCLR
from videodatasets import VideoDataset
from crabs_pipeline import CrabsPipeline


def extract_frames(video_dir, save_dir):
    vidcap = cv2.VideoCapture(video_dir)
    success, image = vidcap.read()
    os.makedirs(save_dir, exist_ok=True)

    frame = 0
    video_name = video_dir.split('/')[-1][:-4]
    
    while success:
        frame_name = video_name + '_' + '{:08d}'.format(frame) + '.png'
        cv2.imwrite(os.path.join(save_dir, frame_name), image)
        
        success, image = vidcap.read()
        frame += 1
            
    # video = VideoDataset(os.path.join(video_dir))
    
    # video_frames = list(video)
    
    # os.makedirs(save_dir, exist_ok=True)
    
    # for i in tqdm(range(len(video_frames))):
    #     video_frames[i][1].save(os.path.join(save_dir, video_frames[i][0] + '.png'))


def decode_video(frame_dir, output_path):

    namelist = sorted(os.listdir(frame_dir))
    tmp_img = cv2.imread(os.path.join(frame_dir, namelist[0]))

    height,width,layers=tmp_img.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(output_path, fourcc, 24, (width, height))

    for name in namelist:
        imgpath = os.path.join(frame_dir, name)
        img = cv2.imread(imgpath)
        video.write(img)

    # cv2.destroyAllWindows()
    video.release()

def main(args):

    if args.parse_video:
        start = time.time()
        extract_frames(args.video_dir, args.frame_dir)
        print('Saving time is ' + str(time.time()-start))
    
    start = time.time()
    pipeline = CrabsPipeline(args.pannel_dir, args.frame_dir, args.device)

    size = (args.size, args.size)
    test_transform = transforms.Compose([transforms.Resize(size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(0.5, 0.5)
                                        ])
    
    arch = args.matching_arch
    out_dim = 128
    matching_model = ResNetSimCLR(arch, out_dim)
    matching_model.load_state_dict(torch.load(args.matching_model_path, map_location='cpu')['state_dict'])

    pipeline.initialize_matching_model(matching_model)

    pipeline.inference_all_frames(args.save_path, transform=test_transform)

    print('time is ' + str(time.time()-start))
    
    if args.make_video_output:
        result_frame_dir = os.path.join(args.save_path, 'draw')
        output_video_path = os.path.join(args.save_path, 'out_video.mp4')
        decode_video(result_frame_dir, output_video_path)


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Image matching')
    # SimCLR test에 사용될 데이터의 dir
    parser.add_argument('--test_dataset', default='../../dataset/crabs_dataset/crops/test/', type=str)
    parser.add_argument('--size', default=224, type=int)
    parser.add_argument('--cuda', default=True, type=bool)

    ################################################################

    parser.add_argument('--pannel_dir', default='../crabs_dataset/eval_videos/fullvideo_test6/tmp_panels', type=str, nargs='+')
    parser.add_argument('--frame_dir', default='../crabs_dataset/eval_videos/fullvideo_test6/sample6', type=str, nargs='+')
    parser.add_argument('--matching_arch', default='resnet18', type=str)
    parser.add_argument('--matching_model_path', default='./models/matching_model003.tar', type=str)
    parser.add_argument('--save_path', default='./output/sample6_001/', type=str)

    parser.add_argument('--caption_model', default='BLIP2', type=str, nargs='+')
    parser.add_argument('--detection_model', default='OWL-VIT', type=str, nargs='+')

    # cpu / cuda
    parser.add_argument('--device', default='cuda:0', help='device to use for training / testing, only cuda supported')

    # if perform video decoding
    parser.add_argument('--parse_video', default=True, type=bool, help='whether parsing video or not')
    parser.add_argument('--make_video_output', default=True, type=bool, help='whether parsing video or not')
    parser.add_argument('--video_dir', default='../crabs_dataset/eval_videos/fullvideo_test6/sample6.mp4', type=str, nargs='+')
    parser.add_argument('--video_save_dir', default='../crabs_dataset/eval_videos/fullvideo_test6/sample6', type=str, nargs='+')

    args = parser.parse_args()
    main(args)

    