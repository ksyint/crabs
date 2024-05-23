
import argparse
import torch

from resnet_simclr import ResNetSimCLR
from video_pipeline import CrabsVideoPipeline

def main(args):
    
    pipeline = CrabsVideoPipeline(args.video_path, args.panel_dir, args.device, args.frame_size)

    arch = 'resnet18'
    out_dim = 128
    matching_model = ResNetSimCLR(arch, out_dim)
    matching_model.load_state_dict(torch.load(args.model_path, map_location='cpu')['state_dict'])

    pipeline.initialize_matching_model(matching_model)

    pipeline.video_inference()
    pipeline.parse_output(args.output_path)

    # pipeline.export_result_video(args.video_path, args.output_path)


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Image matching')

    parser.add_argument('--device', default='cuda:0', help='device to use for training / testing, only cuda supported')

    parser.add_argument('--video_path', default='../crabs_dataset/ssg_video006/ssg_video006.mp4', type=str)
    parser.add_argument('--panel_dir', default='../crabs_dataset/ssg_video006/newpanels/', type=str)
    parser.add_argument('--model_path', default='./models/matching_model007_fewshot2.tar', type=str)
    parser.add_argument('--output_path', default='./output/ssg_video006_4', type=str)
    parser.add_argument('--frame_size', default=(320,180), type=tuple)

    args = parser.parse_args()
    main(args)

    