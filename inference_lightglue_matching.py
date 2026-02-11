import os
import argparse
import torch
from torch.utils.data import DataLoader

from models import SimCLRModel
from core import ContrastiveBackboneExtractor
from matcher import LightGlue
from downstream.matching import LightGlueMatchingEngine
from datasets import ImagePairDataset
from utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simclr_ckpt', default='checkpoints/simclr/best_simclr_model.pth')
    parser.add_argument('--test_dir', default='data/matching_test')
    parser.add_argument('--descriptor_dim', type=int, default=256)
    parser.add_argument('--max_keypoints', type=int, default=1024)
    parser.add_argument('--min_matches', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    simclr_model = SimCLRModel(base_model='resnet50', out_dim=128)
    if os.path.exists(args.simclr_ckpt):
        load_checkpoint(simclr_model, args.simclr_ckpt)
        print(f"Loaded SimCLR backbone from {args.simclr_ckpt}")

    extractor = ContrastiveBackboneExtractor(
        simclr_model, descriptor_dim=args.descriptor_dim
    ).to(device)

    matcher = LightGlue(
        features=None,
        input_dim=args.descriptor_dim,
        descriptor_dim=args.descriptor_dim,
        weights=None,
    ).to(device)

    engine = LightGlueMatchingEngine(
        extractor=_ExtractorWrapper(extractor, args.max_keypoints),
        matcher=matcher,
        device=device,
        min_matches=args.min_matches,
    )

    dataset = ImagePairDataset(root_dir=args.test_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    accuracy = engine.evaluate(dataloader)
    print(f"LightGlue Matching Accuracy: {accuracy:.4f}")


class _ExtractorWrapper:
    def __init__(self, backbone_extractor, max_keypoints):
        self.backbone_extractor = backbone_extractor
        self.max_keypoints = max_keypoints

    def eval(self):
        self.backbone_extractor.eval()
        return self

    def __call__(self, data):
        image = data["image"]
        return self.backbone_extractor.extract_keypoints_and_descriptors(
            image, max_keypoints=self.max_keypoints
        )


if __name__ == '__main__':
    main()
