import os
import torch
from models import SimCLRModel
from datasets import MatchingDataset
from downstream.matching import inference_matching
from augmentation import get_standard_transforms
from utils import load_checkpoint

def main():
    args = {
        'model_path': 'checkpoints/matching/best_matching_model.pth',
        'test_dir': 'data/matching_test',
        'batch_size': 64,
        'threshold': 0.5,
        'num_workers': 4
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SimCLRModel(base_model='resnet50', out_dim=128)
    
    if os.path.exists(args['model_path']):
        load_checkpoint(model, args['model_path'])
        print(f"Loaded model from {args['model_path']}")
    else:
        raise FileNotFoundError(f"Model not found: {args['model_path']}")
    
    transform = get_standard_transforms(size=224)
    
    test_dataset = MatchingDataset(
        root_dir=args['test_dir'],
        transform=transform,
        mode='test'
    )
    
    accuracy = inference_matching(model, test_dataset, args)
    
    print(f"Final Matching Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()
