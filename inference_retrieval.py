import os
import torch
from models import SimCLRModel
from datasets import RetrievalDataset
from downstream.retrieval import inference_retrieval
from augmentation import get_standard_transforms
from utils import load_checkpoint

def main():
    args = {
        'model_path': 'checkpoints/retrieval/best_retrieval_model.pth',
        'query_dir': 'data/retrieval_query',
        'gallery_dir': 'data/retrieval_gallery',
        'batch_size': 64,
        'top_k': 5,
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
    
    query_dataset = RetrievalDataset(
        root_dir=args['query_dir'],
        transform=transform,
        mode='test'
    )
    
    gallery_dataset = RetrievalDataset(
        root_dir=args['gallery_dir'],
        transform=transform,
        mode='test'
    )
    
    accuracy = inference_retrieval(model, query_dataset, gallery_dataset, args)
    
    print(f"Final Retrieval Accuracy@{args['top_k']}: {accuracy:.4f}")

if __name__ == '__main__':
    main()
