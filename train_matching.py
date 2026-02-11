import os
import torch
from torch.utils.data import DataLoader
from models import SimCLRModel
from datasets import MatchingDataset
from downstream.matching import train_matching
from augmentation import get_standard_transforms
from utils import setup_logger, load_checkpoint


def main():
    args = {
        'pretrained_path': 'checkpoints/simclr/best_simclr_model.pth',
        'data_dir': 'data/matching_train',
        'val_dir': 'data/matching_val',
        'save_dir': 'checkpoints/matching',
        'batch_size': 64,
        'epochs': 50,
        'lr': 1e-4,
        'step_size': 10,
        'num_workers': 4
    }

    os.makedirs(args['save_dir'], exist_ok=True)
    logger = setup_logger(args['save_dir'], 'matching_training')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model = SimCLRModel(base_model='resnet50', out_dim=128)

    if os.path.exists(args['pretrained_path']):
        load_checkpoint(model, args['pretrained_path'])
        logger.info(f"Loaded pretrained model from {args['pretrained_path']}")
    else:
        logger.warning("Pretrained model not found, training from scratch")

    transform = get_standard_transforms(size=224)

    train_dataset = MatchingDataset(
        root_dir=args['data_dir'],
        transform=transform,
        mode='train'
    )

    val_dataset = MatchingDataset(
        root_dir=args['val_dir'],
        transform=transform,
        mode='val'
    )

    model = train_matching(model, train_dataset, val_dataset, args)

    logger.info("Downstream matching training completed!")


if __name__ == '__main__':
    main()
