import os
import torch
from models import SimCLRModel
from downstream.matching import train_matching, inference_matching
from datasets import SimCLRDataset, MatchingDataset
from trainers import SimCLRTrainer
from augmentation import get_simclr_transforms, get_standard_transforms, ContrastiveLearningViewGenerator
from torch.utils.data import DataLoader
from utils import setup_logger, load_checkpoint


def train_simclr_stage(args):
    logger = setup_logger(args['simclr_save_dir'], 'simclr')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = ContrastiveLearningViewGenerator(
        get_simclr_transforms(size=224), n_views=2
    )
    dataset = SimCLRDataset(root_dir=args['simclr_data_dir'], transform=transform)
    train_loader = DataLoader(
        dataset, batch_size=args['batch_size'], shuffle=True,
        num_workers=args['num_workers'], drop_last=True
    )

    model = SimCLRModel(base_model='resnet50', out_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['simclr_epochs'])
    criterion = torch.nn.CrossEntropyLoss()

    trainer = SimCLRTrainer(model, optimizer, scheduler, criterion, device, args)

    best_loss = float('inf')
    for epoch in range(args['simclr_epochs']):
        loss = trainer.train_epoch(train_loader, epoch)
        logger.info(f"SimCLR Epoch {epoch}: Loss={loss:.4f}")

        if loss < best_loss:
            best_loss = loss
            save_path = os.path.join(args['simclr_save_dir'], 'best_simclr_model.pth')
            trainer.save_checkpoint(save_path, epoch, best_loss)

    return model


def train_matching_stage(pretrained_model, args):
    logger = setup_logger(args['matching_save_dir'], 'matching')

    transform = get_standard_transforms(size=224)
    train_dataset = MatchingDataset(args['matching_train_dir'], transform, 'train')
    val_dataset = MatchingDataset(args['matching_val_dir'], transform, 'val')

    model = train_matching(pretrained_model, train_dataset, val_dataset, args)
    logger.info("Matching training completed")

    return model


def main():
    args = {
        'simclr_data_dir': 'data/simclr_train',
        'simclr_save_dir': 'checkpoints/simclr',
        'simclr_epochs': 100,
        'matching_train_dir': 'data/matching_train',
        'matching_val_dir': 'data/matching_val',
        'matching_save_dir': 'checkpoints/matching',
        'batch_size': 64,
        'epochs': 50,
        'lr': 3e-4,
        'temperature': 0.5,
        'fp16': True,
        'num_workers': 4,
        'step_size': 10
    }

    for key in ['simclr_save_dir', 'matching_save_dir']:
        os.makedirs(args[key], exist_ok=True)

    print("Stage 1: Training SimCLR (Contrastive Learning)")
    simclr_model = train_simclr_stage(args)

    print("Stage 2: Training Image Matching (Downstream Task)")
    matching_model = train_matching_stage(simclr_model, args)

    print("All stages completed!")


if __name__ == '__main__':
    main()
