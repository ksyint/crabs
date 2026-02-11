import os
import torch
from torch.utils.data import DataLoader
from models import SimCLRModel
from datasets import SimCLRDataset
from trainers import SimCLRTrainer
from augmentation import get_simclr_transforms, ContrastiveLearningViewGenerator
from utils import setup_logger


def main():
    args = {
        'data_dir': 'data/simclr_train',
        'save_dir': 'checkpoints/simclr',
        'batch_size': 64,
        'epochs': 100,
        'lr': 3e-4,
        'temperature': 0.5,
        'fp16': True,
        'log_every_n_steps': 100,
        'num_workers': 4
    }

    os.makedirs(args['save_dir'], exist_ok=True)
    logger = setup_logger(args['save_dir'], 'simclr_training')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    transform = ContrastiveLearningViewGenerator(
        get_simclr_transforms(size=224), n_views=2
    )

    dataset = SimCLRDataset(root_dir=args['data_dir'], transform=transform)
    train_loader = DataLoader(
        dataset,
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=args['num_workers'],
        drop_last=True
    )

    model = SimCLRModel(base_model='resnet50', out_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args['epochs']
    )
    criterion = torch.nn.CrossEntropyLoss()

    trainer = SimCLRTrainer(model, optimizer, scheduler, criterion, device, args)

    best_loss = float('inf')
    for epoch in range(args['epochs']):
        loss = trainer.train_epoch(train_loader, epoch)
        logger.info(f"Epoch {epoch}: Loss={loss:.4f}")

        if loss < best_loss:
            best_loss = loss
            save_path = os.path.join(args['save_dir'], 'best_simclr_model.pth')
            trainer.save_checkpoint(save_path, epoch, best_loss)

    logger.info("Training completed!")


if __name__ == '__main__':
    main()
