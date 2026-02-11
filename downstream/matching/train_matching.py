import os
import torch
from torch.utils.data import DataLoader

def train_matching(model, train_dataset, val_dataset, args):
    from trainers import MatchingTrainer
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=args.get('num_workers', 4)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=args.get('num_workers', 4)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.get('step_size', 10), gamma=0.1
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    
    trainer = MatchingTrainer(model, optimizer, scheduler, criterion, device, args)
    
    best_acc = 0.0
    for epoch in range(args['epochs']):
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")
        
        if train_acc > best_acc:
            best_acc = train_acc
            save_path = os.path.join(args['save_dir'], 'best_matching_model.pth')
            trainer.save_checkpoint(save_path, epoch, best_acc)
    
    return model
