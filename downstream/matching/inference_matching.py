import torch
from torch.utils.data import DataLoader


def inference_matching(model, test_dataset, args):
    from .matching_engine import MatchingEngine

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=args.get('num_workers', 4)
    )

    engine = MatchingEngine(model, device, threshold=args.get('threshold', 0.5))
    accuracy = engine.evaluate(test_loader)

    print(f"Matching Accuracy: {accuracy:.4f}")

    return accuracy
