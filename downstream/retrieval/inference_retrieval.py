import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

def inference_retrieval(model, query_dataset, gallery_dataset, args):
    from .retrieval_engine import RetrievalEngine
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    query_loader = DataLoader(
        query_dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=args.get('num_workers', 4)
    )
    
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=args.get('num_workers', 4)
    )
    
    engine = RetrievalEngine(model, device)
    
    query_features, query_labels = engine.extract_features(query_loader)
    gallery_features, gallery_labels = engine.extract_features(gallery_loader)
    
    accuracy = engine.evaluate(
        query_features, query_labels,
        gallery_features, gallery_labels,
        top_k=args.get('top_k', 5)
    )
    
    print(f"Retrieval Accuracy@{args.get('top_k', 5)}: {accuracy:.4f}")
    
    return accuracy
