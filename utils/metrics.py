import torch

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res

def compute_recall_at_k(query_labels, gallery_labels, top_k_indices, k=1):
    correct = 0
    total = len(query_labels)
    
    for i in range(total):
        query_label = query_labels[i]
        retrieved_labels = gallery_labels[top_k_indices[i][:k]]
        
        if query_label in retrieved_labels:
            correct += 1
    
    recall = correct / total
    return recall
