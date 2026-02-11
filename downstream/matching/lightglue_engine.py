import torch
import torch.nn.functional as F
from tqdm import tqdm


class LightGlueMatchingEngine:
    def __init__(self, extractor, matcher, device, min_matches=10):
        self.extractor = extractor
        self.matcher = matcher
        self.device = device
        self.min_matches = min_matches
        self.extractor.eval()
        self.matcher.eval()

    def extract_and_match(self, img0, img1):
        with torch.no_grad():
            feats0 = self.extractor({"image": img0.to(self.device)})
            feats1 = self.extractor({"image": img1.to(self.device)})

            result = self.matcher({"image0": feats0, "image1": feats1})
        return result

    def is_match(self, img0, img1):
        result = self.extract_and_match(img0, img1)
        matches = result["matches"]
        if isinstance(matches, list):
            num_matches = sum(len(m) for m in matches)
        else:
            num_matches = (result["matches0"] > -1).sum().item()
        return num_matches >= self.min_matches, num_matches

    def evaluate(self, dataloader):
        correct = 0
        total = 0

        for img0, img1, labels in tqdm(dataloader, desc='LightGlue Eval'):
            batch_size = img0.shape[0]
            for i in range(batch_size):
                matched, num_matches = self.is_match(
                    img0[i:i+1], img1[i:i+1]
                )
                pred = 1.0 if matched else 0.0
                if pred == float(labels[i]):
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0.0
