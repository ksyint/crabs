import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveBackboneExtractor(nn.Module):
    def __init__(self, simclr_model, descriptor_dim=256, freeze_backbone=True):
        super().__init__()
        self.descriptor_dim = descriptor_dim

        backbone = simclr_model.backbone
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        self.adapter = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, descriptor_dim, 1),
        )

        self.score_net = nn.Sequential(
            nn.Conv2d(1024, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, image):
        feat_map = self.features(image)
        desc_map = self.adapter(feat_map)
        score_map = self.score_net(feat_map)
        return desc_map, score_map

    def extract_keypoints_and_descriptors(self, image, max_keypoints=1024):
        desc_map, score_map = self.forward(image)
        b, c, h, w = desc_map.shape

        scores_flat = score_map.view(b, -1)
        num_kpts = min(max_keypoints, h * w)
        topk_scores, topk_indices = torch.topk(scores_flat, num_kpts, dim=1)

        all_kpts = []
        all_descs = []
        all_scores = []

        for i in range(b):
            indices = topk_indices[i]
            kpts_y = (indices // w).float()
            kpts_x = (indices % w).float()

            scale_x = image.shape[-1] / w
            scale_y = image.shape[-2] / h
            kpts_x = kpts_x * scale_x
            kpts_y = kpts_y * scale_y

            kpts = torch.stack([kpts_x, kpts_y], dim=-1)

            grid_x = 2.0 * (kpts_x / scale_x) / max(w - 1, 1) - 1.0
            grid_y = 2.0 * (kpts_y / scale_y) / max(h - 1, 1) - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, -1, 2)
            descs = F.grid_sample(
                desc_map[i:i+1], grid, mode='bilinear', align_corners=True
            ).reshape(c, -1).t()
            descs = F.normalize(descs, p=2, dim=-1)

            all_kpts.append(kpts)
            all_descs.append(descs)
            all_scores.append(topk_scores[i])

        max_n = max(len(k) for k in all_kpts)
        padded_kpts = torch.zeros(b, max_n, 2, device=image.device)
        padded_descs = torch.zeros(b, max_n, c, device=image.device)
        padded_scores = torch.zeros(b, max_n, device=image.device)

        for i in range(b):
            n = len(all_kpts[i])
            padded_kpts[i, :n] = all_kpts[i]
            padded_descs[i, :n] = all_descs[i]
            padded_scores[i, :n] = all_scores[i]

        return {
            "keypoints": padded_kpts,
            "descriptors": padded_descs,
            "keypoint_scores": padded_scores,
            "image_size": torch.tensor([[image.shape[-1], image.shape[-2]]] * b, device=image.device).float(),
        }
